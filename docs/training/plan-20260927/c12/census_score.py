#!/usr/bin/env python3
"""Score the C12 census with reward v2.1 and build the R4 inputs (PC side, CPU only).

Reads the fetched Kaggle census shards of ``scripts/kaggle/vllm_census_entry.py
--census`` (``<run>/chunks/chunk-NNNN.jsonl``, one record per prompt and
sample) and the DAT10 v4-braceindent TRAIN rows that reward v2.1 now binds
(``reward_v21.load_train_specs``). Checks before scoring:

- every census prompt hash equals the hash of the TRAIN row's prompt ids
  (``input_ids[:target_start]``), so the samples belong to the rows scored;
- every TRAIN row has exactly k samples, numbered 0..k-1, and nothing else
  appears (no DEV row, no unknown id).

Every sample is scored with ``RewardV21Scorer`` (default configuration:
partial credit "gain", weight 0.5, cap 1,024). Rules and pass rates use
``exact`` only. Outputs in ``--out-dir``:

- ``census-prompts.jsonl``: id, family, census global index and seed, prompt
  and target token ids, prompt sha256;
- ``samples.jsonl``: the merged generations (ids include the stop id);
- ``scores.jsonl``: one reward v2.1 record per sample, plus its sample index,
  global index and decoded text;
- ``pass-rates.jsonl``: per prompt, k, exact count, pass rate, mean reward,
  outcome counts, cap hits and mean completion length;
- ``rft-candidates.jsonl``: verified-correct samples (``exact``), deduplicated
  per prompt by generated ids, at most 2 per prompt, with families balanced to
  the TRAIN mix (largest total for which every family's quota fits its
  supply; prompts drawn in a seeded order within each family; a family with
  no verified-correct sample is dropped and the shares renormalized over the
  rest, recorded in the summary);
- ``dpo-pair-candidates.jsonl``: chosen is a distinct verified-correct sample
  of the same prompt, or the gold target if the prompt has none; rejected is a
  distinct incorrect sample that is protocol-valid (parses under the PRM03
  output contract: near misses, missed edits, syntax-invalid edits and, on
  no-op prompts, any suggested edit). Cap hits and protocol-invalid outputs
  are never rejected samples. At most 2 pairs per prompt, model-generated
  chosen samples first; rejected samples whose applied edit does not parse in
  R are used only after the others;
- ``census-summary.json``: the census statistics overall and per family
  (exact pass-rate distribution, ``M`` = share of prompts with 0 < pass < 1,
  all-correct and all-wrong shares, no-op false-suggestion rate, cap hits,
  mean completion length), the candidate counts, the R4 step 1 inputs, the
  scorer identity and input hashes.

Example::

    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python \\
        docs/training/plan-20260927/c12/census_score.py \\
        --run ~/.local/state/sepalith/kaggle/runs/c12-census-s0/c12-census-s0 \\
        --run ~/.local/state/sepalith/kaggle/runs/c12-census-s1/c12-census-s1 \\
        --run ~/.local/state/sepalith/kaggle/runs/c12-census-s2/c12-census-s2 \\
        --tokenizer <checkpoint>/tokenizer.json --out-dir ~/.local/state/sepalith/c12/census-arm-a
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from sepalith.training.rl.air_canonical import AirFormatter
from sepalith.training.rl.campaign_r_parse_probe import DEFAULT_HARNESS, RParseOnlyProbe, sha_file
from sepalith.training.rl.reward_v21 import (
    DATA_ROOT,
    TRAIN_INPUTS,
    RewardV21Config,
    RewardV21Scorer,
    Sample,
    load_train_specs,
    tokenizer_decoder,
)

SCHEMA = "sepalith.c12.census-summary.v1"
TRAIN_ROWS = 15006
SEED = 3407
MAX_PER_PROMPT = 2
# R4 step 1 (RULES.md): the offline route needs these counts.
R4_MIN_VERIFIED = 2000
R4_MIN_PAIRS = 1000
REJECT_PREFERRED = ("near_miss", "wrong_parse_unverified", "missed_edit", "false_positive_edit")
REJECT_LAST = ("applied_syntax_invalid",)


def sha256_ids(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sha_path(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_samples(runs):
    samples, shards = [], []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        chunks = sorted((run / "chunks").glob("chunk-*.jsonl"))
        for path in chunks:
            samples.extend(read_jsonl(path))
        shards.append({"run": str(run), "status": summary.get("status"), "shard": summary["inputs"]["shard"],
                       "chunks_files": len(chunks), "chunks": summary.get("chunks"),
                       "order": summary["inputs"]["order"], "train_rows": summary["inputs"]["train_rows"],
                       "model": summary.get("model"), "throughput": summary.get("throughput"),
                       "vllm": [w.get("vllm") for w in summary.get("vllm_workers", [])],
                       "engine": [w.get("config", {}).get("name") for w in summary.get("vllm_workers", [])],
                       "elapsed_seconds": summary.get("elapsed_seconds")})
    return samples, shards


def check_coverage(samples, rows, k):
    by_prompt = collections.defaultdict(list)
    for sample in samples:
        if sample["id"] not in rows:
            raise SystemExit(f"census sample for unknown TRAIN id {sample['id']}")
        by_prompt[sample["id"]].append(sample)
    problems = []
    for row_id, row in rows.items():
        got = by_prompt.get(row_id, [])
        if sorted(s["sample"] for s in got) != list(range(k)):
            problems.append(row_id)
            continue
        prompt = row["prompt_ids"]
        if any(s["prompt_sha256"] != sha256_ids(prompt) or s["prompt_tokens"] != len(prompt) for s in got):
            raise SystemExit(f"census prompt hash for {row_id} differs from the TRAIN row's prompt")
    return by_prompt, problems


def distribution(counts, k):
    hist = collections.Counter(counts)
    n = len(counts)
    return {
        "prompts": n,
        "pass_count_histogram": {str(i): hist.get(i, 0) for i in range(k + 1)},
        "M_mixed_share": round(sum(0 < c < k for c in counts) / n, 6) if n else None,
        "all_correct_share": round(sum(c == k for c in counts) / n, 6) if n else None,
        "all_wrong_share": round(sum(c == 0 for c in counts) / n, 6) if n else None,
        "mean_pass_rate": round(sum(counts) / (k * n), 6) if n else None,
    }


def group_stats(prompt_rows, score_rows, k):
    out = distribution([p["exact_count"] for p in prompt_rows], k)
    lengths = [s["generated_tokens"] for s in score_rows]
    noop = [s for s in score_rows if s["expected_operation"] == "no_op"]
    out.update(
        samples=len(score_rows),
        exact_samples=sum(bool(s["exact"]) for s in score_rows),
        mean_reward=round(statistics.fmean(float(s["reward"]) for s in score_rows), 6) if score_rows else None,
        cap_hits=sum(bool(s["cap_hit"]) for s in score_rows),
        cap_hit_rate=round(sum(bool(s["cap_hit"]) for s in score_rows) / len(score_rows), 6) if score_rows else None,
        mean_completion_tokens=round(statistics.fmean(lengths), 2) if lengths else None,
        median_completion_tokens=statistics.median(lengths) if lengths else None,
        protocol_invalid=sum(s["outcome"] == "protocol_invalid" for s in score_rows),
        outcomes=dict(collections.Counter(str(s["outcome"]) for s in score_rows)),
        comparisons=dict(collections.Counter(str(s["comparison"]) for s in score_rows)),
        canonical_fallback=sum(bool(s["canonical_fallback"]) for s in score_rows),
    )
    if noop:
        out["noop_samples"] = len(noop)
        out["noop_false_suggestions"] = sum(s["outcome"] == "false_positive_edit" for s in noop)
        out["noop_false_suggestion_rate"] = round(out["noop_false_suggestions"] / len(noop), 6)
    return out


def distinct(rows):
    seen, out = set(), []
    for row in rows:
        key = tuple(row["generated_ids"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def rft_candidates(by_prompt_scores, family_of, train_mix):
    supply = {}
    per_prompt = {}
    for row_id, rows in by_prompt_scores.items():
        correct = distinct(sorted((r for r in rows if r["exact"]), key=lambda r: r["sample"]))[:MAX_PER_PROMPT]
        if correct:
            per_prompt[row_id] = correct
            supply[family_of[row_id]] = supply.get(family_of[row_id], 0) + len(correct)
    # A family with no verified-correct sample at all cannot be represented; it is dropped and the TRAIN
    # shares are renormalized over the families that have supply (recorded as "families_without_supply").
    present = {family: count for family, count in train_mix.items() if supply.get(family, 0) > 0}
    total_rows = sum(present.values())
    share = {family: count / total_rows for family, count in present.items()}
    # Largest total N with floor(N * share_f) <= supply_f for every family.
    limit = min((supply.get(family, 0) / s for family, s in share.items() if s > 0), default=0)
    quota = {family: int(limit * s) for family, s in share.items()}
    rng = random.Random(SEED)
    chosen = []
    for family in sorted(share):
        prompts = sorted(row_id for row_id in per_prompt if family_of[row_id] == family)
        rng.shuffle(prompts)
        left = quota[family]
        for row_id in prompts:
            if left <= 0:
                break
            for row in per_prompt[row_id][:left]:
                chosen.append(row)
                left -= 1
    return chosen, {"supply_distinct_capped": supply, "quota": quota, "balanced_total": len(chosen),
                    "families_without_supply": sorted(set(train_mix) - set(present)),
                    "renormalized_train_share": {f: round(s, 6) for f, s in share.items()},
                    "binding_family": min(share, key=lambda f: supply[f] / share[f]) if share else None}


def dpo_pairs(by_prompt_scores, gold):
    pairs = []
    for row_id in sorted(by_prompt_scores):
        rows = sorted(by_prompt_scores[row_id], key=lambda r: r["sample"])
        correct = distinct([r for r in rows if r["exact"]])
        wrong = [r for r in rows if r["protocol_valid"] and not r["exact"]]
        rejected = distinct([r for r in wrong if r["outcome"] in REJECT_PREFERRED]
                            + [r for r in wrong if r["outcome"] in REJECT_LAST])
        if not rejected:
            continue
        for index, bad in enumerate(rejected[:MAX_PER_PROMPT]):
            if correct:
                good = correct[index % len(correct)]
                chosen = {"source": "model", "sample": good["sample"], "generated_ids": good["generated_ids"],
                          "text": good["text"]}
            else:
                chosen = {"source": "gold", "sample": None, "generated_ids": gold[row_id], "text": None}
            pairs.append({"id": row_id, "family": bad["family"], "expected_operation": bad["expected_operation"],
                          "chosen": chosen,
                          "rejected": {"sample": bad["sample"], "generated_ids": bad["generated_ids"],
                                       "text": bad["text"], "outcome": bad["outcome"], "reward": bad["reward"],
                                       "parse": bad["parse"]}})
    return pairs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, action="append", required=True, help="fetched census shard run directory")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--samples-per-prompt", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--air", default="air")
    p.add_argument("--model-id", default="sepalith-2b-edit-sft/arm-a")
    args = p.parse_args(argv)
    k = args.samples_per_prompt
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    specs, targets, data_identity = load_train_specs()
    spec_by_id = {spec.row_id: spec for spec in specs}
    raw_rows = {}
    rows_path = DATA_ROOT / TRAIN_INPUTS["rows"][0]
    if sha_path(rows_path) != TRAIN_INPUTS["rows"][1]:
        raise SystemExit("TRAIN rows changed after loading")
    with rows_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            raw_rows[row["id"]] = row
    rows = {row_id: {"prompt_ids": row["input_ids"][:row["target_start"]], "family": row["family"]}
            for row_id, row in raw_rows.items()}
    if len(rows) != TRAIN_ROWS or set(rows) != set(spec_by_id):
        raise SystemExit("TRAIN rows and reward specs disagree")
    samples, shards = load_samples(args.run)
    by_prompt, missing = check_coverage(samples, rows, k)
    if missing:
        raise SystemExit(f"census incomplete: {len(missing)} TRAIN prompts lack exactly {k} samples (first: {missing[:5]})")
    loaded = time.perf_counter()

    probe = RParseOnlyProbe(DEFAULT_HARNESS, sha_file(DEFAULT_HARNESS))
    scorer = RewardV21Scorer(probe, AirFormatter((args.air,)), RewardV21Config(), workers=args.workers)
    decoder = tokenizer_decoder(args.tokenizer)
    covered = [row_id for row_id in sorted(by_prompt) if row_id not in set(missing)]
    references = scorer.build_references([spec_by_id[row_id] for row_id in covered])
    built = time.perf_counter()
    todo = [s for row_id in covered for s in sorted(by_prompt[row_id], key=lambda s: s["sample"])]
    records = scorer.score(references, [Sample(s["id"], completion_ids=tuple(s["generated_ids"]),
                                               sample_id=f"{s['id']}:{s['sample']}") for s in todo], decoder)
    scored = time.perf_counter()

    score_rows = []
    for sample, record in zip(todo, records, strict=True):
        ids = sample["generated_ids"]
        text = decoder(ids[:-1] if ids and ids[-1] in (1, 130073) else ids)
        score_rows.append({**record, "id": sample["id"], "sample": sample["sample"],
                           "global_index": sample["global_index"], "generated_ids": ids,
                           "generated_tokens": len(ids), "finish_reason": sample["finish_reason"],
                           "sampler_cap_hit": sample["cap_hit"], "text": text})
    by_prompt_scores = collections.defaultdict(list)
    for row in score_rows:
        by_prompt_scores[row["id"]].append(row)
    prompt_rows = []
    for row_id in covered:
        group = by_prompt_scores[row_id]
        exact = sum(bool(r["exact"]) for r in group)
        prompt_rows.append({"id": row_id, "family": rows[row_id]["family"],
                            "expected_operation": group[0]["expected_operation"], "k": len(group),
                            "exact_count": exact, "pass_rate": exact / len(group),
                            "mean_reward": round(statistics.fmean(float(r["reward"]) for r in group), 6),
                            "outcomes": dict(collections.Counter(str(r["outcome"]) for r in group)),
                            "cap_hits": sum(bool(r["cap_hit"]) for r in group),
                            "mean_completion_tokens": round(statistics.fmean(r["generated_tokens"] for r in group), 2),
                            "global_index": group[0]["global_index"]})

    family_of = {row_id: rows[row_id]["family"] for row_id in rows}
    train_mix = dict(collections.Counter(family_of.values()))
    rft, rft_info = rft_candidates(by_prompt_scores, family_of, train_mix)
    gold = {row_id: list(targets[row_id]) for row_id in covered}
    pairs = dpo_pairs(by_prompt_scores, gold)

    families = sorted(train_mix)
    stats = {"overall": group_stats(prompt_rows, score_rows, k)}
    for family in families:
        stats[family] = group_stats([p for p in prompt_rows if p["family"] == family],
                                    [s for s in score_rows if s["family"] == family], k)
    verified = sum(bool(s["exact"]) for s in score_rows)
    verified_distinct = sum(len(distinct([r for r in g if r["exact"]])) for g in by_prompt_scores.values())
    verified_distinct_capped = sum(min(MAX_PER_PROMPT, len(distinct([r for r in g if r["exact"]])))
                                   for g in by_prompt_scores.values())
    pair_sources = collections.Counter(pair["chosen"]["source"] for pair in pairs)
    r4_inputs = {"verified_correct_samples": verified, "verified_correct_distinct": verified_distinct,
                 "verified_correct_distinct_max2_per_prompt": verified_distinct_capped,
                 "rft_candidates_balanced": len(rft), "dpo_pair_candidates": len(pairs),
                 "dpo_pairs_model_chosen": pair_sources.get("model", 0),
                 "dpo_pairs_gold_chosen": pair_sources.get("gold", 0),
                 "min_verified": R4_MIN_VERIFIED, "min_pairs": R4_MIN_PAIRS,
                 "M": stats["overall"]["M_mixed_share"]}
    meets = min(verified, verified_distinct_capped, len(rft)) >= R4_MIN_VERIFIED and len(pairs) >= R4_MIN_PAIRS

    out = args.out_dir
    prompt_file = []
    for row_id in sorted(rows, key=lambda r: by_prompt[r][0]["global_index"] if r in by_prompt else 1 << 30):
        first = by_prompt.get(row_id, [{}])[0]
        prompt_file.append({"id": row_id, "family": rows[row_id]["family"], "split": "train",
                            "global_index": first.get("global_index"), "seed": first.get("seed"),
                            "prompt_token_ids": rows[row_id]["prompt_ids"],
                            "prompt_sha256": sha256_ids(rows[row_id]["prompt_ids"]),
                            "target_ids": list(targets[row_id])})
    write_jsonl(out / "census-prompts.jsonl", prompt_file)
    write_jsonl(out / "samples.jsonl", sorted(samples, key=lambda s: (s["global_index"], s["sample"])))
    write_jsonl(out / "scores.jsonl", sorted(score_rows, key=lambda s: (s["global_index"], s["sample"])))
    write_jsonl(out / "pass-rates.jsonl", sorted(prompt_rows, key=lambda p: p["global_index"]))
    write_jsonl(out / "rft-candidates.jsonl", [
        {"id": r["id"], "family": r["family"], "sample": r["sample"], "generated_ids": r["generated_ids"],
         "text": r["text"], "comparison": r["comparison"]} for r in rft])
    write_jsonl(out / "dpo-pair-candidates.jsonl", pairs)
    files = {name: {"sha256": sha_path(out / name), "bytes": (out / name).stat().st_size}
             for name in ("census-prompts.jsonl", "samples.jsonl", "scores.jsonl", "pass-rates.jsonl",
                          "rft-candidates.jsonl", "dpo-pair-candidates.jsonl")}
    summary = {
        "schema": SCHEMA, "model_id": args.model_id, "samples_per_prompt": k,
        "census": {"train_rows": TRAIN_ROWS, "prompts_scored": len(covered), "prompts_missing": len(missing),
                   "missing_ids": missing[:50], "samples_scored": len(score_rows)},
        "shards": shards, "data": data_identity, "scorer": scorer.identity(),
        "infrastructure_failures": {"r": scorer.probe.infrastructure_failures,
                                    "air": scorer.formatter.infrastructure_failures},
        "statistics": stats, "rft": rft_info,
        "dpo": {"pairs": len(pairs), "chosen_sources": dict(pair_sources),
                "rejected_outcomes": dict(collections.Counter(pair["rejected"]["outcome"] for pair in pairs)),
                "prompts_with_pairs": len({pair["id"] for pair in pairs})},
        "r4_step1": {"inputs": r4_inputs, "offline_route_counts_met": meets},
        "files": files,
        "seconds": {"load": round(loaded - t0, 1), "references": round(built - loaded, 1),
                    "score": round(scored - built, 1), "total": round(time.perf_counter() - t0, 1)},
        "final_set_access": False,
    }
    (out / "census-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"census": summary["census"], "overall": stats["overall"], "r4_step1": summary["r4_step1"],
                      "seconds": summary["seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
