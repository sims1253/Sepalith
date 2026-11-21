"""Difficulty probe: arm A (Q8_0 GGUF, llama.cpp CPU) greedy on a stratified
SYNTH sample, scored with reward v2.1 exactly as the gold self-check.

CPU only (CUDA hidden), 8 threads, nice 10, one request at a time; prompts are
sent as the rows' token ids (no re-tokenisation). Q8_0 is close to but not
identical with the bf16 checkpoint, so read rates as approximate.

v3.1: the receipt keeps one record per case (generation, score, packaging
variant, prefix lines and tokens, target lines and tokens, determinacy class,
seed wave). ``--diagnostic-pairs N`` adds a paired diagnostic: N medium
(9-20 line) ``sibling`` cuts of the build, each rebuilt from the build's
sources under a function-only packaging (``plain``) and under the ``file``
packaging, same cut and target, same model and decoding. ``--heldout`` probes
the held-out manipulation-check set (all rows, no stratification).

    PYTHONPATH=packages/sepalith/src .venv-sft/bin/python docs/training/plan-20260927/synth/probe_arm_a.py \
        --data /mnt/e/.../SYNTH-v3.1 --out docs/training/plan-20260927/receipts/SYNTH-v3.1-arm-a-probe.json \
        --exclude-ids docs/training/plan-20260927/synth/audit-v3.1/audit-sample-60-ids.json --diagnostic-pairs 40
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import tempfile
import time

from sepalith.campaign_protocol import PromptContext
from sepalith.data.synth import finish_block, profile
from sepalith.data.synth.build import load_tokenizer, make_scorer, package_and_path, sha256_file
from sepalith.data.synth.structure import describe
from sepalith.data.synth.verify import render
from sepalith.quant import gguf_dev as gd
from sepalith.quant.llama_tools import gpu_layers, llama_bin, tool_env
from sepalith.training.rl.reward_v21 import Evidence, ReferenceSpec, Sample

GGUF = Path("/mnt/e/sepalith/campaign-20260915/quant-work/arm-a/arm-a-Q8_0.gguf")
STRATA = {("finish_block", "short"): 60, ("finish_block", "medium"): 50, ("finish_block", "long"): 10,
          ("roxygen_drafting", None): 30}
CAP = 1024
SEED = 3407  # sampling seed; the audit pack uses 20261022 and its ids are excluded here
DIAGNOSTIC_SEED = 3408


def target_class(n: int) -> str:
    return "short" if n <= 8 else "medium" if n <= 20 else "long"


def diagnostic_pairs(data: Path, records: dict, n: int, tokenizer: object, skip: set[str]) -> list[dict]:
    """Rebuild N medium sibling cuts under ``plain`` and ``file`` packaging (same target)."""
    pool = sorted(rid for rid, r in records.items()
                  if r["family"] == "finish_block" and r["meta"]["determinacy"] == "sibling"
                  and target_class(profile.target_length(r["context"]["region_old"], r["region_new"])) == "medium"
                  and rid not in skip)
    random.Random(DIAGNOSTIC_SEED).shuffle(pool)
    structures: dict[str, object] = {}
    pairs = []
    for rid in pool:
        if len(pairs) >= n:
            break
        meta = records[rid]["meta"]
        source = data / "sources" / meta["seed"]
        if str(source) not in structures:
            structures[str(source)] = describe([source])[str(source)]
        structure = structures[str(source)]
        lines = source.read_text(encoding="utf-8").rstrip("\n").split("\n")
        unit = next(u for u in structure.units if u.start == meta["unit_lines"][0])
        cut = finish_block.Cut(meta["cut_line"], meta["rule"].removeprefix("fb_"), unit.end - meta["cut_line"] + 1)
        built = {}
        for variant in ("plain", "file"):
            case = finish_block.build_case(unit, cut, variant, lines, structure, records[rid]["package_id"],
                                           package_and_path(meta["seed"])[1], signature_region=meta["signature_region"])
            row, context, error = render(case, tokenizer, f"{rid}-diag-{variant}")
            if error:
                break
            built[variant] = (case, row, context)
        if len(built) != 2 or built["plain"][1]["prompt_text"] == built["file"][1]["prompt_text"]:
            continue
        if built["plain"][0].target != built["file"][0].target:
            continue
        pairs.append({"source_row_id": rid, "selected_variant": meta["variant"], "built": built})
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--exclude-ids", type=Path, default=None, help="JSON list of row ids to leave out")
    parser.add_argument("--heldout", action="store_true", help="probe heldout-manipulation/ (all rows)")
    parser.add_argument("--diagnostic-pairs", type=int, default=0)
    args = parser.parse_args()
    if args.threads > 8:
        raise SystemExit("at most 8 threads")
    if args.heldout:
        base = args.data / "heldout-manipulation"
        cases_path, rows_path = base / "probe-cases.jsonl", base / "probe-token-rows.jsonl"
    else:
        cases_path, rows_path = args.data / "cases.jsonl", args.data / "train-token-rows.jsonl"
    records = {r["row_id"]: r for r in map(json.loads, cases_path.open())}
    rows = {r["id"]: r for r in map(json.loads, rows_path.open())}
    excluded = set(json.loads(args.exclude_ids.read_text())) if args.exclude_ids else set()

    def length_of(rid: str) -> int:
        return profile.target_length(records[rid]["context"]["region_old"], records[rid]["region_new"])

    if args.heldout:
        chosen = [(("finish_block", target_class(length_of(rid))), rid) for rid in sorted(records)]
        strata = sorted({key for key, _ in chosen})
    else:
        pools = defaultdict(list)
        for rid in sorted(records):
            if rid in excluded:
                continue
            family = records[rid]["family"]
            pools[(family, target_class(length_of(rid)) if family == "finish_block" else None)].append(rid)
        rng = random.Random(SEED)
        chosen = [(key, rid) for key, n in STRATA.items() for rid in rng.sample(pools[key], min(n, len(pools[key])))]
        strata = list(STRATA)
    # (generation id, prompt ids) for every generation
    jobs = [(rid, rows[rid]["input_ids"][:rows[rid]["target_start"]]) for _, rid in chosen]
    pairs = []
    tokenizer = None
    if args.diagnostic_pairs:
        tokenizer = load_tokenizer()
        pairs = diagnostic_pairs(args.data, records, args.diagnostic_pairs, tokenizer, set())
        for pair in pairs:
            for _case, row, _context in pair["built"].values():
                jobs.append((row["id"], row["input_ids"][:row["target_start"]]))
    ctx = -(-(max(len(p) for _, p in jobs) + CAP) // 256) * 256
    port = gd.free_port()
    argv = ["nice", "-n", "10", str(llama_bin("llama-server", "cpu")), "-m", str(GGUF), "-c", str(ctx), "-np", "1",
            "-t", str(args.threads), "-tb", str(args.threads), *gpu_layers("cpu"), "--host", "127.0.0.1",
            "--port", str(port), "--no-webui", "--seed", str(SEED), "--no-context-shift", "--no-cache-prompt"]
    generated: dict[str, list[int]] = {}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="synth-probe-") as tmp:
        with gd.llama_server(argv, port, tool_env("cpu"), Path(tmp) / "server.log") as url:
            for index, (job_id, prompt) in enumerate(jobs):
                response = gd.post_json(f"{url}/completion", gd.completion_request(prompt, CAP, SEED), 900)
                generated[job_id] = gd.generated_ids(response, len(prompt), CAP)
                if (index + 1) % 25 == 0:
                    print(f"[{index + 1}/{len(jobs)}] {time.monotonic() - started:.0f}s", flush=True)
    scorer, _formatter, decoder = make_scorer(8)
    specs = [ReferenceSpec(rid, records[rid]["family"], PromptContext.from_dict(records[rid]["context"]), "replace",
                           rows[rid]["target_body_text"], Evidence("unverified")) for _, rid in chosen]
    samples = [Sample(rid, completion_ids=tuple(generated[rid]), sample_id="arm-a-q8-greedy") for _, rid in chosen]
    for pair in pairs:
        for case, row, context in pair["built"].values():
            specs.append(ReferenceSpec(row["id"], case.family, PromptContext.from_dict(context), "replace",
                                       row["target_body_text"], Evidence("unverified")))
            samples.append(Sample(row["id"], completion_ids=tuple(generated[row["id"]]), sample_id="arm-a-q8-greedy"))
    scored = {r["row_id"]: r for r in scorer.score(scorer.build_references(specs), samples, decoder)}

    def score_fields(job_id: str) -> dict:
        s = scored[job_id]
        return {"exact": bool(s["exact"]), "reward": round(float(str(s["reward"])), 4), "outcome": str(s["outcome"]),
                "generation": decoder(generated[job_id]), "generated_tokens": len(generated[job_id])}

    cases = []
    for key, rid in chosen:
        record, row = records[rid], rows[rid]
        meta = record["meta"]
        cases.append({"row_id": rid, "stratum": f"{key[0]}:{key[1] or 'all'}", "family": record["family"],
                      "variant": meta.get("variant"), "determinacy": meta.get("determinacy"),
                      "seed_wave": meta.get("seed_wave"),
                      "batch": meta.get("batch"),
                      "prefix_lines": len(record["context"]["prefix"]), "prompt_tokens": row["prompt_token_count"],
                      "target_lines": length_of(rid), "target_tokens": row["target_body_token_count"],
                      **score_fields(rid)})
    table: dict[str, dict] = {}
    for key in strata:
        name = f"{key[0]}:{key[1] or 'all'}"
        members = [c for c in cases if c["stratum"] == name]
        outcomes: dict[str, int] = defaultdict(int)
        for c in members:
            outcomes[c["outcome"]] += 1
        table[name] = {"cases": len(members), "exact": sum(c["exact"] for c in members),
                       "mean_reward": round(sum(c["reward"] for c in members) / max(1, len(members)), 4),
                       "outcomes": dict(outcomes)}
    receipt = {
        "schema": "sepalith.synth.arm-a-probe.v2",
        "model": {"gguf": str(GGUF), "sha256": sha256_file(GGUF), "note": "arm A checkpoint 536, Q8_0"},
        "decoding": {"greedy": True, "cap": CAP, "engine": "llama.cpp CPU llama-server", "threads": args.threads},
        "sample": {"set": "heldout-manipulation" if args.heldout else "training rows",
                   "seed": SEED, "strata": {f"{k[0]}:{k[1] or 'all'}": n for k, n in STRATA.items()}
                   if not args.heldout else "all held-out rows",
                   "length_definition": "profile.target_length; short <= 8, medium 9-20, long > 20 lines",
                   "excluded_ids": len(excluded), "overlap_with_excluded": len(excluded & {rid for _, rid in chosen})},
        "by_stratum": table,
        "total": {"cases": len(cases), "exact": sum(c["exact"] for c in cases)},
        "cases": cases,
        "data": {rows_path.name: sha256_file(rows_path), cases_path.name: sha256_file(cases_path)},
    }
    if pairs:
        pair_records = []
        for pair in pairs:
            entry = {"source_row_id": pair["source_row_id"], "selected_variant": pair["selected_variant"]}
            for variant, (case, row, _context) in pair["built"].items():
                entry[variant] = {"prefix_lines": case.region_start - case.window_start,
                                  "prompt_tokens": row["prompt_token_count"], **score_fields(row["id"])}
            entry["target_lines"] = profile.target_length(pair["built"]["plain"][0].region_old,
                                                          pair["built"]["plain"][0].target)
            pair_records.append(entry)
        both = [(p["plain"]["exact"], p["file"]["exact"]) for p in pair_records]
        receipt["diagnostic_function_only_vs_file"] = {
            "design": "same medium sibling cut and target; packaging plain (roxygen + function head) vs file "
                      "(up to 60 preceding file lines); same model, greedy decoding, cap",
            "pairs": len(pair_records),
            "exact_plain": sum(a for a, _ in both), "exact_file": sum(b for _, b in both),
            "only_plain": sum(a and not b for a, b in both), "only_file": sum(b and not a for a, b in both),
            "both": sum(a and b for a, b in both), "neither": sum(not a and not b for a, b in both),
            "mean_reward_plain": round(statistics.mean(p["plain"]["reward"] for p in pair_records), 4),
            "mean_reward_file": round(statistics.mean(p["file"]["reward"] for p in pair_records), 4),
            "median_prefix_lines": {v: statistics.median(p[v]["prefix_lines"] for p in pair_records)
                                    for v in ("plain", "file")},
            "median_prompt_tokens": {v: statistics.median(p[v]["prompt_tokens"] for p in pair_records)
                                     for v in ("plain", "file")},
            "records": pair_records}
    receipt["wall_seconds"] = round(time.monotonic() - started, 1)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"by_stratum": {k: (v["exact"], v["cases"]) for k, v in table.items()},
                      "total": receipt["total"],
                      "diagnostic": {k: v for k, v in receipt.get("diagnostic_function_only_vs_file", {}).items()
                                     if k != "records"}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
