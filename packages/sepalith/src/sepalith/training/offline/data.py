#!/usr/bin/env python3
"""Card C13 data: the RFT cohort and the DPO pair file, built from one C12 census.

Inputs are TRAIN only: the census outputs of one model (``census-summary.json``
with ``scores.jsonl``, ``pass-rates.jsonl``, ``dpo-pair-candidates.jsonl`` and
``census-prompts.jsonl``, all hash-checked against the summary) and the TRAIN
rows the census sampled. Selection rules (user decisions of 2026-10-17 in
docs/training/plan-20260927/status/C12.json and C13.json):

RFT samples
  1. Verified-correct pool: the census samples with ``exact`` (reward v2.1),
     made distinct per prompt by generated ids, ranked per prompt by the
     smallest edit (line-level diff against the prompt's CURRENT region; ties:
     raw-identical to the gold target first, then fewer tokens, then the lower
     sample index), at most 2 per prompt. This is the census's 7,528-sample
     distinct pool; only the order within a prompt is new.
  2. Family caps: each family keeps at most its TRAIN share of that pool,
     floor(train_rows_f / train_rows * pool), and at most its supply. No family
     is scaled down to the scarcest one (the user's family-capped choice).
  3. Within a family: prompts ordered by census pass rate, lowest first
     (mixed prompts carry the most signal), ties in a seeded order; first each
     prompt's best sample, then second samples, until the cap.
  4. Each sample becomes a TRAIN-shaped row: the TRAIN row's prompt, the
     sample's body, the TRAIN row's protocol terminal and EOS 1 (the SFT
     render, masking and termination). The row id is ``<train id>:rft:s<k>``.

RFT cohort
  The RFT rows plus a uniform seeded sample of original TRAIN rows, sized so
  that the TRAIN rows are 30% of the cohort (card C13 step 1), drawn from the
  TRAIN rows whose prompt has no RFT row (no duplicate sequences). The draw
  schedule is schedule v2 unchanged (13 edits plus 3 no-ops per update,
  18.75% no-ops, no-op rows at most 3 exposures, seed 3407).

DPO pairs
  Every census pair candidate (chosen: a distinct verified-correct sample, else
  the gold target; rejected: a protocol-valid incorrect sample). Completions
  end in EOS 1 like the SFT targets (an alternative end-of-generation id 130073
  is normalized to 1 and counted). The pair schedule is one seeded pass,
  16 pairs per update; the last update is completed with the earliest drawn
  pairs.

Run with the training interpreter or any Python 3.10+ (stdlib only), CPU only:

    PYTHONPATH=packages/sepalith/src nice -n 10 python3 -m sepalith.training.offline.data build \\
        --census-dir ~/.local/state/sepalith/c12/census-arm-a --census-summary-sha256 <sha> \\
        --model-id sepalith-2b-edit-sft-arm-a --output /mnt/e/.../data-work/C13-offline-v1/sepalith-2b-edit-sft-arm-a
"""
from __future__ import annotations

import argparse
import collections
from collections.abc import Iterable, Iterator, Mapping, Sequence
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import cast

from sepalith.training.sft.full_coverage_schedule import stable
from sepalith.training.sft.schedule_v2 import write_schedule

SEED = 3407
EOS = 1
EOG_IDS = (1, 130073)
BOS = 0
MAX_PER_PROMPT = 2
TRAIN_MIX_SHARE = 0.3
EFFECTIVE_BATCH = 16
MAX_SEQUENCE_TOKENS = 4096
NOOP_MAX_EXPOSURES = 3
CURRENT_OPEN = "<<<<<<< CURRENT\n"
CURRENT_CLOSE = "\n======="
TERMINAL_TEXT = ">>>>>>> UPDATED"
CURSOR = "<|user_cursor|>"
TRAIN_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl")
TRAIN_ROWS_SHA256 = "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef"
CENSUS_FILES = ("scores.jsonl", "pass-rates.jsonl", "dpo-pair-candidates.jsonl", "census-prompts.jsonl")
MANIFEST_SCHEMA = "sepalith.c13.offline-data-manifest.v1"
PAIR_SCHEMA = "sepalith.c13.dpo-pair.v1"
PAIR_SCHEDULE_SCHEMA = "sepalith.c13.dpo-pair-schedule.v1"

Json = dict[str, object]


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> Iterator[Json]:
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield cast(Json, json.loads(line))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    require(not path.exists(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: object) -> None:
    require(not path.exists(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _ids(value: object, label: str) -> list[int]:
    require(isinstance(value, list) and all(type(x) is int for x in value), f"{label} must be a list of token ids")
    return cast(list[int], value)


def _str(value: object, label: str) -> str:
    require(isinstance(value, str), f"{label} must be a string")
    return cast(str, value)


def _int(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer")
    return cast(int, value)


def _obj(value: object, label: str) -> Json:
    require(isinstance(value, dict), f"{label} must be an object")
    return cast(Json, value)


# Selection ---------------------------------------------------------------------------------------

def current_region(prompt_text: str) -> str | None:
    """The editable CURRENT region of a PRM03 prompt, without the cursor marker."""
    start = prompt_text.rfind(CURRENT_OPEN)
    if start < 0:
        return None
    start += len(CURRENT_OPEN)
    end = prompt_text.find(CURRENT_CLOSE, start)
    if end < 0:
        return None
    return prompt_text[start:end].replace(CURSOR, "")


def completion_body_text(text: str) -> str:
    """A decoded completion without the protocol terminal line."""
    body = text[:-len(TERMINAL_TEXT)] if text.endswith(TERMINAL_TEXT) else text
    return body[:-1] if body.endswith("\n") else body


def diff_lines(before: str, after: str) -> int:
    """Changed lines between two texts: the sum over non-equal opcodes of the larger side."""
    a, b = before.split("\n"), after.split("\n")
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")


def rank_correct_samples(samples: Sequence[Mapping[str, object]], current: str | None) -> list[Json]:
    """Distinct verified-correct samples of one prompt, smallest edit first."""
    seen: set[tuple[int, ...]] = set()
    distinct: list[Json] = []
    for sample in sorted(samples, key=lambda s: _int(s["sample"], "sample")):
        key = tuple(_ids(sample["generated_ids"], "generated_ids"))
        if key in seen:
            continue
        seen.add(key)
        text = _str(sample["text"], "text")
        size = diff_lines(current, completion_body_text(text)) if current is not None else 0
        distinct.append({**sample, "diff_lines": size, "diff_basis": "current_region" if current is not None else "none"})
    distinct.sort(key=lambda s: (_int(s["diff_lines"], "diff"), not s.get("exact_raw"),
                                 len(_ids(s["generated_ids"], "ids")), _int(s["sample"], "sample")))
    return distinct


def family_caps(train_mix: Mapping[str, int], supply: Mapping[str, int], pool: int) -> dict[str, int]:
    """Each family's TRAIN share of the distinct pool, floored, and at most its supply."""
    total = sum(train_mix.values())
    require(total > 0 and pool >= 0, "empty TRAIN mix")
    return {family: min(supply.get(family, 0), math.floor(count * pool / total)) for family, count in sorted(train_mix.items())}


def select_rft(pool: Mapping[str, Sequence[Mapping[str, object]]], family_of: Mapping[str, str],
               pass_rate: Mapping[str, float], caps: Mapping[str, int], seed: int = SEED) -> list[Json]:
    """Apply the family caps: lowest pass rate first, best samples before second samples."""
    chosen: list[Json] = []
    for family in sorted(caps):
        prompts = sorted((rid for rid in pool if family_of[rid] == family),
                         key=lambda rid: (pass_rate[rid], stable(seed, "c13-rft-prompt-order", family, rid)))
        left = caps[family]
        for round_index in range(MAX_PER_PROMPT):
            for rid in prompts:
                if left <= 0:
                    break
                ranked = pool[rid]
                if round_index < len(ranked):
                    chosen.append({**ranked[round_index], "id": rid, "family": family, "round": round_index + 1,
                                   "pass_rate": pass_rate[rid]})
                    left -= 1
        require(left == 0, f"{family}: supply below its cap")
    return chosen


def split_completion(generated: Sequence[int], terminal: Sequence[int]) -> tuple[list[int], bool]:
    """Body tokens of a protocol-valid completion; True if its end-of-generation id was normalized to EOS 1."""
    ids = list(generated)
    require(ids and ids[-1] in EOG_IDS, "completion does not end with an end-of-generation id")
    normalized = ids[-1] != EOS
    ids = ids[:-1]
    require(ids[len(ids) - len(terminal):] == list(terminal), "completion does not end with the protocol terminal")
    body = ids[:len(ids) - len(terminal)]
    require(BOS not in ids and not any(x in EOG_IDS for x in ids), "completion holds BOS or an end-of-generation id inside")
    return body, normalized


def rft_row(train_row: Mapping[str, object], sample: Mapping[str, object], model_id: str) -> Json:
    """A TRAIN-shaped row: the TRAIN prompt, the sample's body, the TRAIN terminal and EOS."""
    rid = _str(train_row["id"], "id")
    start = _int(train_row["target_start"], "target_start")
    ids = _ids(train_row["input_ids"], "input_ids")
    terminal = _ids(train_row["target_terminal_tokens"], "terminal")
    body, normalized = split_completion(_ids(sample["generated_ids"], "generated_ids"), terminal)
    require(not normalized, "an RFT sample must end with EOS 1")
    text = _str(sample["text"], "text")
    index = _int(sample["sample"], "sample")
    row = {key: value for key, value in train_row.items()
           if key not in ("id", "input_ids", "target_body_tokens", "target_body_token_count", "target_token_count",
                          "target_body_text", "target_text")}
    row.update({
        "id": f"{rid}:rft:s{index}",
        "input_ids": ids[:start] + body + terminal + [EOS],
        "target_body_tokens": body,
        "target_body_token_count": len(body),
        "target_token_count": len(body) + len(terminal),
        "target_body_text": completion_body_text(text),
        "target_text": text,
        "rft": {"source_row_id": rid, "model_id": model_id, "census_sample": index, "round": sample["round"],
                "pass_rate": sample["pass_rate"], "diff_lines": sample["diff_lines"], "diff_basis": sample["diff_basis"],
                "exact_raw": bool(sample.get("exact_raw")), "comparison": sample.get("comparison"),
                "same_tokens_as_gold": body == _ids(train_row["target_body_tokens"], "gold body")},
    })
    require(row["target_terminal_token_count"] == len(terminal) and len(_ids(row["input_ids"], "ids")) <= MAX_SEQUENCE_TOKENS,
            f"{rid}: RFT row geometry differs")
    return row


def train_mix_ids(train_ids: Iterable[str], excluded: set[str], count: int, seed: int = SEED) -> list[str]:
    """A uniform seeded sample of TRAIN rows whose prompt has no RFT row."""
    candidates = sorted((rid for rid in train_ids if rid not in excluded), key=lambda rid: stable(seed, "c13-train-mix", rid))
    require(len(candidates) >= count, "too few TRAIN rows for the mix")
    return sorted(candidates[:count])


def train_mix_count(rft_rows: int, share: float = TRAIN_MIX_SHARE) -> int:
    return round(rft_rows * share / (1 - share))


# DPO pairs ---------------------------------------------------------------------------------------

def pair_row(candidate: Mapping[str, object], index: int, train_row: Mapping[str, object], model_id: str) -> tuple[Json, int]:
    """One pair with SFT-terminated completions; returns the row and the number of normalized ends."""
    rid = _str(candidate["id"], "id")
    require(rid == train_row["id"] and train_row.get("split") == "train", f"{rid}: pair is not on its TRAIN row")
    start = _int(train_row["target_start"], "target_start")
    ids = _ids(train_row["input_ids"], "input_ids")
    terminal = _ids(train_row["target_terminal_tokens"], "terminal")
    chosen, rejected = _obj(candidate["chosen"], "chosen"), _obj(candidate["rejected"], "rejected")
    normalized = 0
    completions = {}
    for name, part in (("chosen", chosen), ("rejected", rejected)):
        body, fixed = split_completion(_ids(part["generated_ids"], f"{name} ids"), terminal)
        normalized += int(fixed)
        completions[name] = body + terminal + [EOS]
    if chosen["source"] == "gold":
        require(completions["chosen"] == ids[start:], f"{rid}: gold chosen differs from the TRAIN target")
    require(completions["chosen"] != completions["rejected"], f"{rid}: chosen equals rejected")
    prompt = ids[:start]
    require(max(len(prompt) + len(c) for c in completions.values()) <= MAX_SEQUENCE_TOKENS, f"{rid}: pair exceeds the context")
    gold_text = _str(train_row["target_text"], "target_text")
    row = {"schema": PAIR_SCHEMA, "pair_id": f"{rid}:p{index}", "id": rid, "split": "train", "model_id": model_id,
           "family": candidate["family"], "expected_operation": candidate["expected_operation"],
           "prompt_ids": prompt, "chosen_ids": completions["chosen"], "rejected_ids": completions["rejected"],
           "chosen": {"source": chosen["source"], "sample": chosen.get("sample"),
                      "text": chosen.get("text") if chosen["source"] == "model" else gold_text},
           "rejected": {"sample": rejected.get("sample"), "outcome": rejected.get("outcome"),
                        "reward": rejected.get("reward"), "parse": rejected.get("parse"), "text": rejected.get("text")}}
    return row, normalized


def pair_schedule(pair_ids: Sequence[str], pairs_sha256: str, seed: int = SEED, batch: int = EFFECTIVE_BATCH) -> Json:
    """One seeded pass over the pairs; the last update is completed with the earliest drawn pairs."""
    require(len(set(pair_ids)) == len(pair_ids) and pair_ids, "pair ids must be unique and present")
    order = sorted(range(len(pair_ids)), key=lambda i: stable(seed, "c13-dpo-pair-order", pair_ids[i]))
    updates = math.ceil(len(order) / batch)
    pad = updates * batch - len(order)
    replays = order[:pad]
    require(not set(replays) & set(order[(updates - 1) * batch:]), "a replay would repeat a pair inside the last update")
    draws = order + replays
    return {"schema": PAIR_SCHEDULE_SCHEMA, "pairs_sha256": pairs_sha256, "seed": seed, "effective_batch": batch,
            "pairs": len(order), "max_steps": updates, "draw_count": len(draws), "replays": len(replays),
            "replay_policy": "the last update is completed with the earliest drawn pairs (each seen twice)",
            "gate_updates": halves(updates), "draws": draws}


def halves(updates: int) -> list[int]:
    """Gates at half and at the end of the horizon (card C13), rounded half to even."""
    return [round(updates / 2), updates]


# Build -------------------------------------------------------------------------------------------

def load_census(directory: Path, summary_sha256: str) -> Json:
    summary_path = directory / "census-summary.json"
    require(sha256(summary_path) == summary_sha256, "census summary differs from the bound hash")
    summary = _obj(json.loads(summary_path.read_text()), "census summary")
    require(summary.get("final_set_access") is False, "census summary must record no final-set access")
    files = _obj(summary.get("files"), "census files")
    for name in CENSUS_FILES:
        record = _obj(files.get(name), name)
        require(sha256(directory / name) == record.get("sha256"), f"census file differs: {name}")
    return summary


def build(*, census_dir: Path, census_summary_sha256: str, model_id: str, output: Path, train_rows: Path = TRAIN_ROWS,
          train_rows_sha256: str = TRAIN_ROWS_SHA256, seed: int = SEED, train_mix_share: float = TRAIN_MIX_SHARE,
          expected_rft_total: int | None = None) -> Json:
    require(not output.exists(), f"output directory must be fresh: {output}")
    summary = load_census(census_dir, census_summary_sha256)
    require(sha256(train_rows) == train_rows_sha256, "TRAIN rows differ from the bound hash")
    rows: dict[str, Json] = {}
    for row in read_jsonl(train_rows):
        require(row.get("split") == "train", "a non-TRAIN row is in the TRAIN file")
        rows[_str(row["id"], "id")] = row
    prompt_hash = {_str(r["id"], "id"): _str(r["prompt_sha256"], "prompt hash") for r in read_jsonl(census_dir / "census-prompts.jsonl")}
    require(set(prompt_hash) == set(rows), "census prompts and TRAIN rows differ")
    for rid, row in rows.items():
        require(sha256_ids(_ids(row["input_ids"], "ids")[:_int(row["target_start"], "start")]) == prompt_hash[rid],
                f"{rid}: census prompt differs from the TRAIN prompt")
    family_of = {rid: _str(row["family"], "family") for rid, row in rows.items()}
    pass_rate = {_str(p["id"], "id"): float(_int(p["exact_count"], "exact") / _int(p["k"], "k"))
                 for p in read_jsonl(census_dir / "pass-rates.jsonl")}
    require(set(pass_rate) == set(rows), "pass rates and TRAIN rows differ")

    correct: dict[str, list[Json]] = collections.defaultdict(list)
    keep = ("sample", "generated_ids", "text", "exact_raw", "comparison")
    for score in read_jsonl(census_dir / "scores.jsonl"):
        if score.get("exact") is True:
            require(score.get("protocol_valid") is True and not score.get("cap_hit"), "an exact sample is invalid or capped")
            correct[_str(score["id"], "id")].append({key: score.get(key) for key in keep})
    pool: dict[str, list[Json]] = {}
    for rid, samples in correct.items():
        ranked = rank_correct_samples(samples, current_region(_str(rows[rid]["prompt_text"], "prompt_text")))
        pool[rid] = ranked[:MAX_PER_PROMPT]
    pool_total = sum(len(v) for v in pool.values())
    supply = dict(collections.Counter(family_of[rid] for rid, v in pool.items() for _ in v))
    train_mix = dict(collections.Counter(family_of.values()))
    caps = family_caps(train_mix, supply, pool_total)
    selected = select_rft(pool, family_of, pass_rate, caps, seed)
    if expected_rft_total is not None:
        require(len(selected) == expected_rft_total, f"RFT total {len(selected)} differs from the expected {expected_rft_total}")
    rft_rows = sorted((rft_row(rows[_str(s["id"], "id")], s, model_id) for s in selected), key=lambda r: _str(r["id"], "id"))
    rft_prompts = {_str(_obj(r["rft"], "rft")["source_row_id"], "source") for r in rft_rows}
    mix_ids = train_mix_ids(rows, rft_prompts, train_mix_count(len(rft_rows), train_mix_share), seed)
    cohort = rft_rows + [rows[rid] for rid in mix_ids]
    sequences = collections.Counter(tuple(_ids(r["input_ids"], "ids")) for r in cohort)
    require(max(sequences.values()) == 1, "the RFT cohort holds a duplicate sequence")

    rft_dir, dpo_dir = output / "rft", output / "dpo"
    cohort_path = rft_dir / "rft-cohort-rows.jsonl"
    write_jsonl(cohort_path, cohort)
    cohort_sha = sha256(cohort_path)
    ledger_path = rft_dir / "rft-selection.jsonl"
    write_jsonl(ledger_path, ({"row_id": r["id"], **_obj(r["rft"], "rft"), "family": r["family"]} for r in rft_rows))
    mix_path = rft_dir / "train-mix-ids.json"
    write_json(mix_path, {"seed": seed, "share": train_mix_share, "rows": len(mix_ids), "ids": mix_ids,
                          "drawn_from": "TRAIN rows whose prompt has no RFT row", "train_rows_sha256": train_rows_sha256})
    schedule_path = rft_dir / "draw-schedule-v2-rft.json"
    schedule = write_schedule(cohort_path, cohort_sha, schedule_path, seed=seed)
    coverage = _obj(schedule["coverage"], "coverage")
    noop_hist = _obj(_obj(coverage["exposure_histogram_by_pool"], "histogram")["noop"], "noop histogram")
    require(max(int(k) for k in noop_hist) <= NOOP_MAX_EXPOSURES, "no-op exposures exceed the schedule v2 cap of 3")
    from sepalith.training.sft.recipe_v2 import build_length_profile

    profile_path = rft_dir / "length-profile.json"
    write_json(profile_path, build_length_profile(cohort_path, cohort_sha))

    pairs: list[Json] = []
    normalized = 0
    per_prompt: collections.Counter[str] = collections.Counter()
    for candidate in read_jsonl(census_dir / "dpo-pair-candidates.jsonl"):
        rid = _str(candidate["id"], "id")
        row, fixed = pair_row(candidate, per_prompt[rid], rows[rid], model_id)
        per_prompt[rid] += 1
        normalized += fixed
        pairs.append(row)
    pairs_path = dpo_dir / "dpo-pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    pairs_sha = sha256(pairs_path)
    pair_schedule_path = dpo_dir / "pair-schedule.json"
    pair_plan = pair_schedule([_str(p["pair_id"], "pair id") for p in pairs], pairs_sha, seed)
    write_json(pair_schedule_path, pair_plan)

    def record(path: Path) -> Json:
        return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

    rft_by_family = collections.Counter(_str(r["family"], "family") for r in rft_rows)
    mix_by_family = collections.Counter(family_of[rid] for rid in mix_ids)
    manifest: Json = {
        "schema": MANIFEST_SCHEMA, "model_id": model_id, "seed": seed, "split": "train", "final_set_access": False,
        "inputs": {"census_summary": {"path": str(census_dir / "census-summary.json"), "sha256": census_summary_sha256},
                   "census_files": {name: _obj(_obj(summary["files"], "files")[name], name) for name in CENSUS_FILES},
                   "train_rows": {"path": str(train_rows), "sha256": train_rows_sha256, "rows": len(rows)}},
        "rft": {
            "verified_correct_samples": sum(len(v) for v in correct.values()),
            "distinct_pool_max2_per_prompt": pool_total, "prompts_with_correct": len(pool),
            "supply_by_family": dict(sorted(supply.items())), "train_rows_by_family": dict(sorted(train_mix.items())),
            "caps_by_family": caps, "selected": len(rft_rows), "selected_by_family": dict(sorted(rft_by_family.items())),
            "selected_by_round": dict(collections.Counter(str(s["round"]) for s in selected)),
            "selected_exact_raw": sum(bool(s.get("exact_raw")) for s in selected),
            "selected_same_tokens_as_gold": sum(bool(_obj(r["rft"], "rft")["same_tokens_as_gold"]) for r in rft_rows),
            "selected_mixed_pass_rate": sum(0 < float(cast(float, s["pass_rate"])) < 1 for s in selected),
            "selection_rule": ("distinct exact samples per prompt ranked by line diff against the CURRENT region (then raw "
                               "gold identity, fewer tokens, lower sample index), at most 2 per prompt; family cap = "
                               "min(supply, floor(train share x pool)); within a family prompts by pass rate ascending, "
                               "ties seeded; best samples before second samples"),
        },
        "cohort": {"rows": len(cohort), "rft_rows": len(rft_rows), "train_mix_rows": len(mix_ids),
                   "train_mix_share": round(len(mix_ids) / len(cohort), 6),
                   "train_mix_by_family": dict(sorted(mix_by_family.items())),
                   "noop_rows": sum(r.get("target_operation") == "no_op" for r in cohort),
                   "max_sequence_tokens": max(len(_ids(r["input_ids"], "ids")) for r in cohort)},
        "schedule": {"updates": schedule["max_steps"], "draws": schedule["draw_count"],
                     "gates": halves(_int(schedule["max_steps"], "updates")),
                     "noop_share": coverage["noop_share"], "exposure_histogram_by_pool": coverage["exposure_histogram_by_pool"],
                     "checks": schedule["checks"], "token_denominators": schedule["token_denominators"]},
        "dpo": {"pairs": len(pairs), "prompts": len(per_prompt),
                "chosen_sources": dict(collections.Counter(str(_obj(p["chosen"], "chosen")["source"]) for p in pairs)),
                "rejected_outcomes": dict(collections.Counter(str(_obj(p["rejected"], "rejected")["outcome"]) for p in pairs)),
                "by_family": dict(sorted(collections.Counter(str(p["family"]) for p in pairs).items())),
                "normalized_end_of_generation": normalized,
                "max_sequence_tokens": max(len(_ids(p["prompt_ids"], "p")) + max(len(_ids(p["chosen_ids"], "c")),
                                                                                  len(_ids(p["rejected_ids"], "r"))) for p in pairs),
                "tokens": {"prompt": sum(len(_ids(p["prompt_ids"], "p")) for p in pairs),
                           "chosen": sum(len(_ids(p["chosen_ids"], "c")) for p in pairs),
                           "rejected": sum(len(_ids(p["rejected_ids"], "r")) for p in pairs)},
                "schedule": {key: pair_plan[key] for key in ("max_steps", "draw_count", "replays", "gate_updates", "effective_batch")}},
        "outputs": {"rft_cohort_rows": record(cohort_path), "rft_selection": record(ledger_path),
                    "train_mix_ids": record(mix_path), "rft_draw_schedule": record(schedule_path),
                    "rft_length_profile": record(profile_path), "dpo_pairs": record(pairs_path),
                    "dpo_pair_schedule": record(pair_schedule_path)},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("build")
    run.add_argument("--census-dir", type=Path, required=True)
    run.add_argument("--census-summary-sha256", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--train-rows", type=Path, default=TRAIN_ROWS)
    run.add_argument("--train-rows-sha256", default=TRAIN_ROWS_SHA256)
    run.add_argument("--train-mix-share", type=float, default=TRAIN_MIX_SHARE)
    run.add_argument("--expected-rft-total", type=int, help="fail unless the family-capped RFT set has this many samples")
    args = parser.parse_args(argv)
    manifest = build(census_dir=args.census_dir, census_summary_sha256=args.census_summary_sha256, model_id=args.model_id,
                     output=args.output, train_rows=args.train_rows, train_rows_sha256=args.train_rows_sha256,
                     train_mix_share=args.train_mix_share, expected_rft_total=args.expected_rft_total)
    print(json.dumps({key: manifest[key] for key in ("rft", "cohort", "schedule", "dpo")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
