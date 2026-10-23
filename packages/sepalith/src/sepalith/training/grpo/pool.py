#!/usr/bin/env python3
"""Card C14 step 1: the GRPO prompt pool from the C12 census (TRAIN only).

Rule (card C14 step 1): every prompt with 0.125 <= pass <= 0.875 at k = 8
(1 to 7 exact samples of 8, which is the census ``M`` set), plus prompts sampled
from the all-correct set (8 of 8 exact) to guard against forgetting. The
all-correct prompts make up 10% of the finished pool, so their count is
``round(band / 9)``. They are allocated to families in proportion to the
all-correct set (largest remainder) and drawn with a seeded RNG inside each
family. No-op prompts keep their natural share inside the band: nothing is
re-weighted.

Inputs are the C12 arm A census at temperature 1.0 (``pass-rates.jsonl`` and its
summary, hash-pinned) and the v4 TRAIN rows. Outputs:

- the pool (``id``, family, stratum, census pass rate) as JSON lines, committed;
- a manifest with the inputs, the rule, per-family counts and the pool hash;
- the bundle: per pool prompt, the prompt token IDs and the reward v2.1
  reference spec (context, target, reward-buffer evidence), so the trainer
  needs neither the 15,006-row load nor the reward-buffer blobs at night.

    python -m sepalith.training.grpo.pool build --docs-dir docs/training/plan-20260927/c14
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import cast

Json = dict[str, object]

POOL_SCHEMA = "sepalith.c14.grpo-pool.v1"
POOL_ROW_SCHEMA = "sepalith.c14.grpo-pool-row.v1"
BUNDLE_ROW_SCHEMA = "sepalith.c14.grpo-pool-bundle-row.v1"
SAMPLES_PER_PROMPT = 8
BAND_MIN_EXACT, BAND_MAX_EXACT = 1, 7  # 0.125 <= pass <= 0.875 at k = 8
ALL_CORRECT_SHARE = 0.10
SEED = 3407
MODEL_ID = "sepalith-2b-edit-sft/arm-a"
CENSUS_DIR = Path.home() / ".local/state/sepalith/c12/census-arm-a"
# The arm A census at temperature 1.0 (status/C12.json, status/C13.json).
CENSUS_SUMMARY_SHA256 = "eaeadf7b9640ddd069cff093d92fedb8831042ca53f79415b0e0187bf3269171"
PASS_RATES_SHA256 = "e41534e952000c085671f59f8a09cb57aeb288ba97f8872ecb0f3ba4bb37a5ac"
CENSUS_PROMPTS_SHA256 = "55194312168f7d636d78c8c1728b7f7457b06b07bd7bdad7fb4d65b93f445025"
CENSUS_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "samples": 8, "max_new_tokens": 1024, "seed": 3407,
                   "source": "docs/training/plan-20260927/status/C12.json outputs.census.sampling"}
BUNDLE_ROOT = Path("/mnt/e/sepalith/campaign-20260915/data-work/C14-grpo-pool-v1/sepalith-2b-edit-sft-arm-a")
POOL_NAME = "C14-grpo-pool-arm-a-536"


class PoolError(ValueError):
    """An input differs from its pin or the selection cannot be made."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: object, message: str) -> None:
    if not condition:
        raise PoolError(message)


def write_new_json(path: Path, value: object) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_new_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


# Selection ---------------------------------------------------------------------------------------

def largest_remainder(weights: Mapping[str, int], total: int) -> dict[str, int]:
    """Integer allocation of ``total`` proportional to ``weights``; ties go to the larger weight, then the name."""
    _require(total >= 0 and all(value >= 0 for value in weights.values()), "allocation needs nonnegative counts")
    whole = sum(weights.values())
    if total == 0 or whole == 0:
        return dict.fromkeys(weights, 0)
    _require(total <= whole, f"cannot allocate {total} from {whole}")
    exact = {name: value * total / whole for name, value in weights.items()}
    out = {name: math.floor(value) for name, value in exact.items()}
    order = sorted(weights, key=lambda name: (-(exact[name] - out[name]), -weights[name], name))
    for name in order[:total - sum(out.values())]:
        out[name] += 1
    return out


def check_census_row(row: Mapping[str, object]) -> tuple[str, str, int]:
    identifier, family, count = row.get("id"), row.get("family"), row.get("exact_count")
    _require(isinstance(identifier, str) and identifier and isinstance(family, str) and family, f"bad census row {row!r}")
    _require(row.get("k") == SAMPLES_PER_PROMPT and type(count) is int and 0 <= count <= SAMPLES_PER_PROMPT,
             f"census row {identifier} is not a k = {SAMPLES_PER_PROMPT} pass count")
    _require(abs(float(cast(float, row.get("pass_rate"))) - cast(int, count) / SAMPLES_PER_PROMPT) < 1e-9,
             f"census row {identifier} pass rate differs from its count")
    return cast(str, identifier), cast(str, family), cast(int, count)


def select_pool(rows: Sequence[Mapping[str, object]], *, share: float = ALL_CORRECT_SHARE, seed: int = SEED) -> Json:
    """The card's pool: the 1-7 of 8 band plus a family-proportional seeded sample of all-correct prompts."""
    _require(0 <= share < 1, "the all-correct share must lie in [0, 1)")
    seen: set[str] = set()
    band: list[Mapping[str, object]] = []
    all_correct: dict[str, list[Mapping[str, object]]] = {}
    families: dict[str, int] = {}
    for row in rows:
        identifier, family, count = check_census_row(row)
        _require(identifier not in seen, f"duplicate census id {identifier}")
        seen.add(identifier)
        families[family] = families.get(family, 0) + 1
        if BAND_MIN_EXACT <= count <= BAND_MAX_EXACT:
            band.append(row)
        elif count == SAMPLES_PER_PROMPT:
            all_correct.setdefault(family, []).append(row)
    target = math.floor(len(band) * share / (1 - share) + 0.5)
    allocation = largest_remainder({family: len(items) for family, items in sorted(all_correct.items())}, target)
    rng = random.Random(seed)
    sampled: list[Mapping[str, object]] = []
    for family in sorted(allocation):
        candidates = sorted(all_correct[family], key=lambda row: str(row["id"]))
        sampled.extend(rng.sample(candidates, allocation[family]))
    pool = [pool_row(row, "band") for row in band] + [pool_row(row, "all_correct") for row in sampled]
    pool.sort(key=lambda row: str(row["id"]))
    by_family: dict[str, Json] = {}
    for family in sorted(families):
        in_band = sum(1 for row in band if row["family"] == family)
        drawn = allocation.get(family, 0)
        by_family[family] = {"train_prompts": families[family], "band": in_band,
                             "all_correct_available": len(all_correct.get(family, [])),
                             "all_correct_sampled": drawn, "pool": in_band + drawn}
    total = len(seen)
    noop = by_family.get("no_op") or {}
    noop_band, noop_pool = int(cast(int, noop.get("band", 0))), int(cast(int, noop.get("pool", 0)))
    return {
        "pool": pool,
        "counts": {
            "train_prompts": total, "band": len(band), "all_correct_available": sum(map(len, all_correct.values())),
            "all_correct_sampled": len(sampled), "pool": len(pool),
            "M": round(len(band) / total, 6) if total else None,
            "all_correct_share_of_pool": round(len(sampled) / len(pool), 6) if pool else None,
            "noop_share_in_band": round(noop_band / len(band), 6) if band else None,
            "noop_share_of_pool": round(noop_pool / len(pool), 6) if pool else None,
            "by_family": by_family,
        },
    }


def pool_row(row: Mapping[str, object], stratum: str) -> Json:
    return {"schema": POOL_ROW_SCHEMA, "id": row["id"], "family": row["family"], "stratum": stratum,
            "exact_count": row["exact_count"], "pass_rate": row["pass_rate"],
            "census_mean_completion_tokens": row.get("mean_completion_tokens"),
            "census_mean_reward": row.get("mean_reward")}


def read_jsonl(path: Path) -> list[Json]:
    with Path(path).open() as stream:
        return [cast(Json, json.loads(line)) for line in stream if line.strip()]


def read_pool(path: Path, expected_sha256: str) -> list[Json]:
    """The committed pool, hash-checked, with one row per distinct TRAIN prompt id."""
    _require(sha256_file(path) == expected_sha256, f"pool {path} differs from its pinned hash")
    rows = read_jsonl(path)
    ids = [str(row["id"]) for row in rows]
    _require(rows and len(set(ids)) == len(ids) and all(row.get("schema") == POOL_ROW_SCHEMA for row in rows),
             "pool rows are empty, duplicated or of the wrong schema")
    return rows


# Bundle --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BundleEntry:
    id: str
    family: str
    prompt_ids: tuple[int, ...]
    spec: Json


def bundle_row(spec: object, prompt_ids: Sequence[int]) -> Json:
    """One bundle row from a reward v2.1 ``ReferenceSpec`` and the prompt's token IDs."""
    from sepalith.training.rl.reward_v21 import ReferenceSpec

    value = cast(ReferenceSpec, spec)
    evidence = value.evidence
    return {"schema": BUNDLE_ROW_SCHEMA, "id": value.row_id, "family": value.family, "prompt_ids": list(prompt_ids),
            "spec": {"target_operation": value.target_operation, "target_body_text": value.target_body_text,
                     "context": value.context.to_dict(),
                     "evidence": {"mode": evidence.mode, "baseline_text": evidence.baseline_text,
                                  "diagnostic_suffix": evidence.diagnostic_suffix,
                                  "parser_r_version": evidence.parser_r_version}}}


def reference_spec(entry: BundleEntry) -> object:
    """The reward v2.1 ``ReferenceSpec`` of a bundle entry."""
    from sepalith.campaign_protocol import PromptContext
    from sepalith.training.rl.reward_v21 import Evidence, ReferenceSpec

    spec = entry.spec
    evidence = cast(Json, spec["evidence"])
    return ReferenceSpec(entry.id, entry.family, PromptContext.from_mapping(spec["context"]), str(spec["target_operation"]),
                         str(spec["target_body_text"]),
                         Evidence(str(evidence["mode"]), cast("str | None", evidence["baseline_text"]),
                                  str(evidence["diagnostic_suffix"]), cast("str | None", evidence["parser_r_version"])))


def load_bundle(path: Path, expected_sha256: str, ids: Sequence[str]) -> dict[str, BundleEntry]:
    """Bundle rows for exactly ``ids``, hash-checked; prompts keep the PRM03 one-BOS geometry."""
    _require(sha256_file(path) == expected_sha256, f"bundle {path} differs from its pinned hash")
    out: dict[str, BundleEntry] = {}
    for row in read_jsonl(path):
        _require(row.get("schema") == BUNDLE_ROW_SCHEMA, "bundle row schema differs")
        prompt = tuple(int(token) for token in cast(list[int], row["prompt_ids"]))
        _require(prompt and prompt[0] == 0 and prompt.count(0) == 1 and 1 not in prompt, f"bundle prompt {row['id']} geometry")
        entry = BundleEntry(str(row["id"]), str(row["family"]), prompt, cast(Json, row["spec"]))
        _require(entry.id not in out, f"duplicate bundle id {entry.id}")
        out[entry.id] = entry
    _require(sorted(out) == sorted(ids), "bundle ids differ from the pool")
    return out


def build_bundle(pool: Sequence[Mapping[str, object]], census_prompts: Path) -> list[Json]:
    """Bundle rows for the pool from the hash-pinned TRAIN inputs (loads all 15,006 rows once)."""
    from sepalith.training.rl.length_policy import TRAIN_ROWS_V4_BRACEINDENT_SHA256
    from sepalith.training.rl.reward_v21 import DATA_ROOT, TRAIN_INPUTS, load_train_specs

    wanted = {str(row["id"]) for row in pool}
    specs, _targets, _identity = load_train_specs()
    by_id = {spec.row_id: spec for spec in specs if spec.row_id in wanted}
    _require(set(by_id) == wanted, "pool ids missing from the TRAIN reward specs")
    prompts: dict[str, list[int]] = {}
    rows_path = DATA_ROOT / TRAIN_INPUTS["rows"][0]
    _require(sha256_file(rows_path) == TRAIN_ROWS_V4_BRACEINDENT_SHA256, "TRAIN rows differ from the v4 pin")
    with rows_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in wanted:
                _require(row.get("split") == "train", f"row {row['id']} is not TRAIN")
                prompts[row["id"]] = list(row["input_ids"][:row["target_start"]])
    _require(set(prompts) == wanted, "pool ids missing from the TRAIN rows")
    _require(sha256_file(census_prompts) == CENSUS_PROMPTS_SHA256, "census prompts differ from their pin")
    with census_prompts.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in wanted:
                _require(row["prompt_token_ids"] == prompts[row["id"]], f"census prompt {row['id']} differs from TRAIN")
    out = []
    for identifier in sorted(wanted):
        value = bundle_row(by_id[identifier], prompts[identifier])
        again = reference_spec(BundleEntry(identifier, str(value["family"]), tuple(prompts[identifier]), cast(Json, value["spec"])))
        _require(again == by_id[identifier], f"bundle spec for {identifier} does not round-trip")
        out.append(value)
    return out


def build(docs_dir: Path, *, census_dir: Path = CENSUS_DIR, bundle_root: Path = BUNDLE_ROOT,
          repository_root: Path | None = None, with_bundle: bool = True) -> Json:
    summary_path, rates_path = census_dir / "census-summary.json", census_dir / "pass-rates.jsonl"
    _require(sha256_file(summary_path) == CENSUS_SUMMARY_SHA256, "census summary differs from the T = 1.0 arm A pin")
    _require(sha256_file(rates_path) == PASS_RATES_SHA256, "pass rates differ from their pin")
    summary = cast(Json, json.loads(summary_path.read_text()))
    _require(summary.get("model_id") == MODEL_ID and summary.get("samples_per_prompt") == SAMPLES_PER_PROMPT
             and summary.get("final_set_access") is False, "census summary identity differs")
    selection = select_pool(read_jsonl(rates_path))
    counts = cast(Json, selection["counts"])
    _require(abs(cast(float, counts["M"]) - 0.125083) < 5e-7, "census M differs from the recorded 12.5%")
    pool = cast(list[Json], selection["pool"])
    root = (repository_root or Path(__file__).resolve().parents[6]).resolve()
    pool_path = docs_dir.resolve() / "pool" / f"{POOL_NAME}.jsonl"
    pool_sha = write_new_jsonl(pool_path, pool)
    manifest: Json = {
        "schema": POOL_SCHEMA, "card": "C14", "step": 1, "name": POOL_NAME,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "census": {"directory": str(census_dir), "model_id": MODEL_ID,
                   "model_safetensors_sha256": "befe587a0d4969e33b0daaa8fe66247371ec32f331616db74724616878ad6030",
                   "summary_sha256": CENSUS_SUMMARY_SHA256, "pass_rates_sha256": PASS_RATES_SHA256,
                   "census_prompts_sha256": CENSUS_PROMPTS_SHA256, "sampling": CENSUS_SAMPLING,
                   "reward": cast(Json, summary["scorer"])["policy"]},
        "rule": {"band": f"{BAND_MIN_EXACT} <= exact <= {BAND_MAX_EXACT} of {SAMPLES_PER_PROMPT} (0.125 <= pass <= 0.875)",
                 "all_correct": f"{ALL_CORRECT_SHARE:.0%} of the finished pool, round(band / 9), allocated to families "
                                "in proportion to the all-correct set (largest remainder), seeded sample per family",
                 "noop": "natural share inside the band (no re-weighting)", "seed": SEED},
        "counts": counts,
        "pool": {"path": str(pool_path.relative_to(root)), "sha256": pool_sha, "rows": len(pool)},
        "scope": "TRAIN only: C12 census pass rates on TRAIN prompts and the v4 TRAIN rows; no DEV row, no sealed-set access",
        "final_set_access": False,
    }
    if with_bundle:
        bundle = build_bundle(pool, census_dir / "census-prompts.jsonl")
        bundle_path = bundle_root / "bundle.jsonl"
        manifest["bundle"] = {"path": str(bundle_path), "sha256": write_new_jsonl(bundle_path, bundle), "rows": len(bundle),
                              "content": "prompt token IDs (TRAIN input_ids[:target_start], equal to the census prompts) "
                                         "and reward v2.1 reference specs; local only"}
    manifest_path = docs_dir.resolve() / f"{POOL_NAME}.json"
    write_new_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), "sha256": sha256_file(manifest_path), "counts": counts}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("build")
    run.add_argument("--docs-dir", type=Path, required=True)
    run.add_argument("--census-dir", type=Path, default=CENSUS_DIR)
    run.add_argument("--bundle-root", type=Path, default=BUNDLE_ROOT)
    args = parser.parse_args(argv)
    result = build(args.docs_dir, census_dir=args.census_dir, bundle_root=args.bundle_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
