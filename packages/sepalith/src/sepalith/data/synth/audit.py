"""Audit pack: a stratified random sample of SYNTH-v1 cases as readable Markdown.

For each sampled case: identity and generator metadata, the verification
record, a unified diff of the edited window (before -> after), the target, and
the exact rendered prompt (``prompt_text`` of the drop-in row).

    python -m sepalith.data.synth.audit --data /mnt/e/.../SYNTH-v1 --out AUDIT.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import difflib
import hashlib
import json
from pathlib import Path
import random

# family -> stratum key -> sample count (60 in total). Strata with fewer cases
# than requested are taken whole and the remainder goes to the largest stratum.
STRATA = {
    "finish_block": {"constrained_short": 22, "sibling:short": 6, "sibling:medium": 16, "sibling:long": 4},
    "roxygen_drafting": {"roxy_full": 12},
}
# Independent of the arm A probe's seed (3407); the probe also excludes these ids.
SEED = 20261022


def stratum(record: dict) -> str:
    meta = record["meta"]
    if record["family"] != "finish_block":
        return meta["rule"]
    if meta.get("determinacy") == "constrained_short":
        return "constrained_short"
    return f"{meta.get('determinacy')}:{meta.get('bucket')}"


def windows(record: dict) -> tuple[list[str], list[str]]:
    context = record["context"]
    before = [*context["prefix"], *context["region_old"], *context["suffix_lines"]]
    after = [*context["prefix"], *record["region_new"], *context["suffix_lines"]]
    return before, after


def render_case(index: int, record: dict, row: dict) -> str:
    meta = record["meta"]
    checks = record["verification"]
    before, after = windows(record)
    diff = "\n".join(difflib.unified_diff(before, after, "before", "after", lineterm="", n=3))
    keep = {k: meta[k] for k in ("rule", "variant", "bucket", "target_lines", "novelty", "novel_tokens",
                                 "structural_support", "determinacy", "behaviour", "full_body", "test_file",
                                 "determined_share", "tag_order", "unit", "seed", "signature_region") if k in meta}
    verdict = {k: checks[k] for k in ("passed", "parse_after", "parse_before_status", "air_idempotent",
                                      "reward_exact", "reward", "reward_comparison", "reward_flags",
                                      "prompt_tokens", "completion_tokens", "family_problems") if k in checks}
    parts = [
        f"## {index}. `{record['row_id']}` {record['family']} / {stratum(record)}",
        "",
        f"- package `{record['package_id']}`, path `{record['context']['path']}`",
        f"- generator: `{json.dumps(keep, sort_keys=True)}`",
        f"- verification: `{json.dumps(verdict, sort_keys=True)}`",
        "",
        "**Edit (window diff, before -> after)**",
        "",
        "```diff",
        diff,
        "```",
        "",
        "**Target (gold region)**",
        "",
        "```r",
        "\n".join(record["region_new"]),
        "```",
        "",
        "<details><summary>Rendered prompt (prompt_text)</summary>",
        "",
        "````text",
        row["prompt_text"].rstrip("\n"),
        "````",
        "",
        "</details>",
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids-out", type=Path, default=None)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.data / "cases.jsonl").open()]
    rows = {}
    with (args.data / "train-token-rows.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["id"]] = row
    pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        pools[(record["family"], stratum(record))].append(record)
    rng = random.Random(SEED)
    chosen = []
    for family, strata in STRATA.items():
        spare = 0
        for key, count in strata.items():
            pool = sorted(pools.get((family, key), []), key=lambda r: r["row_id"])
            take = min(count, len(pool))
            spare += count - take
            chosen += rng.sample(pool, take)
        if spare:
            largest = max(strata, key=lambda k: len(pools.get((family, k), [])))
            rest = [r for r in sorted(pools.get((family, largest), []), key=lambda r: r["row_id"]) if r not in chosen]
            chosen += rng.sample(rest, min(spare, len(rest)))
    manifest_sha = hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    summary_sha = hashlib.sha256((args.data / "build-summary.json").read_bytes()).hexdigest()
    header = [
        f"# {args.data.name} audit sample",
        "",
        f"Bound to `{args.data}/manifest.json` sha256 `{manifest_sha}` "
        f"(build-summary.json `{summary_sha}`).",
        "",
        f"Stratified random sample (seed {SEED}) of {len(chosen)} of {len(records)} cases. "
        f"Strata: {json.dumps(STRATA)}.",
        "",
        "Each case shows the edited window as a diff (prefix + region + suffix, before and after the gold edit),",
        "the gold target, the verification record and the exact rendered prompt of the drop-in TRAIN row.",
        "",
    ]
    body = [render_case(i + 1, record, rows[record["row_id"]]) for i, record in enumerate(chosen)]
    args.out.write_text("\n".join(header + body), encoding="utf-8")
    if args.ids_out:
        args.ids_out.write_text(json.dumps([r["row_id"] for r in chosen], indent=1) + "\n")
    print(f"wrote {len(chosen)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
