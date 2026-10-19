#!/usr/bin/env python3
"""Pool the record streams of several bench runs into one summary.

Interleaved A/B passes (one server restart each) are measured as separate runs
so that each arm is sampled in both halves of the session; this pools them with
the *same* aggregation the bench itself uses (``bench.summarize``), so the
numbers in the report are computed exactly like a multi-pass run's.

Usage::

    python pool_runs.py --label m0pad-eager-6t ARTIFACTS/run-m0pad-eager-6t-*.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench  # noqa: E402  (path bootstrap lives in the bench module)


def load_records(paths: list[Path]) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    sources: list[str] = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not rows:
            continue
        sources.append(f"{path.name} (n={len(rows)})")
        records.extend(rows)
    return records, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--label", default="pooled")
    parser.add_argument(
        "--reference",
        type=Path,
        default=HERE / "reference_outputs.json",
        help="token-identity reference (default: the frozen session-flow reference)",
    )
    args = parser.parse_args()

    records, sources = load_records(args.paths)
    if not records:
        print("no records", file=sys.stderr)
        return 1
    reference = None
    if args.reference.exists():
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
    summary = bench.summarize(records, reference)
    summary["label"] = args.label
    summary["sources"] = sources
    summary["host_load1"] = float(open("/proc/loadavg").read().split()[0])

    edit = summary["edit"]
    by_type = summary["by_type"]
    for name in sorted(by_type):
        row = by_type[name]
        print(
            f"METRIC {name}_ttft_p50_ms={row['ttft_p50_ms']} n={row['n']} "
            f"cached={row['cached_frac_p50']} processed_p50={row['processed_n_p50']} "
            f"prompt_tok_p50={row['prompt_tokens_p50']}"
        )
    print(f"METRIC edit_ttft_p50_ms={edit['ttft_p50_ms']} ci={edit['ttft_p50_ci']}")
    print(f"METRIC edit_ttft_p95_ms={edit['ttft_p95_ms']}")
    print(f"METRIC slow_edit_ttft_p50_ms={edit['slow_ttft_p50_ms']}")
    if edit.get("parity_rate") is not None:
        print(f"METRIC parity_rate={edit['parity_rate']} parity_n={edit['parity_n']}")
    print(f"METRIC warmup_total_s={edit['warmup_total_s']} warmup_count={edit['warmup_count']}")
    print(f"METRIC host_load1={summary['host_load1']}")
    print(f"SUMMARY {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())