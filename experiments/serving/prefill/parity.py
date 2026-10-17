#!/usr/bin/env python3
"""Recompute output parity of a bench records file against a reference.

Usage: parity.py RECORDS.jsonl REFERENCE.json [--decode]
Prints agreement per event type and overall, plus (with --decode) the first
few textual diffs for eyeballing whether mismatches are benign.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "sepalith" / "src"))

DEFAULT_TOKENIZER = Path(
    "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/"
    "full/checkpoint-536/tokenizer.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()

    reference: Mapping[str, Sequence[int]] = json.loads(
        args.reference.read_text(encoding="utf-8")
    )
    tokenizer = None
    if args.decode:
        import tokenizers

        tokenizer = tokenizers.Tokenizer.from_file(str(DEFAULT_TOKENIZER))

    by_type: dict[str, list[int]] = {}
    diffs: list[tuple[str, list[int], list[int]]] = []
    total = match = 0
    positional: dict[tuple[int, str], int] = {}
    for line in args.records.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "index" in record:
            key = f'{record["session_id"]}:{record["index"]}:{record["pass"]}'
        else:
            counter = positional.setdefault((record["pass"], record["session_id"]), 0)
            key = f'{record["session_id"]}:{counter}:{record["pass"]}'
            positional[(record["pass"], record["session_id"])] = counter + 1
        expected = reference.get(key)
        if expected is None:
            continue
        total += 1
        ok = list(record["output_ids"]) == list(expected)
        match += int(ok)
        hits = by_type.setdefault(record["type"], [0, 0])
        hits[0] += int(ok)
        hits[1] += 1
        if not ok and len(diffs) < 5 and tokenizer is not None:
            diffs.append((key, record["output_ids"], list(expected)))

    print(json.dumps({
        "overall": {"match": match, "total": total, "rate": round(match / total, 4) if total else None},
        "by_type": {k: {"match": v[0], "total": v[1]} for k, v in sorted(by_type.items())},
    }, indent=1))
    for key, got, want in diffs:
        print(f"--- {key}")
        print("got :", repr(tokenizer.decode(got, skip_special_tokens=False)[:220]))
        print("want:", repr(tokenizer.decode(want, skip_special_tokens=False)[:220]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
