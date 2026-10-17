#!/usr/bin/env python3
"""Simulate byte-prefix cache reuse under alternative PRM03 section orders.

The shipped renderer (zeta2-prm03-v1) emits sections in a fixed order with
edit_history in the middle, so every accepted edit invalidates the whole tail.
This script does NOT change the shipped renderer: it re-assembles the same
session contexts under candidate orderings and measures, per event, the
longest-common-prefix fraction that llama.cpp's ``--cache-prompt`` would reuse
(token-identical prefix semantics), plus the estimated prefill milliseconds at
a measured per-token rate.  Output feeds the retraining proposal in the
prefill report.

Orderings:
  prm03       shipped order (verified byte-identical to render_prompt)
  stable_first contract, path, prefix, references, retrieval, diagnostics,
              history, scope+suffix, region  (volatility monotonically last)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "packages" / "sepalith" / "src"))

from sepalith.campaign_protocol import (  # noqa: E402
    CURSOR_MARKER,
    TASK_CONTRACT,
    PromptContext,
    render_prompt,
)

HERE = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = Path(
    "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/"
    "full/checkpoint-536/tokenizer.json"
)
# measured pp512 rate, Q8_0, 8 threads, contended host (conservative)
MS_PER_TOKEN = 1000.0 / 100.0


def region_lines(context: dict[str, Any]) -> list[str]:
    region = list(context["region_old"])
    cursor = context["cursor"]
    index = cursor["region_line_index"]
    if 0 <= index < len(region):
        line = region[index]
        column = cursor["code_point_column"] or 0
        region[index] = line[:column] + CURSOR_MARKER + line[column:]
    return region


def _parts(context: dict[str, Any], order: str) -> list[str]:
    """Section parts for an ordering (mirrors render_prompt join semantics)."""
    contract = TASK_CONTRACT
    filename = f"<filename>{context['path']}"
    prefix = list(context["prefix"])
    refs_head = "<filename>selected_references"
    refs = _records(context["selected_references"])
    hist_head = "<filename>edit_history"
    hist = [e["event_diff"] for e in context["history"]]
    diag_head = "<filename>diagnostics"
    diag = _records(context["diagnostics"])
    ret_head = "<filename>retrieval"
    ret = _records(context["retrieval"])
    suffix_head = "<[fim-suffix]>"
    suffix = list(context["scope_lines"]) + list(context["suffix_lines"])
    current = "<<<<<<< CURRENT"
    region = region_lines(context)
    tail = ["=======", "<[fim-middle]>"]
    if order == "prm03":
        return [
            contract, filename, *prefix,
            refs_head, *refs,
            hist_head, *hist,
            diag_head, *diag,
            ret_head, *ret,
            suffix_head, *suffix, current,
            *region, *tail,
        ]
    if order == "stable_first":
        return [
            contract, filename, *prefix,
            refs_head, *refs,
            ret_head, *ret,
            diag_head, *diag,
            hist_head, *hist,
            suffix_head, *suffix, current,
            *region, *tail,
        ]
    if order == "history_cap1":
        # PRM03 order with the edit history truncated to its newest event —
        # an extension-side budget policy (changes what the model sees:
        # proposal-only, needs a quality gate before adoption).
        return [
            contract, filename, *prefix,
            refs_head, *refs,
            hist_head, *hist[-1:],
            diag_head, *diag,
            ret_head, *ret,
            suffix_head, *suffix, current,
            *region, *tail,
        ]
    raise ValueError(f"unknown order {order}")


def _records(records: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for record in records:
        if record.get("path"):
            out.append(f"<filename>{record['path']}")
        out.append(record["content"])
    return out


def render_custom(context: dict[str, Any], order: str) -> str:
    return "\n".join(_parts(context, order)) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=HERE / "sessions.jsonl")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=HERE / "layout_simulation.json")
    args = parser.parse_args()

    import tokenizers

    tokenizer = tokenizers.Tokenizer.from_file(str(args.tokenizer))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pinned = {(e["session_id"], e["index"]): e for e in manifest["events"]}

    orders = ["prm03", "stable_first", "history_cap1"]
    per_order: dict[str, dict[str, list[float]]] = {o: {} for o in orders}
    tokens_per_order: dict[str, dict[str, list[int]]] = {o: {} for o in orders}
    prm03_verified = 0
    for line in args.sessions.read_text(encoding="utf-8").splitlines():
        session = json.loads(line)
        prev: dict[str, list[int]] = {}
        for index, event in enumerate(session["events"]):
            pin = pinned[(session["session_id"], index)]
            for order in orders:
                text = (
                    render_prompt(PromptContext.from_mapping(event["context"]))
                    if order == "prm03"
                    else render_custom(event["context"], order)
                )
                if order == "prm03":
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if digest == pin["prompt_sha256"]:
                        prm03_verified += 1
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                previous = prev.get(order)
                if previous is not None:
                    k = 0
                    for a, b in zip(previous, ids):
                        if a != b:
                            break
                        k += 1
                    frac = k / len(ids) if ids else 0.0
                    per_order[order].setdefault(event["type"], []).append(frac)
                    tokens_per_order[order].setdefault(event["type"], []).append(
                        len(ids) - k
                    )
                prev[order] = ids
    result: dict[str, Any] = {
        "schema": "sepalith.prefill.layout-simulation.v1",
        "prm03_render_verified_events": prm03_verified,
        "ms_per_token_estimate": MS_PER_TOKEN,
        "orders": {},
    }
    for order in orders:
        rows = {}
        for event_type, fracs in sorted(per_order[order].items()):
            toks = tokens_per_order[order][event_type]
            rows[event_type] = {
                "n": len(fracs),
                "cached_frac_p50": round(statistics.median(fracs), 3),
                "cached_frac_mean": round(statistics.mean(fracs), 3),
                "uncached_tokens_p50": int(statistics.median(toks)),
                "uncached_tokens_mean": round(statistics.mean(toks), 1),
                "est_prefill_ms_p50": round(statistics.median(toks) * MS_PER_TOKEN, 0),
            }
        all_fracs = [f for fracs in per_order[order].values() for f in fracs]
        all_toks = [t for toks in tokens_per_order[order].values() for t in toks]
        rows["_pooled"] = {
            "n": len(all_fracs),
            "cached_frac_p50": round(statistics.median(all_fracs), 3),
            "uncached_tokens_p50": int(statistics.median(all_toks)),
            "est_prefill_ms_p50": round(statistics.median(all_toks) * MS_PER_TOKEN, 0),
        }
        result["orders"][order] = rows
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
