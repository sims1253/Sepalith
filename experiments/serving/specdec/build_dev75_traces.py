#!/usr/bin/env python3
"""Build the DEV75 final-report trace set (THE one allowed DEV use).

Converts the corrected DEV75 panel (PRM03 contexts, sha-pinned — the same
panel the prefill mission recorded) into token-id traces for the specdec
bench. Every use of DEV data is recorded: this script writes a manifest with
the panel sha256 and the derived row ids next to the traces.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "packages" / "sepalith" / "src"))

from sepalith.campaign_protocol import PromptContext, render_prompt  # noqa: E402

HERE = Path(__file__).resolve().parent
DEV75 = Path(
    "/mnt/e/sepalith/campaign-20260915/campaign-overlay/work/lead/"
    "corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
)
DEV75_SHA = "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035"
TOKENIZER = Path(
    "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/"
    "full/checkpoint-536/tokenizer.json"
)


def main() -> None:
    digest = hashlib.sha256(DEV75.read_bytes()).hexdigest()
    if digest != DEV75_SHA:
        raise SystemExit(f"panel sha mismatch: {digest}")

    import tokenizers

    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    rows = []
    for line in DEV75.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        text = render_prompt(PromptContext.from_mapping(case["context"]))
        ids = [0] + tok.encode(text, add_special_tokens=False).ids  # BOS + prompt
        n = len(ids)
        rows.append(
            {
                "trace_id": f"dev75-{case['id']}",
                "family": case.get("family"),
                "package_id": case.get("package_id"),
                "band": "short" if n < 1000 else "long",
                "is_noop": case.get("operation") == "no_op"
                or case.get("family") == "no_op",
                "prompt_ids": ids,
                "prompt_token_count": n,
                "target_token_count": case.get("target_body_token_count", 0) + 5,
                "target_ids": [],
            }
        )

    out = HERE / "dev75_traces.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    manifest = {
        "schema": "sepalith.specdec.dev75.v1",
        "panel": str(DEV75),
        "panel_sha256": DEV75_SHA,
        "purpose": "final report of at most 3 configurations (brief §8) — speed only",
        "n": len(rows),
        "ids": [r["trace_id"] for r in rows],
        "prompt_tokens": {
            "min": min(r["prompt_token_count"] for r in rows),
            "median": sorted(r["prompt_token_count"] for r in rows)[len(rows) // 2],
            "max": max(r["prompt_token_count"] for r in rows),
        },
    }
    (HERE / "dev75_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["prompt_tokens"]), f"-> {out}")


if __name__ == "__main__":
    main()
