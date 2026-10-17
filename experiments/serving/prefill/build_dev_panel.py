#!/usr/bin/env python3
"""Convert a DEV panel (PRM03 contexts) into bench sessions for the ONE
allowed final latency report on real prompts (brief §5: DEV75/DEV250 may be
used for a single final latency report; every use is recorded).

Each case becomes its own session with a single ``open`` event: per-prompt
cold latency, as a reviewer walking unrelated files would see.
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEV75)
    parser.add_argument("--out-dir", type=Path, default=HERE / "dev75-latency")
    parser.add_argument("--limit", type=int, default=0, help="0 = all cases")
    args = parser.parse_args()

    import tokenizers

    tokenizer = tokenizers.Tokenizer.from_file(
        str(
            Path(
                "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/"
                "full/checkpoint-536/tokenizer.json"
            )
        )
    )
    digest = hashlib.sha256(args.panel.read_bytes()).hexdigest()
    if digest != DEV75_SHA:
        raise SystemExit(f"panel sha mismatch: {digest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sessions = []
    events = []
    count = 0
    for line in args.panel.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        context = case["context"]
        text = render_prompt(PromptContext.from_mapping(context))
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        session_id = f"dev75-{count:03d}"
        sessions.append(
            {
                "session_id": session_id,
                "package_id": case.get("package_id"),
                "family": case.get("family"),
                "base_row_id": session_id,
                "switch_row_id": None,
                "events": [{"type": "open", "context": context}],
            }
        )
        events.append(
            {
                "session_id": session_id,
                "index": 0,
                "type": "open",
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_tokens": len(ids),
            }
        )
        count += 1
        if args.limit and count >= args.limit:
            break
    sessions_path = args.out_dir / "sessions.jsonl"
    with sessions_path.open("w", encoding="utf-8") as out:
        for session in sessions:
            out.write(json.dumps(session) + "\n")
    manifest = {
        "schema": "sepalith.prefill.sessions.devpanel.v1",
        "panel": {"path": str(args.panel), "sha256": digest, "cases": count},
        "seed": None,
        "n_sessions": count,
        "sessions_sha256": hashlib.sha256(sessions_path.read_bytes()).hexdigest(),
        "events": events,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"sessions": count, "out": str(args.out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
