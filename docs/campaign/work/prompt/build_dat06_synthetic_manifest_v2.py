"""Build/check the bounded synthetic DAT-06 v2 48k manifest.

This is a functional sampler fixture, not a source-data or training-data
builder.  It creates admission metadata only and imports the sampler from the
read-only execution source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


EXECUTION_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
DEFAULT_OUTPUT = Path(__file__).with_name("DAT-06-48k-draw-manifest-v2.json")
sys.path.insert(0, str(EXECUTION_ROOT / "experiments" / "training"))

from campaign_sampling import build_draw_manifest, validate_draw_manifest  # noqa: E402


def metadata(row_id, family, *, source_id, noop=False, length="short"):
    prompt_tokens, target_tokens = (80, 20)
    if length == "long":
        prompt_tokens, target_tokens = (2480, 20)
    return {
        "row_id": row_id,
        "family": family,
        "source_id": source_id,
        "package_id": "synthetic-package",
        "split": "train",
        "semantic_noop": noop,
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "total_tokens": prompt_tokens + target_tokens,
        "length_bucket": length,
        "naturally_long": length == "long",
        "source_kind": "ordinary",
    }


def synthetic_metadata():
    rows = [
        metadata(
            f"noop-{index:04d}", "no_op",
            source_id=f"noop-source-{index % 30}", noop=True,
        ) for index in range(600)
    ]
    for family in ("alpha", "beta", "gamma", "delta", "epsilon"):
        rows.extend(
            metadata(
                f"{family}-{index:04d}", family,
                source_id=f"{family}-source-{index % 30}",
                length="long" if index < 120 else "short",
            ) for index in range(1080)
        )
    return rows


def render():
    manifest = build_draw_manifest(
        synthetic_metadata(), max_steps=3000, effective_batch=16,
        split_id="synthetic-48k-v1", seed=20260912,
        ordinary_replay_cap=8, naturally_long_fraction=0.20,
    )
    validate_draw_manifest(manifest)
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")
    rendered = render()
    if args.write:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {len(rendered.encode('utf-8'))} bytes: {args.output}")
    else:
        actual = args.output.read_text(encoding="utf-8")
        if actual != rendered:
            raise SystemExit(f"manifest differs: {args.output}")
        print(f"checked {args.output}")
    print(f"sha256={hashlib.sha256(rendered.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
