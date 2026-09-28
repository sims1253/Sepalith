#!/usr/bin/env python3
"""Exercise the proposed RL loader against every admitted 15,006 row.

This verifier intentionally streams the 134 MB row file and 159 MB context
sidecar together.  It calls the same fail-closed row, context, semantic and
reward-buffer validators as the proposed training loader without retaining
both decoded JSONL inputs in memory.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import resource
import sys
import time


PACKET = Path(__file__).resolve().parent
TRAINING = PACKET / "source" / "experiments" / "training"
sys.path.insert(0, str(TRAINING))

import campaign_rl_data as data  # noqa: E402


ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl")
CONTEXT = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/context-sidecar.jsonl")
SELECTED = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/selected-train-ids.json")
BUFFER_MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-buffer-v5/materialization.json")
EXPECTED = {
    "rows": "65b2feb2e53970f02e7cfbe8947d628d584c204f741dad3a5a8b3254af5c1cd7",
    "context": "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a",
    "selected": "986a1f7910c44423e50988e76abb6671dedf8cd70f5542ff9efd7a03ce504462",
    "buffer_manifest": "a49b1ef464773817d7a18f76cdec68b82eaa4abb960a10a7d919c7535d7add84",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    started = time.monotonic()
    actual = {
        "rows": sha256(ROWS),
        "context": sha256(CONTEXT),
        "selected": sha256(SELECTED),
        "buffer_manifest": sha256(BUFFER_MANIFEST),
    }
    if actual != EXPECTED:
        raise RuntimeError(f"input identity mismatch: {actual!r}")
    selected = data.load_selected_ids(SELECTED, EXPECTED["selected"])
    buffer_index = data.RewardBufferIndex.load(
        BUFFER_MANIFEST, EXPECTED["buffer_manifest"], selected,
    )
    selected_iter = iter(selected)
    counts: Counter[str] = Counter()
    with ROWS.open(encoding="utf-8") as rows_stream, CONTEXT.open(encoding="utf-8") as contexts_stream:
        for position, pair in enumerate(zip_longest(rows_stream, contexts_stream), 1):
            row_line, context_line = pair
            if row_line is None or context_line is None:
                raise RuntimeError(f"row/context cardinality mismatch at {position}")
            row, prompt = data._validate_row(json.loads(row_line))
            sidecar = json.loads(context_line)
            expected_id = next(selected_iter)
            if row["id"] != expected_id or sidecar.get("row_id") != expected_id:
                raise RuntimeError(f"ordered ID mismatch at {position}")
            (_row_id, context, _capture, source_identity, _geometry) = data._validate_sidecar_record(sidecar, row)
            data._validate_semantics(row, context)
            envelope = buffer_index.envelope_for(expected_id)
            if envelope["row_id"] != expected_id:
                raise RuntimeError(f"buffer join mismatch at {position}")
            counts[f"expanded_context_source:{sidecar.get('expanded_context_source', 'absent')}"] += 1
            counts[f"syntax_mode:{envelope['syntax_evidence_mode']}"] += 1
            if "source_ref" not in sidecar["source_identity"]:
                counts["document_level_source_identity"] += 1
                if source_identity["source_ref"]["source"] != "accepted_short_packet_source_document":
                    raise RuntimeError(f"short source identity normalization mismatch at {position}")
            if len(prompt) > 2048:
                counts["prompt_gt_prior_2048"] += 1
            completion = len(row["input_ids"]) - row["target_start"]
            if completion > 192:
                counts["completion_gt_prior_192"] += 1
            counts["rows"] += 1
    try:
        next(selected_iter)
    except StopIteration:
        pass
    else:
        raise RuntimeError("selected ID list has trailing rows")
    if counts["rows"] != 15006:
        raise RuntimeError(f"expected 15006 rows, found {counts['rows']}")
    result = {
        "schema": "sepalith.rl11.context15006.loader-compat-verification.v1",
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_r_executed": False,
        "inputs": {name: {"path": str(path), "sha256": actual[name]} for name, path in {
            "rows": ROWS, "context": CONTEXT, "selected": SELECTED,
            "buffer_manifest": BUFFER_MANIFEST,
        }.items()},
        "checks": [
            "all input hashes exact",
            "selected/row/context order exact",
            "canonical training-row and token geometry valid",
            "PromptContext render and semantic geometry valid",
            "source identity and selection geometry valid",
            "reward buffer joins every eligible row and refuses held IDs",
        ],
        "counts": dict(sorted(counts.items())),
        "limits": {
            "prompt_max_tokens": data.PROMPT_MAX_TOKENS,
            "completion_max_tokens": data.COMPLETION_MAX_TOKENS,
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    output = PACKET / "loader-compatibility.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
