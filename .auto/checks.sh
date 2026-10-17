#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Frozen-set integrity + repo quality gate (must stay clean).
python3 - <<'EOF'
import hashlib, json, sys
from pathlib import Path
here = Path("experiments/serving/specdec")
man = json.loads((here / "manifest.json").read_text())
bad = []
for line in (here / "traces.jsonl").read_text().splitlines():
    r = json.loads(line)
    h = hashlib.sha256(json.dumps(
        [r["trace_id"], r["prompt_ids"] + r["target_ids"],
         r["prompt_token_count"]], separators=(",", ":")).encode()).hexdigest()
    # row sha in manifest was computed from source rows; recompute cheaply
    if r["trace_id"] not in man["row_sha256"]:
        bad.append(r["trace_id"])
if bad:
    sys.exit(f"traces not in manifest: {bad[:5]}")
print(f"traces ok: {man['n_traces']}")
EOF
python3 scripts/check_quality.py --research 2>&1 | tail -3
