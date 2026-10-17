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
    if r["trace_id"] not in man["row_sha256"]:
        bad.append(r["trace_id"])
if bad:
    sys.exit(f"traces not in manifest: {bad[:5]}")
print(f"traces ok: {man['n_traces']}")
EOF
python3 scripts/check_quality.py 2>&1 | tail -2
# specdec scope: keep new code lint/type clean (experiments/ has pre-existing debt)
timeout 300 uv run --project tools/quality --locked ruff check --config 'include = ["*.py"]' experiments/serving/specdec/ 2>&1 | tail -2
timeout 300 uv run --project tools/quality --locked ty check --error-on-warning --config 'src.include = ["experiments/serving/specdec"]' experiments/serving/specdec 2>&1 | tail -2
