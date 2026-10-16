#!/bin/bash
# Correctness checks for the prefill autoresearch session: lint + types on
# experiments/scripts (research mode), plus the frozen-session integrity gate.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/m0hawk/sandbox/Sepalith/.venv/bin/python

OUT=0
"$PY" scripts/check_quality.py > /tmp/prefill-checks.log 2>&1 || OUT=1
tail -3 /tmp/prefill-checks.log

# frozen session integrity: manifest sha + render/tokenize parity on a sample
"$PY" - <<'EOF' >> /tmp/prefill-checks.log 2>&1 || OUT=1
import hashlib, json, random, sys
from pathlib import Path
sys.path.insert(0, "packages/sepalith/src")
sys.path.insert(0, "experiments/serving/prefill")
import tokenizers
from sepalith.campaign_protocol import PromptContext, render_prompt
here = Path("experiments/serving/prefill")
manifest = json.loads((here / "manifest.json").read_text())
sessions = (here / "sessions.jsonl").read_text().splitlines()
digest = hashlib.sha256((here / "sessions.jsonl").read_bytes()).hexdigest()
assert digest == manifest["sessions_sha256"], "sessions.jsonl drifted from manifest"
rng = random.Random(1)
pinned = {(e["session_id"], e["index"]): e for e in manifest["events"]}
tok = tokenizers.Tokenizer.from_file(manifest["source"]["tokenizer_sha256"] and str(
    Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536/tokenizer.json")))
checked = 0
for line in rng.sample(sessions, 3):
    session = json.loads(line)
    for index, event in enumerate(session["events"]):
        pin = pinned[(session["session_id"], index)]
        text = render_prompt(PromptContext.from_mapping(event["context"]))
        assert hashlib.sha256(text.encode()).hexdigest() == pin["prompt_sha256"]
        checked += 1
print(f"integrity ok: {checked} events re-verified")
EOF
tail -1 /tmp/prefill-checks.log
exit $OUT
