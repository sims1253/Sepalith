#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01
DRIVER=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-sourcewalk-independent-replay-v3/full_replay.py
EXPECTED_DRIVER=63d165ae1488c1f38174f7e24f68a51e4c80c2d48e5aec44a0446fa062a61f7f
EXPECTED_INDEX=65637a9e05c66647de042d46f42bf9afa197a0b068f63680ec9f3ac0dfe922a1
DEPS=/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/parser-dependencies-root-v1

[[ $(sha256sum "$DRIVER" | cut -d' ' -f1) == "$EXPECTED_DRIVER" ]] || { echo driver_hash_mismatch >&2; exit 65; }
[[ -f "$ROOT/index/manifest.json" ]] || { echo replay_index_missing >&2; exit 66; }
[[ $(sha256sum "$ROOT/index/manifest.json" | cut -d' ' -f1) == "$EXPECTED_INDEX" ]] || { echo replay_index_hash_mismatch >&2; exit 67; }
[[ -f "$DEPS/manifest.json" ]] || { echo parser_dependency_manifest_missing >&2; exit 68; }

exec 9>"$ROOT/.owner.lock"
flock -n 9 || { echo output_owned_by_another_process >&2; exit 69; }

missing=$(ROOT="$ROOT" /usr/bin/python3 -B - <<'PY'
import hashlib, json, os
from pathlib import Path

root = Path(os.environ['ROOT'])
expected = list(range(27, 41))
missing = []
for shard in expected:
    receipt_path = root / 'shards' / f'shard-{shard:04d}' / 'receipt.json'
    if not receipt_path.is_file():
        missing.append(shard)
        continue
    value = json.loads(receipt_path.read_text())
    outputs = value.get('outputs')
    if value.get('status') != 'complete' or value.get('shard') != shard or not isinstance(outputs, list) or len(outputs) != 1:
        raise SystemExit(f'future_receipt_invalid:{shard}')
    output = outputs[0]
    path = Path(output.get('path', ''))
    if path != root / 'shards' / f'shard-{shard:04d}' / 'ledger.jsonl':
        raise SystemExit(f'future_receipt_path_invalid:{shard}')
    if not path.is_file() or path.stat().st_size != output.get('bytes'):
        raise SystemExit(f'future_receipt_output_invalid:{shard}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != output.get('sha256'):
        raise SystemExit(f'future_receipt_output_hash_invalid:{shard}')
print(','.join(map(str, missing)))
PY
)

if [[ -z "$missing" ]]; then
  echo all_27_40_receipts_already_valid
  exit 0
fi

echo "replay_missing_shards=$missing" >&2
exec /usr/bin/python3 -B "$DRIVER" replay --output "$ROOT" --shards "$missing"
