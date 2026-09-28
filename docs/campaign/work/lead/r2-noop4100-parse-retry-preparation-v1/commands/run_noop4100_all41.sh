#!/usr/bin/env bash
set -euo pipefail
PACKET="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_ROOT="/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-reconstruction-v1/inputs-full41"
OUTPUT_ROOT="/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/render16-parse-retry-v1"
OLD_OUTPUT_ROOT="/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/render16"
PLAN="$PACKET/inputs/frozen-plan16.json"
PLAN_SHA256="093fd437f6cc4a364b7273e7936459c836a18a31bc06fd84ec39ec88faacd725"
[[ "$OUTPUT_ROOT" != "$OLD_OUTPUT_ROOT" ]] || { echo 'refusing to use preserved old output root' >&2; exit 3; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "fresh retry root required: $OUTPUT_ROOT" >&2; exit 3; }
python3 - "$PACKET/source-manifest.json" "$PACKET" "$PLAN" "$PLAN_SHA256" "$INPUT_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest,packet,plan_path,plan_sha,input_root=sys.argv[1:]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()
m=json.load(open(manifest,encoding='utf-8')); p=json.load(open(plan_path,encoding='utf-8'))
assert sha(plan_path)==plan_sha==m['frozen_plan16_sha256']
for item in m['files']:
    path=f"{packet}/{item['path']}"
    assert sha(path)==item['sha256'] and Path(path).stat().st_size==item['bytes']
assert m['status']=='prepared_no_launch' and m['input_root']==input_root
assert p['status']=='prepared_no_launch' and p['rows']==4100 and p['shards']==41
assert sum(int(x['rows']) for x in p['entries'])==4100
assert sum(int(x['rows'])>0 for x in p['entries'])==31
assert p['context_size']==16384 and p['generation_reserve']==2048
print('PASS pinned 41-shard input plan: 4100 rows, 31 nonempty shards')
PY
if [[ "${1:-}" == "--check-only" ]]; then
  echo 'PASS source and plan gates; no output directory created and no provider launched'
  exit 0
fi
mkdir -p "$OUTPUT_ROOT"

# Keep infrastructure failures visible: each lane attempts every nonempty
# shard, records the shard terminal, and returns aggregate failure if needed.
run_lane() {
  local core="$1"; shift
  local failures=()
  for shard in "$@"; do
    if ! bash "$PACKET/run_shard.sh" "$shard" "$core" 16384 2048 "$INPUT_ROOT" "$OUTPUT_ROOT" "$PLAN"; then
      failures+=("$shard")
    fi
  done
  if ((${#failures[@]})); then
    printf 'failed shards on core %s: %s\n' "$core" "${failures[*]}" >&2
    return 1
  fi
}

run_lane 4 0012 0015 0016 0018 0020 0024 0025 0026 0027 0028 0029 0031 0034 0036 0037 & lane0=$!
run_lane 6 0010 0011 0013 0014 0017 0019 0021 0022 0023 0030 0032 0033 0035 0038 0039 0040 & lane1=$!
if wait "$lane0"; then rc0=0; else rc0=$?; fi
if wait "$lane1"; then rc1=0; else rc1=$?; fi
((rc0==0 && rc1==0))
