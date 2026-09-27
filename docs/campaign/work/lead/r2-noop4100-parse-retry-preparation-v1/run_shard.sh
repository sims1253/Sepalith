#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 7 ]]; then echo 'usage: run_shard.sh SHARD CORE CONTEXT RESERVE INPUT_ROOT OUTPUT_ROOT PLAN' >&2; exit 2; fi
shard="$1"; core="$2"; context="$3"; reserve="$4"; input_root="$5"; output_root="$6"; plan_path="$7"
packet="$(cd "$(dirname "$0")" && pwd)"
old_output_root="/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/render16"
expected_input_root="/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-reconstruction-v1/inputs-full41"
[[ "$output_root" != "$old_output_root" ]] || { echo 'refusing to write the preserved old output root' >&2; exit 3; }
[[ "$input_root" == "$expected_input_root" ]] || { echo 'input root differs from pinned reconstruction root' >&2; exit 3; }
mkdir -p "$output_root"
input="$input_root/shard-$shard.jsonl"; output="$output_root/shard-$shard.jsonl"; log="$output_root/shard-$shard.log"; terminal="$output_root/shard-$shard.terminal.json"
[[ -f "$input" && ! -e "$output" && ! -e "$terminal" ]] || { echo 'fresh shard paths required' >&2; exit 3; }
python3 - "$packet/source-manifest.json" "$plan_path" "$shard" "$core" "$context" "$reserve" "$input_root" "$input" <<'PY'
import hashlib,json,sys
manifest,plan_path,shard,core,context,reserve,input_root,input_path=sys.argv[1:]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()
m=json.load(open(manifest,encoding='utf-8')); p=json.load(open(plan_path,encoding='utf-8'))
assert m['status']=='prepared_no_launch' and m['input_root']==input_root
assert p['input_root']==input_root and p['rows']==4100 and p['context_size']==int(context)==16384 and p['generation_reserve']==int(reserve)==2048
assert sha(plan_path)==m['frozen_plan16_sha256']
item=next((x for x in p['entries'] if int(x['shard'])==int(shard)),None)
assert item is not None and item['rows']>0 and item['path']==f'shard-{int(shard):04d}.jsonl'
assert item['core']==int(core) and item['lane']==(0 if int(core)==4 else 1)
assert sha(input_path)==item['sha256']
assert sum(1 for _ in open(input_path,encoding='utf-8'))==item['rows']
PY
started="$(date --iso-8601=ns)"; start_ns="$(date +%s%N)"; exit_code=0
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=30s 1800 taskset -c "$core" nice -n 10 ionice -c 3 \
  node --no-warnings=ExperimentalWarning --experimental-strip-types "$packet/render_shard.ts" \
  "$input" "$output" /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python \
  "$packet/tokenize_bridge.py" /home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json \
  "$packet/source/namespace_evidence.R" "$context" "$reserve" >"$log" 2>&1 || exit_code=$?
end_ns="$(date +%s%N)"
python3 - "$terminal" "$shard" "$core" "$context" "$reserve" "$started" "$start_ns" "$end_ns" "$exit_code" "$input" "$output" "$log" <<'PY'
import hashlib,json,os,sys
p,shard,core,context,reserve,started,start,end,exit_code,input_path,output,log=sys.argv[1:]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()
value={
    'schema':'sepalith.dat10.noop4100.parse_retry.render_shard_terminal.v1',
    'status':'complete' if exit_code=='0' else 'failed',
    'shard':int(shard),'core':int(core),'context_size':int(context),'generation_reserve':int(reserve),
    'started_at':started,'elapsed_seconds':(int(end)-int(start))/1e9,'exit_code':int(exit_code),
    'input':{'path':input_path,'sha256':sha(input_path),'rows':sum(1 for _ in open(input_path,encoding='utf-8'))},
    'output':None if not os.path.exists(output) else {'path':output,'sha256':sha(output),'bytes':os.path.getsize(output),'rows':sum(1 for _ in open(output,encoding='utf-8'))},
    'log':{'path':log,'sha256':sha(log)},
}
with open(p,'x',encoding='utf-8') as stream:
    json.dump(value,stream,indent=2,sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
PY
exit "$exit_code"
