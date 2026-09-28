#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then echo 'usage: run_shard.sh SHARD CORE CONTEXT OUTPUT_ROOT' >&2; exit 2; fi
shard="$1"; core="$2"; context="$3"; output_root="$4"
packet="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$output_root"
input="/mnt/e/sepalith/campaign-20260915/data-work/Semantic4551-provider-materialization-v1/input-shards/shard-${shard}.jsonl"
output="$output_root/shard-${shard}.jsonl"; log="$output_root/shard-${shard}.log"; terminal="$output_root/shard-${shard}.terminal.json"
started="$(date --iso-8601=ns)"; start_ns="$(date +%s%N)"; status=0
taskset -c "$core" nice -n 10 ionice -c 3 node --no-warnings=ExperimentalWarning --experimental-strip-types "$packet/render_shard.ts" "$input" "$output" /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python "$packet/tokenize_bridge.py" /home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json "$packet/source/namespace_evidence.R" "$context" >"$log" 2>&1 || status=$?
end_ns="$(date +%s%N)"; python3 - "$terminal" "$shard" "$core" "$context" "$started" "$start_ns" "$end_ns" "$status" "$input" "$output" "$log" <<'PY'
import hashlib,json,os,sys
p,shard,core,context,started,start,end,status,input_path,output,log=sys.argv[1:]
def sha(q):
 h=hashlib.sha256()
 with open(q,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
value={'schema':'sepalith.dat10.semantic4551.render_shard_terminal.v1','status':'complete' if status=='0' else 'failed','shard':shard,'core':int(core),'context_size':int(context),'started_at':started,'elapsed_seconds':(int(end)-int(start))/1e9,'exit_code':int(status),'input':{'path':input_path,'sha256':sha(input_path)},'output':None if not os.path.exists(output) else {'path':output,'sha256':sha(output),'bytes':os.path.getsize(output),'rows':sum(1 for _ in open(output))},'log':{'path':log,'sha256':sha(log)}}
with open(p,'x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
PY
exit "$status"
