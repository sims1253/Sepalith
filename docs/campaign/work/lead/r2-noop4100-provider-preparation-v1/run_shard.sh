#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then echo 'usage: run_shard.sh SHARD CORE CONTEXT RESERVE INPUT_ROOT OUTPUT_ROOT' >&2; exit 2; fi
shard="$1"; core="$2"; context="$3"; reserve="$4"; input_root="$5"; output_root="$6"
packet="$(cd "$(dirname "$0")" && pwd)"
base="/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic10948-provider-materialization-v2"
mkdir -p "$output_root"
input="$input_root/shard-${shard}.jsonl"; output="$output_root/shard-${shard}.jsonl"; log="$output_root/shard-${shard}.log"; terminal="$output_root/shard-${shard}.terminal.json"
[[ -f "$input" && ! -e "$output" && ! -e "$terminal" ]] || { echo 'fresh shard paths required' >&2; exit 3; }
started="$(date --iso-8601=ns)"; start_ns="$(date +%s%N)"; status=0
timeout --signal=TERM --kill-after=10s 1800 taskset -c "$core" nice -n 10 ionice -c 3 node --no-warnings=ExperimentalWarning --experimental-strip-types "$packet/render_shard.ts" "$input" "$output" /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python "$base/tokenize_bridge.py" /home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json "$base/source/namespace_evidence.R" "$context" "$reserve" >"$log" 2>&1 || status=$?
end_ns="$(date +%s%N)"
python3 - "$terminal" "$shard" "$core" "$context" "$reserve" "$started" "$start_ns" "$end_ns" "$status" "$input" "$output" "$log" <<'PY'
import hashlib,json,os,sys
p,shard,core,context,reserve,started,start,end,status,input_path,output,log=sys.argv[1:]
def sha(q):
 h=hashlib.sha256()
 with open(q,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
value={'schema':'sepalith.dat10.noop4100.render-shard-terminal.v1','status':'complete'if status=='0'else'failed','shard':int(shard),'core':int(core),'context_size':int(context),'generation_reserve':int(reserve),'started_at':started,'elapsed_seconds':(int(end)-int(start))/1e9,'exit_code':int(status),'input':{'path':input_path,'sha256':sha(input_path)},'output':None if not os.path.exists(output)else{'path':output,'sha256':sha(output),'bytes':os.path.getsize(output),'rows':sum(1 for _ in open(output))},'log':{'path':log,'sha256':sha(log)}}
with open(p,'x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
PY
exit "$status"
