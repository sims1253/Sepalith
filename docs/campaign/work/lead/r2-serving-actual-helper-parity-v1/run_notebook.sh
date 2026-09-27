#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/.local/state/sepalith/campaign-20260915/run06-actual-helper-parity-v1-20260914T2340Z"
TASK="$ROOT/r2-serving-actual-helper-parity-v1"
cd "$TASK"
started="$(date --iso-8601=ns)"
timeout 300s taskset -c 0,2 node --no-warnings=ExperimentalWarning --experimental-strip-types check_parity.ts actual-helper-fixture.json parity.json >node.stdout 2>node.stderr &
node_pid=$!
node_tick="$(awk '{print $22}' "/proc/$node_pid/stat")"
wait "$node_pid"
node_rc=$?
timeout 300s taskset -c 0,2 Rscript --vanilla parse_only.R actual-helper-fixture.json parse-only.json >r.stdout 2>r.stderr &
r_pid=$!
r_tick="$(awk '{print $22}' "/proc/$r_pid/stat")"
wait "$r_pid"
r_rc=$?
ended="$(date --iso-8601=ns)"
python3 - "$started" "$ended" "$node_pid" "$node_tick" "$node_rc" "$r_pid" "$r_tick" "$r_rc" <<'PY'
import json,sys
keys=('started_at','ended_at','node_timeout_pid','node_start_tick','node_exit_code','r_timeout_pid','r_start_tick','r_exit_code')
value=dict(zip(keys,sys.argv[1:]));value['node_timeout_pid']=int(value['node_timeout_pid']);value['node_start_tick']=int(value['node_start_tick']);value['node_exit_code']=int(value['node_exit_code']);value['r_timeout_pid']=int(value['r_timeout_pid']);value['r_start_tick']=int(value['r_start_tick']);value['r_exit_code']=int(value['r_exit_code']);value['cpu_affinity']='0,2';value['timeout_seconds_each']=300;value['generated_r_executed']=False
open('processes.json','w').write(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
sha256sum parity.json parse-only.json processes.json node.stdout node.stderr r.stdout r.stderr >output-sha256.txt
test ! -e "/proc/$node_pid"
test ! -e "/proc/$r_pid"
echo NOTEBOOK_HELPER_PARITY_TERMINAL
