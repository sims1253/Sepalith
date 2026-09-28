#!/usr/bin/env bash
set -euo pipefail
run_id=$1
admission=$2
base=$HOME/.local/share/sepalith-e750-notebook-cpu-quality-v1
run=$base/recovery-pilot-runs/$run_id
log=$base/recovery-pilot-logs/$run_id.log
identity=$base/recovery-pilot-logs/$run_id.runner-identity.json
test ! -e "$run";test ! -e "$identity"
pid=$$;tick=$(awk '{print $22}' /proc/$$/stat)
printf '{"pid":%s,"start_tick":"%s","run_id":"%s"}\n' "$pid" "$tick" "$run_id" > "$identity"
exec /usr/bin/taskset --cpu-list 0,2,4,6,8,10,1,3 /usr/bin/python3 -B "$base/recovery-pilot-v1/run_failed_row_pilot.py" --packet "$base/recovery-pilot-v1/packet.json" --row "$base/recovery-pilot-v1/failed-row.json" --admission "$admission" --run-root "$run" >>"$log" 2>&1
