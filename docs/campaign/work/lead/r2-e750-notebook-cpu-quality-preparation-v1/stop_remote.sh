#!/usr/bin/env bash
set -euo pipefail
run_id=$1
base=$HOME/.local/share/sepalith-e750-notebook-cpu-quality-v1
identity=$base/logs/$run_id.runner-identity.json
test -f "$identity"
read -r pid tick < <(python3 - "$identity" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));print(x['pid'],x['start_tick'])
PY
)
if test -r "/proc/$pid/stat"; then
  current=$(awk '{print $22}' "/proc/$pid/stat")
  test "$current" = "$tick"
  kill -TERM "$pid"
fi
