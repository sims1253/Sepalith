#!/usr/bin/env bash
set -euo pipefail
run_id=$1
local_admission=$2
host=m0hawk@192.168.178.40
base=/home/m0hawk/.local/share/sepalith-e750-notebook-cpu-quality-v1
remote_admission=$base/admissions/$run_id.json
local_watchdog=$(dirname "$0")/watchdog_remote.py
watchdog_sha=424a641b599cf793e58d0ec142d0a07d3ac0def375b2bc3db6faa9dc070b2fff
[[ $run_id =~ ^[A-Za-z0-9._-]+$ ]]
test -f "$local_admission"
test -f "$local_watchdog"
test "$(sha256sum "$local_watchdog" | cut -d\  -f1)" = "$watchdog_sha"
python3 - "$local_admission" "$run_id" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='admitted' and x['run_id']==sys.argv[2]
assert x['packet_sha256']=='2b6dd8210ede1db19b00fa1aa8e0815eb4034618acfbc920a5facd31ae27cfb3'
assert x['remote_packet_source_manifest_sha256']=='fa18fefe0d771f8493b59ad08b0968ab7cc62e6532691c2e9ec73f331221db9c'
PY
ssh "$host" /usr/bin/bash -s -- "$run_id" <<'SH'
set -euo pipefail
run_id=$1
base=/home/m0hawk/.local/share/sepalith-e750-notebook-cpu-quality-v1
mkdir -p -m 700 "$base/admissions"
mkdir -p -m 700 "$base/root-admission-prep-v1"
mkdir -p -m 700 "$base/logs" "$base/runs"
test ! -e "$base/admissions/$run_id.json"
test ! -e "$base/admissions/.$run_id.json.new"
test ! -e "$base/root-admission-prep-v1/watchdog_remote.py"
test ! -e "$base/root-admission-prep-v1/.watchdog_remote.py.new"
SH
rsync -a --chmod=F400 "$local_admission" "$host:$base/admissions/.$run_id.json.new"
rsync -a --chmod=F500 "$local_watchdog" "$host:$base/root-admission-prep-v1/.watchdog_remote.py.new"
ssh "$host" /usr/bin/bash -s -- "$run_id" <<'SH'
set -euo pipefail
run_id=$1
base=/home/m0hawk/.local/share/sepalith-e750-notebook-cpu-quality-v1
packet=$base/packet-final
admission=$base/admissions/$run_id.json
new=$base/admissions/.$run_id.json.new
watchdog=$base/root-admission-prep-v1/watchdog_remote.py
watchdog_new=$base/root-admission-prep-v1/.watchdog_remote.py.new
test ! -e "$admission"
python3 - "$packet" "$new" "$run_id" <<'PY'
import hashlib,json,socket,stat,sys
from pathlib import Path
root=Path(sys.argv[1]); admission=Path(sys.argv[2]); run_id=sys.argv[3]
assert hashlib.sha256((root/'packet.json').read_bytes()).hexdigest()=='2b6dd8210ede1db19b00fa1aa8e0815eb4034618acfbc920a5facd31ae27cfb3'
assert hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest()=='fa18fefe0d771f8493b59ad08b0968ab7cc62e6532691c2e9ec73f331221db9c'
m=json.loads((root/'source-manifest.json').read_text()); seen=set()
for x in m['files']:
 p=root/x['path']; assert p.is_file() and x['path'] not in seen; seen.add(x['path'])
 assert p.stat().st_size==x['bytes'] and oct(stat.S_IMODE(p.stat().st_mode))==x['mode']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='source-manifest.json'}
assert actual==seen
a=json.loads(admission.read_text()); assert a['status']=='admitted' and a['run_id']==run_id
assert a['packet_sha256']=='2b6dd8210ede1db19b00fa1aa8e0815eb4034618acfbc920a5facd31ae27cfb3'
assert hashlib.sha256(Path('/home/m0hawk/.local/share/sepalith-e750-notebook-cpu-quality-v1/root-admission-prep-v1/.watchdog_remote.py.new').read_bytes()).hexdigest()=='424a641b599cf793e58d0ec142d0a07d3ac0def375b2bc3db6faa9dc070b2fff'
model=Path('/home/m0hawk/.local/share/sepalith-e750-notebook-cpu-quality-v1/models/model-Q8_0.gguf').stat()
assert (model.st_dev,model.st_ino,model.st_size,int(model.st_mtime),int(model.st_ctime),stat.S_IMODE(model.st_mode))==(56,16500077,2679710496,1789384145,1789443342,0o400)
s=socket.socket(); s.bind(('127.0.0.1',18527)); s.close()
PY
if pgrep -af '[/]home/m0hawk/.local/share/sepalith-e750-notebook-cpu-quality-v1/packet-final/run_suite.py|[/]home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server.*sepalith-e750-notebook-cpu-quality-v1'; then exit 1; fi
test ! -e "$base/runs/$run_id"
test ! -e "$base/logs/$run_id.runner-identity.json"
! tmux has-session -t "e750-cap-$run_id" 2>/dev/null
mv "$new" "$admission"
chmod 400 "$admission"
mv "$watchdog_new" "$watchdog"
chmod 500 "$watchdog"
python3 - "$base/admissions" <<'PY'
import os,sys
fd=os.open(sys.argv[1],os.O_DIRECTORY);os.fsync(fd);os.close(fd)
PY
command="/usr/bin/taskset --cpu-list 0,2 /usr/bin/python3 -B $watchdog --run-id $run_id --admission $admission"
tmux new-session -d -s "e750-cap-$run_id" "$command"
printf 'launched session=%s run=%s admission=%s watchdog=%s maximum_seconds=28800\n' "e750-cap-$run_id" "$base/runs/$run_id" "$admission" "$watchdog"
SH
