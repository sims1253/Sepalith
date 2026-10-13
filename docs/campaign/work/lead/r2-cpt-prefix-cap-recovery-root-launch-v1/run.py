import subprocess,json,time,os
from pathlib import Path
cmd=['timeout', '--signal=TERM', '--kill-after=30s', '7200', 'ionice', '-c3', 'nice', '-n', '10', 'taskset', '-c', '0,2', '/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python', '-B', '/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v3/materialize_prefix_cap_recovery.py', '--frontier', '/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v3/combined-frontier.jsonl', '--output', '/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-cap-recovery-v3']
base=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-cap-recovery-v3-root')
started=time.time()
with Path(str(base)+".log").open("xb") as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 Path(str(base)+".launch.json").write_text(json.dumps({"controller_pid":os.getpid(),"child_pid":p.pid,"command":cmd,"at":started})+"\n")
 rc=p.wait()
Path(str(base)+".terminal.json").write_text(json.dumps({"exit_code":rc,"elapsed_seconds":time.time()-started,"at":time.time()})+"\n")
raise SystemExit(rc)
