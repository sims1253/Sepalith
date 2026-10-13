import sys,json,pathlib,datetime,os,time
sys.path.insert(0,'/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src')
from sepalith.runner import Runner
W=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/target-only25-a')
s=time.monotonic();(W/'supervisor-launch.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'start_tick':pathlib.Path('/proc/self/stat').read_text().split()[21],'owner':'lead','task':'SFT-06'})+'\n')
try:
 result=Runner('/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-target-only-v1').run_next()
 (W/'supervisor-terminal.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-s,'result':result},indent=2)+'\n')
except Exception as e:
 (W/'supervisor-terminal.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-s,'failure':repr(e)},indent=2)+'\n');raise
