import os,json,time,subprocess,hashlib
from pathlib import Path
packet=Path(__file__).resolve().parent
E=Path('/mnt/e/sepalith/campaign-20260915')
terminal=E/'training/SFT11-cpt82-native-continuation-root-v1/guard-v1/terminal.json'
start=time.monotonic()
while not terminal.exists():
 if time.monotonic()-start>5400:raise SystemExit('prior training did not terminate within wait budget')
 time.sleep(2)
value=json.loads(terminal.read_text());assert value['child_exit_code']==0,value
assert not Path('/proc/1763848').exists()
result=json.loads((E/'checkpoints/SFT11-full-weight-CPT-expandable-pilot-v1/run-result.json').read_text());assert result['global_step']==90,result['global_step']
cp=E/'checkpoints/SFT11-full-weight-CPT-expandable-pilot-v1/full/checkpoint-90'
m=json.loads((cp/'campaign-manifest.json').read_text());assert m['step']==90 and m['full'] is True
checkpoint_observation={'checkpoint':str(cp),'manifest_sha256':hashlib.sha256((cp/'campaign-manifest.json').read_bytes()).hexdigest(),'prior_terminal':value,'observed_at':time.time()}
(packet/'predecessor-terminal.json').write_text(json.dumps(checkpoint_observation,indent=2)+'\n')
guard=packet.parent/'host-memory-guard-v4/cuda_host_guard.py';assert hashlib.sha256(guard.read_bytes()).hexdigest()=='25f85d1da5bfee8da085a209982617e39e176b89dda11febc851c248a75240e0'
env={**os.environ,**json.loads((packet/'environment.json').read_text())}
command=['python3',str(guard),'--command-json',str(packet/'command.json'),'--output',str(E/'training/SFT11-sm120-kernel-root-v1/guard-v1'),'--seconds','300','--admission-free-mib','14336']
child=subprocess.Popen(command,env=env)
(packet/'guard-controller-start.json').write_text(json.dumps({'guard_pid':child.pid,'at':time.time(),'command':command},indent=2)+'\n')
raise SystemExit(child.wait())
