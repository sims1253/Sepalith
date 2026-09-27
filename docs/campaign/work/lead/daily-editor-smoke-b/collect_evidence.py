from pathlib import Path
import subprocess,shlex,json,tarfile,hashlib,datetime,socket,fcntl
root=Path(__file__).resolve().parent;out=root/'notebook-evidence';out.mkdir(exist_ok=False)
remote='/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/daily-editor-smoke-b'
script=f'''import pathlib,tarfile,sys,hashlib,json,io
root=pathlib.Path({remote!r});guard=pathlib.Path(str(root)+'-supervision');assert (guard/'terminal.json').exists()
selected=[(p,'run/'+p.name) for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.png','.log')]
selected += [(root/'workspace/primary-accept.R','workspace/primary-accept.R')]
selected += [(guard/n,'supervision/'+n) for n in ['launch.json','owned-process-history.json','terminal.json','stdout.log','stderr.log']]
assert sum(p.stat().st_size for p,n in selected)<40*1024*1024
pins={{n:{{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}} for p,n in selected}}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for p,n in selected:t.add(p,arcname=n,recursive=False)
 raw=json.dumps(pins).encode();info=tarfile.TarInfo('source-hashes.json');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
'''
base=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','m0hawk@192.168.178.40']
with (root/'notebook-evidence.tar').open('xb') as f:subprocess.run(base+[shlex.join(['python3','-c',script])],stdout=f,stderr=subprocess.PIPE,timeout=25,check=True)
with tarfile.open(root/'notebook-evidence.tar') as t:
 for m in t.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 t.extractall(out)
pins=json.loads((out/'source-hashes.json').read_text())
for name,row in pins.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==row['sha256'] and (out/name).stat().st_size==row['bytes']
d=json.loads((root/'controller-terminal.json').read_text());assert d['failure'] is None and d['ssh_exit']==0 and d['daily_exit']==0
assert d['daily_terminal']['failure'] is None and d['daily_terminal']['resourceReleaseProven'] and d['daily_terminal']['ledgerRelease']['status']=='released'
ids=[x['identity'] for x in d['daily_terminal']['cleanup']];ids+=[json.loads((root/'root-launch.json').read_text())[k] for k in ['controller','daily_supervisor']];ids.append(json.loads((root/'notebook-launch.json').read_text())['local_ssh'])
for x in ids:
 p=Path('/proc')/str(x['pid']);assert not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[19]!=x['startTick'],x
for port in (18403,18423):
 with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
with open('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock','r+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
history=json.loads((out/'supervision/owned-process-history.json').read_text());launch=json.loads((out/'supervision/launch.json').read_text());remoteids=[(x['pid'],x['start_ticks']) for x in history]
assert not any(x['argv'].startswith('fish -c ') for x in history)
script=f'''from pathlib import Path
import socket,json
ids={remoteids!r};alive=[]
for pid,tick in ids:
 p=Path('/proc')/str(pid)
 if p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==tick:alive.append(pid)
assert not alive
assert not Path('/proc/{launch['supervisor_pid']}').exists()
for port in (18403,19403):
 with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
print(json.dumps({{'all_recorded_absent':True,'recorded_count':len(ids),'supervisor_absent':True,'ports_reusable':[18403,19403]}}))
'''
r=subprocess.run(base+[shlex.join(['python3','-c',script])],capture_output=True,text=True,check=True,timeout=15)
receipt={'task':'RUN-01','owner':'lead','observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'terminal_process_and_cleanup_pass_editor_review_pending','controller':d,'desktop_identities_absent':ids,'notebook':json.loads(r.stdout),'global_lock_available':True,'source_destination_hashes_match':len(pins),'notebook_archive_sha256':hashlib.sha256((root/'notebook-evidence.tar').read_bytes()).hexdigest(),'parent_shell_in_owned_history':False,'next':'Review actual editor identities, screenshot, accepted buffer and R parse; no full quality or representative p95 claim.'}
outfile=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/receipts/RUN-01-daily-editor-smoke-b-resource-release.json');outfile.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(outfile),'desktop_identities':len(ids),'notebook':json.loads(r.stdout),'files':len(pins),'seconds':d['seconds'],'remote_terminal':json.loads((out/'supervision/terminal.json').read_text())}))
