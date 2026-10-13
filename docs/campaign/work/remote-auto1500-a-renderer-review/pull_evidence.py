"""Read-only SSH pull of named, terminal notebook run evidence. No remote writes."""
from pathlib import Path
import hashlib,io,json,shlex,subprocess,tarfile
HERE=Path(__file__).resolve().parent
RUN_NAMES=['preflight.json','install.json','launch.json','renderer-ready.json','renderer-frames.jsonl','renderer-inputs.jsonl','harness-events.jsonl','gateway-observations.jsonl','renderer-observer.json','renderer-result.json','host-result.json','host-process.json','run-result.json','retained-evidence.json','input-request.json','input-result.json','runtime-status.json','host.stdout.log','host.stderr.log','install.stdout.log','install.stderr.log']+[f'ghost-{n:02}.png' for n in range(1,13)]
GUARD_NAMES=['terminal.json','launch.json','owned-processes.json','owned-process-history.json','stdout.log','stderr.log']
REMOTE_CODE=r'''
from pathlib import Path
import hashlib,io,json,sys,tarfile
base=Path('/home/m0hawk/.local/share/sepalith-campaign-20260915/runs')
run=base/'remote-auto1500-a';guard=base/'remote-auto1500-a-supervision'
terminal=json.loads((guard/'terminal.json').read_text())
assert terminal['survivors']==[],'guard must prove writer cleanup before pull'
run_names=json.loads(sys.argv[1]);guard_names=json.loads(sys.argv[2]);rows=[];missing=[]
for prefix,root,names in [('remote',run,run_names),('supervision',guard,guard_names)]:
 for name in names:
  p=root/name
  if not p.is_file():missing.append(prefix+'/'+name);continue
  assert not p.is_symlink() and p.stat().st_size<=50*1024*1024
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  rows.append({'archive_name':prefix+'/'+name,'source_path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
assert sum(r['bytes'] for r in rows)<=120*1024*1024
manifest=json.dumps({'guard_terminal':terminal,'files':rows,'missing_named_files':missing,'remote_writes':False},indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 info=tarfile.TarInfo('source-manifest.json');info.size=len(manifest);tar.addfile(info,io.BytesIO(manifest))
 for row in rows:
  p=Path(row['source_path']);info=tarfile.TarInfo(row['archive_name']);info.size=row['bytes']
  with p.open('rb') as f:tar.addfile(info,f)
'''
def main():
    cmd=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','m0hawk@192.168.178.40',shlex.join(['python3','-c',REMOTE_CODE,json.dumps(RUN_NAMES),json.dumps(GUARD_NAMES)])]
    result=subprocess.run(cmd,capture_output=True,timeout=45)
    (HERE/'pull-stderr.log').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError('exact evidence pull failed: '+result.stderr.decode(errors='replace'))
    expected={'source-manifest.json'}|{'remote/'+n for n in RUN_NAMES}|{'supervision/'+n for n in GUARD_NAMES}
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:') as tar:
        seen=set()
        for member in tar:
            assert member.isfile() and member.name in expected and member.name not in seen
            seen.add(member.name);p=HERE/member.name;p.parent.mkdir(exist_ok=True)
            assert not p.exists()
            with p.open('xb') as f:
                stream=tar.extractfile(member)
                for block in iter(lambda:stream.read(1024*1024),b''):f.write(block)
    manifest=json.loads((HERE/'source-manifest.json').read_text())
    for row in manifest['files']:
        p=HERE/row['archive_name'];h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        assert p.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256']
    print(json.dumps({'files':len(manifest['files']),'bytes':sum(r['bytes'] for r in manifest['files']),'terminal':manifest['guard_terminal'],'remote_writes':False}))
if __name__=='__main__':main()
