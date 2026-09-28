"""Root-run file transfer only. Default checks local files; never launches editor/native."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, io, json, shlex, subprocess, tarfile
HERE=Path(__file__).resolve().parent
HOST='m0hawk@192.168.178.40'
REMOTE_BASE='/home/m0hawk/.local/share/sepalith-campaign-20260915'
MAX_BYTES=2*1024*1024

def validate_archive(data, expected):
    if len(data)>MAX_BYTES:raise ValueError('archive exceeds 2 MiB bound')
    files={}
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as tar:
        for member in tar.getmembers():
            name=member.name;p=PurePosixPath(name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or str(p)!=name or name in files:
                raise ValueError('unsafe or duplicate archive member')
            if name not in expected:raise ValueError('unexpected archive member')
            content=tar.extractfile(member).read(MAX_BYTES+1)
            if hashlib.sha256(content).hexdigest()!=expected[name]:raise ValueError('file hash mismatch')
            files[name]=content
    if set(files)!=set(expected):raise ValueError('missing archive member')
    return files

# The receiver reads only this fixed capsule path and the pinned VS Code source.
# It never checks ports, profiles, processes or model files, and runs no child.
REMOTE_RECEIVER=r'''
from pathlib import Path, PurePosixPath
import sys,json,hashlib,io,tarfile
root=Path(sys.argv[1]);renderer=Path(sys.argv[2]);renderer_sha=sys.argv[3];expected=json.loads(sys.argv[4]);archive_sha=sys.argv[5]
assert str(root.parent)=='/home/m0hawk/.local/share/sepalith-campaign-20260915'
assert root.name in ['remote-auto1500-b-capsule','remote-auto350-b-capsule']
assert not root.is_symlink() and root.parent.is_dir() and root.parent.resolve()==root.parent
assert str(renderer)=='/usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js'
assert hashlib.sha256(renderer.read_bytes()).hexdigest()==renderer_sha,'fresh notebook renderer hash mismatch'
data=sys.stdin.buffer.read(2*1024*1024+1)
assert len(data)<=2*1024*1024 and hashlib.sha256(data).hexdigest()==archive_sha
files={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as tar:
 for m in tar.getmembers():
  p=PurePosixPath(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts and str(p)==m.name
  assert m.name in expected and m.name not in files and m.size<=2*1024*1024
  raw=tar.extractfile(m).read(2*1024*1024+1)
  assert hashlib.sha256(raw).hexdigest()==expected[m.name]
  files[m.name]=raw
assert set(files)==set(expected)
if root.exists():
 assert root.is_dir()
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
 assert actual==set(files),'existing capsule differs; no overwrite'
 for name,raw in files.items():
  p=root/name
  assert not p.is_symlink() and p.resolve().is_relative_to(root)
  assert p.read_bytes()==raw,'existing capsule differs; no overwrite'
 status='existing_identical_capsule_verified'
else:
 root.mkdir(mode=0o700)
 for name,raw in files.items():
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
  p.chmod(0o600)
 status='transferred_and_verified'
print(json.dumps({'status':status,'capsule':str(root),'files':len(files),'archive_sha256':archive_sha,'renderer_sha256':renderer_sha,'launch_performed':False}))
'''

def check_plan(plan):
    arm=plan['arm_ms'];assert arm in [1500,350]
    assert plan['name']==f'remote-auto{arm}-b' and plan['host']==HOST
    assert plan['remote_capsule']==REMOTE_BASE+f'/remote-auto{arm}-b-capsule'
    archive=Path(plan['archive']['path']);assert archive.parent==HERE and not archive.is_symlink()
    data=archive.read_bytes();assert hashlib.sha256(data).hexdigest()==plan['archive']['sha256']
    expected=dict(plan['files']);expected['capsule-manifest.json']=plan['capsule_manifest']['sha256']
    files=validate_archive(data,expected)
    assert 'asset-key.pem' not in files and all(not name.endswith(('.pem','.gguf','.safetensors')) for name in files)
    assert json.loads(files['capsule-manifest.json'])==plan['files']
    binding=json.loads(files['binding.json']);assert binding['instanceId']==plan['instance_id'] and binding['endpoint']=='http://127.0.0.1:18403'
    assert hashlib.sha256(files['binding.json']).hexdigest()==plan['binding_sha256']
    argv=plan['editor_argv'];assert argv[argv.index('--debounce-ms')+1]==str(arm)
    assert argv[argv.index('--instance-id')+1]==binding['instanceId']
    return data,expected

def command(plan,expected):
    remote=shlex.join(['python3','-c',REMOTE_RECEIVER,plan['remote_capsule'],plan['renderer_path'],plan['renderer_sha256'],json.dumps(expected,separators=(',',':')),plan['archive']['sha256']])
    return ['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',HOST,remote]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--transfer',choices=['1500','350','both']);args=parser.parse_args()
    plans=json.loads((HERE/'staging-plan.json').read_text())['arms']
    checked=[(p,*check_plan(p)) for p in plans]
    if not args.transfer:
        print(json.dumps({'status':'local_capsules_verified','arms':[p['arm_ms'] for p,_,_ in checked],'network':False,'launch':False}));return
    for plan,data,expected in checked:
        if args.transfer!='both' and plan['arm_ms']!=int(args.transfer):continue
        result=subprocess.run(command(plan,expected),input=data,capture_output=True,timeout=60)
        receipt={'arm_ms':plan['arm_ms'],'returncode':result.returncode,'stdout':result.stdout.decode(errors='replace'),'stderr':result.stderr.decode(errors='replace'),'launch_performed':False}
        (HERE/f"transfer-{plan['arm_ms']}-result.json").write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))
        if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
