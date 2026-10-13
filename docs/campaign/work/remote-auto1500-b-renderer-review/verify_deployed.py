from pathlib import Path
import json,hashlib,shlex,subprocess
P=Path(__file__).resolve().parent;C=P.parents[1];capsule=C/'work/remote-auto-v2-staging/remote-auto1500-b';expected=json.loads((capsule/'capsule-manifest.json').read_text());expected['capsule-manifest.json']=hashlib.sha256((capsule/'capsule-manifest.json').read_bytes()).hexdigest()
code=r'''
from pathlib import Path
import json,hashlib,sys
root=Path('/home/m0hawk/.local/share/sepalith-campaign-20260915/remote-auto1500-b-capsule');expected=json.loads(sys.argv[1]);rows=[]
for name,digest in expected.items():
 p=root/name;assert not p.is_symlink() and p.stat().st_size<1024*1024
 actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==digest,name
 rows.append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size})
print(json.dumps({'status':'PASS','files':rows,'remote_writes':False},indent=2))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','m0hawk@192.168.178.40',shlex.join(['python3','-c',code,json.dumps(expected)])],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr;(P/'remote-capsule-verification.json').write_text(r.stdout);print(json.dumps({'status':'PASS','files':len(expected),'observer_sha256':expected['observe_renderer.mjs'],'capsule_sha256':expected['capsule-manifest.json']}))
