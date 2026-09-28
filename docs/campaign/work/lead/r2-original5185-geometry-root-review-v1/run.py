import pathlib,json,hashlib,subprocess,datetime,os
root=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');p=pathlib.Path(__file__).resolve().parent
src=root/'docs/campaign/work/lead/r2-original5185-geometry-input-registry-preparation-v1'
r=json.loads((src/'registry.json').read_text());pins=[]
for c in r['cohorts']:
 for group in ('prediction_inputs','selected_contexts'):
  pins.extend(c[group])
size=sum(pathlib.Path(x['path']).stat().st_size for x in pins)
assert size<350_000_000,size
for x in pins:
 q=pathlib.Path(x['path']);h=hashlib.sha256();count=0
 with q.open('rb') as f:
  for line in f: h.update(line);count+=bool(line.strip())
 assert h.hexdigest()==x['sha256'],q
 assert count==x['rows'],q
(p/'payload-pins.json').write_text(json.dumps(dict(bytes=size,files=pins,verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat())))
with (p/'run.log').open('w') as log:
 t=subprocess.run(['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(src/'verify_registry.py'),str(src/'registry.json'),'--mode','full-join','--report',str(p/'full-join.json')],cwd=root,stdout=log,stderr=subprocess.STDOUT)
(p/'terminal.json').write_text(json.dumps(dict(returncode=t.returncode,at=datetime.datetime.now(datetime.timezone.utc).isoformat())))
raise SystemExit(t.returncode)
