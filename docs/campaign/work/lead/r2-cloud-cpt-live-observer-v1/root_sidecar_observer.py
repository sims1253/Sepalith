import json,datetime
from pathlib import Path
import live_observer as o
extra = """
out['checkpoint_files']={}
for rel in ('artifacts/archive/full/checkpoint-317','artifacts/cloud-persistence','artifacts/training/evaluations'):
 d=os.path.join(p,rel)
 out['checkpoint_files'][rel]=[{'name':n,'bytes':os.stat(os.path.join(d,n)).st_size} for n in os.listdir(d) if os.path.isfile(os.path.join(d,n))] if os.path.isdir(d) else []
q=os.path.join(p,'artifacts/archive/full/checkpoint-317/campaign-manifest.json')
if os.path.isfile(q):
 m=json.load(open(q));files=m.get('files');out['checkpoint_validation']={'full':m.get('full'),'step':m.get('step'),'files_type':type(files).__name__,'invalid_files':[]}
 if isinstance(files,dict):
  import hashlib
  for name,rec in files.items():
   f=os.path.join(os.path.dirname(q),name)
   good=os.path.isfile(f) and os.stat(f).st_size==rec.get('bytes')
   if good:
    h=hashlib.sha256()
    with open(f,'rb') as stream:
     for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    good=h.hexdigest()==rec.get('sha256')
   if not good:out['checkpoint_validation']['invalid_files'].append(name)
out['sidecar_processes']=[]
for procpath in glob.glob('/proc/[0-9]*/cmdline'):
 try:
  cmd=open(procpath,'rb').read().decode(errors='replace')
  if 'checkpoint_sidecar.py' not in cmd or run not in cmd:continue
  status=open(os.path.dirname(procpath)+'/status').read().splitlines()
  out['sidecar_processes'].append({'pid':int(procpath.split('/')[2]),'state':next((line for line in status if line.startswith('State:')),None)})
 except (OSError,ValueError):pass
log=os.path.join(p,'sidecar.log')
if os.path.isfile(log):
 import re
 text=open(log).read()[-4000:];out['sidecar_log']={'bytes':os.stat(log).st_size,'error_classes':re.findall(r'([A-Za-z]+Error):',text)}
print(json.dumps(out,sort_keys=True))
"""
o.REMOTE=o.REMOTE.replace('print(json.dumps(out,sort_keys=True))',extra)
v=o.observe(use_hf=False)
f=Path(__file__).with_name('root-checkpoint-observation-sidecar.json')
assert not f.exists()
f.write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'status':v['provider']['provider_run_status'],'checkpoint_files':v['remote'].get('checkpoint_files',{}),'validation':v['remote'].get('checkpoint_validation'),'sidecar_log':v['remote'].get('sidecar_log'),'sidecar_processes':v['remote'].get('sidecar_processes')}))
