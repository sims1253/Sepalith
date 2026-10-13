"""Persist each completed full-cadence checkpoint while training is running."""
import argparse,json,time
from pathlib import Path
from artifact_upload import checkpoint_complete,upload_checkpoint
from cloud_contract import write

def ready(archive,done):
 out=[]
 for p in Path(archive).glob('checkpoint-*'):
  try:step=int(p.name.split('-')[-1])
  except ValueError:continue
  if step not in done and checkpoint_complete(p) is not None:out.append((step,p))
 return sorted(out)

def monitor(binding,run,stop,poll=5,uploader=upload_checkpoint):
 b=json.loads(Path(binding).read_text());done=set();failures=[];remote=[]
 while True:
  candidates=ready(Path(run)/'artifacts/archive/full',done)
  for step,path in candidates:
   try:remote.append(uploader(Path(run),b,path));done.add(step)
   except Exception as e:failures.append({'step':step,'error_type':type(e).__name__})
  if Path(stop).exists() and not ready(Path(run)/'artifacts/archive/full',done):break
  time.sleep(poll)
 result={'status':'complete' if not failures else 'failed','uploaded_steps':sorted(done),'remote_checkpoints':remote,'failures':failures}
 write(Path(run)/'artifacts/cloud-persistence/sidecar-terminal.json',result)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('binding',type=Path);p.add_argument('run',type=Path);p.add_argument('--stop-file',type=Path,required=True);a=p.parse_args()
 result=monitor(a.binding,a.run,a.stop_file)
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
