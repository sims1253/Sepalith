import json,sys
from pathlib import Path
p=Path('docs/campaign/work/lead/r2-cloud-cpt-live-observer-v1').resolve();sys.path.insert(0,str(p));import live_observer as o
o.REMOTE=r'''import glob,hashlib,json,os,time
run='4f899bbc6e9d46c0a88985d64e6d40e2';pid=50853;expected_start=686825;paths=glob.glob('/mnt/local_storage/sepalith-cpt-'+run+'-*');root=paths[0] if len(paths)==1 else '';scratch=root+'/persistence-staging'
def proc():
 base='/proc/'+str(pid)
 try:
  parts=[x.decode(errors='replace') for x in open(base+'/cmdline','rb').read().split(b'\0') if x];stat=open(base+'/stat').read().split();io={}
  for line in open(base+'/io'):
   k,v=line.split(':',1)
   if k in ('rchar','wchar','read_bytes','write_bytes'):io[k]=int(v)
  return {'alive':stat[2]!='Z','state':stat[2],'pid':pid,'ppid':int(stat[3]),'start_tick':int(stat[21]),'start_matches':int(stat[21])==expected_start,'argv':parts,'io':io}
 except Exception as e:return {'alive':False,'error_type':type(e).__name__}
a=proc();time.sleep(3);b=proc();events=[]
try:
 for line in open(scratch+'/events.jsonl'):
  try:x=json.loads(line)
  except Exception:continue
  events.append({k:x[k] for k in ('schema','at','status','step','attempt','error_type','error_message','revision','receipt_revision','tar_sha256') if k in x})
except OSError:pass
entries=[]
if os.path.isdir(scratch):
 for name in sorted(os.listdir(scratch)):
  if '/' in name or name.startswith('.'):continue
  p=scratch+'/'+name
  try:s=os.stat(p);entries.append({'name':name,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'is_file':os.path.isfile(p),'sha256_if_small':hashlib.sha256(open(p,'rb').read()).hexdigest() if os.path.isfile(p) and s.st_size<200000 else None})
  except OSError:pass
tele=root+'/artifacts/training/telemetry.jsonl';last=None;count=0;tele_errors=[]
try:
 for line in open(tele):
  try:x=json.loads(line)
  except Exception:continue
  if x.get('event')=='optimizer_step':count+=1;last={k:x[k] for k in ('at','step','seconds','remaining_seconds') if k in x}
  if x.get('event') in ('error','train_error','exception'):tele_errors.append({k:x[k] for k in ('at','event','step','error_type','error_message') if k in x})
except OSError:pass
print(json.dumps({'run_id':run,'run_directory_match_count':len(paths),'process_a':a,'process_b':b,'events':events[-20:],'persistence_entries':entries,'optimizer_records':count,'latest_optimizer':last,'telemetry_errors':tele_errors[-10:],'credential_persisted':False},sort_keys=True))'''
client=o.load_sdk();provider=o.provider(client);print(json.dumps({'provider':provider,'remote':o.ssh_read(client,provider['cluster_id'])},sort_keys=True))
