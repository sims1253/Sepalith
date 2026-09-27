#!/usr/bin/env python3
"""Run reviewed v3 semantic analysis over the remaining committed shards 12+.

The wrapper owns fresh intake/launch/terminal telemetry and an exact disjoint
check against the already processed semantic 0-11 scope. The reviewed worker
is reused byte-for-byte and remains review-only.
"""
from __future__ import annotations
import datetime,hashlib,json,os,pathlib,subprocess,time
PACKET=pathlib.Path(__file__).resolve().parent
PLAN=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
V3=PLAN/'docs/campaign/work/lead/r2-semantic-queue-root-launch-v3'
REPLAY=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01')
OUTPUT=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards12plus-v1')
SELECTED=list(range(12,27));PRIOR=list(range(12))
PRIOR_OUTPUTS=[pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-v2/shard5-root-03/shard-0005/semantic-ledger.jsonl'),pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-v2/shards0to4-root-01/shard-0003/semantic-ledger.jsonl')]+[pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1')/f'shard-{n:04d}'/'semantic-ledger.jsonl' for n in range(6,12)]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def stable_hash(p,expected,bytes_):
 a=p.stat();raw=p.read_bytes();b=p.stat();require(a.st_size==bytes_ and len(raw)==bytes_,'size changed '+str(p));require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'changed during preflight '+str(p));require(hashlib.sha256(raw).hexdigest()==expected,'hash changed '+str(p))
def jsonl_ids(p, status=None):
 ids=[]
 with p.open() as f:
  for line in f:
   if not line.strip():continue
   x=json.loads(line)
   if status is None or x.get('status')==status:ids.append(x.get('row_id'))
 return ids
def preflight():
 sm=json.loads((V3/'source-manifest.json').read_text())
 for x in sm['files']:
  p=V3/x['path'];require(p.stat().st_size==x['bytes'],'review source bytes changed');require(sha(p)==x['sha256'],'review source hash changed '+str(p))
 ip=REPLAY/'index/manifest.json';idx=json.loads(ip.read_text());require(idx.get('schema')=='sepalith.dat10.sourcewalk_raw_index.v3' and idx.get('status')=='complete','replay index not complete v3');entries={x['shard']:x for x in idx['index_files']};require(len(entries)==len(idx['index_files']),'duplicate index shard')
 selected=[];new_ids=[]
 for n in SELECTED:
  require(n in entries,'selected shard absent '+str(n));e=entries[n];p=pathlib.Path(e['path']);require(p.is_file(),'selected index absent');stable_hash(p,e['sha256'],e['bytes']);rp=REPLAY/'shards'/f'shard-{n:04d}'/'receipt.json';require(rp.is_file(),'selected receipt absent');rx=json.loads(rp.read_text());require(rx.get('status')=='complete' and rx.get('rows')==e.get('rows'),'selected receipt incomplete/row mismatch');o=rx['outputs'][0];lp=pathlib.Path(o['path']);require(lp.is_file() and lp.stat().st_size==o['bytes'] and sha(lp)==o['sha256'],'selected provenance ledger changed');ids=jsonl_ids(lp,'provenance_pass_semantic_analyzer_queued');require(len(ids)==len(set(ids)),'selected source IDs duplicate '+str(n));new_ids.extend(ids);selected.append({'shard':n,'index_rows':e['rows'],'index_bytes':e['bytes'],'index_sha256':e['sha256'],'receipt_sha256':sha(rp),'provenance_rows':rx.get('rows'),'queued_rows':len(ids),'provenance_ledger_sha256':o['sha256']})
 require(len(new_ids)==len(set(new_ids)),'selected queued IDs overlap within 12+');old=[]
 for p in PRIOR_OUTPUTS:
  require(p.is_file(),'prior semantic ledger absent '+str(p));ids=jsonl_ids(p);require(len(ids)==len(set(ids)),'prior semantic IDs duplicate '+str(p));old.extend(ids)
 require(len(old)==len(set(old)),'prior semantic IDs overlap 0-11');overlap=set(new_ids)&set(old);require(not overlap,'selected IDs overlap prior semantic scope: '+str(len(overlap)))
 require(not OUTPUT.exists(),'fresh semantic output exists');return {'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'replay_index_path':str(ip),'replay_index_sha256':sha(ip),'index_status':idx['status'],'selected':selected,'selected_provenance_rows':sum(x['provenance_rows'] for x in selected),'selected_queued_rows':len(new_ids),'selected_queued_ids_sha256':hashlib.sha256(json.dumps(sorted(new_ids),separators=(',',':')).encode()).hexdigest(),'prior_scope_shards':PRIOR,'prior_semantic_rows':len(old),'prior_semantic_ids_sha256':hashlib.sha256(json.dumps(sorted(old),separators=(',',':')).encode()).hexdigest(),'prior_overlap':0,'replay_process_status':'historical PIDs absent; no replay restart'}
def main():
 pre=preflight();OUTPUT.mkdir(parents=True,exist_ok=False);log=OUTPUT.parent/'Sourcewalk-semantic-streaming-queue-shards12plus-v1.log';cmd=['timeout','--signal=TERM','--kill-after=30s','3000','ionice','-c3','nice','-n','10','taskset','-c','2,3','/usr/bin/python3','-B',str(V3/'source/run_streaming_semantic_queue.py'),'--replay-root',str(REPLAY),'--output',str(OUTPUT),'--shards',','.join(map(str,SELECTED)),'--max-workers','2'];env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/parser-dependencies-root-v1');start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
 with log.open('xb') as s:
  child=subprocess.Popen(cmd,env=env,stdout=s,stderr=subprocess.STDOUT,start_new_session=True);launch={'schema':'sepalith.dat10.semantic_shards12plus_launch.v1','status':'running','training_admission':False,'started_at_utc':start,'controller_pid':os.getpid(),'timeout_pid':child.pid,'command':cmd,'environment':{k:env[k] for k in ('PYTHONNOUSERSITE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','PYTHONPATH')},'preflight':pre,'selected_shards':SELECTED,'time_limit_seconds':3000,'output':str(OUTPUT),'log':str(log),'source_reuse_packet':str(V3)};write(PACKET/'launch.json',launch);write(PACKET/'status.json',{**launch,'status':'running','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});code=child.wait()
 term={'schema':'sepalith.dat10.semantic_shards12plus_terminal.v1','status':'terminal','training_admission':False,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code,'elapsed_seconds':time.monotonic()-t,'launch':str(PACKET/'launch.json'),'output':str(OUTPUT),'log':str(log)};sm=OUTPUT/'streaming-manifest.json';
 if sm.is_file():term['streaming_manifest']={'path':str(sm),'bytes':sm.stat().st_size,'sha256':sha(sm)}
 write(PACKET/'terminal.json',term);write(PACKET/'status.json',{**term,'status':'terminal','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});return code
if __name__=='__main__':raise SystemExit(main())
