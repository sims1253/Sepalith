"""Root closure check for all41 sourcewalk shards, then publish reviewed merge."""
import fcntl,hashlib,json,os,pathlib,subprocess,datetime
P=pathlib.Path(__file__).resolve().parent
OUT=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01')
DRIVER=P.parent/'r2-sourcewalk-independent-replay-v3/full_replay.py'
INDEX=OUT/'index/manifest.json'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 assert sha(DRIVER)=='63d165ae1488c1f38174f7e24f68a51e4c80c2d48e5aec44a0446fa062a61f7f'
 assert sha(INDEX)=='65637a9e05c66647de042d46f42bf9afa197a0b068f63680ec9f3ac0dfe922a1'
 assert not (OUT/'manifest.json').exists(),'merge already exists; review existing artifact'
 lock=os.open(OUT/'.owner.lock',os.O_RDWR|os.O_NOFOLLOW);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 index=json.loads(INDEX.read_text());assert index['requested_shards']==list(range(41));assert [x['shard'] for x in index['index_files']]==list(range(41))
 total=0;receipt_pins=[]
 for x in index['index_files']:
  path=pathlib.Path(x['path']);assert path.stat().st_size==x['bytes'] and sha(path)==x['sha256'];n=sum(1 for line in path.open('rb') if line.strip());assert n==x['rows'];total+=n
  rp=OUT/'shards'/f"shard-{x['shard']:04d}"/'receipt.json';r=json.loads(rp.read_text());assert r['status']=='complete' and r['shard']==x['shard'] and r['rows']==n;assert len(r['outputs'])==1
  o=r['outputs'][0];ledger=rp.parent/'ledger.jsonl';assert pathlib.Path(o['path'])==ledger and o['rows']==n and ledger.stat().st_size==o['bytes'] and sha(ledger)==o['sha256'];receipt_pins.append({'shard':x['shard'],'sha256':sha(rp),'rows':n})
 assert total==76279
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/parser-dependencies-root-v1')
 with (P/'merge.log').open('x') as log:r=subprocess.run(['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(DRIVER),'merge','--output',str(OUT)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200,pass_fds=(lock,))
 assert r.returncode==0,f'merge failed {r.returncode}'
 result=json.loads((OUT/'manifest.json').read_text());assert result['status']=='complete_review_only_no_admission' and result['training_admission'] is False;assert result['denominators']['all']==total and len(result['shard_receipts'])==41
 report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'DAT-10','index_rows_verified':total,'shards_verified':41,'receipt_pins':receipt_pins,'merged_manifest_sha256':sha(OUT/'manifest.json'),'denominators':result['denominators'],'status_counts':result['status_counts'],'training_admitted':False}
 with (P/'root-result.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))
if __name__=='__main__':main()
