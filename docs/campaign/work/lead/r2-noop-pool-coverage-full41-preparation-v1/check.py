import pathlib,json,collections,hashlib,datetime
E=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work');REPLAY=E/'Sourcewalk-independent-replay-v3/full-01';P=pathlib.Path(__file__).resolve().parent
existing={}
for line in (E/'DAT10-finish-source-repair-v3/train-token-rows.jsonl').open():
 r=json.loads(line)
 if r['family']=='no_op':existing[r['id']]={'family':r['family'],'tokens':len(r['input_ids'])}
assert len(existing)==1094
stats=collections.Counter();overlap=[];new=[];missing=[];receipts=[]
EXPECTED_SOURCE_NOOPS=4227
SOURCE_PROFILE=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-sourcewalk-roxy-semantic-preparation-v1/metadata-profile.json')
source_profile=json.loads(SOURCE_PROFILE.read_text());assert source_profile['converted_by_family']['no_op']==EXPECTED_SOURCE_NOOPS and source_profile['shards']==41
for shard in range(41):
 rp=REPLAY/'shards'/f'shard-{shard:04d}'/'receipt.json'
 if not rp.exists():missing.append(shard);continue
 rec=json.loads(rp.read_text());assert rec['status']=='complete';out=rec['outputs'][0];f=pathlib.Path(out['path']);h=hashlib.sha256();n=0
 for raw in f.open('rb'):
  h.update(raw);r=json.loads(raw);n+=1
  if r['family']!='no_op':continue
  stats[r['status']]+=1
  item={'row_id':r['row_id'],'shard':shard,'status':r['status'],'upstream_mechanical_hold_reasons':r['upstream_mechanical_hold_reasons'],'reasons':r.get('reasons',[])}
  (overlap if r['row_id'] in existing else new).append(item)
 assert n==out['rows']==rec['rows'] and h.hexdigest()==out['sha256'];receipts.append({'shard':shard,'sha256':hashlib.sha256(rp.read_bytes()).hexdigest()})
assert missing==[] and len(receipts)==41
assert sum(stats.values())==EXPECTED_SOURCE_NOOPS
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'DAT-10','existing_pool_noop_rows':1094,'source_replay_expected_noops':EXPECTED_SOURCE_NOOPS,'source_index':{'path':str(SOURCE_PROFILE),'sha256':hashlib.sha256(SOURCE_PROFILE.read_bytes()).hexdigest()},'completed_shards':len(receipts),'pending_shards':missing,'partial':bool(missing),'observed_replay_noop_statuses':dict(stats),'same_row_id_overlap':len(overlap),'not_in_existing_pool_by_id':len(new),'new_by_status':dict(collections.Counter(x['status'] for x in new)),'receipts':receipts,'caveat':'Row-ID nonoverlap is a coverage lead, not proof of semantic novelty or training eligibility. Source/prompt-target dedup and current prompt-contract parity still required.','training_admitted':False}
(P/'result.json').write_text(json.dumps(r,indent=2)+'\n')
with (P/'candidate-ids.jsonl').open('w') as f:
 for item in new:f.write(json.dumps(item,sort_keys=True)+'\n')
print(json.dumps(r))
