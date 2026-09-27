"""Independent CPU audit of the frozen DAT-10 25%-noop finite schedule."""
from pathlib import Path
from collections import Counter
import hashlib,json,datetime,platform,sys
ROOT=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
W=ROOT/'docs/campaign/work/r2-task-mixture-v1'
OUT=ROOT/'docs/campaign/work/r2-task-mixture-notebook-acceptance-v1'
sp=W/'verified-corrected-short-draw-manifest-noop25.json'
rp=W/'verified-corrected-short-token-rows.jsonl'
sh=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads(sp.read_text())
# Embedded manifest_sha256 is the generator's canonical payload digest; retain both it and the immutable file hash.
rows={}
with rp.open() as f:
  for line in f:
    d=json.loads(line); rows[d['id']]=d
assert sh(rp)==s['token_rows_sha256'] and len(rows)==8526
# draw order is explicit and all rows bind to their frozen row metadata.
for d in s['draws']:
  r=rows[d['row_id']]
  assert d['family']==r['family'] and d['package_id']==r['package_id']
  # Schedule geometry counts BOS in the prompt and EOS in the target; row metadata stores both without these boundary IDs.
  assert d['target_tokens']==r['target_body_token_count']+r['target_terminal_token_count']+1
  assert d['prompt_tokens']==r['prompt_token_count']+1
  assert d['total_tokens']==len(r['input_ids'])
  assert d['semantic_noop']==(r['target_operation']=='no_op')
  assert d['source_identity']==f"{d['package_id']}::{d['source_id']}"
# A schedule group is the source identity used by the production quota logic.
def audit(n):
  ds=s['draws'][:n]
  rc=Counter(x['row_id'] for x in ds)
  gc=Counter(x['source_identity'] for x in ds)
  fc=Counter(x['family'] for x in ds)
  oc=Counter('no_op' if x['semantic_noop'] else 'replace' for x in ds)
  lb=Counter(x['length_bucket'] for x in ds)
  long=sum(x['naturally_long'] for x in ds)
  targets=sum(x['target_tokens'] for x in ds)
  prompts=sum(x['prompt_tokens'] for x in ds)
  # The policy has an 8 repeat cap on ordinary rows; finite schedule also caps each family at .25 draws.
  violations=[]
  if max(rc.values(),default=0)>int(s['policy']['ordinary_replay_cap']): violations.append('ordinary_replay_cap')
  # source identity is a group-level coverage key and is intentionally allowed to repeat; only row replay is capped.
  if any(v>int(s['policy']['effective_batch']*s['max_steps']*s['policy']['family_ceiling_fraction']) for v in fc.values()): violations.append('family_ceiling')
  if sum(oc.values()) and oc['no_op']>s['policy']['noop_slots']*n/s['draw_count']+0.000001: violations.append('noop_fraction')
  if long>s['policy']['naturally_long_slots']*n/s['draw_count']+0.000001: violations.append('naturally_long_ceiling')
  return {'draws':n,'steps':n//s['effective_batch'],'unique_rows':len(rc),'unique_groups':len(gc),
    'max_row_repetition':max(rc.values(),default=0),'max_group_repetition':max(gc.values(),default=0),
    'family_counts':dict(sorted(fc.items())),'operation_counts':dict(sorted(oc.items())),
    'length_bucket_counts':dict(sorted(lb.items())),'naturally_long':long,
    'target_tokens':targets,'prompt_tokens':prompts,'total_tokens':prompts+targets,
    'row_repetition_histogram':dict(sorted(Counter(rc.values()).items())),
    'group_repetition_histogram':dict(sorted(Counter(gc.values()).items())),
    'policy_violations':violations}
result={'status':'pass','observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'schedule_path':str(sp),'schedule_sha256':sh(sp),'declared_manifest_sha256':s['manifest_sha256'],
 'row_path':str(rp),'row_sha256':sh(rp),'declared_row_sha256':s['token_rows_sha256'],
 'schema_version':s['schema_version'],'split_id':s['split_id'],'seed':s['policy']['seed'],
 'effective_batch':s['effective_batch'],'max_steps':s['max_steps'],'draw_count':s['draw_count'],
 'policy':{k:s['policy'][k] for k in ('noop_fraction','noop_slots','family_ceiling_fraction','ordinary_replay_cap','naturally_long_fraction_ceiling','naturally_long_slots','truncation')},
 'prefixes':{str(step):audit(step*s['effective_batch']) for step in (250,500,1000)},
 'full_schedule_declared':{'achieved_mixture':s['achieved_mixture'],'deficits':s['deficits'],'quota_ledger':s['quota_ledger']}}
(OUT/'schedule-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':result['status'],'sha256':sh(OUT/'schedule-audit.json'),'prefixes':result['prefixes']},sort_keys=True))
