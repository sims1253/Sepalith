#!/usr/bin/env python3
import collections,datetime as dt,hashlib,json,os
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');W=PLAN/'docs/campaign/work/lead/r2-cpt-all-eligible-v1';E=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
progress=json.loads((E/'progress.json').read_text())
snapshot_boundary=progress['first_uncommitted_seeded_index']
receipts=[];queued_repairs={}
for folder in sorted((E/'groups').iterdir()):
 try: seeded_index=int(folder.name.split('-',1)[0])
 except ValueError: continue
 if seeded_index >= snapshot_boundary: continue
 p=folder/'receipt.json'
 if p.is_file(): receipts.append(json.loads(p.read_text()))
counts=collections.Counter();categories=collections.Counter();reasons=collections.Counter();statuses=collections.Counter()
for r in receipts:
 counts.update(r['counts']);categories.update(r['source_categories']);statuses[r['status']]+=1
 for key,value in r['counts'].items():
  if key.startswith('excluded_') and key!='excluded_documents': reasons[key.removeprefix('excluded_')]+=value
 repair_path=E/'groups'/f'{r["seeded_index"]:06d}-{r["group_id"]}'/'repair-queue.jsonl'
 for line_number,line in enumerate(repair_path.read_text().splitlines(),1):
  row=json.loads(line);record_sha=hashlib.sha256((line+'\n').encode()).hexdigest();receipt_sha=sha(repair_path.parent/'receipt.json')
  queued_repairs[(r['seeded_index'],line_number,record_sha,receipt_sha)]=row;reasons['repair:'+row['reason']]+=1
resolution_record=json.loads((W/'repair-resolutions.json').read_text());resolved=set()
for item in resolution_record['items']:
 key=(item['seeded_index'],item['line'],item['repair_record_sha256'],item['group_receipt_sha256'])
 if key not in queued_repairs:raise ValueError('repair resolution does not bind a queued item')
 resolved.add(key);reasons['resolved:'+item['resolution']]+=1
if len(receipts) != progress['groups_committed']:
 raise ValueError('receipt snapshot does not match committed-group boundary')
if dict(counts) != progress['counts']:
 raise ValueError('receipt aggregate counts do not match progress snapshot')
progress['repair_items_recorded']=len(queued_repairs);progress['repair_items_resolved']=len(resolved);progress['repair_items_pending']=len(set(queued_repairs)-resolved)
launch=json.loads((W/'launch.json').read_text());alive=False
try:
 stat=Path(f'/proc/{launch["pid"]}/stat').read_text().rsplit(')',1)[1].split();alive=stat[0] not in ('Z','X') and stat[19]==launch['start_tick']
except (FileNotFoundError,IndexError):pass
status='in_progress' if alive else progress['status']
migrations={p.name:sha(p) for p in sorted(E.glob('source-migration-*.json'))}
value={'schema':'sepalith.campaign.receipt.v1','task':'DAT-10-all-eligible-cpt-materialization','owner':'lead/r2-cpt-all-eligible-v1','status':status,'updated_at':dt.datetime.now(dt.timezone.utc).isoformat(),'launch':{**launch,'identity_alive':alive},
 'scope':{'order_groups':8867,'start_seeded_index':775,'expected_remaining_groups':8092,'supported_payload_category':'regular .R under normalized package R trees','caps':{'wall_time':None,'package_tokens':None,'group_tokens':None,'file_bytes':None},'heldout_cpt_validation_groups':556,'non_TRAIN_payload_read':False,'final_content_accessed':False},
 'progress':{**progress,'committed_statuses':dict(statuses)},'actual_counts':dict(counts),'source_category_inventory':dict(categories),'per_reason_ledger':dict(reasons),
 'source':{'materializer_sha256':sha(W/'materialize_all_eligible.py'),'source_manifest_sha256':sha(W/'source-manifest.json'),'source_migration_sha256':sha(W/'source-migration.json'),'repair_resolutions_sha256':sha(W/'repair-resolutions.json'),'preflight_sha256':sha(W/'preflight.log'),'tests_sha256':sha(W/'tests.log'),'finalizer_sha256':sha(W/'finalize_materialization.py')},
 'resumability':{'unit':'one seeded group','commit':'fsynced immutable group directory renamed atomically after artifact hashes and receipt','reconciliation':'all committed artifact hashes and document identities verified before resume','signal':'SIGTERM/SIGINT finishes current group then stops','source_migrations_applied':migrations},
 'acceptance':'partial until all 8,092 group receipts exist, repair queues are resolved or documented with exact remaining payload, and a final ordered merge/validation receipt is produced.'}
out=PLAN/'docs/campaign/receipts/DAT-10-all-eligible-cpt-materialization.json';tmp=out.with_suffix('.json.tmp')
with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
tmp.replace(out);print(json.dumps({'receipt':str(out),'sha256':sha(out),'status':status,'groups':len(receipts),'remaining':progress['groups_remaining'],'counts':dict(counts),'reasons':dict(reasons)},sort_keys=True))
