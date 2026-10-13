#!/usr/bin/env python3
"""Freeze the terminal metadata for the bounded shards-6+ semantic queue.

This consumes only queue/replay metadata and semantic ledgers. It never marks
training admission and it does not alter any producer or accepted artifact.
"""
from __future__ import annotations
import datetime,hashlib,json,os
from pathlib import Path
PACKET=Path(__file__).resolve().parent
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
V3=PLAN/'docs/campaign/work/lead/r2-semantic-queue-root-launch-v3'
REPLAY=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01')
OUTPUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1')
RECEIPT=PLAN/'docs/campaign/receipts/DAT-10-semantic-shards6plus-preparation.json'
SELECTED=[6,7,8,9,10,11]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def lines(p):
 with p.open() as f:
  for n,line in enumerate(f,1):
   if line.strip(): yield json.loads(line)
def ids_digest(values):
 return hashlib.sha256(json.dumps(sorted(values),separators=(',',':')).encode()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def ledger_summary(path):
 rows=list(lines(path));ids=[x.get('row_id') for x in rows];require(all(isinstance(x,str) and x for x in ids),'bad ledger id');require(len(ids)==len(set(ids)),'duplicate ledger id '+str(path));
 from collections import Counter
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'rows':len(rows),'row_ids_sha256':ids_digest(ids),'status_counts':dict(Counter(x.get('status') for x in rows)),'reason_counts':dict(Counter(r for x in rows for r in (x.get('reasons') or []))),'ids':ids}
def main():
 launch=json.loads((PACKET/'launch.json').read_text());term=json.loads((PACKET/'terminal.json').read_text())
 require(term.get('exit_code')==0,'queue did not exit zero')
 sm=OUTPUT/'streaming-manifest.json';require(sm.is_file(),'queue streaming manifest missing');stream=json.loads(sm.read_text());require(stream.get('training_admission') is False,'queue admission flag changed');require(stream.get('requested_shards')==SELECTED,'queue selected scope differs');require(stream.get('status')=='partial_review_only','bounded queue must remain partial review only')
 require(stream.get('replay_index_sha256')==launch['preflight']['index_sha256'],'replay index changed across queue run')
 summaries=[];source_ids=[];output_ids=[];old_ids=[]
 for n in SELECTED:
  d=OUTPUT/f'shard-{n:04d}';mp=d/'manifest.json';require(mp.is_file(),'child manifest missing '+str(n));m=json.loads(mp.read_text());out=m.get('output');require(isinstance(out,dict) and out.get('path')==str(d/'semantic-ledger.jsonl'),'child output binding missing');s=ledger_summary(d/'semantic-ledger.jsonl');require(s['sha256']==out.get('sha256') and s['bytes']==out.get('bytes') and s['rows']==out.get('rows'),'child output hash binding differs');summaries.append({'shard':n,'manifest_path':str(mp),'manifest_bytes':mp.stat().st_size,'manifest_sha256':sha(mp),'manifest_status':m.get('status'),'source_provenance_binding_sha256':m.get('source_provenance_binding_sha256'),'queued_ids_sha256':m.get('queued_ids_sha256'),'semantic_output':{k:v for k,v in s.items() if k!='ids'}});output_ids.extend(s['ids'])
  # The queue's source ledger is immutable and its child manifest binds its hash.
  src=Path(m['streaming_binding']['provenance_ledger']['path']);ss=ledger_summary(src);queued=[x['row_id'] for x in lines(src) if x.get('family')=='roxygen_drafting' and x.get('status')=='provenance_pass_semantic_analyzer_queued'];require(ids_digest(queued)==m['queued_ids_sha256'],'source queued ID digest differs');source_ids.extend(queued)
 for old in [OUTPUT.parent/'Sourcewalk-semantic-streaming-queue-v2/shard5-root-03/shard-0005/semantic-ledger.jsonl',OUTPUT.parent/'Sourcewalk-semantic-streaming-queue-v2/shards0to4-root-01/shard-0003/semantic-ledger.jsonl']:
  if old.is_file():old_ids.extend(x.get('row_id') for x in lines(old))
 require(len(source_ids)==len(set(source_ids)),'duplicate source queued IDs across selected shards')
 require(len(output_ids)==len(set(output_ids)),'duplicate output IDs across selected shards')
 require(sorted(source_ids)==sorted(output_ids),'source/output row ID closure differs')
 overlap=set(output_ids)&set(old_ids)
 # bind selected source receipts from launch preflight and current immutable outputs
 inp=json.loads((PACKET/'input-pin.json').read_text());source_sha={x['shard']:x['receipt_sha256'] for x in inp['shards']}
 source_rows={x['shard']:x['index_rows'] for x in inp['shards']}
 require(sum(source_rows.values())==stream['provenance_rows'],'stream source row count differs')
 receipt={'schema':'sepalith.dat10.semantic_shards6plus_preparation.v1','status':'complete_review_only_pending_root_review','training_admission':False,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':{'selected_shards':SELECTED,'already_processed_shards':[0,1,2,3,4,5],'remaining_committed_not_selected':list(range(12,27)),'not_yet_receipted_at_intake':list(range(27,41))},'replay':{'root':str(REPLAY),'index_manifest':inp['replay_index'],'index_status_at_intake':'complete','selected_source_rows':sum(source_rows.values()),'selected_source_index_bytes':sum(x['index_bytes'] for x in inp['shards']),'selected_source_receipts':source_sha,'controller_pids_at_intake':[1689900,1689901,1691118],'controller_owner':'pre-existing independent replay controller; not restarted or interrupted','queue_ran_against_intake_index_sha256':launch['preflight']['index_sha256']},'source_reuse':{'packet':str(V3),'source_manifest_path':str(V3/'source-manifest.json'),'source_manifest_sha256':sha(V3/'source-manifest.json'),'source_files':json.loads((V3/'source-manifest.json').read_text())['files']},'launch':{'path':str(PACKET/'launch.json'),'bytes':(PACKET/'launch.json').stat().st_size,'sha256':sha(PACKET/'launch.json'),'controller_pid':launch['controller_pid'],'timeout_pid':launch['timeout_pid'],'worker_scope':launch['selected_shards'],'max_workers':2,'time_limit_seconds':launch['time_limit_seconds']},'terminal':{'path':str(PACKET/'terminal.json'),'bytes':(PACKET/'terminal.json').stat().st_size,'sha256':sha(PACKET/'terminal.json'),'exit_code':term['exit_code'],'elapsed_seconds':term['elapsed_seconds'],'output_streaming_manifest':{'path':str(sm),'bytes':sm.stat().st_size,'sha256':sha(sm)}},'queue_manifest':{'path':str(sm),'bytes':sm.stat().st_size,'sha256':sha(sm),'status':stream['status'],'semantic_rows':stream['semantic_rows'],'provenance_rows':stream['provenance_rows'],'source_pool_closed':stream.get('source_pool_closed'),'closure':stream.get('closure'),'code':stream.get('code')},'children':summaries,'conservation':{'source_queued_rows':len(source_ids),'semantic_output_rows':len(output_ids),'source_output_id_equal':True,'source_queued_ids_sha256':ids_digest(source_ids),'semantic_output_ids_sha256':ids_digest(output_ids),'old_processed_semantic_ids_considered':len(old_ids),'overlap_with_old_processed_semantic_ids':len(overlap),'old_processed_overlap_ids_sha256':ids_digest(sorted(overlap)) if overlap else ids_digest([])},'review_gates':{'source_license_global_split_and_protected_holdout_review':'pending root admission review','exact_prompt_target_dedup':'not established by this semantic queue; root admission review required','geometry_and_terminal_protocol':'not established by this semantic queue; root admission review required','semantic_statuses_are_review_evidence_only':True},'next_frontier':{'committed_shards_not_selected':list(range(12,27)),'committed_rows_not_selected':sum(json.loads((REPLAY/'index/manifest.json').read_text())['index_files'][n]['rows'] for n in range(12,27)),'source_receipts_not_present_at_intake':list(range(27,41))},'artifacts_unchanged':True}
 write(RECEIPT,receipt)
 print(json.dumps({'receipt':str(RECEIPT),'receipt_sha256':sha(RECEIPT),'selected_rows':len(source_ids),'status':receipt['status'],'old_overlap':len(overlap)}))
if __name__=='__main__':main()
