#!/usr/bin/env python3
"""Freeze terminal metadata for the remaining committed semantic queue."""
from __future__ import annotations
import datetime, hashlib, json, os
from collections import Counter
from pathlib import Path
PACKET=Path(__file__).resolve().parent
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
V3=PLAN/'docs/campaign/work/lead/r2-semantic-queue-root-launch-v3'
REPLAY=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01')
OUTPUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards12plus-v1')
RECEIPT=PLAN/'docs/campaign/receipts/DAT-10-semantic-shards12plus-preparation.json'
SELECTED=list(range(12,27))
PRIOR_OUTPUTS=[Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-v2/shard5-root-03/shard-0005/semantic-ledger.jsonl'),Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-v2/shards0to4-root-01/shard-0003/semantic-ledger.jsonl')]+[Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1')/f'shard-{n:04d}'/'semantic-ledger.jsonl' for n in range(6,12)]
QUEUED='provenance_pass_semantic_analyzer_queued'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def rows(p):
 with p.open() as f:
  for line in f:
   if line.strip(): yield json.loads(line)
def digest(values):
 return hashlib.sha256(json.dumps(sorted(values),separators=(',',':')).encode()).hexdigest()
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def summarize_ledger(path):
 xs=list(rows(path));ids=[x.get('row_id') for x in xs];require(all(isinstance(i,str) and i for i in ids),'invalid row ID '+str(path));require(len(ids)==len(set(ids)),'duplicate row ID '+str(path));return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'rows':len(xs),'row_ids_sha256':digest(ids),'status_counts':dict(Counter(x.get('status') for x in xs)),'reason_counts':dict(Counter(r for x in xs for r in (x.get('reasons') or []))),'ids':ids}
def main():
 launch=json.loads((PACKET/'launch.json').read_text());terminal=json.loads((PACKET/'terminal.json').read_text());require(terminal.get('exit_code')==0,'queue exit code not zero')
 stream_path=OUTPUT/'streaming-manifest.json';require(stream_path.is_file(),'stream manifest absent');stream=json.loads(stream_path.read_text());require(stream.get('status')=='partial_review_only' and stream.get('training_admission') is False,'unexpected queue admission/status');require(stream.get('requested_shards')==SELECTED,'selected scope differs');require(stream.get('replay_index_sha256')==launch['preflight']['replay_index_sha256'],'replay index changed')
 children=[];source_ids=[];output_ids=[];status_counts=Counter();reason_counts=Counter()
 for n in SELECTED:
  d=OUTPUT/f'shard-{n:04d}';mp=d/'manifest.json';lp=d/'semantic-ledger.jsonl';require(mp.is_file() and lp.is_file(),'missing child '+str(n));m=json.loads(mp.read_text());out=m.get('output');require(isinstance(out,dict) and out.get('path')==str(lp),'output path binding missing '+str(n));s=summarize_ledger(lp);require(s['sha256']==out.get('sha256') and s['bytes']==out.get('bytes') and s['rows']==out.get('rows'),'output hash binding differs '+str(n));binding=m.get('streaming_binding');require(isinstance(binding,dict),'source binding missing '+str(n));src=Path(binding['provenance_ledger']['path']);require(src.is_file() and sha(src)==binding['provenance_ledger']['sha256'],'source ledger changed '+str(n));queued=list(binding.get('queued_ids') or []);require(len(queued)==binding.get('queued_rows') and digest(queued)==binding.get('queued_ids_sha256'),'source queued binding invalid '+str(n));require(sorted(queued)==sorted(s['ids']),'source/output exact ID join differs '+str(n));source_ids.extend(queued);output_ids.extend(s['ids']);status_counts.update(s['status_counts']);reason_counts.update(s['reason_counts']);children.append({'shard':n,'manifest':{'path':str(mp),'bytes':mp.stat().st_size,'sha256':sha(mp),'status':m.get('status'),'source_binding_sha256':m.get('source_provenance_binding_sha256'),'queued_ids_sha256':m.get('queued_ids_sha256')},'semantic_output':{k:v for k,v in s.items() if k!='ids'},'source_provenance_ledger':{'path':str(src),'bytes':src.stat().st_size,'sha256':binding['provenance_ledger']['sha256'],'rows':binding['provenance_ledger'].get('rows')}})
 require(len(source_ids)==len(set(source_ids)),'duplicate selected source IDs');require(len(output_ids)==len(set(output_ids)),'duplicate selected output IDs');require(sorted(source_ids)==sorted(output_ids),'selected source/output IDs differ')
 old=[]
 for p in PRIOR_OUTPUTS:
  require(p.is_file(),'prior semantic ledger absent '+str(p));s=summarize_ledger(p);old.extend(s['ids'])
 require(len(old)==len(set(old)),'prior semantic IDs overlap');overlap=set(source_ids)&set(old);require(not overlap,'selected IDs overlap prior scope')
 inp=json.loads((PACKET/'input-pin.json').read_text());selected_source_rows=sum(x['index_rows'] for x in inp['shards']);selected_source_bytes=sum(x['index_bytes'] for x in inp['shards']);require(selected_source_rows==stream['provenance_rows'],'provenance count differs')
 receipt = {
  'schema': 'sepalith.dat10.semantic_shards12plus_preparation.v1',
  'status': 'complete_review_only_pending_root_review',
  'training_admission': False,
  'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'scope': {
   'selected_shards': SELECTED,
   'prior_semantic_scope_shards': list(range(12)),
   'remaining_unreceipted_at_intake': list(range(27, 41)),
  },
  'replay': {
   'root': str(REPLAY),
   'index_manifest': inp['replay_index'],
   'index_status_at_intake': 'complete',
   'terminal_exit_code': 0,
   'selected_source_rows': selected_source_rows,
   'selected_source_index_bytes': selected_source_bytes,
   'selected_source_receipts': {x['shard']: x['receipt_sha256'] for x in inp['shards']},
   'queue_ran_against_intake_index_sha256': launch['preflight']['replay_index_sha256'],
   'replay_controller': 'historical PIDs 1689900/1689901/1691118 absent at intake; no restart attempted',
  },
  'source_reuse': {
   'packet': str(V3),
   'source_manifest_path': str(V3 / 'source-manifest.json'),
   'source_manifest_sha256': sha(V3 / 'source-manifest.json'),
   'source_files': json.loads((V3 / 'source-manifest.json').read_text())['files'],
  },
  'launch': {
   'path': str(PACKET / 'launch.json'),
   'bytes': (PACKET / 'launch.json').stat().st_size,
   'sha256': sha(PACKET / 'launch.json'),
   'controller_pid': launch['controller_pid'],
   'timeout_pid': launch['timeout_pid'],
   'selected_shards': SELECTED,
   'max_workers': 2,
   'time_limit_seconds': launch['time_limit_seconds'],
  },
  'terminal': {
   'path': str(PACKET / 'terminal.json'),
   'bytes': (PACKET / 'terminal.json').stat().st_size,
   'sha256': sha(PACKET / 'terminal.json'),
   'exit_code': terminal['exit_code'],
   'elapsed_seconds': terminal['elapsed_seconds'],
   'streaming_manifest': {
    'path': str(stream_path), 'bytes': stream_path.stat().st_size, 'sha256': sha(stream_path),
   },
  },
  'queue_manifest': {
   'path': str(stream_path),
   'bytes': stream_path.stat().st_size,
   'sha256': sha(stream_path),
   'status': stream['status'],
   'training_admission': stream['training_admission'],
   'provenance_rows': stream['provenance_rows'],
   'semantic_rows': stream['semantic_rows'],
   'source_pool_closed': stream.get('source_pool_closed'),
   'closure': stream.get('closure'),
   'code': stream.get('code'),
  },
  'children': children,
  'aggregate_semantic_status_counts': dict(status_counts),
  'aggregate_reason_counts': dict(reason_counts),
  'conservation': {
   'source_queued_rows': len(source_ids),
   'semantic_output_rows': len(output_ids),
   'source_output_id_equal': True,
   'source_queued_ids_sha256': digest(source_ids),
   'semantic_output_ids_sha256': digest(output_ids),
   'prior_semantic_rows_considered': len(old),
   'prior_overlap_rows': len(overlap),
   'prior_overlap_ids_sha256': digest(sorted(overlap)),
  },
  'review_gates': {
   'source_license_global_split_protected_holdout': 'pending root admission review',
   'exact_prompt_target_dedup': 'pending root admission review',
   'geometry_terminal_protocol': 'pending root admission review',
   'semantic_statuses': 'review evidence only; no admission',
  },
  'next_frontier': {
   'committed_shards_not_selected': list(range(27, 41)),
   'source_receipts_not_present_at_intake': list(range(27, 41)),
  },
  'artifacts_unchanged': True,
 }
 write(RECEIPT,receipt);print(json.dumps({'receipt':str(RECEIPT),'receipt_sha256':sha(RECEIPT),'selected_source_rows':selected_source_rows,'selected_queued_rows':len(source_ids),'semantic_rows':len(output_ids),'prior_overlap':len(overlap),'status_counts':dict(status_counts)}))
if __name__=='__main__':main()
