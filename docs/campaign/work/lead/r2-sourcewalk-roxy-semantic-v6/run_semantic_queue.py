#!/usr/bin/env python3
"""Serial atomic/resumable semantic analysis of a completed provenance replay."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,subprocess,sys,uuid
from pathlib import Path

HERE=Path(__file__).parent
BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1')
ANALYZER=HERE/'analyze_semantics.py'
SCOPE_HELPER=HERE/'semantic_scope.R'
NAMESPACE_HELPER=HERE/'namespace_scope.R'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(path,value):
 q=path.with_name(path.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
 with q.open('rb') as f:os.fsync(f.fileno())
 q.replace(path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def reusable(path,provenance_sha,packet_sha,analyzer_sha,helper_sha,namespace_helper_sha):
 try:m=json.loads((path/'manifest.json').read_text())
 except Exception:return False
 out=m.get('output',{});p=Path(out.get('path',''))
 return (m.get('status')=='complete_review_only' and m.get('inputs',{}).get('provenance_ledger',{}).get('sha256')==provenance_sha and
         m.get('inputs',{}).get('candidate_packets',[{}])[0].get('sha256')==packet_sha and m.get('code',{}).get('analyzer_sha256')==analyzer_sha and
         m.get('code',{}).get('scope_helper_sha256')==helper_sha and m.get('code',{}).get('namespace_helper_sha256')==namespace_helper_sha and
         m.get('exact_id_closure') is True and p.is_file() and p.stat().st_size==out.get('bytes') and sha(p)==out.get('sha256'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--replay-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 replay=json.loads((a.replay_root/'manifest.json').read_text())
 if replay.get('status')!='complete_review_only_no_admission':raise RuntimeError('provenance replay is not terminal')
 expected=set();observed=set();counts=collections.Counter();receipts=[];analyzer_sha=sha(ANALYZER);helper_sha=sha(SCOPE_HELPER);namespace_helper_sha=sha(NAMESPACE_HELPER)
 for item in replay['shard_receipts']:
  shard=item['shard'];receipt_path=a.replay_root/'shards'/f'shard-{shard:04d}/receipt.json'
  if sha(receipt_path)!=item['receipt_sha256']:raise RuntimeError(f'provenance receipt changed:{shard}')
  receipt=json.loads(receipt_path.read_text());provenance=Path(receipt['outputs'][0]['path']);provenance_sha=sha(provenance)
  if provenance_sha!=receipt['outputs'][0]['sha256']:raise RuntimeError(f'provenance ledger changed:{shard}')
  shard_expected=set()
  with provenance.open() as f:
   for line in f:
    x=json.loads(line)
    if x.get('family')=='roxygen_drafting' and x.get('status')=='provenance_pass_semantic_analyzer_queued':expected.add(x['row_id']);shard_expected.add(x['row_id'])
  packet=BASE/f'shard-{shard:04d}/structured-materialization-v1/candidate-packets.jsonl';packet_sha=receipt.get('binding',{}).get('candidate_packets_sha256')
  if not isinstance(packet_sha,str) or len(packet_sha)!=64:raise RuntimeError(f'provenance packet binding missing:{shard}')
  target=a.output/f'shard-{shard:04d}'
  if not reusable(target,provenance_sha,packet_sha,analyzer_sha,helper_sha,namespace_helper_sha):
   if target.exists():raise RuntimeError(f'nonreusable terminal:{shard}')
   stage=a.output/f'.shard-{shard:04d}-{uuid.uuid4().hex}'
   done=subprocess.run([sys.executable,str(ANALYZER),'--provenance-ledger',str(provenance),'--candidate-packets',str(packet),'--expected-candidate-packets-sha256',packet_sha,'--output',str(stage)])
   if done.returncode:raise RuntimeError(f'analyzer failed:{shard}:{done.returncode}')
   child=json.loads((stage/'manifest.json').read_text());child_ledger=stage/'semantic-ledger.jsonl'
   with child_ledger.open() as stream:child_ids={json.loads(line)['row_id'] for line in stream}
   if child_ids!=shard_expected or child.get('rows')!=len(shard_expected) or child.get('code')!={'analyzer_sha256':analyzer_sha,'scope_helper_sha256':helper_sha,'namespace_helper_sha256':namespace_helper_sha} or child.get('inputs',{}).get('candidate_packets',[{}])[0].get('sha256')!=packet_sha:raise RuntimeError(f'new shard closure invalid:{shard}')
   stage.replace(target);child['output']['path']=str(target/'semantic-ledger.jsonl');atomic(target/'manifest.json',child);fd=os.open(a.output,os.O_RDONLY);os.fsync(fd);os.close(fd)
  manifest=json.loads((target/'manifest.json').read_text());ledger=Path(manifest['output']['path'])
  # Staging paths are embedded before atomic rename; resolve to the terminal file.
  if not ledger.exists():ledger=target/'semantic-ledger.jsonl'
  shard_observed=set()
  with ledger.open() as stream:
   for line in stream:
    x=json.loads(line)
    if x['row_id'] in observed:raise RuntimeError('duplicate semantic ID')
    observed.add(x['row_id']);shard_observed.add(x['row_id']);counts[x['status']]+=1
  if shard_observed!=shard_expected:raise RuntimeError(f'exact per-shard ID closure mismatch:{shard}')
  receipts.append({'shard':shard,'manifest_sha256':sha(target/'manifest.json'),'rows':manifest['rows']})
 if observed!=expected:raise RuntimeError(f'semantic queue ID closure mismatch:{len(observed)}/{len(expected)}')
 result={'schema':'sepalith.dat10.sourcewalk_roxy_semantic_queue.v6','status':'complete_review_only','provenance_manifest_sha256':sha(a.replay_root/'manifest.json'),'code':{'analyzer_sha256':analyzer_sha,'scope_helper_sha256':helper_sha,'namespace_helper_sha256':namespace_helper_sha,'queue_runner_sha256':sha(Path(__file__).resolve())},'rows':len(observed),'status_counts':dict(counts),'shards':receipts,'exact_id_closure':True,'training_admission':False}
 atomic(a.output/'manifest.json',result);print(json.dumps({'rows':len(observed),'status_counts':dict(counts)}))
if __name__=='__main__':main()
