#!/usr/bin/env python3
"""Tightened independent gates on the frozen v1 20-row bounded sample."""
import argparse, collections, importlib.util, json, time
from pathlib import Path

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('full_replay',HERE/'full_replay.py'); m=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(m)
OLD=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v1/attempt-06/sample-ledger.jsonl')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise ValueError('fresh output required')
 a.output.mkdir(parents=True);started=time.monotonic(); old=[json.loads(x) for x in OLD.open()]; ids={x['row_id'] for x in old}
 packets={}
 for shard in sorted({x['shard'] for x in old}):
  p=m.BASE/f'shard-{shard:04d}'/'structured-materialization-v1/candidate-packets.jsonl'
  with p.open() as stream:
   for line in stream:
    item=json.loads(line);rid=item.get('row_ref',{}).get('row_id')
    if rid in ids:packets[rid]=item
 if set(packets)!=ids:raise RuntimeError('sample packet join incomplete')
 wanted=collections.defaultdict(lambda:collections.defaultdict(list))
 for x in old:wanted[Path(x['raw_source_path'])][x['raw_source_line']].append(x['row_id'])
 raw_rows={};raw_evidence=[]
 for path, lines in wanted.items():
  before=path.stat();identity=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns);digest=m.hashlib.sha256();found=set()
  with path.open('rb',buffering=4<<20) as stream:
   for number,line in enumerate(stream,1):
    digest.update(line)
    if number in lines:
     found.add(number)
     for rid in lines[number]:raw_rows[rid]=json.loads(line)
  after=path.stat();stable=identity==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
  expected={x['raw_source_sha256'] for x in old if x['raw_source_path']==str(path)}
  if not stable or expected!={digest.hexdigest()} or found!=set(lines):raise RuntimeError('raw source stability/hash/line gate failed')
  raw_evidence.append({'path':str(path),'sha256':digest.hexdigest(),'bytes':after.st_size,'stat_stable':stable,'selected_lines':len(lines)})
 parser=m.load_module('bounded_license',m.LICENSE,m.LICENSE_SHA);results=[]
 for oldrow in old:
  rid=oldrow['row_id'];packet=packets[rid];raw=raw_rows[rid];source,source_ev=m.stable_read(Path(oldrow['normalized_source_path']));desc,desc_ev=m.stable_read(Path(oldrow['license_path']))
  license_decision=m.positive_license(desc,oldrow['package_id'],parser);window,method=m.source_window(packet,raw,oldrow['family']);occ,occurrence_method=m.window_occurrence(source,window)
  noop=m.noop_geometry(packet,raw,source) if oldrow['family']=='no_op' else None
  gates=source_ev['stat_stable'] and desc_ev['stat_stable'] and source_ev['sha256']==oldrow['normalized_source_sha256'] and desc_ev['sha256']==oldrow['license_sha256'] and license_decision['ok'] and method in {'packet_selection_source_exact','raw_post_edit_source_support_window'} and occ==1
  status=('provenance_supported_candidate_root_review_required' if gates and noop and noop['ok'] else
          'provenance_pass_semantic_analyzer_queued' if gates and noop is None else 'hold_independent_provenance_failure')
  results.append({'row_id':rid,'family':oldrow['family'],'package_id':oldrow['package_id'],'shard':oldrow['shard'],'source_stat_stable':source_ev['stat_stable'],'description_stat_stable':desc_ev['stat_stable'],'license_decision':license_decision,'window_method':method,'window_occurrences':occ,'window_occurrence_method':occurrence_method,'noop_geometry':noop,'prior_v1_status':oldrow['status'],'status':status})
 ledger=a.output/'bounded-ledger.jsonl';n,h=m.atomic_lines(ledger,sorted(results,key=lambda x:x['row_id']))
 manifest={'schema':'sepalith.dat10.sourcewalk_independent_replay_bounded.v3','status':'complete_review_only','rows':n,'families':dict(collections.Counter(x['family'] for x in results)),'status_counts':dict(collections.Counter(x['status'] for x in results)),'raw_sources':raw_evidence,'old_sample':{'path':str(OLD),'sha256':m.sha(OLD)},'held_824a_investigation':next(x for x in results if x['row_id']=='824a351dc66d530f239aa3d9'),'outputs':{'ledger':{'path':str(ledger),'sha256':h,'bytes':ledger.stat().st_size,'rows':n}},'elapsed_seconds':time.monotonic()-started,'training_admission':False}
 m.atomic_json(a.output/'manifest.json',manifest);print(m.canonical({'rows':n,'status_counts':manifest['status_counts'],'elapsed_seconds':manifest['elapsed_seconds']}))

if __name__=='__main__':main()
