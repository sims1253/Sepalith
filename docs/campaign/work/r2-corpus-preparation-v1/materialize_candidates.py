"""Root-run CPU candidate materialization through the existing DAT04 builders.

Output remains unadmitted. Finish packets still require the accepted strict-v2
outer-brace materialization and all rows require PRM03/token/registry review.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
HERE=Path(__file__).resolve().parent
EXEC=Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
STRUCTURED=Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/04ee1310deac54a2af6916755d9f1d22d3433b17878a787be99457e3f994ab70/source/experiments/training/campaign_structured_batch.py')
SPLIT=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['structured','completion']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if a.output.exists():raise ValueError('output_must_be_fresh')
 pins=json.loads((HERE/'candidate-inputs.json').read_text())
 for file,digest in pins.items():
  if sha(Path(file))!=digest:raise ValueError('input_identity_changed:'+file)
 roster=HERE/'novel-train-audit-roster.jsonl'
 with roster.open() as f:
  rows=[json.loads(line) for line in f]
 if not rows or any(r['split']!='train_group' for r in rows):raise ValueError('TRAIN_only_roster_required')
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
 if a.kind=='completion':
  sys.path[:0]=[str(EXEC/'experiments/training'),str(EXEC/'packages/sepalith/src')]
  from campaign_completion_batch import materialize_completion_batch
  a.output.mkdir(parents=True)
  result=materialize_completion_batch(roster,a.output/'completion-packets.jsonl',max_rows=4096,profile_limit=512,summary_path=a.output/'completion-summary.json')
 else:
  source=STRUCTURED.parents[2]
  sys.path[:0]=[str(STRUCTURED.parent),str(source/'packages/sepalith/src')]
  spec=importlib.util.spec_from_file_location('r2_existing_structured',STRUCTURED);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  # Only the identity of the bounded TRAIN metadata input changes. The source,
  # full split check, raw row hashes, reconstruction and validators are unchanged.
  module.AUDIT_SHA=sha(roster)
  sys.argv=[str(STRUCTURED),'--audit',str(roster),'--split',str(SPLIT),'--output',str(a.output),'--per-family','1024','--package-family-cap','4','--max-seconds','1200']
  module.main();result={'status':'existing_structured_builder_finished_requires_source_token_registry_review'}
 print(json.dumps({'kind':a.kind,'output':str(a.output),'roster_sha256':sha(roster),'admitted':False,'result_status':result.get('status','candidate_packets_only')}))
if __name__=='__main__':main()
