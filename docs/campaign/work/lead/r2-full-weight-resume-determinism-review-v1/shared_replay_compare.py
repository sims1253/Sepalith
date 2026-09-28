#!/usr/bin/env python3
"""Compare two deterministic continuations from one exact checkpoint."""
import argparse,json
from pathlib import Path
FIELDS=("global_step","model_state_sha256","optimizer_state_sha256","scheduler_state_sha256","cpu_rng_sha256","cuda_rng_sha256","draw_cursor","optimizer_dispatch_sha256")
def compare(a,b):
 a=json.loads(Path(a).read_text()); b=json.loads(Path(b).read_text())
 if a.get('resume_from') != b.get('resume_from'): raise ValueError('replays did not use one shared checkpoint')
 values={key:a.get(key)==b.get(key) for key in FIELDS}
 if not all(values.values()): raise ValueError('shared-checkpoint continuations differ exactly')
 return {'schema':'sepalith.sft11.shared-checkpoint-replay-comparison.v1','status':'exact_match','shared_checkpoint':a['resume_from'],'comparisons':values}
def main():
 p=argparse.ArgumentParser();p.add_argument('--a',type=Path,required=True);p.add_argument('--b',type=Path,required=True);p.add_argument('--output',type=Path,required=True);x=p.parse_args();r=compare(x.a,x.b);x.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
