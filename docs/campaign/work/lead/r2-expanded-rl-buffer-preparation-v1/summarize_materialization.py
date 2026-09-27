#!/usr/bin/env python3
from __future__ import annotations
from collections import Counter,defaultdict
import argparse,hashlib,json
from pathlib import Path

def category(reason:str)->str:
 for name in ('pinned_snapshot_unavailable','no_complete_or_prefix_buffer_materialized','reconstructed_context_hash_mismatch',
              'complete_baseline_parse_failed','gold_applied_parse_failed','framed_gold_projection_parse_failed'):
  if name in reason:return name
 return reason.split(':',1)[0]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('destination',type=Path);a=ap.parse_args()
 counts=Counter();by_family=Counter();by_origin=Counter();ids=defaultdict(list);rows=supported=0;positions=[]
 side=a.output/'reward-buffer-sidecar.jsonl'
 with side.open() as f:
  for line in f:
   x=json.loads(line);rows+=1;positions.append(x['position'])
   if x['supported']:supported+=1;continue
   c=category(x['repair_reason']);counts[c]+=1;by_family[(x['family'],c)]+=1;by_origin[(x['origin'],c)]+=1;ids[c].append(x['row_id'])
 raw=side.read_bytes()
 result={'schema':'sepalith.rl11.expanded-buffer-repair-summary.v1','output':str(a.output),'rows':rows,'supported':supported,
  'repair':rows-supported,'positions_exact':positions==list(range(rows)),'sidecar_sha256':hashlib.sha256(raw).hexdigest(),
  'reason_counts':dict(sorted(counts.items())),'family_reason_counts':{f'{a}|{b}':n for (a,b),n in sorted(by_family.items())},
  'origin_reason_counts':{f'{a}|{b}':n for (a,b),n in sorted(by_origin.items())},'repair_ids_by_reason':dict(sorted(ids.items())),
  'policy':'Every unsupported admitted ID remains explicit; no row is truncated, silently omitted, or converted to an inferred syntax claim.'}
 a.destination.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
