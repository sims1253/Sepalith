#!/usr/bin/env python3
"""Materialize the exact admitted-minus-two contradiction view; never rewrite source."""
import hashlib,json,os
from pathlib import Path
SOURCE=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/combined-token-rows.jsonl')
OUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl')
EXCLUDED={
 '8451310ff8e0c5d9f3e77bbc':'R4.6.1 unsupported [[ on native-pipe RHS outside editable selected region',
 'dd65bd2cd11f38e729a9c712':'R4.6.1 unsupported * on native-pipe RHS outside editable selected region',
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.parent.mkdir(parents=True,exist_ok=True);tmp=OUT.with_suffix('.jsonl.new');seen=[];kept=0;kept_ids=set()
 with SOURCE.open() as src,tmp.open('x') as dst:
  for line in src:
   r=json.loads(line);rid=r['id']
   if rid in EXCLUDED:seen.append(rid);continue
   dst.write(line);kept+=1;kept_ids.add(rid)
 candidate=SOURCE.with_name('candidate-token-rows.jsonl')
 candidate_ids={json.loads(line)['id'] for line in candidate.open()}
 finish=len(candidate_ids & kept_ids)
 if set(seen)!=set(EXCLUDED) or kept!=15006 or len(candidate_ids)!=3503 or finish!=3503:
  tmp.unlink();raise ValueError({'seen':seen,'kept':kept,'new_finish':finish,'candidate_rows':len(candidate_ids)})
 os.replace(tmp,OUT)
 result={'schema':'sepalith.dat10.expanded-15006.named-exclusion.v1','status':'prepared_root_exclusion_admission_required','source':{'path':str(SOURCE),'sha256':sha(SOURCE),'rows':15008},'output':{'path':str(OUT),'sha256':sha(OUT),'rows':kept},'excluded':[{'id':i,'reason':EXCLUDED[i],'repair':'not invented; selected-region geometry cannot yield a valid native pipe'} for i in sorted(EXCLUDED)],'new_finish_rows_retained':finish,'other_exclusions':0,'checks':{'source_immutable':True,'exact_two_named_rows_removed':True,'all3503_finish_rows_retained':True}}
 Path(__file__).with_name('full15006-exclusion-ledger.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result['output']))
if __name__=='__main__':main()
