#!/usr/bin/env python3
import argparse,collections,hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();m=json.loads((a.output/'manifest.json').read_text())
 assert m['status']=='complete_review_only_root_admission_required' and m['training_admission'] is False and m['semantic_denominator']==763 and m['existing_denominator']==15006
 for name,record in m['outputs'].items():
  q=a.output/name;assert q.stat().st_size==record['bytes'] and sha(q)==record['sha256'] and len(rows(q))==record['rows']
 candidates=rows(a.output/'candidate-tokenrows.jsonl');provenance=rows(a.output/'candidate-provenance.jsonl');ledger=rows(a.output/'exclusion-ledger.jsonl')
 ids=[x['id'] for x in candidates];pids=[x['row_id'] for x in provenance];lids=[x['row_id'] for x in ledger];assert len(ids)==len(set(ids))==616 and ids==pids and len(lids)==len(set(lids))==763
 accepted={x['row_id'] for x in ledger if x['status']=='candidate'};assert accepted==set(ids) and sum(x['status']!='candidate' for x in ledger)==147
 assert all(x['split']=='train' and x['family']=='roxygen_drafting' and x['target_token_count']<=1024 for x in candidates)
 assert all(x['target_truncated'] is False and x['selection_target_or_gold_used'] is False and x['source_identity']['split']=='train_group' and all(x['source_identity']['checks'].values()) for x in provenance)
 context=collections.Counter(x['context_size'] for x in provenance);mode=collections.Counter((x['context_size'],x['mode']) for x in provenance);helpers=collections.Counter((x['context_size'],'helper' if x['required_helper_spans'] else 'no_helper') for x in provenance);packages={x['source_identity']['package_id'] for x in provenance};groups={x['source_identity']['group_id'] for x in provenance}
 result={'schema':'sepalith.dat10.semantic763.dedup_verification.v1','status':'pass','manifest_sha256':sha(a.output/'manifest.json'),'candidate_rows':616,'held_or_excluded_rows':147,'semantic_rows':763,'candidate_ids_sha256':hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),'context_counts':dict(context),'context_mode_counts':{f'{k[0]}:{k[1]}':v for k,v in mode.items()},'context_helper_counts':{f'{k[0]}:{k[1]}':v for k,v in helpers.items()},'packages':len(packages),'source_groups':len(groups),'checks':['output byte/hash/row closure','exact candidate/provenance order','candidate and ledger ID conservation','TRAIN-only family and source membership','complete targets at or below reserve','no target truncation','selection did not use target/gold','all source/provenance gates remain true']}
 a.result.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
