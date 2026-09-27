#!/usr/bin/env python3
import collections,hashlib,json,sys
from pathlib import Path
base=Path(sys.argv[1]);prep=base/'prepared-02';cand=base/'candidates-01';rendered=base/'rendered-03.jsonl'
def rows(path,key=lambda x:x.get('row_id') or x.get('id')):
 out={}
 for line in path.open():
  if not line.strip():continue
  x=json.loads(line);rid=key(x);assert isinstance(rid,str) and rid and rid not in out;out[rid]=x
 return out
def sha(b):return hashlib.sha256(b).hexdigest()
pm=json.loads((prep/'preparation-manifest.json').read_text());cm=json.loads((cand/'manifest.json').read_text());pred=rows(prep/'prediction-inputs.jsonl');side=rows(prep/'training-sidecar.jsonl');preholds=rows(prep/'preparation-holds.jsonl');rend=rows(rendered)
assert pm['semantic_supported']==763 and len(pred)==len(side)==len(rend)==762 and len(preholds)==1 and set(pred)==set(side)==set(rend) and not (set(pred)&set(preholds));checks=8
for x in pred.values():
 assert not any(k in x for k in ('target','target_lines','target_body','target_text','gold','target_tokens'));assert sha(x['preedit_text'].encode())==x['preedit_sha256'];checks+=2
for cap in (16384,32768):
 token=rows(cand/f'candidate-tokenrows-{cap}.jsonl');prof=rows(cand/f'profiles-{cap}.jsonl');holds=rows(cand/f'holds-{cap}.jsonl');assert set(token)==set(prof);assert not(set(token)&set(holds));assert set(token)|set(holds)|set(preholds)==set(pred)|set(preholds);checks+=3
 prompt_groups=collections.defaultdict(set)
 for rid,row in token.items():
  p=prof[rid];s=side[rid];assert row['id']==rid and row['bos_token_id']==0 and row['eos_token_id']==1 and len(row['input_ids'])==p['sequence_tokens']<=cap;assert all(type(v) is int and 0<=v<130560 for v in row['input_ids']);assert row['input_ids'][0]==0 and row['input_ids'][-1]==1;assert row['target_start']==1+row['prompt_token_count'] and row['target_token_count']==row['target_body_token_count']+row['target_terminal_token_count'];assert sha(row['prompt_text'].encode())==p['prompt_sha256'];assert row['target_body_text'].split('\n')==s['target_lines'];assert not p['external_import_dependencies'];assert row['target_token_count']<=1024;prompt_groups[p['prompt_sha256']].add(p['target_sha256']);checks+=9
 assert all(len(v)==1 for v in prompt_groups.values());assert not any(h['reason'] in ('duplicate_prompt_target','prompt_target_contradiction') for h in holds.values());checks+=2
 assert len(token)+len(holds)+len(preholds)==763 and cm['profiles'][str(cap)]['exact_763_accounting'];checks+=2
print(json.dumps({'status':'pass','checks':checks,'prediction_rows':len(pred),'preparation_holds':len(preholds),'profile_counts':{c:cm['profiles'][str(c)] for c in (16384,32768)}},sort_keys=True))
