#!/usr/bin/env python3
import argparse,hashlib,json,os
from pathlib import Path
MANIFEST_SHA='38736227196ba22a5c55d1826411a20a1a50a8f3db277bb9c36ead2b696a9e65'
HOLDS_SHA='f82903981cc331a71bd7e23f1336af7a35704a0e2670282e86cfb86534937be8'
FORBIDDEN={'target','target_text','gold','label','no_op'}
def req(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def prepare(input_root,holds_path,out):
 manifest=input_root/'manifest.json';req(sha(manifest)==MANIFEST_SHA,'input manifest hash');m=json.loads(manifest.read_text());req(m['prediction_inputs']==4100 and m['holds']==127,'input accounting')
 req(sha(holds_path)==HOLDS_SHA,'holds hash');holds=load(holds_path);req(len(holds)==67 and len({x['row_id'] for x in holds})==67,'67 hold ids');wanted={x['row_id'] for x in holds};found={}
 for s in m['shards']:
  if s['prediction_inputs']==0:continue
  p=input_root/f"shard-{s['shard']:04d}.jsonl";req(sha(p)==s['prediction_sha256'],f'shard hash:{s["shard"]}')
  rows=load(p);req(len(rows)==s['prediction_inputs'],f'shard rows:{s["shard"]}')
  for x in rows:
   rid=x['row_id']
   if rid in wanted:
    req(rid not in found,'duplicate selected id');req(not(FORBIDDEN&set(x)),'target-shaped provider field');found[rid]=x
 req(set(found)==wanted,'missing held input ids');req(not out.exists(),'fresh output required');tmp=out.with_name(out.name+'.tmp-'+str(os.getpid()));tmp.mkdir(parents=True)
 ordered=[found[x['row_id']] for x in holds];data=''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in ordered);(tmp/'inputs67.jsonl').write_text(data)
 om={'schema':'sepalith.dat10.noop67.targetfree-inputs.v1','status':'prepared_root_review_required','rows':67,'inputs_sha256':hashlib.sha256(data.encode()).hexdigest(),'row_ids_sha256':hashlib.sha256((''.join(x['row_id']+'\n' for x in ordered)).encode()).hexdigest(),'source_manifest_sha256':MANIFEST_SHA,'holds_sha256':HOLDS_SHA,'target_or_gold_present':False,'training_admission':False};(tmp/'manifest.json').write_text(json.dumps(om,sort_keys=True,indent=2)+'\n');os.rename(tmp,out);return om
def main():
 a=argparse.ArgumentParser();a.add_argument('--input-root',type=Path,required=True);a.add_argument('--holds',type=Path,required=True);a.add_argument('--output',type=Path,required=True);n=a.parse_args();print(json.dumps(prepare(n.input_root,n.holds,n.output),sort_keys=True))
if __name__=='__main__':main()
