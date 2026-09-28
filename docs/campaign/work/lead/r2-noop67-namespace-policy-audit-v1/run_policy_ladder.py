#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess
from pathlib import Path
CONTEXTS=(16384,32768,65536,131072); TOKENIZER_SHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
def req(x,m):
 if not x:raise ValueError(m)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def split_stage(inputs,outputs):
 req(len(inputs)==len(outputs),'stage count');by={x['row_id']:x for x in outputs};req(len(by)==len(outputs) and set(by)=={x['row_id'] for x in inputs},'stage ids');supported=[];held=[]
 for x in inputs:
  y=by[x['row_id']];req(y['status'] in ('supported','hold'),'stage status')
  (supported if y['status']=='supported' else held).append(y)
 return supported,held
def run(a):
 inp_manifest=json.loads((a.input_root/'manifest.json').read_text());req(inp_manifest['rows']==67 and inp_manifest['target_or_gold_present'] is False,'input binding');current=load(a.input_root/'inputs67.jsonl');req(len(current)==67,'input rows');source={x['row_id']:x for x in current};req(not a.output.exists(),'fresh output required');tmp=a.output.with_name(a.output.name+'.tmp-'+str(os.getpid()));tmp.mkdir(parents=True);selected={};stages=[]
 for context in CONTEXTS:
  inpath=tmp/f'inputs-{context}.jsonl';inpath.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in current));outpath=tmp/f'render-{context}.jsonl'
  cmd=[a.node,'--no-warnings=ExperimentalWarning','--experimental-strip-types',str(a.renderer),str(inpath),str(outpath),str(a.python),str(a.bridge),str(a.tokenizer),str(a.namespace_helper),str(context),'2048']
  subprocess.run(cmd,check=True,timeout=a.timeout)
  outputs=load(outpath);supported,held=split_stage(current,outputs)
  for y in supported:req(y['row_id'] not in selected,'selected twice');req(y['mode']=='full_document','unavailable namespace must use full document');req(y['selected_context']['selected_references']==[],'fabricated namespace evidence');selected[y['row_id']]=y
  stages.append({'context':context,'input_rows':len(current),'supported':len(supported),'holds':len(held),'input_sha256':sha_bytes(inpath.read_bytes()),'output_sha256':sha_bytes(outpath.read_bytes())})
  current=[source[x['row_id']] for x in held]
 final=[selected.get(x['row_id'],{'row_id':x['row_id'],'status':'hold','reasons':['namespace_unavailable_full_document_not_selected_through_128k']}) for x in load(a.input_root/'inputs67.jsonl')];data=''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in final);(tmp/'selected-or-held.jsonl').write_text(data)
 m={'schema':'sepalith.dat10.noop67.namespace-unavailable-ladder.v1','status':'complete_review_only','rows':67,'supported':len(selected),'holds':67-len(selected),'contexts':list(CONTEXTS),'generation_reserve':2048,'stages':stages,'selected_sha256':sha_bytes(data.encode()),'provider_source_manifest_sha256':a.source_manifest_sha256,'tokenizer_sha256':TOKENIZER_SHA,'target_or_gold_used':False,'fixed_target_joined':False,'training_admission':False};(tmp/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n');os.rename(tmp,a.output);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--renderer',type=Path,required=True);p.add_argument('--node',default='node');p.add_argument('--python',type=Path,required=True);p.add_argument('--bridge',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--namespace-helper',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--timeout',type=int,default=3600);a=p.parse_args();req(hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()==TOKENIZER_SHA,'tokenizer hash');req(sha_file(a.source_manifest)==a.source_manifest_sha256,'source manifest hash');print(json.dumps(run(a),sort_keys=True))
if __name__=='__main__':main()
