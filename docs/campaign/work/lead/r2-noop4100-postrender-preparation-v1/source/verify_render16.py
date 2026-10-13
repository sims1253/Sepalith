#!/usr/bin/env python3
"""Verify all 31 nonempty no-op 16K parse-retry shards before fallback."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
PLAN_SHA='093fd437f6cc4a364b7273e7936459c836a18a31bc06fd84ec39ec88faacd725';INPUT_MANIFEST_SHA='38736227196ba22a5c55d1826411a20a1a50a8f3db277bb9c36ead2b696a9e65';EXPECTED=tuple(range(10,41));TERMINAL_SCHEMA='sepalith.dat10.noop4100.parse_retry.render_shard_terminal.v1'
class Error(ValueError):pass
def req(x,m):
 if not x:raise Error(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 try:x=json.loads(Path(p).read_text())
 except Exception as e:raise Error(f'metadata_read:{p}:{type(e).__name__}') from e
 req(isinstance(x,dict),f'metadata_object:{p}');return x
def rows(path):
 out=[]
 with Path(path).open() as f:
  for n,line in enumerate(f,1):
   if not line.strip():continue
   try:x=json.loads(line)
   except json.JSONDecodeError as e:raise Error(f'jsonl:{path}:{n}') from e
   rid=x.get('row_id');req(isinstance(rid,str) and rid,f'row_id:{path}:{n}');out.append((rid,x))
 req(len({x[0] for x in out})==len(out),f'duplicate_id:{path}');return out
def idsha(ids):return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
def validate(plan_path,input_manifest_path,render,output):
 req(sha(plan_path)==PLAN_SHA,'plan_sha');req(sha(input_manifest_path)==INPUT_MANIFEST_SHA,'input_manifest_sha');p=load(plan_path);m=load(input_manifest_path)
 req(p.get('rows')==m.get('prediction_inputs')==4100 and p.get('upstream_holds')==m.get('holds')==127,'denominator');req(m.get('candidate_rows')==4227 and m.get('status')=='complete_review_only' and m.get('prediction_target_free') is True and m.get('training_admission') is False,'input_manifest_contract')
 entries=p.get('entries');req(isinstance(entries,list) and len(entries)==41,'plan_entries');non=[x for x in entries if x.get('rows')];req(tuple(x['shard'] for x in non)==EXPECTED and sum(x['rows'] for x in non)==4100,'nonempty_shards')
 records=[];allids=[];supported=held=0
 for item in non:
  shard=item['shard'];inp=Path(p['input_root'])/item['path'];out=Path(render)/f'shard-{shard:04d}.jsonl';term=Path(render)/f'shard-{shard:04d}.terminal.json';t=load(term)
  req(t.get('schema')==TERMINAL_SCHEMA and t.get('status')=='complete' and t.get('exit_code')==0 and t.get('shard')==shard,'terminal_status');req(t.get('context_size')==16384 and t.get('generation_reserve')==2048,'terminal_policy');req(t.get('input',{}).get('path')==str(inp) and t['input'].get('sha256')==item['sha256']==sha(inp) and t['input'].get('rows')==item['rows'],'terminal_input')
  req(t.get('output',{}).get('path')==str(out) and t['output'].get('sha256')==sha(out) and t['output'].get('rows')==item['rows'] and t['output'].get('bytes')==out.stat().st_size,'terminal_output');log=Path(t.get('log',{}).get('path',''));req(log.is_file() and t['log'].get('sha256')==sha(log),'terminal_log')
  ins=rows(inp);outs=rows(out);req([x[0] for x in ins]==[x[0] for x in outs],'ordered_id_closure')
  for rid,x in outs:
   req(x.get('status') in ('supported','hold'),f'output_status:{rid}');reasons=x.get('reasons',[]);req(not any(str(v).startswith('prediction_namespace_infrastructure:') or v in ('source_import_inventory_invalid','namespace_helper_output_count') for v in reasons),f'infrastructure_as_hold:{rid}');req(x.get('selection_target_or_gold_used') in (None,False),f'target_used:{rid}')
   supported+=x.get('status')=='supported';held+=x.get('status')=='hold'
  ids=[x[0] for x in outs];allids+=ids;records.append({'shard':shard,'rows':len(ids),'input_sha256':item['sha256'],'output_sha256':sha(out),'terminal_sha256':sha(term),'row_ids_sha256':idsha(ids)})
 req(len(allids)==len(set(allids))==4100,'global_id_closure')
 # Exact malformed-preedit control is retained and must be a supported explicit evidence-unavailable row.
 e=None
 for item in non:
  for rid,x in rows(Path(render)/f"shard-{item['shard']:04d}.jsonl"):
   if rid=='e677ee6a8da38436f4bdb6b6':e=x;break
  if e:break
 req(e is not None and e.get('status')=='supported' and e.get('source_import_evidence')=='unavailable' and e.get('source_inventory_status')=='source_import_evidence_unavailable','e677_fallback_contract')
 result={'schema':'sepalith.dat10.noop4100.render16-terminal-review.v1','status':'complete_review_only','plan_sha256':PLAN_SHA,'input_manifest_sha256':INPUT_MANIFEST_SHA,'render_root':str(Path(render)),'terminals':records,'terminal_count':31,'provider_rows':4100,'supported':supported,'policy_holds':held,'upstream_holds':127,'candidate_denominator':4227,'all_ids_sha256':idsha(allids),'malformed_preedit_retained':True,'infrastructure_as_hold':False,'target_or_gold_used':False,'training_admission':False}
 req(not Path(output).exists(),'fresh_output');Path(output).parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=Path(output).parent,prefix='.'+Path(output).name+'.',delete=False) as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno());tmp=f.name
 os.replace(tmp,output);return result
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--input-manifest',type=Path,required=True);a.add_argument('--render',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();print(json.dumps(validate(x.plan,x.input_manifest,x.render,x.output),sort_keys=True))
if __name__=='__main__':main()
