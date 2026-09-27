#!/usr/bin/env python3
"""Select 16K/32K contexts and preserve the complete no-op denominator."""
import argparse,collections,hashlib,json,os,shutil,tempfile
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def req(v,m):
 if not v:raise ValueError(m)
def keyed(path):
 out={}
 with Path(path).open() as stream:
  for line in stream:
   x=json.loads(line);req(x['row_id'] not in out,'duplicate row');out[x['row_id']]=x
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan16',type=Path,required=True);p.add_argument('--fallback',type=Path,required=True);p.add_argument('--render16',type=Path,required=True);p.add_argument('--render32',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan16.read_text());fallback=json.loads((a.fallback/'manifest.json').read_text());req(plan['rows']==fallback['provider_denominator']==4100 and plan['upstream_holds']==fallback['upstream_holds']==127,'policy denominator differs');req(fallback['source_plan_sha256']==sha(a.plan16),'fallback plan differs');req(not a.output.exists(),'fresh output required');a.output.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent));selected=[];counts=collections.Counter();all_ids=set();fallback_by={x['shard']:x for x in fallback['entries']}
 try:
  for item in plan['entries']:
   if not item['rows']:continue
   shard=item['shard'];source=keyed(Path(plan['input_root'])/item['path']);r16_path=a.render16/f'shard-{shard:04d}.jsonl';t16=json.loads((a.render16/f'shard-{shard:04d}.terminal.json').read_text());req(t16.get('schema')=='sepalith.dat10.noop4100.parse_retry.render_shard_terminal.v1' and t16['status']=='complete' and t16['context_size']==16384 and t16['generation_reserve']==2048 and t16['input']['sha256']==item['sha256'] and t16['output']['sha256']==sha(r16_path),'16K terminal differs');r16=keyed(r16_path);req(set(source)==set(r16) and all(x.get('status')in ('supported','hold') for x in r16.values()),'16K closure differs');fb=fallback_by[shard];r32={}
   if fb['rows']:
    r32_path=a.render32/f'shard-{shard:04d}.jsonl';t=json.loads((a.render32/f'shard-{shard:04d}.terminal.json').read_text());req(t.get('schema')=='sepalith.dat10.semantic9534.render_shard_rich_terminal.v1' and t['status']=='complete' and t['context_size']==32768 and t['generation_reserve']==2048 and t['input']['sha256']==fb['sha256'] and t['output']['sha256']==sha(r32_path),'32K terminal differs');r32=keyed(r32_path);req(set(r32)==set(keyed(a.fallback/fb['path'])) and all(x.get('status')in ('supported','hold') for x in r32.values()),'32K closure differs')
   for rid in sorted(source):
    if r16[rid]['status']=='supported':value={**r16[rid],'policy_resolution':'selected_16k'};counts['selected_16k']+=1
    else:
     req(rid in r32,'missing 32K fallback result');value=r32[rid]
     if value['status']=='supported':value={**value,'policy_resolution':'fallback_supported_32k'};counts['fallback_supported_32k']+=1
     else:value={**value,'policy_resolution':'hold_after_32k'};counts['hold_after_32k']+=1
    selected.append(value);all_ids.add(rid)
  req(len(selected)==len(all_ids)==4100,'provider selection denominator differs');path=tmp/'selected-contexts.jsonl'
  with path.open('x') as stream:
   for x in selected:stream.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
   stream.flush();os.fsync(stream.fileno())
  manifest={'schema':'sepalith.dat10.noop4100.context-policy.v1','status':'complete_review_only','source_candidates':4227,'upstream_holds':127,'upstream_hold_classes':{'provenance':121,'mixed_eol_geometry':6},'provider_denominator':4100,'provider_supported':sum(x['status']=='supported' for x in selected),'provider_holds_after_32k':sum(x['status']!='supported' for x in selected),'exact_accounting':127+len(selected)==4227,'counts':dict(counts),'selected':{'path':path.name,'rows':4100,'sha256':sha(path),'bytes':path.stat().st_size},'render16_source_manifest_sha256':'114b0a410cf15c821ce83c8d569b16d831640f744f3e53d303d849929e503919','render32_source_manifest_sha256':'9725d968382c98fe5313985f5407fd6aa01a3e34063393b88c3bfb45193572e8','contexts':[16384,32768],'generation_reserve':2048,'fixed_noop_target_joined_after_selection':True,'fixed_training_target_protocol':'NO_EDIT','target_truncated':False,'target_or_gold_used_for_selection':False,'dedup_admission_required':True,'dedup_bindings_required':['accepted_current_20191','finalized_semantic10948','eventual_semantic9535'],'training_admission':False};(tmp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.rename(tmp,a.output);print(json.dumps(manifest,sort_keys=True))
 except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
