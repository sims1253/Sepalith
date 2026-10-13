#!/usr/bin/env python3
"""Materialize fixed NO_EDIT rows after target-free provider selection."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import campaign_protocol as protocol
import audit_authoritative_stream as strict

TOKENIZER_SHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
PREPARATION_MANIFEST_SHA='38736227196ba22a5c55d1826411a20a1a50a8f3db277bb9c36ead2b696a9e65'
TARGET=f'{protocol.NO_EDIT}\n{protocol.TERMINAL}'

def req(v,m):
 if not v:raise ValueError(m)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def digest_text(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def read_rows(paths,key='row_id'):
 out={}
 for path in paths:
  with Path(path).open()as stream:
   for n,line in enumerate(stream,1):
    if not line.strip():continue
    row=json.loads(line);rid=row.get(key);req(isinstance(rid,str)and rid and rid not in out,f'row identity:{path}:{n}');out[rid]=row
 return out
class Tok:
 def __init__(self,path):
  req(sha(path)==TOKENIZER_SHA,'tokenizer pin');from tokenizers import Tokenizer;self.backend=Tokenizer.from_file(str(path));self.backend.encode_special_tokens=True
 def encode(self,text,*,add_special_tokens=False,split_special_tokens=True):req(add_special_tokens is False and split_special_tokens is True,'token policy');return self.backend.encode(text,add_special_tokens=False).ids

def utf16_offset(line,column):
 req(type(column)is int and column>=0,'UTF16 column')
 used=0
 for i,ch in enumerate(line):
  if used==column:return i
  used+=len(ch.encode('utf-16-le'))//2;req(used<=column,'UTF16 surrogate split')
 req(used==column,'UTF16 column outside line');return len(line)
def position_offset(text,position):
 lines=text.split('\n');line=position.line;req(0<=line<len(lines),'line outside source');return sum(len(x)+1 for x in lines[:line])+utf16_offset(lines[line],position.character)
def geometry_identity(context,prediction,sidecar):
 raw=prediction['preedit_text'].encode('utf-8');raw_sha=hashlib.sha256(raw).hexdigest();req(raw_sha==prediction['preedit_sha256']==sidecar['full_source_reapplication_sha256'],'raw no-op replay hash')
 req(context.replacement_range.content_sha256==raw_sha,'context/preedit binding')
 req(prediction['cursor']==context.replacement_range.start.to_dict(),'authoritative cursor/replacement start')
 text=prediction['preedit_text'];view=text.replace('\r\n','\n')if prediction['document_eol']=='crlf'else text
 if prediction['document_eol']=='crlf':req('\r\n'in text and '\r'not in text.replace('\r\n','')and '\n'not in text.replace('\r\n',''),'uniform CRLF')
 else:req('\r'not in text,'LF source')
 start=position_offset(view,context.replacement_range.start);end=position_offset(view,context.replacement_range.end);req(end>=start,'range reversed');selected=view[start:end];expected='\n'.join(context.region_old);req(selected==expected,'selected raw source geometry differs')
 # NO_EDIT applies no replacement, so exact raw bytes and digest remain fixed.
 req(hashlib.sha256(prediction['preedit_text'].encode()).hexdigest()==raw_sha,'full raw reapplication')
 identity=sidecar['identity'];window=identity.get('window_sha256');req(isinstance(window,str)and len(window)==64,'window identity')
 return digest_text(canonical({'source_sha256':raw_sha,'path':prediction['path'],'cursor':prediction['cursor'],'replacement_range':context.replacement_range.to_dict(),'region_old':list(context.region_old),'window_sha256':window}))

def validate_token_row(row,tok):
 req(row['family']=='no_op'and row['split']=='train'and row['target_operation']=='no_op'and row['target_body_text']==protocol.NO_EDIT and row['target_text']==TARGET,'NO_EDIT target contract')
 req(not strict.validate_token_row(row,full_text=True),'strict token row')
 req(tok.encode(row['prompt_text'])==row['input_ids'][1:row['target_start']],'prompt retokenization')
 req(tok.encode(row['target_text'])==row['input_ids'][row['target_start']:-1],'target retokenization')
 req(row['input_ids'][0]==0 and row['input_ids'][-1]==1 and row['input_ids'].count(0)==1 and row['input_ids'].count(1)==1,'terminal BOS/EOS')

def registry_rows(registry_path,registry_sha):
 req(sha(registry_path)==registry_sha,'dedup registry pin');registry=json.loads(Path(registry_path).read_text());req(registry.get('schema')=='sepalith.dat10.noop4100.dedup-registry.v1','registry schema')
 datasets=registry.get('datasets');req(isinstance(datasets,dict)and set(datasets)=={'accepted_current_20191','finalized_semantic10948','eventual_semantic9534'},'dedup dataset names')
 all_rows=[];pins=[]
 for name in ('accepted_current_20191','finalized_semantic10948','eventual_semantic9534'):
  entry=datasets[name];req(isinstance(entry,dict)and isinstance(entry.get('manifest'),str)and isinstance(entry.get('manifest_sha256'),str),'unbound dedup dataset:'+name);manifest=Path(entry['manifest']);req(sha(manifest)==entry['manifest_sha256'],'dedup manifest pin:'+name);m=json.loads(manifest.read_text());req(m.get('training_admission')is False or name=='accepted_current_20191','review-only dedup input')
  token_sets=entry.get('token_rows');req(isinstance(token_sets,list)and token_sets,'dedup token row list:'+name);seen=0
  if name=='accepted_current_20191':
   req(m.get('schema')=='sepalith.dat10.review-union-20191.v1'and m.get('rows')==20191 and sum(x['rows']for x in m.get('cohorts',[]))==20191,'current union manifest');expected=[{'path':str(Path(x['path']).resolve()),'sha256':x['sha256'],'rows':x['rows']}for x in m['cohorts']]
  else:
   req(m.get('schema')=='sepalith.dat10.semantic10948.provider_materialization.v1'and m.get('status')=='complete_review_only_root_admission_required','semantic materialization manifest');x=m['outputs']['candidate-tokenrows.jsonl'];expected=[{'path':str((manifest.parent/x['path']).resolve()),'sha256':x['sha256'],'rows':x['rows']}]
  observed=[{'path':str(Path(x['path']).resolve()),'sha256':x['sha256'],'rows':x['rows']}for x in token_sets];req(observed==expected,'dedup manifest/token rows binding:'+name)
  for item in token_sets:
   path=Path(item['path']);req(sha(path)==item['sha256'],'dedup token rows pin:'+name);rows=[]
   with path.open()as stream:
    for line in stream:
     if line.strip():rows.append(json.loads(line))
   req(len(rows)==item['rows'],'dedup token rows count:'+name);seen+=len(rows);all_rows.extend((name,x)for x in rows);pins.append({'dataset':name,'path':str(path),'sha256':item['sha256'],'rows':len(rows)})
  req(seen==entry['rows'],'dedup dataset count:'+name)
 return all_rows,pins

def build_indexes(existing):
 ids=set();pairs=collections.defaultdict(list);prompts=collections.defaultdict(dict);geometries=collections.defaultdict(dict)
 for cohort,row in existing:
  rid=row['id'];req(rid not in ids,'duplicate existing id');ids.add(rid);prompt=row['prompt_text'];target=row['target_text'];p=digest_text(prompt);t=digest_text(target);pairs[digest_text(prompt+'\0'+target)].append((cohort,rid));prompts[p][t]=(cohort,rid)
  g=row.get('source_cursor_geometry_sha256')
  if g is not None:req(isinstance(g,str)and len(g)==64,'existing geometry digest');geometries[g][t]=(cohort,rid)
 return ids,pairs,prompts,geometries

def dedup_decision(row,geometry,existing_indexes,candidate_indexes):
 existing_ids,pairs,prompts,geometries=existing_indexes;seen_pairs,seen_prompts,seen_geometry=candidate_indexes;rid=row['id'];prompt_sha=digest_text(row['prompt_text']);target_sha=digest_text(row['target_text']);pair=digest_text(row['prompt_text']+'\0'+row['target_text'])
 if rid in existing_ids:return 'excluded','duplicate_existing_id',[rid]
 if pair in pairs:return 'excluded','duplicate_existing_prompt_target',pairs[pair]
 conflicts=[v for t,v in prompts.get(prompt_sha,{}).items()if t!=target_sha]
 if conflicts:return 'hold','contradiction_existing_prompt',conflicts
 if geometry in geometries and target_sha in geometries[geometry]:return 'excluded','duplicate_existing_source_cursor_geometry_target',[geometries[geometry][target_sha]]
 conflicts=[v for t,v in geometries.get(geometry,{}).items()if t!=target_sha]
 if conflicts:return 'hold','contradiction_existing_geometry',conflicts
 if pair in seen_pairs:return 'excluded','duplicate_candidate_prompt_target',[seen_pairs[pair]]
 if prompt_sha in seen_prompts and seen_prompts[prompt_sha][0]!=target_sha:return 'hold','contradiction_candidate_prompt',[seen_prompts[prompt_sha][1]]
 if geometry in seen_geometry:
  if seen_geometry[geometry][0]==target_sha:return 'excluded','duplicate_candidate_source_cursor_geometry_target',[seen_geometry[geometry][1]]
  return 'hold','contradiction_candidate_geometry',[seen_geometry[geometry][1]]
 return 'candidate','novel_after_all_bound_dedup',[]

def main():
 p=argparse.ArgumentParser();p.add_argument('--selected',type=Path,required=True);p.add_argument('--selected-manifest',type=Path,required=True);p.add_argument('--selected-manifest-sha256',required=True);p.add_argument('--preparation-root',type=Path,required=True);p.add_argument('--dedup-registry',type=Path,required=True);p.add_argument('--dedup-registry-sha256',required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 req(sha(a.preparation_root/'manifest.json')==PREPARATION_MANIFEST_SHA,'reconstruction manifest pin');prep=json.loads((a.preparation_root/'manifest.json').read_text());req((prep['schema'],prep['status'],prep['prediction_inputs'],prep['holds'])==('sepalith.dat10.noop4100.reconstruction.v1','complete_review_only',4100,127),'reconstruction closure')
 req(sha(a.selected_manifest)==a.selected_manifest_sha256,'selected manifest pin');sm=json.loads(a.selected_manifest.read_text());req(sm.get('schema')=='sepalith.dat10.noop4100.context-policy.v1'and sm.get('status')=='complete_review_only'and sm.get('selected',{}).get('rows')==4100,'selected policy closure');req(a.selected.resolve()==(a.selected_manifest.parent/sm['selected']['path']).resolve()and sha(a.selected)==sm['selected']['sha256'],'selected path/hash')
 side=[];pred=[]
 for item in prep['shards']:
  shard=item['shard'];pp=a.preparation_root/f'shard-{shard:04d}.jsonl';sp=a.preparation_root/f'shard-{shard:04d}.sidecar.jsonl';req(sha(pp)==item['prediction_sha256']and sha(sp)==item['sidecar_sha256'],'reconstruction shard pin');pred.append(pp);side.append(sp)
 predictions=read_rows(pred);sidecars=read_rows(side);selected=read_rows([a.selected]);req(set(predictions)==set(sidecars)==set(selected)and len(selected)==4100,'provider/reconstruction join')
 existing,pins=registry_rows(a.dedup_registry,a.dedup_registry_sha256);indexes=build_indexes(existing);tok=Tok(a.tokenizer);kept=[];provenance=[];ledger=[];seen_pair={};seen_prompt={};seen_geometry={}
 for rid in sorted(selected):
  choice=selected[rid];prediction=predictions[rid];meta=sidecars[rid]
  if choice.get('status')!='supported':ledger.append({'row_id':rid,'status':'hold','reason':choice.get('reason','provider_hold'),'matches':[],'silent_drop':False});continue
  req(choice.get('selection_target_or_gold_used')is False,'provider selection target-free attestation')
  context=protocol.PromptContext.from_mapping(choice['selected_context']);prompt=protocol.render_prompt(context);req(digest_text(prompt)==choice['prompt_sha256']and len(tok.encode(prompt))==choice['prompt_tokens'],'provider prompt parity');geometry=geometry_identity(context,prediction,meta)
  row=protocol.build_training_row(context,operation='no_op',region_new=list(context.region_old),tokenizer=tok,row_id=rid,family='no_op',package_id=meta['identity']['package_id'],split='train');validate_token_row(row,tok)
  status,reason,matches=dedup_decision(row,geometry,indexes,(seen_pair,seen_prompt,seen_geometry));ledger.append({'row_id':rid,'status':status,'reason':reason,'matches':matches,'silent_drop':False})
  if status!='candidate':continue
  prompt_sha=digest_text(row['prompt_text']);target_sha=digest_text(row['target_text']);pair=digest_text(row['prompt_text']+'\0'+row['target_text']);seen_pair[pair]=rid;seen_prompt[prompt_sha]=(target_sha,rid);seen_geometry[geometry]=(target_sha,rid);kept.append(row);provenance.append({'row_id':rid,'source_identity':meta['identity'],'preedit_sha256':prediction['preedit_sha256'],'authoritative_cursor':prediction['cursor'],'source_cursor_geometry_sha256':geometry,'prompt_sha256':prompt_sha,'target_sha256':target_sha,'target_free_selection':True,'full_raw_noop_reapplication':True})
 req(len(ledger)==4100,'decision accounting');a.output.parent.mkdir(parents=True,exist_ok=True);req(not a.output.exists(),'fresh output');tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent))
 try:
  outputs={}
  for name,values in [('candidate-tokenrows.jsonl',kept),('candidate-provenance.jsonl',provenance),('decision-ledger.jsonl',ledger)]:
   path=tmp/name
   with path.open('x')as f:
    for x in values:f.write(canonical(x)+'\n')
    f.flush();os.fsync(f.fileno())
   outputs[name]={'rows':len(values),'bytes':path.stat().st_size,'sha256':sha(path)}
  counts=collections.Counter(x['status']+':'+x['reason']for x in ledger);manifest={'schema':'sepalith.dat10.noop4100.materialization.v1','status':'complete_review_only_root_admission_required','provider_denominator':4100,'candidate_rows':len(kept),'decision_rows':len(ledger),'target_protocol':'NO_EDIT','target_only_dedup_forbidden':True,'dedup_dimensions':['row_id','exact prompt+target','same prompt different target contradiction','source+authoritative cursor+matched geometry+target'],'dedup_input_pins':pins,'decision_counts':dict(counts),'outputs':outputs,'full_raw_noop_reapplication':True,'target_truncated':False,'training_admission':False};(tmp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.rename(tmp,a.output);print(json.dumps(manifest,sort_keys=True))
 except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
