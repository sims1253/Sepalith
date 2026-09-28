#!/usr/bin/env python3
"""Recover canonical geometry only from target-free authoritative contexts."""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path
SCHEMA='sepalith.sft11.expanded-union-geometry-recovery.v1';GEOMETRY='sepalith.source-cursor-geometry.v1'
class RecoveryError(RuntimeError):pass
def req(v,m):
 if not v:raise RecoveryError(m)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest_text(x):return hashlib.sha256(x.encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def pos(x,label):
 req(isinstance(x,dict) and set(x)=={'line','character'},label);req(type(x['line'])is int and x['line']>=0 and type(x['character'])is int and x['character']>=0,label);return {'line':x['line'],'character':x['character']}
def recover(record,context_key,expected_row_id):
 req(record.get('row_id')==expected_row_id,'row_id_join')
 if 'selection_target_or_gold_used' in record:req(record['selection_target_or_gold_used'] is False,'target_used_for_selection')
 if 'context_has_target_or_reward_keys' in record:req(record['context_has_target_or_reward_keys'] is False,'context_has_target')
 c=record.get(context_key);req(isinstance(c,dict),context_key);rr=c.get('replacement_range');req(isinstance(rr,dict),'replacement_range');start=pos(rr.get('start'),'start');end=pos(rr.get('end'),'end');req((end['line'],end['character'])>=(start['line'],start['character']),'range_reversed')
 content=rr.get('content_sha256');req(isinstance(content,str) and len(content)==64,'content_sha256');path=c.get('path');req(isinstance(path,str) and path,'source_path')
 # Cursor for application is the authoritative LSP replacement start. The
 # auxiliary context.cursor describes a position inside region_old and is not
 # substituted for a document-global cursor.
 eol={'lf':'\n','crlf':'\r\n'}.get(c.get('document_eol'));req(eol is not None,'document_eol')
 parts=[c.get('prefix'),c.get('region_old'),c.get('suffix_lines')];req(all(isinstance(x,list) and all(isinstance(y,str) for y in x) for x in parts),'context_lines')
 raw=eol.join(sum(parts,[])).encode();observed=hashlib.sha256(raw).hexdigest();availability=record.get('selection_geometry',{}).get('availability')
 full_claim=availability=='full_snapshot';req(not full_claim or observed==content,'claimed_full_snapshot_hash_mismatch')
 window=digest_text(canonical({'document_eol':c['document_eol'],'prefix':parts[0],'region_old':parts[1],'suffix_lines':parts[2]}))
 geometry={'schema':GEOMETRY,'source_path':path,'source_sha256':content,'preedit_sha256':content,'cursor':start,'replacement_range':{'start':start,'end':end},'window_sha256':window}
 return {'row_id':expected_row_id,'status':'recovered' if full_claim else 'recovered_context_geometry_buffer_unavailable','source_cursor_geometry':geometry,'source_cursor_geometry_sha256':digest_text(canonical(geometry)),'full_buffer_hash_verified':full_claim,'target_or_gold_used':False}
def stream_map(sidecar,expected_pin,ids_path,ids_pin,context_key,output):
 req(sha(sidecar)==expected_pin,'sidecar_sha');req(sha(ids_path)==ids_pin,'candidate_ids_sha');wanted=[]
 with Path(ids_path).open() as f:
  for line in f:
   if line.strip():wanted.append((json.loads(line).get('row_id') or json.loads(line).get('id')))
 req(len(wanted)==len(set(wanted)),'candidate_ids_duplicate');wanted=set(wanted);seen=set();rows=[]
 with Path(sidecar).open() as f:
  for line in f:
   if not line.strip():continue
   record=json.loads(line);rid=record.get('row_id')
   if rid in wanted:
    req(rid not in seen,'sidecar_duplicate_id');seen.add(rid)
    try:rows.append(recover(record,context_key,rid))
    except RecoveryError as e:rows.append({'row_id':rid,'status':'unresolved','reason':str(e),'silent_drop':False,'target_or_gold_used':False})
 missing=sorted(wanted-seen);rows.extend({'row_id':rid,'status':'unresolved','reason':'sidecar_row_missing','silent_drop':False,'target_or_gold_used':False} for rid in missing);rows.sort(key=lambda x:x['row_id'])
 output=Path(output);req(not output.exists(),'fresh_output');output.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent));ledger=tmp/'geometry.jsonl'
 with ledger.open('x') as f:
  for x in rows:f.write(canonical(x)+'\n')
 manifest={'schema':SCHEMA,'status':'review_only','input_ids':len(wanted),'recovered':sum(x['status'].startswith('recovered') for x in rows),'unresolved':sum(x['status']=='unresolved' for x in rows),'outputs':{'geometry.jsonl':{'path':'geometry.jsonl','rows':len(rows),'sha256':sha(ledger)}},'training_admission':False};(tmp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');os.rename(tmp,output);return manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--sidecar',type=Path,required=True);p.add_argument('--sidecar-sha256',required=True);p.add_argument('--candidate-ids',type=Path,required=True);p.add_argument('--candidate-ids-sha256',required=True);p.add_argument('--context-key',choices=('context','selected_context'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(canonical(stream_map(a.sidecar,a.sidecar_sha256,a.candidate_ids,a.candidate_ids_sha256,a.context_key,a.output)))
if __name__=='__main__':main()
