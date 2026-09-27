#!/usr/bin/env python3
"""Recover geometry while keeping immutable source and current pre-edit identities distinct."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
from urllib.parse import unquote,urlparse
SCHEMA='sepalith.sft11.expanded-union-geometry-recovery.v2';GEOMETRY='sepalith.source-cursor-geometry.v1'
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
 req(isinstance(x,dict) and set(x)=={'line','character'},label);req(type(x['line']) is int and x['line']>=0 and type(x['character']) is int and x['character']>=0,label);return {'line':x['line'],'character':x['character']}
def utf16_len(s):return len(s.encode('utf-16-le'))//2
def valid_utf16_position(lines,p,label,allow_eof=False):
 if allow_eof and p['line']==len(lines):req(p['character']==0,label+':eof_character');return
 req(p['line']<len(lines),label+':line_outside_buffer');line=lines[p['line']];req(p['character']<=utf16_len(line),label+':column_outside_buffer')
 # Round-trip the requested UTF-16 boundary; a position inside a surrogate pair is invalid.
 units=line.encode('utf-16-le')
 try:units[:2*p['character']].decode('utf-16-le')
 except UnicodeDecodeError as exc:raise RecoveryError(label+':surrogate_boundary') from exc
def uri_path(uri):
 req(isinstance(uri,str) and uri.startswith('file://'),'replacement_uri');return unquote(urlparse(uri).path)
def identity(record,provenance,layout,rr):
 original=provenance.get('source_identity');req(isinstance(original,dict),'source_identity')
 preedit=provenance.get('preedit_sha256',rr.get('content_sha256'));req(preedit==rr.get('content_sha256'),'preedit_range_hash_join')
 if layout=='direct':
  source_path=original.get('source_path');source_sha=original.get('source_sha256');req(isinstance(source_path,str) and source_path.startswith('/'),'direct_source_path');req(isinstance(source_sha,str) and len(source_sha)==64,'direct_source_sha');req(Path(uri_path(rr['uri']))==Path(source_path),'direct_uri_source_path_join');basis={'layout':'direct','source_path_field':'source_identity.source_path','source_sha256_field':'source_identity.source_sha256','preedit_sha256_field':'preedit_sha256'}
 else:
  nested=original.get('source_provenance');req(isinstance(nested,dict),'nested_source_provenance');source_path=nested.get('source_snapshot_path');source_sha=nested.get('source_snapshot_sha256');req(isinstance(source_path,str) and source_path.startswith('/'),'nested_source_path');req(isinstance(source_sha,str) and len(source_sha)==64,'nested_source_sha');current=nested.get('after_snapshot_sha256');req(current==preedit,'nested_after_snapshot_preedit_join');logical=uri_path(rr['uri']);req(source_path.endswith(logical.removeprefix('/sepalith/')),'nested_uri_source_suffix_join');basis={'layout':'nested_original15006','source_path_field':'source_identity.source_provenance.source_snapshot_path','source_sha256_field':'source_identity.source_provenance.source_snapshot_sha256','preedit_sha256_field':'source_identity.source_provenance.after_snapshot_sha256'}
 return original,source_path,source_sha,preedit,basis
def recover(record,provenance,context_key,layout,expected_row_id):
 req(record.get('row_id')==expected_row_id,'row_id_join');req((provenance.get('row_id') or provenance.get('id'))==expected_row_id,'provenance_row_id_join')
 if 'selection_target_or_gold_used' in record:req(record['selection_target_or_gold_used'] is False,'target_used_for_selection')
 if 'context_has_target_or_reward_keys' in record:req(record['context_has_target_or_reward_keys'] is False,'context_has_target')
 c=record.get(context_key);req(isinstance(c,dict),context_key);rr=c.get('replacement_range');req(isinstance(rr,dict),'replacement_range');start=pos(rr.get('start'),'start');end=pos(rr.get('end'),'end');req((end['line'],end['character'])>=(start['line'],start['character']),'range_reversed')
 content=rr.get('content_sha256');req(isinstance(content,str) and len(content)==64,'content_sha256');original,source_path,source_sha,preedit,basis=identity(record,provenance,layout,rr)
 eol={'lf':'\n','crlf':'\r\n'}.get(c.get('document_eol'));req(eol is not None,'document_eol');parts=[c.get('prefix'),c.get('region_old'),c.get('suffix_lines')];req(all(isinstance(x,list) and all(isinstance(y,str) for y in x) for x in parts),'context_lines')
 lines=sum(parts,[]);availability=record.get('selection_geometry',{}).get('availability');full_claim=availability=='full_snapshot';observed=digest_text(eol.join(lines));req(not full_claim or observed==preedit,'claimed_full_snapshot_hash_mismatch')
 # A full snapshot proves document-global positions. A partial full-document context
 # still proves a zero-width boundary at prefix/suffix or its retained region only.
 if full_claim:
  valid_utf16_position(lines,start,'start',True);valid_utf16_position(lines,end,'end',True)
 else:
  req(start==end,'partial_nonempty_range_unverified');req(start['line']==len(parts[0]),'partial_cursor_prefix_boundary');
  if parts[2]:req(start['character']<=utf16_len(parts[2][0]),'partial_cursor_column')
 window=digest_text(canonical({'document_eol':c['document_eol'],'prefix':parts[0],'region_old':parts[1],'suffix_lines':parts[2]}))
 geometry={'schema':GEOMETRY,'source_path':source_path,'source_sha256':source_sha,'preedit_sha256':preedit,'cursor':start,'replacement_range':{'start':start,'end':end},'window_sha256':window}
 return {'row_id':expected_row_id,'status':'recovered' if full_claim else 'recovered_context_geometry_buffer_unavailable','source_identity':original,'source_cursor_geometry':geometry,'source_cursor_geometry_sha256':digest_text(canonical(geometry)),'geometry_identity_basis':basis,'full_buffer_hash_verified':full_claim,'target_or_gold_used':False}
def rows(path):
 with Path(path).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def stream_map(sidecar,sidepin,provenance,provpin,ids_path,idspin,context_key,layout,output):
 req(sha(sidecar)==sidepin,'sidecar_sha');req(sha(provenance)==provpin,'provenance_sha');req(sha(ids_path)==idspin,'candidate_ids_sha');wanted=[]
 for x in rows(ids_path):wanted.append(x.get('row_id') or x.get('id'))
 req(all(isinstance(x,str) and x for x in wanted) and len(wanted)==len(set(wanted)),'candidate_ids');wanted_set=set(wanted)
 prov={};
 for x in rows(provenance):
  rid=x.get('row_id') or x.get('id')
  if rid in wanted_set:req(rid not in prov,'provenance_duplicate_id');prov[rid]=x
 seen=set();out=[]
 for record in rows(sidecar):
  rid=record.get('row_id')
  if rid in wanted_set:
   req(rid not in seen,'sidecar_duplicate_id');seen.add(rid)
   try:req(rid in prov,'provenance_row_missing');out.append(recover(record,prov[rid],context_key,layout,rid))
   except RecoveryError as e:out.append({'row_id':rid,'status':'unresolved','reason':str(e),'silent_drop':False,'target_or_gold_used':False})
 for rid in sorted(wanted_set-seen):out.append({'row_id':rid,'status':'unresolved','reason':'sidecar_row_missing','silent_drop':False,'target_or_gold_used':False})
 out.sort(key=lambda x:x['row_id']);output=Path(output);req(not output.exists(),'fresh_output');output.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent));ledger=tmp/'geometry.jsonl'
 with ledger.open('x') as f:
  for x in out:f.write(canonical(x)+'\n')
 manifest={'schema':SCHEMA,'status':'review_only','input_ids':len(wanted),'recovered':sum(x['status'].startswith('recovered') for x in out),'unresolved':sum(x['status']=='unresolved' for x in out),'outputs':{'geometry.jsonl':{'path':'geometry.jsonl','rows':len(out),'sha256':sha(ledger)}},'training_admission':False};(tmp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');os.rename(tmp,output);return manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--sidecar',type=Path,required=True);p.add_argument('--sidecar-sha256',required=True);p.add_argument('--provenance',type=Path,required=True);p.add_argument('--provenance-sha256',required=True);p.add_argument('--candidate-ids',type=Path,required=True);p.add_argument('--candidate-ids-sha256',required=True);p.add_argument('--context-key',choices=('context','selected_context'),required=True);p.add_argument('--identity-layout',choices=('direct','nested_original15006'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(canonical(stream_map(a.sidecar,a.sidecar_sha256,a.provenance,a.provenance_sha256,a.candidate_ids,a.candidate_ids_sha256,a.context_key,a.identity_layout,a.output)))
if __name__=='__main__':main()
