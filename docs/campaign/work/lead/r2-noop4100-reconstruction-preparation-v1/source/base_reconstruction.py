#!/usr/bin/env python3
"""Reconstruct full-document, target-free no-op prediction inputs from reviewed provenance."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,shutil,tempfile
from pathlib import Path
SUPPORTED='provenance_supported_candidate_root_review_required'
class RowValidationError(ValueError):pass
def req(v,m):
 if not v:raise ValueError(m)
def row_req(v,m):
 if not v:raise RowValidationError(m)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def rows(path):
 out={}
 with Path(path).open() as stream:
  for n,line in enumerate(stream,1):
   x=json.loads(line);rid=x.get('row_id')or x.get('row_ref',{}).get('row_id');req(isinstance(rid,str)and rid and rid not in out,f'row identity:{n}');out[rid]=x
 return out
def atomic_jsonl(path,values):
 tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
 with tmp.open('x')as f:
  for x in values:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 os.rename(tmp,path)
def eol_view(raw,method):
 if method=='exact_bytes':return raw
 if method=='uniform_crlf_to_lf':
  row_req(b'\r\n'in raw and b'\r'not in raw.replace(b'\r\n',b''),'mixed EOL unsupported');return raw.replace(b'\r\n',b'\n')
 raise RowValidationError('unsupported occurrence method')
def is_full_closure(coverage):return coverage.get('partial')is False and coverage.get('completed_shards')==41 and coverage.get('pending_shards')==[]and len(coverage.get('receipts',[]))==41
def duplicate_geometry_groups(predictions):
 groups=collections.defaultdict(list)
 for prediction in predictions:
  cursor=prediction['cursor'];key=(prediction['preedit_sha256'],cursor['line'],cursor['character'],prediction['path'])
  groups[key].append(prediction)
 out=[]
 for (preedit_sha256,line,character,path),members in sorted(groups.items()):
  if len(members)<2:continue
  out.append({'schema':'sepalith.dat10.sourcewalk-noop.duplicate-geometry.v1','geometry':{'preedit_sha256':preedit_sha256,'cursor':{'line':line,'character':character},'path':path},'count':len(members),'row_ids':sorted(x['row_id']for x in members),'workspace_roots':sorted(set(x['workspace_root']for x in members)),'absolute_document_paths':sorted(set(x['absolute_document_path']for x in members)),'disposition':'retain_all_until_provider_prompt_target_dedup'})
 return out
def validate_prediction_set(predictions):
 ids=[x['row_id']for x in predictions]
 req(len(ids)==len(set(ids)),'internal duplicate row identity')
 return duplicate_geometry_groups(predictions)
def recover(packet,provenance):
 rid=provenance['row_id'];result=packet.get('result',{});validation=packet.get('validation',{});geom=provenance.get('noop_geometry',{});row_req(packet.get('family')=='no_op'and packet.get('row_ref',{}).get('row_id')==rid,'packet join');row_req(result.get('operation')=='no_op'and result.get('target_body')==[]and result.get('status')=='converted','no-op target contract');row_req(geom.get('ok')is True and geom.get('geometry_supported')is True and geom.get('zero_width_cursor_geometry')is True and geom.get('unchanged_target')is True and geom.get('window_occurrences')==1,'provenance geometry');source=Path(validation['source_path']);before=source.stat();raw=source.read_bytes();after=source.stat();req((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)and sha(source)==provenance['source_path_sha256']==validation['source_sha256'],'source identity');view=eol_view(raw,provenance['source_window_occurrence_method']);window=result['selection_source']['text'].encode();row_req(hashlib.sha256(window).hexdigest()==geom['window_sha256']==result['selection_source']['content_sha256']and view.count(window)==1,'window identity');offset=view.index(window);base_line=view[:offset].count(b'\n');ctx=result['context'];rr=ctx['replacement_range'];row_req(rr['start']==rr['end']and rr['start']['character']==0 and ctx.get('region_old')==[],'zero-width context');global_line=base_line+rr['start']['line'];lines=view.decode('utf-8').split('\n');row_req(0<=global_line<len(lines),'global cursor');expected_eol='crlf'if b'\r\n'in raw else 'lf';row_req(ctx['document_eol']==expected_eol,'document EOL');preedit=raw.decode('utf-8');pre_sha=hashlib.sha256(raw).hexdigest();package_root=source.parent.parent;identity={'row_id':rid,'package_id':provenance['package_id'],'group_id':provenance['group_id'],'family':'no_op','source_path':str(source),'source_sha256':pre_sha,'normalized_relative_source_path':provenance['normalized_relative_source_path'],'window_sha256':geom['window_sha256'],'window_start_line':base_line,'local_cursor_line':rr['start']['line'],'global_cursor_line':global_line,'document_eol':expected_eol}
 prediction={'schema':'sepalith.dat10.sourcewalk-noop.prediction_input.v1','row_id':rid,'path':provenance['normalized_relative_source_path'],'preedit_text':preedit,'preedit_sha256':pre_sha,'cursor':{'line':global_line,'character':0},'document_eol':expected_eol,'absolute_document_path':str(source),'workspace_root':str(package_root),'expected_dependencies':[],'selection_target_or_gold_used':False}
 sidecar={'schema':'sepalith.dat10.sourcewalk-noop.training_sidecar.v1','row_id':rid,'identity':identity,'target_operation':'no_op','target_body_lines':[],'full_source_reapplication_sha256':pre_sha,'prediction_target_free':True}
 req(prediction['preedit_text'].encode()==raw and sidecar['full_source_reapplication_sha256']==hashlib.sha256(prediction['preedit_text'].encode()).hexdigest(),'full source replay');return prediction,sidecar

def candidate_output(item,packet,provenance):
 req(item.get('status')==provenance.get('status'),'candidate status differs from pinned ledger')
 rid=provenance['row_id'];shard=item['shard']
 if item['status']!=SUPPORTED:
  return None,None,{'row_id':rid,'shard':shard,'status':'hold','reason':'provenance_not_supported','source_reasons':item.get('reasons',[]),'silent_drop':False}
 try:
  prediction,sidecar=recover(packet,provenance);return prediction,sidecar,None
 except RowValidationError as error:
  return None,None,{'row_id':rid,'shard':shard,'status':'hold','reason':type(error).__name__+':'+str(error),'silent_drop':False}


def main():
 a=argparse.ArgumentParser();a.add_argument('--coverage',type=Path,required=True);a.add_argument('--coverage-sha256',required=True);a.add_argument('--candidate-ids',type=Path,required=True);a.add_argument('--candidate-ids-sha256',required=True);a.add_argument('--replay-root',type=Path,required=True);a.add_argument('--packet-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();req(sha(x.coverage)==x.coverage_sha256 and sha(x.candidate_ids)==x.candidate_ids_sha256,'coverage pins');coverage=json.loads(x.coverage.read_text());candidates=rows(x.candidate_ids);req(type(coverage.get('partial'))is bool and coverage['completed_shards']==len(coverage['receipts'])and len(candidates)==coverage['not_in_existing_pool_by_id'],'coverage closure');full_closure=is_full_closure(coverage);by_shard=collections.defaultdict(dict)
 for rid,item in candidates.items():by_shard[item['shard']][rid]=item
 parent=x.output.parent;parent.mkdir(parents=True,exist_ok=True);req(not x.output.exists(),'fresh output');tmp=Path(tempfile.mkdtemp(prefix='.'+x.output.name+'.',dir=parent));all_pred=[];all_side=[];holds=[];per=[]
 try:
  receipt_pins={z['shard']:z['sha256']for z in coverage['receipts']};req(set(by_shard)<=set(receipt_pins),'candidate shard outside receipts')
  for shard in sorted(by_shard):
   receipt=x.replay_root/f'shard-{shard:04d}/receipt.json';req(sha(receipt)==receipt_pins[shard],'receipt pin');rv=json.loads(receipt.read_text());req(rv['status']=='complete'and rv['shard']==shard,'receipt status');ledger_path=x.replay_root/f'shard-{shard:04d}/ledger.jsonl';packet_path=x.packet_root/f'shard-{shard:04d}/structured-materialization-v1/candidate-packets.jsonl';req(sha(ledger_path)==rv['outputs'][0]['sha256']and sha(packet_path)==rv['binding']['candidate_packets_sha256'],'shard artifact pin');ledger=rows(ledger_path);packets=rows(packet_path);pred=[];side=[];held=[]
   for rid,item in sorted(by_shard[shard].items()):
    req(rid in ledger and rid in packets,'shard row join')
    p,s,h=candidate_output(item,packets[rid],ledger[rid])
    if h is not None:held.append(h)
    else:pred.append(p);side.append(s)
   atomic_jsonl(tmp/f'shard-{shard:04d}.jsonl',pred);atomic_jsonl(tmp/f'shard-{shard:04d}.sidecar.jsonl',side);atomic_jsonl(tmp/f'shard-{shard:04d}.holds.jsonl',held);all_pred+=pred;all_side+=side;holds+=held;per.append({'shard':shard,'candidates':len(by_shard[shard]),'prediction_inputs':len(pred),'holds':len(held),'prediction_sha256':sha(tmp/f'shard-{shard:04d}.jsonl'),'sidecar_sha256':sha(tmp/f'shard-{shard:04d}.sidecar.jsonl'),'holds_sha256':sha(tmp/f'shard-{shard:04d}.holds.jsonl')})
  duplicate_groups=validate_prediction_set(all_pred);req(len(all_pred)+len(holds)==len(candidates),'exact candidate accounting');atomic_jsonl(tmp/'duplicate-geometries.jsonl',duplicate_groups);duplicate_rows=sum(x['count']for x in duplicate_groups);manifest={'schema':'sepalith.dat10.sourcewalk-noop-expansion-preparation.v3','status':'complete_review_only'if full_closure else'partial_review_only','coverage_scope_shards':coverage['completed_shards'],'pending_shards':coverage.get('pending_shards',[]),'candidate_rows':len(candidates),'prediction_inputs':len(all_pred),'holds':len(holds),'full_source_reapplication':True,'prediction_target_free':True,'training_admission':False,'full_41_shard_closure':full_closure,'duplicate_geometry':{'policy':'retain_all_until_provider_prompt_target_dedup','key_fields':['preedit_sha256','cursor.line','cursor.character','path'],'groups':len(duplicate_groups),'rows_in_groups':duplicate_rows,'excess_rows':duplicate_rows-len(duplicate_groups),'artifact':'duplicate-geometries.jsonl','sha256':sha(tmp/'duplicate-geometries.jsonl')},'provider_contract':{'source_manifest_sha256':'7546d5457c299edfbf69bc0681327f08bad5041e2a146fe60fd66278dd58c770','generation_reserve':2048,'context_caps':[16384,32768]},'shards':per};(tmp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.rename(tmp,x.output);print(json.dumps(manifest,sort_keys=True))
 except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
