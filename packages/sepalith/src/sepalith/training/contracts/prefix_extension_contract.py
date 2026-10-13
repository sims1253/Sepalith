#!/usr/bin/env python3
"""Lossless cache/schedule prefix contract for an in-place CPT corpus extension."""
from __future__ import annotations
import hashlib,json,os,sqlite3,struct
from pathlib import Path

def require(ok,message):
 if not ok:raise ValueError(message)
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pinned(record,label):
 p=Path(record.get('path',''));require(p.is_file() and sha256(p)==record.get('sha256'),label+' differs');return p
def cache(recipe):
 record=recipe['cohort']['streaming_cache'];root=Path(record['path']);mp=root/'manifest.json'
 require(mp.is_file() and sha256(mp)==record['manifest_sha256'],'streaming cache manifest differs')
 m=json.loads(mp.read_text());require(m.get('schema')=='sepalith.sft11.cpt-streaming-cache.v1' and m.get('status')=='complete','streaming cache is not complete')
 return root,m
def schedule(recipe):
 p=pinned(recipe['cohort']['draw_schedule'],'draw schedule');v=json.loads(p.read_text());ids=v.get('row_ids',[])
 require(v.get('effective_batch')==16 and v.get('coverage',{}).get('draws')==len(ids),'draw schedule geometry differs')
 u=v['coverage'].get('unique_rows');require(type(u)is int and len(set(ids[:u]))==u,'unique draw prefix differs')
 tail=ids[u:];require(tail==v.get('replay_row_ids',[]) and len(tail)<=15 and all(x in set(ids[:u]) for x in tail),'named replay tail differs')
 require(len(ids)%16==0 and v['coverage'].get('updates')*16==len(ids),'draw schedule update count differs')
 return v
def verify_files(root,manifest,bundle_id=None):
 for name,record in manifest.get('files',{}).items():
  p=root/name
  if bundle_id:
   from sepalith.training.checkpoint.native_attestation import require_attested
   require_attested(p,record.get('sha256'),record.get('bytes'),bundle_id)
  else:require(p.is_file() and p.stat().st_size==record.get('bytes') and sha256(p)==record.get('sha256'),'cache payload differs:'+name)
 c=manifest['counts'];require((root/'input_ids.i32le').stat().st_size==c['input_tokens']*4,'input token bytes differ');require((root/'labels.i32le').stat().st_size==c['input_tokens']*4,'label token bytes differ');require((root/'draw_ordinals.i64le').stat().st_size==c['draws']*8,'draw ordinal bytes differ')
 db=sqlite3.connect(f'file:{root/"index.sqlite3"}?mode=ro',uri=True)
 try:
  rows=db.execute('select count(*),coalesce(sum(length),0),coalesce(max(ordinal),-1) from rows').fetchone();docs=db.execute('select count(*),count(distinct package) from documents').fetchone()
  require(rows==(c['rows'],c['input_tokens'],c['rows']-1),'cache row index/counts differ');require(docs==(c['documents'],c['packages']),'cache document/package counts differ')
 finally:db.close()
def verify_source_conservation(root,manifest,rows_alias=None):
 source=manifest.get('source',{}).get('rows',{});path=Path(rows_alias or source.get('path',''))
 require(path.is_file(),'cache source rows missing');raw_hash=hashlib.sha256();ids_hash=hashlib.sha256();labels_hash=hashlib.sha256();rows=input_tokens=payload_tokens=loss_tokens=0;docs=0;last_doc=None;packages=set()
 db=sqlite3.connect(f'file:{root/"index.sqlite3"}?mode=ro',uri=True);indexed=db.execute('select row_id,length,document_id,package,source_sha256,row_sha256 from rows order by ordinal')
 with path.open('rb') as f:
  before=os.fstat(f.fileno())
  for line in f:
   raw_hash.update(line);require(line.endswith(b'\n'),'cache source JSONL final line differs');row=json.loads(line);record=indexed.fetchone();require(record is not None,'cache index ends before source rows')
   ids,labels=row.get('input_ids'),row.get('labels');require(isinstance(ids,list) and isinstance(labels,list) and len(ids)==len(labels)>0,'source token/label geometry differs');require(len(ids)<=manifest.get('max_sequence_tokens'),'source row exceeds cache context without truncation');require(ids[0]==0 and ids[-1]==1 and labels[0]==-100,'source BOS/EOS/label boundary differs');require(row.get('attention_mask',[1]*len(ids))==[1]*len(ids),'source attention mask differs');require(sum(x!=-100 for x in labels)==row['supervised_tokens'],'source supervised count differs')
   require(record==(row['row_id'],len(ids),row['document_id'],row['package'],row['source_sha256'],hashlib.sha256(line).hexdigest()),'source row/index provenance differs')
   ids_hash.update(struct.pack(f'<{len(ids)}i',*ids));labels_hash.update(struct.pack(f'<{len(labels)}i',*labels));rows+=1;input_tokens+=len(ids);payload_tokens+=len(ids)-2-row['overlap_context_tokens'];loss_tokens+=row['supervised_tokens'];packages.add(row['package'])
   if row['document_id']!=last_doc:docs+=1;last_doc=row['document_id']
  after=os.fstat(f.fileno())
 require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'cache source rows changed during validation');require(indexed.fetchone() is None,'cache index has rows absent from source');db.close()
 counts=manifest['counts'];require(raw_hash.hexdigest()==source.get('sha256'),'cache source streamed bytes differ')
 require(ids_hash.hexdigest()==manifest['files']['input_ids.i32le']['sha256'] and labels_hash.hexdigest()==manifest['files']['labels.i32le']['sha256'],'cache arrays are not exact source tokens/labels')
 require((rows,docs,len(packages),input_tokens,payload_tokens,loss_tokens)==tuple(counts[x] for x in ('rows','documents','packages','input_tokens','payload_tokens','loss_tokens')),'cache source conservation counts differ')
def prefix_hash(path,size):
 h=hashlib.sha256();left=size
 with Path(path).open('rb') as f:
  while left:
   b=f.read(min(8<<20,left));require(b,'destination binary prefix is short');h.update(b);left-=len(b)
 return h.hexdigest()
def compare_physical_prefix(old_root,new_root,old_manifest):
 for name in ('input_ids.i32le','labels.i32le'):
  r=old_manifest['files'][name];require(prefix_hash(new_root/name,r['bytes'])==r['sha256'],'old binary prefix changed:'+name)
 db=sqlite3.connect(f'file:{new_root/"index.sqlite3"}?mode=ro',uri=True)
 try:
  db.execute('attach database ? as old',(str(old_root/'index.sqlite3'),))
  mismatch=db.execute('select count(*) from old.rows o left join main.rows n on n.ordinal=o.ordinal where n.ordinal is null or (o.row_id,o.token_offset,o.length,o.document_id,o.package,o.source_sha256,o.row_sha256)!=(n.row_id,n.token_offset,n.length,n.document_id,n.package,n.source_sha256,n.row_sha256)').fetchone()[0]
  require(mismatch==0,'old indexed row/provenance prefix changed')
  dm=db.execute('select count(*) from old.documents o left join main.documents n on n.document_id=o.document_id where n.document_id is null or (o.package,o.source_sha256,o.chunks,o.tokens,o.token_stream_sha256)!=(n.package,n.source_sha256,n.chunks,n.tokens,n.token_stream_sha256)').fetchone()[0]
  require(dm==0,'old document provenance changed')
 finally:db.close()
def verify_draw_ordinals(root,sched):
 db=sqlite3.connect(f'file:{root/"index.sqlite3"}?mode=ro',uri=True);mapping={r[0]:r[1] for r in db.execute('select row_id,ordinal from rows')};db.close()
 raw=(root/'draw_ordinals.i64le').read_bytes();actual=list(struct.unpack(f'<{len(raw)//8}Q',raw))
 require(len(actual)==len(sched['row_ids']),'draw ordinal count differs')
 require(all(rid in mapping and actual[i]==mapping[rid] for i,rid in enumerate(sched['row_ids'])),'draw ordinal/schedule join differs')
def verify_consumed_prefix(source_recipe,destination_recipe,source_state):
 t=destination_recipe['transition'];require(t.get('destination_sampler')=='preserve_verified_prefix','destination sampler does not preserve a verified prefix')
 offset=t.get('global_optimizer_step_offset');step=t.get('source_global_step');cursor=t.get('source_cursor')
 require(type(offset)is int and type(step)is int and type(cursor)is int and cursor>0 and cursor%16==0,'source offset/step/cursor fields differ')
 require(step==offset+cursor//16,'source global step does not match offset and cursor')
 sampler=source_state.get('sampler',{});prior_offset=source_recipe['transition']['global_optimizer_step_offset']
 require(prior_offset==offset and sampler.get('global_optimizer_step_offset')==offset,'source offset differs from checkpoint state')
 require(sampler.get('global_step')==step and sampler.get('stage_cursor')==sampler.get('cursor')==cursor,'source cursor/global step differs from checkpoint state')
 require(sampler.get('draw_schedule_sha256')==source_recipe['cohort']['draw_schedule']['sha256'] and sampler.get('ignore_data_skip')is True,'source sampler schedule/state differs')
 old_s,new_s=schedule(source_recipe),schedule(destination_recipe);old_u=old_s['coverage']['unique_rows'];new_u=new_s['coverage']['unique_rows']
 require(cursor<=old_u and new_u>old_u,'extension must retain consumed unique rows and add new unique rows')
 require(new_s['row_ids'][:old_u]==old_s['row_ids'][:old_u],'old unique draw prefix changed')
 require(new_s['row_ids'][:cursor]==old_s['row_ids'][:cursor],'consumed draw prefix changed')
 require(len(set(new_s['row_ids'][:new_u]))==new_u,'destination unique coverage differs')
 old_root,old_m=cache(source_recipe);new_root,new_m=cache(destination_recipe);bundle=destination_recipe.get('storage_relocation',{}).get('bundle_id')
 verify_files(old_root,old_m);verify_files(new_root,new_m,bundle);verify_source_conservation(new_root,new_m,destination_recipe['cohort']['rows']['path'])
 require(old_m['counts']['rows']==old_u and new_m['counts']['rows']==new_u,'cache unique row count differs from schedule')
 require(new_m['counts']['rows']>old_m['counts']['rows'] and new_m['counts']['documents']>=old_m['counts']['documents'],'destination cache is not an extension')
 compare_physical_prefix(old_root,new_root,old_m);verify_draw_ordinals(new_root,new_s)
 require(destination_recipe['runtime']['mandatory_stop_step']>step and destination_recipe['runtime']['max_steps']>step,'future stop must follow source checkpoint')
 return cursor
