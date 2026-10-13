#!/usr/bin/env python3
"""Build and read a bounded-memory immutable CPT token cache."""
from __future__ import annotations
import argparse, array, hashlib, json, mmap, os, shutil, sqlite3, struct, sys, tempfile
from pathlib import Path
from campaign_cpt_data import validate_materialized_row

SCHEMA="sepalith.sft11.cpt-streaming-cache.v1"

def require(v,m):
 if not v: raise ValueError(m)
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def canonical_sha(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def fsync_dir(path):
 fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try: os.fsync(fd)
 finally: os.close(fd)
def i32_bytes(values):
 a=array.array('i',values);require(a.itemsize==4,'platform int is not32-bit')
 if sys.byteorder!='little':a.byteswap()
 return a.tobytes()

def _doc_start(row):
 return {'key':(row['package'],row['document_id'],row['source_sha256']),'next_chunk':0,'next_start':0,'declared':row.get('document_token_count'),'declared_hash':row.get('token_stream_sha256'),'tokens':0,'chunks':0,'terminal':False,'hash':hashlib.sha256(b'['),'first':True,'tail':[]}
def _doc_add(state,row):
 key=(row['package'],row['document_id'],row['source_sha256']);require(key==state['key'],'document rows must be contiguous')
 require(not state['terminal'],'row follows terminal document chunk');require(row['chunk_index']==state['next_chunk'],'chunk indexes must be contiguous from zero');require(row['source_token_start']==state['next_start'],'document source span has a gap or overlap')
 require(row.get('token_stream_sha256')==state['declared_hash'],'declared token-stream hash changes across chunks')
 overlap=row['overlap_context_tokens'];context=row['input_ids'][1:1+overlap]
 require(overlap==0 or (overlap<=len(state['tail']) and context==state['tail'][-overlap:]),'continuation carry token differs from preceding owned tokens')
 owned=row['input_ids'][1+row['overlap_context_tokens']:-1]
 for token in owned:
  if not state['first']:state['hash'].update(b',')
  state['hash'].update(str(token).encode('ascii'));state['first']=False
 state['tail']=(state['tail']+owned)[-32768:];state['tokens']+=len(owned);state['chunks']+=1;state['next_chunk']+=1;state['next_start']=row['source_token_end'];state['terminal']=row['is_document_end']
 if state['declared'] is not None:require(row.get('document_token_count')==state['declared'],'document token count changes across chunks')
def _doc_finish(db,state):
 require(state['terminal'],'document has no terminal chunk');require(state['next_start']==state['tokens'],'document owned-token count differs from spans')
 if state['declared'] is not None:require(state['tokens']==state['declared'],'document terminal span is incomplete')
 state['hash'].update(b']');computed=state['hash'].hexdigest();require(state['declared_hash'] is None or state['declared_hash']==computed,'declared token-stream hash differs from reconstructed document');pkg,doc,source_sha=state['key']
 try:db.execute('insert into documents(document_id,package,source_sha256,chunks,tokens,token_stream_sha256) values(?,?,?,?,?,?)',(doc,pkg,source_sha,state['chunks'],state['tokens'],computed))
 except sqlite3.IntegrityError as exc:raise ValueError('document reappears after its contiguous group') from exc

def build(rows_path,schedule_path,output,max_sequence_tokens,expected_rows_sha256,expected_schedule_sha256):
 rows_path,schedule_path,output=map(Path,(rows_path,schedule_path,output));require(str(output.resolve()).startswith('/mnt/e/'),'bulk cache must be on E');require(not output.exists(),'cache output must be fresh')
 stage=output.with_name('.'+output.name+f'.building-{os.getpid()}');require(not stage.exists(),'staging path exists');stage.mkdir(parents=True)
 db=sqlite3.connect(stage/'index.sqlite3');db.execute('pragma journal_mode=DELETE');db.execute('pragma synchronous=FULL');db.executescript('create table rows(ordinal integer primary key,row_id text not null unique,token_offset integer not null,length integer not null,document_id text not null,package text not null,source_sha256 text not null,row_sha256 text not null);create table documents(document_id text primary key,package text not null,source_sha256 text not null unique,chunks integer not null,tokens integer not null,token_stream_sha256 text not null);')
 counts={'rows':0,'documents':0,'input_tokens':0,'payload_tokens':0,'loss_tokens':0};packages=set();token_offset=0;doc=None
 try:
  with rows_path.open('rb') as source,(stage/'input_ids.i32le').open('wb') as ids_out,(stage/'labels.i32le').open('wb') as labels_out:
   rows_stat_before=os.fstat(source.fileno());rows_digest=hashlib.sha256()
   for line_number,line in enumerate(source,1):
    rows_digest.update(line)
    require(line.endswith(b'\n'),'source JSONL final line lacks newline')
    try:raw=json.loads(line)
    except Exception as exc:raise ValueError(f'invalid row JSON line {line_number}') from exc
    require(raw.get('cpt_partition')=='cpt_train','cache accepts TRAIN rows only')
    validate_materialized_row(raw,max_sequence_tokens=max_sequence_tokens)
    key=(raw['package'],raw['document_id'],raw['source_sha256'])
    if doc is None:doc=_doc_start(raw)
    elif key!=doc['key']:_doc_finish(db,doc);counts['documents']+=1;doc=_doc_start(raw)
    _doc_add(doc,raw)
    ids_out.write(i32_bytes(raw['input_ids']));labels_out.write(i32_bytes(raw['labels']))
    try:db.execute('insert into rows values(?,?,?,?,?,?,?,?)',(counts['rows'],raw['row_id'],token_offset,len(raw['input_ids']),raw['document_id'],raw['package'],raw['source_sha256'],hashlib.sha256(line).hexdigest()))
    except sqlite3.IntegrityError as exc:raise ValueError('duplicate row ID') from exc
    counts['rows']+=1;counts['input_tokens']+=len(raw['input_ids']);counts['payload_tokens']+=len(raw['input_ids'])-2-raw['overlap_context_tokens'];counts['loss_tokens']+=raw['supervised_tokens'];packages.add(raw['package']);token_offset+=len(raw['input_ids'])
   require(doc is not None,'source rows empty');_doc_finish(db,doc);counts['documents']+=1
   rows_stat_after=os.fstat(source.fileno());require((rows_stat_before.st_dev,rows_stat_before.st_ino,rows_stat_before.st_size,rows_stat_before.st_mtime_ns)==(rows_stat_after.st_dev,rows_stat_after.st_ino,rows_stat_after.st_size,rows_stat_after.st_mtime_ns),'source row file changed during streaming');require(rows_digest.hexdigest()==expected_rows_sha256,'source row streamed bytes differ')
   ids_out.flush();os.fsync(ids_out.fileno());labels_out.flush();os.fsync(labels_out.fileno())
  db.commit();db.execute('pragma optimize');db.close()
  with schedule_path.open('rb') as sf:
   schedule_stat_before=os.fstat(sf.fileno());schedule_bytes=sf.read();schedule_stat_after=os.fstat(sf.fileno())
  require((schedule_stat_before.st_dev,schedule_stat_before.st_ino,schedule_stat_before.st_size,schedule_stat_before.st_mtime_ns)==(schedule_stat_after.st_dev,schedule_stat_after.st_ino,schedule_stat_after.st_size,schedule_stat_after.st_mtime_ns),'draw schedule changed during read');require(hashlib.sha256(schedule_bytes).hexdigest()==expected_schedule_sha256,'draw schedule streamed bytes differ')
  schedule=json.loads(schedule_bytes);core=schedule.get('schedule',schedule);draw_ids=core.get('row_ids');require(isinstance(draw_ids,list) and draw_ids,'draw schedule missing row IDs');require(core.get('token_rows_sha256')==expected_rows_sha256,'schedule source identity differs');require(core.get('effective_batch')==16 and core.get('max_steps')==len(draw_ids)//16 and len(draw_ids)%16==0,'schedule horizon/effective batch differs')
  db=sqlite3.connect(stage/'index.sqlite3');seen=set();draw_ord=[]
  for position,rid in enumerate(draw_ids):
   found=db.execute('select ordinal from rows where row_id=?',(rid,)).fetchone();require(found is not None,f'schedule unknown row at {position}');draw_ord.append(found[0])
   if position<counts['rows']:require(rid not in seen,'replay before unique coverage');seen.add(rid)
  require(len(seen)==counts['rows'],'schedule omits unique rows before replay');replays=len(draw_ord)-counts['rows'];require(0<=replays<=15,'alignment replay count exceeds one batch');require(core.get('replay_count')==replays,'declared replay count differs');require(core.get('replay_row_ids')==draw_ids[-replays:] if replays else core.get('replay_row_ids')==[],'named replay tail differs')
  db.close()
  with (stage/'draw_ordinals.i64le').open('wb') as f:
   for ordinal in draw_ord:f.write(struct.pack('<Q',ordinal))
   f.flush();os.fsync(f.fileno())
  files={name:{'bytes':(stage/name).stat().st_size,'sha256':sha256(stage/name)} for name in ('input_ids.i32le','labels.i32le','draw_ordinals.i64le','index.sqlite3')}
  manifest={'schema':SCHEMA,'status':'complete','source':{'rows':{'path':str(rows_path.resolve()),'sha256':expected_rows_sha256},'draw_schedule':{'path':str(schedule_path.resolve()),'sha256':expected_schedule_sha256}},'max_sequence_tokens':max_sequence_tokens,'counts':{**counts,'packages':len(packages),'draws':len(draw_ord),'named_replays':len(draw_ord)-counts['rows'],'updates':len(draw_ord)//16},'contract':{'dtype':'signed little-endian int32','attention_mask':'derived all ones after source validation','complete_documents':True,'target_truncation':False,'unique_rows_before_replay':True},'files':files}
  (stage/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');
  for f in stage.iterdir():
   if f.is_file():
    with f.open('rb') as h:os.fsync(h.fileno())
  fsync_dir(stage);os.rename(stage,output);fsync_dir(output.parent);return manifest
 except Exception:
  try:db.close()
  except Exception:pass
  shutil.rmtree(stage,ignore_errors=True);raise

class StreamingCptDataset:
 def __init__(self,cache,expected_manifest_sha256,initial_cursor=0,native_bundle_id=None):
  self.root=Path(cache);mp=self.root/'manifest.json'
  if native_bundle_id:
   from native_attestation import require_attested_identity,require_attested
   observed=require_attested_identity(mp,native_bundle_id);require(observed['sha256']==expected_manifest_sha256,'cache manifest identity differs')
  else:require(sha256(mp)==expected_manifest_sha256,'cache manifest identity differs')
  self.manifest=json.loads(mp.read_text());require(self.manifest.get('schema')==SCHEMA and self.manifest.get('status')=='complete','cache manifest differs')
  for name,record in self.manifest['files'].items():
   if native_bundle_id:require_attested(self.root/name,record['sha256'],record['bytes'],native_bundle_id)
   else:require(sha256(self.root/name)==record['sha256'],'cache payload differs:'+name)
  self.draws=self.manifest['counts']['draws'];require(type(initial_cursor)is int and 0<=initial_cursor<=self.draws,'resume cursor outside schedule');self.initial_cursor=initial_cursor;self._pid=None;self._db=None;self._maps=[]
 def __len__(self):return self.draws
 def _open(self):
  if self._pid==os.getpid():return
  self.close();self._pid=os.getpid();self._db=sqlite3.connect(f'file:{self.root/"index.sqlite3"}?mode=ro',uri=True)
  for name in ('input_ids.i32le','labels.i32le','draw_ordinals.i64le'):
   f=(self.root/name).open('rb');self._maps.append((f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)))
 def __getitem__(self,position):
  require(type(position)is int and 0<=position<self.draws,'draw position outside schedule');self._open();ids_map,labels_map,draw_map=[x[1] for x in self._maps];ordinal=struct.unpack_from('<Q',draw_map,position*8)[0];row=self._db.execute('select token_offset,length from rows where ordinal=?',(ordinal,)).fetchone();require(row is not None,'draw ordinal absent');offset,length=row
  ids=list(struct.unpack_from(f'<{length}i',ids_map,offset*4));labels=list(struct.unpack_from(f'<{length}i',labels_map,offset*4));return {'input_ids':ids,'labels':labels,'attention_mask':[1]*length,'_draw_position':position}
 def close(self):
  for f,m in getattr(self,'_maps',[]):m.close();f.close()
  self._maps=[]
  if getattr(self,'_db',None) is not None:self._db.close();self._db=None
 def __del__(self):
  try:self.close()
  except Exception:pass

def main():
 a=argparse.ArgumentParser();a.add_argument('--rows',required=True);a.add_argument('--schedule',required=True);a.add_argument('--output',required=True);a.add_argument('--max-sequence-tokens',type=int,required=True);a.add_argument('--rows-sha256',required=True);a.add_argument('--schedule-sha256',required=True);x=a.parse_args();print(json.dumps(build(x.rows,x.schedule,x.output,x.max_sequence_tokens,x.rows_sha256,x.schedule_sha256),sort_keys=True))
if __name__=='__main__':main()
