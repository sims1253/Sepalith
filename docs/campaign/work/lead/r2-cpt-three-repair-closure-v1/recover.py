#!/usr/bin/env python3
"""Targeted, separate recovery for the three historical CPT repair rows."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,shutil,stat,sys,time,uuid
from pathlib import Path
from tokenizers import Tokenizer

RAW_SHA='84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab'
TOKENIZER_SHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
MAIN=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1')
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
BASE=PLAN/'docs/campaign/work/r2-corpus-preparation-v1';GLOBAL=PLAN/'docs/campaign/work/r2-cpt-global-shard-v1'
TARGETS={
 'svars':{'index':2124,'group_id':'g-267771fec5a097997b32','root':Path('/mnt/h/sepalith/normalized/svars/1.3.12/svars'),'license_status':'supported_mit','admission':'candidate_supported_pending_global_dedup'},
 'Rblpapi':{'index':951,'group_id':'g-070cfee0fcfbe5f0a430','root':Path('/mnt/h/sepalith/normalized/Rblpapi/0.3.16/Rblpapi'),'license_status':'GPL3_source_scope_but_DESCRIPTION_License_is_FOSS_no','admission':'provisional_license_root_review_required'},
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fingerprints(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'sha1':hashlib.sha1(raw).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def write_json(path,x):
 with path.open('xb') as f:f.write((json.dumps(x,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
def load_raw(path):
 if sha(path)!=RAW_SHA:raise ValueError('raw chunk source differs')
 s=importlib.util.spec_from_file_location('repair_raw_chunks',path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def prior_sets():
 exact=set();protected=set(json.loads((BASE/'known-nontrain-parent-hashes.json').read_text()))
 for p in (BASE/'broader-shard-v1-2k/documents.jsonl',GLOBAL/'shard/documents.jsonl'):
  with p.open() as f:
   for line in f:exact.add(json.loads(line)['sha256'])
 with (BASE/'profile-shard-v1/documents.jsonl').open() as f:
  for line in f:
   row=json.loads(line)
   if row['cpt_partition']=='cpt_validation':exact.add(row['sha256'])
 return exact,protected
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('fresh output required')
 if sha(a.tokenizer)!=TOKENIZER_SHA:raise ValueError('tokenizer differs')
 rawmod=load_raw(a.raw);tokenizer=Tokenizer.from_file(str(a.tokenizer));prior,protected=prior_sets()
 seen_path=MAIN/'.runtime/seen-sha256.txt';seen_raw=seen_path.read_bytes();main_seen=set(seen_raw.decode().splitlines())
 stage=a.output.with_name(a.output.name+'.tmp-'+uuid.uuid4().hex);stage.mkdir(parents=True)
 totals={};all_docs=[];exclusions=[];inventory=[];local=set()
 try:
  for package,cfg in TARGETS.items():
   folder=stage/('rblpapi-provisional' if package=='Rblpapi' else package);folder.mkdir()
   description=cfg['root']/'DESCRIPTION';license_path=cfg['root']/'LICENSE';rdir=cfg['root']/'R'
   if not all(p.is_file() for p in (description,license_path)) or not rdir.is_dir() or rdir.is_symlink():raise ValueError(f'{package}: targeted metadata/tree invalid')
   license_text=license_path.read_text()
   if package=='svars' and 'COPYRIGHT HOLDER' not in license_text:raise ValueError('svars MIT license evidence differs')
   if package=='Rblpapi':
    if 'source package contains only code' not in license_text or 'GNU General Public License, Version 3' not in license_text or 'proprietary software' not in license_text:raise ValueError('Rblpapi split license evidence differs')
   rows=[];docs=[];counts={'documents':0,'rows':0,'payload_tokens':0,'input_tokens':0,'supervised_tokens':0};files=sorted(rdir.rglob('*.R'))
   for path in files:
    before=path.stat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):raise ValueError(f'{package}: nonregular targeted R source')
    data=path.read_bytes();after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or len(data)!=before.st_size:raise ValueError(f'{package}: source changed')
    fp=fingerprints(data);entry={'package':package,'group_id':cfg['group_id'],'seeded_index':cfg['index'],'path':str(path),'bytes':len(data),**fp,'license_text':license_text.strip(),'license_sha256':sha(license_path),'description_sha256':sha(description),'license_status':cfg['license_status'],'admission':cfg['admission']};inventory.append(entry)
    reason=None
    if not data:reason='empty_R_file_degenerate'
    elif fp['sha256'] in main_seen:reason='exact_duplicate_of_committed_main'
    elif fp['sha256'] in prior:reason='exact_duplicate_of_prior_admitted_or_validation'
    elif protected.intersection(fp.values()):reason='protected_nontrain_hash_match'
    elif fp['sha256'] in local:reason='duplicate_within_recovery'
    if reason:exclusions.append({**entry,'reason':reason});continue
    text=data.decode('utf-8');ids=tokenizer.encode(text,add_special_tokens=False).ids
    if not ids or tokenizer.decode(ids,skip_special_tokens=False)!=text:raise ValueError(f'{package}: tokenizer roundtrip differs: {path}')
    produced=[]
    for index,chunk in enumerate(rawmod.chunks(ids,2048)):
     row={'schema':1,'row_id':fp['sha256']+':'+str(index),'document_id':fp['sha256'],'package':package,'group_id':cfg['group_id'],'cpt_partition':'cpt_train','source_path':str(path),'source_sha256':fp['sha256'],'chunk_index':index,**chunk};rows.append(row);produced.append(row['row_id']);counts['rows']+=1;counts['input_tokens']+=len(row['input_ids']);counts['supervised_tokens']+=row['supervised_tokens']
    doc={**entry,'document_id':fp['sha256'],'source_code_tokens':len(ids),'chunks':len(produced),'row_ids':produced};docs.append(doc);all_docs.append(doc);local.add(fp['sha256']);counts['documents']+=1;counts['payload_tokens']+=len(ids)
   for name,values in [('cpt_train.jsonl',rows),('documents.jsonl',docs)]:
    with (folder/name).open('xb') as f:
     for row in values:f.write((canonical(row)+'\n').encode())
     f.flush();os.fsync(f.fileno())
   result={'schema':'sepalith.cpt.three-repair-package.v1','status':'complete','package':package,'seeded_index':cfg['index'],'group_id':cfg['group_id'],'license_status':cfg['license_status'],'admission':cfg['admission'],'counts':counts,'source_files_examined':len(files),'artifacts':{name:{'bytes':(folder/name).stat().st_size,'sha256':sha(folder/name)} for name in ('cpt_train.jsonl','documents.jsonl')}};write_json(folder/'result.json',result);totals[package]=result
  riv=Path('/mnt/h/sepalith/normalized/RivRetrieve/0.1.9/RivRetrieve/R/data.R');rivraw=riv.read_bytes();exclusions.append({'package':'RivRetrieve','group_id':'g-55d9f2db1c1c36541dea','seeded_index':934,'path':str(riv),'bytes':len(rivraw),'sha256':hashlib.sha256(rivraw).hexdigest(),'reason':'empty_R_file_degenerate','disposition':'exclude_no_payload'})
  for name,values in [('source-inventory.jsonl',inventory),('exclusions.jsonl',exclusions)]:
   with (stage/name).open('xb') as f:
    for row in values:f.write((canonical(row)+'\n').encode())
    f.flush();os.fsync(f.fileno())
  result={'schema':'sepalith.cpt.three-repair-recovery.v1','status':'complete_provisional_global_dedup','main_seen_snapshot':{'path':str(seen_path),'bytes':len(seen_raw),'sha256':hashlib.sha256(seen_raw).hexdigest(),'hashes':len(main_seen)},'raw_chunks':{'path':str(a.raw),'sha256':RAW_SHA},'tokenizer':{'path':str(a.tokenizer),'sha256':TOKENIZER_SHA},'packages':totals,'recovered_documents':len(all_docs),'recovered_payload_tokens':sum(x['source_code_tokens'] for x in all_docs),'exclusions':exclusions,'dedup':{'prior_and_protected_checked':True,'main_snapshot_checked':True,'within_recovery_checked':True,'terminal_global_dedup_pending_while_main_grows':True},'artifacts':{name:{'bytes':(stage/name).stat().st_size,'sha256':sha(stage/name)} for name in ('source-inventory.jsonl','exclusions.jsonl')}}
  write_json(stage/'result.json',result);fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);stage.rename(a.output);fd=os.open(a.output.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(canonical(result))
 except BaseException:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
