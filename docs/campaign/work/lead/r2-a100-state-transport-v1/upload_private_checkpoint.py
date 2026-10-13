#!/usr/bin/env python3
"""Resumable private-HF upload; final closure is the publication marker."""
from __future__ import annotations
import argparse,json,os,traceback
from pathlib import Path
from transport_common import PAYLOAD_REQUIRED,atomic_json,assert_stats_unchanged,hashes,load_checkpoint_manifest,require,safe_message,sha256,validate_small_state
ACTION='upload_full_weight_checkpoint90_private'
def lfs_sha(value):
 l=getattr(value,'lfs',None);return l.get('sha256') if isinstance(l,dict) else getattr(l,'sha256',None)
def remote_matches(api,spec,path,revision,expected):
 got=api.get_paths_info(spec['repo'],path,repo_type='model',revision=revision,expand=True);require(len(got)==1 and getattr(got[0],'path',None)==path,'remote object absent');row=got[0];require(row.size==expected['bytes'],'remote object size differs');ls=lfs_sha(row)
 require(ls==expected['sha256'] if ls else getattr(row,'blob_id',None)==expected['git_blob_sha1'],'remote object hash differs')
def authorization(path,spec):
 x=json.loads(Path(path).read_text());expected={'schema':'sepalith.pre04.full_weight_transport_authorization.v1','status':'admitted','authorized':True,'action':ACTION,'repo':spec['repo'],'prefix':spec['prefix'],'campaign_manifest_sha256':spec['campaign_manifest_sha256'],'checkpoint_path':spec['checkpoint_path'],'maximum_bytes':spec['total_bytes']}
 require(x==expected,'root authorization differs');return x
def client():
 from huggingface_hub import HfApi,get_token,hf_hub_download
 token=get_token();require(bool(token),'cached private token unavailable');api=HfApi(token=token);require(api.repo_info('scholzmx/sepalith-lora',repo_type='model').private is True,'repository is not private');return api,hf_hub_download,token
def run(spec_path,authorization_path,journal_path,*,api=None,download=None,token=None,stop_after=None):
 spec=json.loads(Path(spec_path).read_text());require(spec.get('schema')=='sepalith.pre04.full_weight_transport_spec.v1' and spec.get('status')=='prepared_upload_not_authorized' and set(spec.get('files',{}))==PAYLOAD_REQUIRED,'transport spec differs');authorization(authorization_path,spec)
 checkpoint=Path(spec['checkpoint_path']);manifest,stats=load_checkpoint_manifest(checkpoint,spec['campaign_manifest_sha256']);state=validate_small_state(checkpoint,manifest)
 if api is None:api,download,token=client()
 journal_path=Path(journal_path);journal=json.loads(journal_path.read_text()) if journal_path.exists() else {'schema':'sepalith.pre04.full_weight_upload_journal.v1','spec_sha256':sha256(spec_path),'completed':{}}
 require(journal.get('schema')=='sepalith.pre04.full_weight_upload_journal.v1' and journal.get('spec_sha256')==sha256(spec_path) and isinstance(journal.get('completed'),dict),'upload journal differs')
 for index,(name,expected0) in enumerate(sorted(spec['files'].items())):
  local=checkpoint/name;observed=hashes(local);expected={**expected0,'git_blob_sha1':observed['git_blob_sha1']};require(observed['bytes']==expected['bytes'] and observed['sha256']==expected['sha256'],'local checkpoint hash differs:'+name);assert_stats_unchanged(checkpoint,{name:stats[name]});remote=spec['prefix']+'/files/'+name
  saved=journal['completed'].get(name)
  if saved:
   require(saved.get('remote_path')==remote and saved.get('expected')==expected,'journal row differs:'+name);remote_matches(api,spec,remote,saved['revision'],expected)
  else:
   commit=api.upload_file(repo_id=spec['repo'],repo_type='model',path_or_fileobj=local,path_in_repo=remote,commit_message='Full-weight checkpoint90 state: '+name);remote_matches(api,spec,remote,commit.oid,expected);journal['completed'][name]={'remote_path':remote,'revision':commit.oid,'expected':expected};atomic_json(journal_path,journal)
  assert_stats_unchanged(checkpoint,{name:stats[name]})
  if stop_after is not None and index+1>=stop_after:raise RuntimeError('injected interruption')
 require(set(journal['completed'])==set(spec['files']),'upload journal file closure incomplete')
 closure={'schema':'sepalith.pre04.full_weight_checkpoint_closure.v1','status':'complete','repo':spec['repo'],'prefix':spec['prefix'],'step':90,'checkpoint_kind':'full_weights','campaign_manifest_sha256':spec['campaign_manifest_sha256'],'files':spec['files'],'total_bytes':spec['total_bytes'],'state':state,'source_world_size':1,'credential_persisted':False}
 closure_path=journal_path.with_name('closure.json');atomic_json(closure_path,closure);closure_sha=sha256(closure_path);remote_closure=spec['prefix']+'/closure.json';commit=api.upload_file(repo_id=spec['repo'],repo_type='model',path_or_fileobj=closure_path,path_in_repo=remote_closure,commit_message='Publish full-weight checkpoint90 closure')
 back=Path(download(spec['repo'],filename=remote_closure,revision=commit.oid,repo_type='model',token=token));require(sha256(back)==closure_sha,'closure readback differs')
 receipt={'schema':'sepalith.pre04.full_weight_upload_receipt.v1','status':'complete','repo':spec['repo'],'prefix':spec['prefix'],'revision':commit.oid,'closure_path':remote_closure,'closure_sha256':closure_sha,'files':len(spec['files']),'bytes':spec['total_bytes'],'campaign_manifest_sha256':spec['campaign_manifest_sha256'],'credential_persisted':False};atomic_json(journal_path.with_name('upload-receipt.json'),receipt);return receipt
def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--journal',type=Path,required=True);a=p.parse_args()
 try:print(json.dumps(run(a.spec,a.authorization,a.journal),sort_keys=True))
 except Exception as e:print(json.dumps({'status':'failed','error_type':type(e).__name__,'error_message':safe_message(e),'stack':[{'file':Path(x.filename).name,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)[-6:]],'credential_persisted':False}),flush=True);raise SystemExit(1)
if __name__=='__main__':main()
