import pathlib,importlib.util,json,hashlib,datetime,sys
P=pathlib.Path(__file__).resolve().parent;L=P.parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
N=load('root_new_semantic',L/'r2-semantic27to40-queue-preparation-v1/run.py');O=load('root_old_semantic',L/'r2-semantic-queue-root-launch-v3/source/run_streaming_semantic_queue.py');pins=N.source_code_pin();_,index,entries,ish=N.load_index(N.REPLAY_DEFAULT)
checks=[]
for shard in (12,27):
 a=N.validate_replay_shard(N.REPLAY_DEFAULT,shard,index,entries,ish);b=O.validate_receipt(N.REPLAY_DEFAULT,shard,index,entries,ish);assert a==b,f'binding mismatch {shard}';checks.append({'shard':shard,'rows':a['receipt_rows'],'queued':a['queued_rows'],'binding_sha256':a['binding_sha256']})
 if shard==12:
  target=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards12plus-v1/shard-0012');reused=N.reusable_child(target,a,pins['semantic_code']);assert reused['status']=='reused_independently_verified'
receipt=json.loads((L.parents[1]/'receipts/DAT-10-semantic27to40-queue-preparation.json').read_text());n=0
for v in receipt['artifacts'].values():
 if not isinstance(v,dict) or 'path' not in v:continue
 f=pathlib.Path(v['path']);assert f.stat().st_size==v['bytes'] and N.sha(f)==v['sha256'];n+=1
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'DAT-10','runner_and_receipt_artifacts_verified':n,'actual_old_new_source_binding_parity':checks,'actual_existing_shard12_semantic_reuse_verified':reused,'root_runner_test_pass':True,'classification_launched':False,'training_admitted':False}
(P/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
