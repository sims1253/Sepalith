"""Metadata-only root binder. Does not authorize, load weights, or launch."""
import argparse, copy, datetime, hashlib, json
from pathlib import Path
H=Path(__file__).resolve().parent
SOURCE='306b50daf21a1c1daa31d849f2f1d64b43dbe4c1349afba1d3793366ab0d4fbb'
PARENTS={'5149a4065285c681c2230352e5847b861e2ad2e416c8c263131a557e48bd23ca':'new_lora_on_midtrain','aaacb632811c5a4639fefe52023e72b72e2b960523dbffe10d0187fd6cef8da8':'new_lora_on_merged_cpt_parent'}
TOK='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
TOKCFG='e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b'
def require(ok,msg):
 if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p):
 require(p.is_absolute() and p.is_file() and not p.is_symlink(),'Regular absolute metadata required')
 return {'path':str(p),'sha256':sha(p)}
def digest_string(s): return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
def validate(m, previous, checkpoint, expected_helper):
 require(m['schema_version']=='sepalith.merged-cpt.parent-manifest.v1' and m['kind']=='merged_cpt','Parent schema')
 require(previous['stage']=='cpt_raw_r_v1','Prior stage')
 i=previous['identity'];require(i['source'] in PARENTS and i['policy']['initialization']==PARENTS[i['source']],'Prior source/policy')
 require(i==m['cpt_identity']==checkpoint['identity'] and checkpoint==m['cpt_checkpoint_manifest'],'Checkpoint identity')
 require(checkpoint['full'] is True and type(checkpoint['step']) is int and checkpoint['step']>0,'Full positive checkpoint')
 require(m['source_cursor']==checkpoint['step']*16==m['previous_sampler']['consumed_draws'],'Source cursor')
 require(m['previous_sampler']['schedule_sha256']==previous['draw_schedule']['sha256'] and m['previous_sampler']['split_id']==previous['split_id'],'Sampler lineage')
 require(digest_string(expected_helper) and m['source_script_sha256']==expected_helper,'Merge helper pin')
 require(m['base_model_revision']==i['parent']['revision']=='8dc5f6055b90fe4b9422340810b270b9569f37f3','Base revision')
 require(m['tokenizer_original_bytes_restored'] is True and m['tokenizer']['tokenizer_json_sha256']==TOK and m['tokenizer']['tokenizer_config_sha256']==TOKCFG,'Tokenizer preservation')
 v=m['merge_verification']
 for k,want in dict(device='cpu',dtype='bfloat16',accumulation_dtype='float32',rounding='one final BF16 cast',adapter_tensors_exact=588,independent_lora_matrix_merges_exact=294,cuda_started=False).items():require(v.get(k)==want,'Merge verification '+k)
 inv=m['weight_inventory'];require(len(inv)==1 and inv[0]['path']=='model.safetensors' and inv[0]['bytes']>0 and inv[0]['sha256']==m['merged_weights_sha256'],'Weight inventory')
 require(hashlib.sha256(json.dumps(inv,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()==m['weight_inventory_sha256'],'Inventory hash')
def bind(manifest,expected,helper,output,archive,deadline,seconds,job):
 require(rec(manifest)['sha256']==expected,'Manifest hash')
 m=json.loads(manifest.read_text());p=Path(m['recipe_path']);c=Path(m['cpt_checkpoint'])/'campaign-manifest.json'
 require(rec(p)['sha256']==m['recipe_sha256'] and rec(c)['sha256']==m['checkpoint_manifest_sha256'],'Prior metadata hashes')
 previous=json.loads(p.read_text());checkpoint=json.loads(c.read_text());validate(m,previous,checkpoint,helper)
 require(output.is_absolute() and archive.is_absolute() and output!=archive and not output.exists() and not archive.exists(),'Fresh distinct absolute outputs')
 require(type(seconds) is int and 361<=seconds<=3300,'Attempt cap 361..3300 seconds')
 d=datetime.datetime.fromisoformat(deadline.replace('Z','+00:00'));require(d.tzinfo is not None and (d-datetime.datetime.now(datetime.timezone.utc)).total_seconds()>seconds+60,'Deadline must cover attempt plus cleanup')
 require(job and all(x.isalnum() or x in '-_' for x in job),'Job ID')
 model=Path(m['merged_model_path']);require(model.is_absolute(),'Absolute model path')
 r=json.loads((H/'task-recipe.template.json').read_text());r.update(id=job,model_path=str(model),output_dir=str(output),archive_dir=str(archive),deadline=deadline,max_attempt_seconds=seconds,status='metadata_bound_root_admission_required')
 r['identity']['parent'].update(weights_sha256=m['merged_weights_sha256'],merged_cpt_manifest_sha256=expected,previous_checkpoint_step=checkpoint['step'],previous_source_cursor=m['source_cursor'])
 r['merged_cpt_parent']={'manifest':rec(manifest),'previous_recipe':rec(p),'previous_checkpoint_manifest':rec(c),'merge_script_sha256':helper}
 pins={'model.safetensors':m['merged_weights_sha256'],'config.json':m['config_sha256'],'generation_config.json':m['generation_config_sha256'],'tokenizer.json':TOK,'tokenizer_config.json':TOKCFG}
 require(all(digest_string(v) for v in pins.values()),'Merged file hashes')
 r['inputs'] += [{'path':str(model/k),'sha256':v} for k,v in pins.items()]+[rec(manifest),rec(p),rec(c)]
 return r
if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__)
 for f in ('manifest','output-dir','archive-dir','output-recipe'):a.add_argument('--'+f,type=Path,required=True)
 for f in ('manifest-sha256','merge-helper-sha256','deadline','job-id'):a.add_argument('--'+f,required=True)
 a.add_argument('--seconds',type=int,default=3300);x=a.parse_args();require(x.output_recipe.is_absolute() and not x.output_recipe.exists(),'Fresh recipe')
 r=bind(x.manifest,x.manifest_sha256,x.merge_helper_sha256,x.output_dir,x.archive_dir,x.deadline,x.seconds,x.job_id);x.output_recipe.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'recipe':rec(x.output_recipe),'launch_authorized':False,'model_bytes_read':False}))
