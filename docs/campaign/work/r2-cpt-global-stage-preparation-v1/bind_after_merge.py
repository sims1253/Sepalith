"""Root-only metadata binding after selected full checkpoint and verified CPU merge.

No model bytes are read here. Actual campaign_cpt --verify-model-files and run
must verify weights before loading. This tool does not authorize or launch.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'source/experiments/training'))
from campaign_cpt import _check_identity_contract,_check_checkpoint_contract

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rec(p):return {'path':str(p),'sha256':sha(p)}
def bind(template,manifest_path,expected_manifest_sha,source_id,output_dir,archive_dir,deadline,max_seconds):
 assert manifest_path.is_absolute() and sha(manifest_path)==expected_manifest_sha
 assert source_id==json.loads((H/'source-manifest.json').read_text())['id'], 'Root snapshot must preserve the frozen source inventory identity'
 assert output_dir.is_absolute() and archive_dir.is_absolute() and not output_dir.exists() and not archive_dir.exists()
 assert type(max_seconds)is int and 211<=max_seconds<=5400
 m=json.loads(manifest_path.read_text());r=copy.deepcopy(template);model=Path(m['merged_model_path'])
 r.update(model_path=str(model),output_dir=str(output_dir),archive_dir=str(archive_dir),deadline=deadline,max_attempt_seconds=max_seconds,resume_from=None,launch_authorized=False)
 r['identity']['source']=source_id
 r['identity']['parent'].update(weights_sha256=m['merged_weights_sha256'],merged_cpt_manifest_sha256=expected_manifest_sha,previous_checkpoint_step=m['cpt_checkpoint_manifest']['step'],previous_source_cursor=m['source_cursor'])
 r['merged_cpt_parent'].update(manifest=rec(manifest_path),previous_recipe=rec(Path(m['recipe_path'])),previous_checkpoint_manifest=rec(Path(m['cpt_checkpoint'])/'campaign-manifest.json'))
 expected={'model.safetensors':m['merged_weights_sha256'],'config.json':m['config_sha256'],'generation_config.json':m['generation_config_sha256'],'tokenizer.json':m['tokenizer']['tokenizer_json_sha256'],'tokenizer_config.json':m['tokenizer']['tokenizer_config_sha256']}
 r['inputs'] += [{'path':str(model/name),'sha256':s} for name,s in expected.items()]
 r['inputs'] += [r['merged_cpt_parent'][key] for key in ('manifest','previous_recipe','previous_checkpoint_manifest')]
 _check_identity_contract(r,r['parameters']);_check_checkpoint_contract(r,r['parameters'])
 return r

def main():
 a=argparse.ArgumentParser(description=__doc__)
 for flag in ('merged-manifest','output-dir','archive-dir','output-recipe'):a.add_argument('--'+flag,type=Path,required=True)
 for flag in ('merged-manifest-sha256','source-id','deadline'):a.add_argument('--'+flag,required=True)
 a.add_argument('--max-seconds',type=int,required=True);x=a.parse_args()
 assert x.output_recipe.is_absolute() and not x.output_recipe.exists()
 r=bind(json.loads((H/'recipe-template.json').read_text()),x.merged_manifest,x.merged_manifest_sha256,x.source_id,x.output_dir,x.archive_dir,x.deadline,x.max_seconds)
 x.output_recipe.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'recipe':rec(x.output_recipe),'launch_authorized':False,'resume_from':None,'model_bytes_read':False,'root_budget_and_source_admission_required':True}))
if __name__=='__main__':main()
