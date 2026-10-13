"""Prepare a root-reviewed runner recipe; never dispatches or reads model bytes."""
import argparse,copy,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent

def rec(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def make(recipe_path):
 r=json.loads(recipe_path.read_text());src=json.loads((H/'source-manifest.json').read_text())
 assert r['identity']['source']==src['id']
 assert r['identity']['policy']['initialization']=='new_lora_on_merged_cpt_parent'
 assert r['resume_from'] is None,'Initial transition only; root separately prepares exact-identity continuations'
 assert r['model_path'].startswith('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/')
 old=json.loads((H.parent/'lead/r2-cpt-broad-a/runner-recipe.json').read_text());v=copy.deepcopy(old)
 v.update(id=r['id']+'-to250',snapshot=src['id'])
 extras=[z for z in old['inputs'] if z['path'].endswith(('/cuda_host_guard.py','/host_memory_policy.py','/post_load_cache.py','/prepare_guard_command.py'))]
 for record in extras:assert rec(Path(record['path']))==record
 v['inputs']=r['inputs']+[rec(recipe_path)]+extras
 v['provenance']={'task':'SFT-11','owner':'lead_required','status':'prepared_not_dispatched','budget_source':'Existing 4h aggregate CPT ceiling; root must deduct all previous CPT attempts, CPU merge and this guard. No new budget authorization.','acceptance':'Fresh merged-CPT parent, new LoRA/optimizer, initial stop250; same661360-loss-token validation baseline before any update; root decides continuation.'}
 v['steps'][0]['argv'][4]=str(recipe_path)
 v['steps'][1]['id']='guarded-cpt-global-to250'
 argv=v['steps'][1]['argv'];argv[argv.index('--output')+1]=r['output_dir']+'-host-supervision';argv[argv.index('--seconds')+1]=str(r['max_attempt_seconds']+60);argv[argv.index('--release-cache-file')+1]=r['model_path']+'/model.safetensors'
 return v
if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--recipe',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 assert x.recipe.is_absolute() and x.output.is_absolute() and not x.output.exists()
 x.output.write_text(json.dumps(make(x.recipe),indent=2,sort_keys=True)+'\n');print(json.dumps({'runner_recipe':rec(x.output),'launched':False}))
