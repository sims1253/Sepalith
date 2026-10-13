"""Prepare guarded runner recipe only; root retains resource admission and dispatch."""
import argparse,copy,json
from pathlib import Path
from bind_task_parent import H,SOURCE,rec,require
BASE=H.parent/'lead/r2-cpt-broad-c/runner-recipe.json'
def make(p):
 r=json.loads(p.read_text());require(r['identity']['source']==SOURCE and r['resume_from'] is None,'Initial task source')
 require(r['mandatory_stop_steps']==[250] and r['decision_steps']==[250] and r['parameters']['max_steps']==1000,'First milestone/horizon')
 old=json.loads(BASE.read_text());v=copy.deepcopy(old);v.update(id=r['id'],snapshot=SOURCE)
 extras=[z for z in old['inputs'] if z['path'].endswith(('/cuda_host_guard.py','/host_memory_policy.py','/post_load_cache.py','/prepare_guard_command.py'))]
 require(len(extras)==4,'Four accepted guard helpers')
 for z in extras: require(rec(Path(z['path']))==z,'Guard helper changed')
 v['inputs']=r['inputs']+[rec(p)]+extras
 v['provenance']={'task':'SFT-11','owner':'lead_required','status':'prepared_not_dispatched','acceptance':'Fresh LoRA on root-selected CPT merge; mandatory full250 stop, same1000 horizon. HF diagnostic only; native Q8 DEV43 edits32 noops decides continuation.','budget_source':'Existing task SFT plus DEV4h ceiling. Root recalculates remaining time and absolute deadline before dispatch.'}
 v['steps'][0]['id']='prepare-task-host-command';v['steps'][0]['argv'][4]=str(p)
 v['steps'][1]['id']='guarded-task-to250';argv=v['steps'][1]['argv'];argv[argv.index('--output')+1]=r['output_dir']+'-host-supervision';argv[argv.index('--seconds')+1]=str(r['max_attempt_seconds']+60);argv[argv.index('--release-cache-file')+1]=r['model_path']+'/model.safetensors'
 return v
if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--recipe',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();require(x.output.is_absolute() and not x.output.exists(),'Fresh absolute output');x.output.write_text(json.dumps(make(x.recipe),indent=2)+'\n');print(json.dumps({'runner_recipe':rec(x.output),'dispatched':False}))
