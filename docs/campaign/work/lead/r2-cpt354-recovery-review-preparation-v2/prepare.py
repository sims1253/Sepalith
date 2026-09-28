"""Verify terminal checkpoint354 and prepare long-only matched evaluation."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path

LEAD=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead')
PACKET=Path(__file__).resolve().parent
ACTIVE=LEAD/'r2-selected330-recovery-root-v2'
RECIPE_SHA='d74c3da8a254eaee572ef80ee12b186d592e91fd1479fae008a60ff84908e3d7'
SOURCE_SHA='d17b49bf5beb26aa6308fba1e1a6fda26b6163e42d636bd6200c55556e3d0f1a'
DESTINATION_IDENTITY='46a5a1aced9f533e8ba4cd25aebe795dedc64d390a0d563cfc7e406c03324739'
SUPERVISOR_LAUNCH=LEAD/'r2-selected330-recovery-supervisor-root-v1/launch.json';SUPERVISOR_LAUNCH_SHA='61e64bd04eda3278ac85c35c107f7a7a7d280cd68a5f60a4b2106099e2c062f6'
GUARD_LAUNCH=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-selected330-recovery-cadence24-v2-host-supervision/launch.json');GUARD_LAUNCH_SHA='8cb63791dbb057599c21e057e41363faf721cae64dbd6de14023fd2a1dfec334'
ALLOCATOR=ACTIVE/'allocator.json';ALLOCATOR_SHA='0acf0de1164b3c091fc45aeef1ed6f97ebcea9166c271361c5d6aca63ac00af6'
ARCHIVE=Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-selected330-recovery-cadence24-v2')
MODEL=ARCHIVE/'full/checkpoint-354'
NATIVE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-selected330-recovery-cadence24-v2/runtime/checkpoint-354')
FULL={'campaign-state.json','chat_template.jinja','config.json','generation_config.json','model.safetensors','optimizer.pt','rng_state.pth','scheduler.pt','tokenizer.json','tokenizer_config.json','trainer_state.json','training_args.bin'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def validate_accounting(result,telemetry):
 assert result['status']=='root_admitted_execution_stopped' and result['global_step']==354 and result['global_optimizer_step_offset']==66
 assert result['initial_cursor']==4224 and result['observed_draws']==384 and result['last_draw_position']==4607
 assert math.isfinite(result['train_loss'])
 gradients=[x for x in telemetry if x.get('event')=='pre_optimizer'];logs=[x for x in telemetry if x.get('event')=='log' and 'loss' in x.get('logs',{})]
 assert [x['next_step'] for x in gradients]==list(range(331,355));assert [x['step'] for x in logs]==list(range(331,355))
 assert all(x['finite_gradient_tensors']==x['nonzero_gradient_tensors']==381 for x in gradients)
 assert all(math.isfinite(x['logs']['loss']) and math.isfinite(x['logs']['grad_norm']) for x in logs)
 return {'updates':len(gradients),'draws':result['observed_draws'],'cursor':result['last_draw_position']+1}

def verify_terminal(handles_path):
 handles=read(handles_path);assert handles['schema']=='sepalith.sft11.cpt354-runtime-handles.v1' and handles['status']=='terminal_success'
 expected_pids={'controller_pid':2630015,'guard_pid':2630027,'attestation_pid':2630218,'model_child_pid':2634519}
 assert {key:handles.get(key) for key in expected_pids}==expected_pids
 assert sha(SUPERVISOR_LAUNCH)==SUPERVISOR_LAUNCH_SHA and sha(GUARD_LAUNCH)==GUARD_LAUNCH_SHA and sha(ALLOCATOR)==ALLOCATOR_SHA
 supervisor=read(SUPERVISOR_LAUNCH);guard_launch=read(GUARD_LAUNCH);allocator=read(ALLOCATOR)
 assert (supervisor['controller_pid'],supervisor['guard_pid'])==(2630015,2630027)
 assert (guard_launch['guard_pid'],guard_launch['child_pid'])==(2630027,2630218) and allocator['pid']==2634519
 assert Path(handles['controller_terminal'])==LEAD/'r2-selected330-recovery-supervisor-root-v1/terminal.json'
 assert Path(handles['guard_terminal'])==GUARD_LAUNCH.parent/'terminal.json'
 for key in expected_pids:
  assert type(handles[key]) is int and handles[key]>1 and not Path(f'/proc/{handles[key]}').exists(),f'live runtime handle:{key}'
 guard=read(handles['guard_terminal']);assert guard.get('status')=='completed' and guard.get('child_exit_code')==0
 controller=read(handles['controller_terminal']);assert controller.get('exit_code')==0
 recipe_path=ACTIVE/'runtime-recipe.json';assert sha(recipe_path)==RECIPE_SHA
 recipe=read(recipe_path);assert recipe['runtime_source']['manifest_sha256']==SOURCE_SHA
 result=read(ARCHIVE/'run-result.json')
 assert Path(result['terminal_checkpoint'])==MODEL
 lineage=result['ordinary_canary_resume_lineage'];transition=ACTIVE/'selected-packed330-transition.admitted.json'
 assert lineage['source_checkpoint_manifest_sha256']=='2b35854537588499fb703af2127029ff649e410cbab65374948723273b520951'
 assert lineage['source_identity_sha256']=='6d5a7a2dff56d74e1d31662e60a2dff6bd81b46827bbf4760bf16d530fd1f0cf'
 assert lineage['destination_identity_sha256']==DESTINATION_IDENTITY and lineage['transition_admission_sha256']==sha(transition)
 telemetry=[json.loads(x) for x in (ARCHIVE/'telemetry.jsonl').open()]
 validate_accounting(result,telemetry)
 manifest=read(MODEL/'campaign-manifest.json');assert manifest['full'] is True and manifest['checkpoint_kind']=='full_weights' and manifest['step']==354 and set(manifest['files'])==FULL
 source=LEAD/'r2-selected330-recovery-preparation-v2/source/experiments/training/full_weight_cpt_trainer.py'
 spec=importlib.util.spec_from_file_location('reviewed354',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify_source(recipe)
 assert manifest['identity']==module.identity(recipe) and canonical_sha(manifest['identity'])==DESTINATION_IDENTITY
 actual={str(p.relative_to(MODEL)) for p in MODEL.rglob('*') if p.is_file()};assert actual==FULL|{'campaign-manifest.json'}
 for name,expected in manifest['files'].items():
  path=MODEL/name;assert not path.is_symlink() and path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'],name
 state=read(MODEL/'campaign-state.json');sampler=state['sampler']
 assert state['identity']==manifest['identity'] and state['step']==354 and state['full'] is True and state['checkpoint_kind']=='full_weights'
 assert sampler['cursor']==sampler['stage_cursor']==4608 and sampler['global_step']==354 and sampler['global_optimizer_step_offset']==66
 assert sampler['draw_schedule_sha256']==recipe['cohort']['draw_schedule']['sha256'] and sampler['resume_lineage']==lineage
 assert read(MODEL/'trainer_state.json')['global_step']==354
 assert manifest['files']['tokenizer.json']['sha256']==recipe['parent']['files']['tokenizer.json']=='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
 assert manifest['files']['tokenizer_config.json']['sha256']==recipe['parent']['files']['tokenizer_config.json']
 native_manifest=read(NATIVE/'campaign-manifest.json');assert native_manifest==manifest
 native_actual={str(p.relative_to(NATIVE)) for p in NATIVE.rglob('*') if p.is_file()};assert native_actual==actual
 for name,expected in manifest['files'].items():
  path=NATIVE/name;assert not path.is_symlink() and path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'],name
 internal=read(ARCHIVE/'evaluations/step-354.json')
 assert internal['schema']=='sepalith.cpt.package-holdout-eval.v1' and internal['step']==354 and Path(internal['checkpoint'])==MODEL
 assert internal['denominators']['validation_rows']==499 and internal['denominators']['validation_loss_tokens']==661360
 assert len(internal['row_metrics'])==499 and math.isfinite(internal['metrics']['mean_causal_nll'])
 anchor=[]
 with Path(recipe['validation']['path']).open() as stream:
  for line in stream:
   row=json.loads(line);anchor.append(row.get('row_id') or row.get('id'))
 assert len(anchor)==499 and all(isinstance(x,str) and x for x in anchor)
 assert [row['row_id'] for row in internal['row_metrics']]==anchor
 assert len(set(anchor))==499
 return recipe,manifest,result,internal

def prepare(handles_path):
 recipe,manifest,result,internal=verify_terminal(handles_path)
 binding=read(LEAD/'r2-cpt90-long-eval-root-v1/binding.json')
 binding.update(status='ROOT_MUST_SET_admitted',model_path=str(NATIVE),checkpoint_step=354,runner_sha256=sha(PACKET/'evaluate_long_only.py'),resource_condition='checkpoint354 terminal and native+durable full-state verification passed; fresh exclusive CUDA guard still required')
 binding.pop('at',None);binding['model_files']={name:manifest['files'][name]['sha256'] for name in binding['model_files']}
 command=read(LEAD/'r2-cpt90-long-eval-root-v1/command.json');command[command.index(command[2])]=str(PACKET/'evaluate_long_only.py');command[command.index('--binding')+1]=str(PACKET/'binding.admitted.json');command[command.index('--output')+1]='/mnt/e/sepalith/campaign-20260915/evaluations/SFT11-cpt354-long-only-root-v1'
 review={'schema':'sepalith.sft11.cpt354-terminal-review.v1','status':'checkpoint_verified_evaluation_not_admitted','step':354,'updates_verified':24,'draws_verified':384,'cursor':4608,'gradient_tensors_each_update':381,'checkpoint_manifest_sha256':sha(MODEL/'campaign-manifest.json'),'native_and_durable_payloads_verified':True,'internal_anchor2k':{'rows':499,'loss_tokens':661360,'mean_causal_nll':internal['metrics']['mean_causal_nll']},'external_evaluation_panels':['8k','16k'],'cuda_launched':False}
 write(PACKET/'binding.review.json',binding);write(PACKET/'command.json',command);write(PACKET/'checkpoint-review.json',review);print(json.dumps(review,sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime-handles',required=True);a=p.parse_args();prepare(a.runtime_handles)
