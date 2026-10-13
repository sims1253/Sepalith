"""Verify terminal checkpoint450 and prepare matched 2K/8K/16K evaluation."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
LEAD=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead');PACKET=Path(__file__).resolve().parent
ACTIVE=LEAD/'r2-cpt354-to450-continuation-root-v1';RECIPE_SHA='26b78532f907ca5e3a34827857ad58a47e0d56a167fb26a61dbf9b555b50a1d5';SOURCE_SHA='d17b49bf5beb26aa6308fba1e1a6fda26b6163e42d636bd6200c55556e3d0f1a';IDENTITY='46a5a1aced9f533e8ba4cd25aebe795dedc64d390a0d563cfc7e406c03324739'
ROOT_LAUNCH=ACTIVE/'launch.json';ROOT_LAUNCH_SHA='7b3a750a54c978d600635bbd6bfe92d112e513cc955f8cad3e65c71df50348d7'
GUARD_LAUNCH=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-recovery354-to450-cadence24-v1-host-supervision/launch.json');GUARD_LAUNCH_SHA='bfd7d0c2e88eb5700b7eaa1c0df9d8fa5a46a6e56e6883d89af3737c54cdf11d'
ALLOCATOR=ACTIVE/'allocator.json';ALLOCATOR_SHA='e476e3a0e0512ae1f3475923ca9d0dc63f0d07038f711b6740cafc90f674c7f0'
ARCHIVE=Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-recovery354-to450-cadence24-v1');MODEL=ARCHIVE/'full/checkpoint-450';NATIVE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-recovery354-to450-cadence24-v1/runtime/checkpoint-450')
FULL={'campaign-state.json','chat_template.jinja','config.json','generation_config.json','model.safetensors','optimizer.pt','rng_state.pth','scheduler.pt','tokenizer.json','tokenizer_config.json','trainer_state.json','training_args.bin'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def validate_accounting(result,telemetry):
 assert result['status']=='root_admitted_execution_stopped' and result['global_step']==450 and result['global_optimizer_step_offset']==66
 assert result['initial_cursor']==4608 and result['observed_draws']==1536 and result['last_draw_position']==6143 and math.isfinite(result['train_loss'])
 gradients=[x for x in telemetry if x.get('event')=='pre_optimizer'];logs=[x for x in telemetry if x.get('event')=='log' and 'loss' in x.get('logs',{})]
 assert [x['next_step'] for x in gradients]==list(range(355,451));assert [x['step'] for x in logs]==list(range(355,451))
 assert all(x['finite_gradient_tensors']==x['nonzero_gradient_tensors']==381 for x in gradients)
 assert all(math.isfinite(x['logs']['loss']) and math.isfinite(x['logs']['grad_norm']) for x in logs)
 return {'updates':len(gradients),'draws':1536,'cursor':6144}
def validate_manifest_contract(manifest,identity=IDENTITY):
 assert manifest['full'] is True and manifest['checkpoint_kind']=='full_weights' and manifest['step']==450 and set(manifest['files'])==FULL
 assert canonical_sha(manifest['identity'])==identity
 return True
def verify_terminal(handles_path):
 handles=read(handles_path);assert handles['schema']=='sepalith.sft11.cpt450-runtime-handles.v1' and handles['status']=='terminal_success'
 pids={'controller_pid':2699989,'guard_pid':2700261,'attestation_pid':2700314,'model_child_pid':2700542};assert {k:handles.get(k) for k in pids}==pids
 assert sha(ROOT_LAUNCH)==ROOT_LAUNCH_SHA and sha(GUARD_LAUNCH)==GUARD_LAUNCH_SHA and sha(ALLOCATOR)==ALLOCATOR_SHA
 launch,guard_launch,allocator=read(ROOT_LAUNCH),read(GUARD_LAUNCH),read(ALLOCATOR)
 assert (launch['controller_pid'],launch['guard_pid'])==(2699989,2700261) and (guard_launch['guard_pid'],guard_launch['child_pid'])==(2700261,2700314) and allocator['pid']==2700542
 assert Path(handles['controller_terminal'])==ACTIVE/'terminal.json' and Path(handles['guard_terminal'])==GUARD_LAUNCH.parent/'terminal.json'
 for k in pids:assert type(handles[k]) is int and handles[k]>1 and not Path(f'/proc/{handles[k]}').exists(),f'live runtime handle:{k}'
 assert read(handles['guard_terminal']).get('status')=='completed' and read(handles['guard_terminal']).get('child_exit_code')==0
 assert read(handles['controller_terminal']).get('exit_code')==0
 recipe_path=ACTIVE/'runtime-recipe.json';assert sha(recipe_path)==RECIPE_SHA;recipe=read(recipe_path);assert recipe['runtime_source']['manifest_sha256']==SOURCE_SHA
 continuation=ACTIVE/'continuation-354.json';stop=ACTIVE/'execution-stop-450.json';assert sha(continuation)=='00f322a7398a7cd0dae19533c9cb7413deca2fe1e9991bb91f8e0c43f729f601' and sha(stop)=='825cb0138506cf3e568e213d5f483c8979d1e7eec68e156a179381d26334e3d8'
 result=read(ARCHIVE/'run-result.json');assert Path(result['terminal_checkpoint'])==MODEL
 lineage=result['ordinary_canary_resume_lineage'];assert lineage['source_checkpoint_manifest_sha256']=='2b35854537588499fb703af2127029ff649e410cbab65374948723273b520951' and lineage['source_identity_sha256']=='6d5a7a2dff56d74e1d31662e60a2dff6bd81b46827bbf4760bf16d530fd1f0cf' and lineage['destination_identity_sha256']==IDENTITY
 assert result['continuation_admission']['sha256']==sha(continuation) and result['execution_stop_admission']['sha256']==sha(stop)
 validate_accounting(result,[json.loads(x) for x in (ARCHIVE/'telemetry.jsonl').open()])
 manifest=read(MODEL/'campaign-manifest.json');validate_manifest_contract(manifest)
 source=LEAD/'r2-selected330-recovery-preparation-v2/source/experiments/training/full_weight_cpt_trainer.py';spec=importlib.util.spec_from_file_location('review450',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify_source(recipe)
 assert manifest['identity']==module.identity(recipe)
 actual={str(p.relative_to(MODEL)) for p in MODEL.rglob('*') if p.is_file()};assert actual==FULL|{'campaign-manifest.json'}
 for name,expected in manifest['files'].items():
  p=MODEL/name;assert not p.is_symlink() and p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],name
 state=read(MODEL/'campaign-state.json');sampler=state['sampler'];assert state['identity']==manifest['identity'] and state['step']==450 and state['full'] is True and state['checkpoint_kind']=='full_weights'
 assert sampler['cursor']==sampler['stage_cursor']==6144 and sampler['global_step']==450 and sampler['global_optimizer_step_offset']==66 and sampler['draw_schedule_sha256']==recipe['cohort']['draw_schedule']['sha256'] and sampler['resume_lineage']==lineage
 assert read(MODEL/'trainer_state.json')['global_step']==450 and manifest['files']['tokenizer.json']['sha256']==recipe['parent']['files']['tokenizer.json']=='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81' and manifest['files']['tokenizer_config.json']['sha256']==recipe['parent']['files']['tokenizer_config.json']
 native_manifest=read(NATIVE/'campaign-manifest.json');assert native_manifest==manifest
 assert {str(p.relative_to(NATIVE)) for p in NATIVE.rglob('*') if p.is_file()}==actual
 for name,expected in manifest['files'].items():
  p=NATIVE/name;assert not p.is_symlink() and p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],name
 assert not (ARCHIVE/'evaluations/step-450.json').exists()
 return recipe,manifest,result
def prepare(handles_path):
 recipe,manifest,result=verify_terminal(handles_path)
 binding=read(LEAD/'r2-cpt90-long-eval-root-v1/binding.json');binding.update(status='ROOT_MUST_SET_admitted',model_path=str(NATIVE),checkpoint_step=450,runner_sha256=sha(PACKET/'evaluate_matched.py'),resource_condition='checkpoint450 terminal and native+durable verification passed; fresh exclusive CUDA guard required');binding.pop('at',None);binding['model_files']={n:manifest['files'][n]['sha256'] for n in binding['model_files']}
 command=read(LEAD/'r2-cpt90-long-eval-root-v1/command.json');command[2]=str(PACKET/'evaluate_matched.py');command[command.index('--binding')+1]=str(PACKET/'binding.admitted.json');command[command.index('--output')+1]='/mnt/e/sepalith/campaign-20260915/evaluations/SFT11-cpt450-from354-matched-root-v1'
 review={'schema':'sepalith.sft11.cpt450-from354-terminal-review.v1','status':'checkpoint_verified_evaluation_not_admitted','step':450,'source_step':354,'updates_verified':96,'draws_verified':1536,'cursor':6144,'gradient_tensors_each_update':381,'checkpoint_manifest_sha256':sha(MODEL/'campaign-manifest.json'),'native_and_durable_payloads_verified':True,'external_evaluation_panels':['anchor2k','8k','16k'],'expected_denominators':{'anchor2k':{'rows':499,'loss_tokens':661360},'8k':{'rows':20,'loss_tokens':126464},'16k':{'rows':6,'loss_tokens':69137}},'cuda_launched':False}
 write(PACKET/'binding.review.json',binding);write(PACKET/'command.json',command);write(PACKET/'checkpoint-review.json',review);print(json.dumps(review,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime-handles',required=True);a=p.parse_args();prepare(a.runtime_handles)
