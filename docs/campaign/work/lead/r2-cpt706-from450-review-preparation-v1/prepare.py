"""Verify terminal checkpoint706 and prepare exact matched 2K/8K/16K evaluation."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path

LEAD=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead')
PACKET=Path(__file__).resolve().parent
ACTIVE=LEAD/'r2-cpt450-to706-cadence64-root-v3'
RECIPE=LEAD/'r2-cpt450-to706-cadence64-root-v1/runtime-recipe.json'
RECIPE_SHA='6e27ffd75187999fa02168124bf7e6d7deb1b4c4158ad7ff1ca4320e930a3fa7'
SOURCE_SHA='7bd1b123346cf1b9aff5f162f6620b18deb4c2010f5501d1aa457a3db602b304'
SOURCE_IDENTITY='46a5a1aced9f533e8ba4cd25aebe795dedc64d390a0d563cfc7e406c03324739'
DESTINATION_IDENTITY='6ac45eec834ee675bfecd880b06972ec028b7314497c47ca0130d6d32c43ce48'
ACTIVE_LAUNCH=ACTIVE/'launch.json';ACTIVE_LAUNCH_SHA='3c7494046b2a9cf27dbb1783cf36a9bdd5ab73f49e2d19d0cef8e0e8f910f2aa'
GUARD_LAUNCH=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-cpt450-cadence64-to706-v1-host-supervision-c/launch.json');GUARD_LAUNCH_SHA='35039dfa5c0c8a2bd72be2d606e601760d6d07b7a33d27f2edcbd532f5ff2fc4'
ALLOCATOR=ACTIVE/'allocator.json';ALLOCATOR_SHA='3122f5b40026c805aa33cc4d0dae316791c25e4e834610112306ffd6617d7a48'
MIGRATION=LEAD/'r2-cpt450-to706-cadence64-root-v1/cadence64-migration-admission.json';MIGRATION_SHA='5ad337ff44557847ab8b72d97e47e2903aee35a76003849266a11393fa4fd1b3'
TRANSITION=LEAD/'r2-cpt450-to706-cadence64-root-v1/cadence64-resume-admission.json';TRANSITION_SHA='3f8d87c6609d9c67af613f11453b2ee474c66c31761aa4a90dc45d6698dc5749'
CONTINUATION=LEAD/'r2-cpt450-to706-cadence64-root-v1/continuation-450.json';CONTINUATION_SHA='6025671fb21a19ff5a03c153331c0af3991ab02ba26913ba0686791e0a37c4de'
STOP=LEAD/'r2-cpt450-to706-cadence64-root-v1/execution-stop-706.json';STOP_SHA='d517826b25c433732345d3cf1f4fa87af38e56711c2a92b9fece7242b2750d50'
ARCHIVE=Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-cpt450-cadence64-to706-v1')
MODEL=ARCHIVE/'full/checkpoint-706'
NATIVE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-cpt450-cadence64-to706-v1/runtime/checkpoint-706')
FULL={'campaign-state.json','chat_template.jinja','config.json','generation_config.json','model.safetensors','optimizer.pt','rng_state.pth','scheduler.pt','tokenizer.json','tokenizer_config.json','trainer_state.json','training_args.bin'}
EXPECTED_PIDS={'controller_pid':3038867,'guard_pid':3040840,'attestation_pid':3042470,'model_child_pid':3044342}

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
 assert result['status']=='root_admitted_execution_stopped' and result['global_step']==706 and result['global_optimizer_step_offset']==66
 assert result['stage_step']==640 and result['initial_cursor']==6144 and result['observed_draws']==4096 and result['last_draw_position']==10239
 assert math.isfinite(result['train_loss'])
 # Failed startup attempts emitted no optimizer event. Only actual optimizer/log events
 # count; their complete ordered range identifies the training attempt.
 gradients=[x for x in telemetry if x.get('event')=='pre_optimizer']
 logs=[x for x in telemetry if x.get('event')=='log' and 'loss' in x.get('logs',{})]
 assert [x['next_step'] for x in gradients]==list(range(451,707))
 assert [x['step'] for x in logs]==list(range(451,707))
 assert all(x['finite_gradient_tensors']==x['nonzero_gradient_tensors']==381 for x in gradients)
 assert all(math.isfinite(x['logs']['loss']) and math.isfinite(x['logs']['grad_norm']) for x in logs)
 return {'updates':len(gradients),'draws':result['observed_draws'],'cursor':result['last_draw_position']+1}

def validate_manifest_contract(manifest,wanted):
 assert manifest['full'] is True and manifest['checkpoint_kind']=='full_weights' and manifest['step']==706
 assert set(manifest['files'])==FULL and canonical_sha(manifest['identity'])==wanted
 return True

def validate_transition(recipe):
 assert sha(MIGRATION)==MIGRATION_SHA and sha(TRANSITION)==TRANSITION_SHA and sha(CONTINUATION)==CONTINUATION_SHA and sha(STOP)==STOP_SHA
 migration=read(MIGRATION);transition=read(TRANSITION);continuation=read(CONTINUATION);stop=read(STOP)
 allowed=['checkpoint cadence 24 to 64 after accepted checkpoint450','mandatory external review boundary 354 to 706','fresh continuation runtime output roots']
 assert migration['status']=='admitted' and migration['launch_authorized'] is True and migration['allowed_changes']==allowed
 assert migration['source_identity_sha256']==SOURCE_IDENTITY and migration['destination_identity_sha256']==DESTINATION_IDENTITY
 assert migration['runtime_source_manifest_sha256']==SOURCE_SHA
 assert transition['status']=='admitted' and transition['launch_authorized'] is True and transition['decision']=='continue_checkpoint450_with_cadence64'
 assert transition['bound_recipe_sha256']==RECIPE_SHA and transition['checkpoint_step']==450
 assert transition['checkpoint_manifest_sha256']=='266d6ef90a736e5047ce3c0e4ce17a55718f981856cb52eafad6bc9bb2457f72'
 assert transition['source_identity_sha256']==SOURCE_IDENTITY and transition['destination_identity_sha256']==DESTINATION_IDENTITY and transition['allowed_changes']==allowed
 assert transition['payload_action']=='load_checkpoint450_full_state_without_rewrite_or_optimizer_reset'
 assert continuation['step']==450 and continuation['checkpoint_manifest_sha256']==transition['checkpoint_manifest_sha256']
 assert stop=={'bound_recipe_sha256':RECIPE_SHA,'launch_authorized':True,'resume_global_step':450,'schema':'sepalith.sft11.native-cpt-execution-stop.v1','status':'admitted','stop_at_global_step':706}
 assert recipe['runtime']['checkpoint_every']==64 and recipe['runtime']['mandatory_stop_step']==706 and recipe['runtime']['evaluation_steps']==[194,354,706,11649]
 return True

def validate_internal(internal,recipe):
 assert internal['schema']=='sepalith.cpt.package-holdout-eval.v1' and internal['step']==706 and Path(internal['checkpoint'])==MODEL
 fixture=Path(recipe['validation']['path']);assert sha(fixture)==recipe['validation']['sha256']=='efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8'
 ids=[]
 with fixture.open() as stream:
  for line in stream:
   row=json.loads(line);ids.append(row.get('row_id') or row.get('id'))
 assert len(ids)==len(set(ids))==499 and [x['row_id'] for x in internal['row_metrics']]==ids
 assert internal['denominators']['validation_rows']==499 and internal['denominators']['validation_loss_tokens']==661360
 loss_sum=0.0;tokens=0
 for row in internal['row_metrics']:
  assert type(row['loss_tokens']) is int and row['loss_tokens']>0 and math.isfinite(row['loss_sum']) and math.isfinite(row['mean_causal_nll'])
  assert math.isclose(row['mean_causal_nll'],row['loss_sum']/row['loss_tokens'],rel_tol=0,abs_tol=1e-12)
  tokens+=row['loss_tokens'];loss_sum+=row['loss_sum']
 assert tokens==661360
 nll=loss_sum/tokens;assert math.isclose(internal['metrics']['mean_causal_nll'],nll,rel_tol=0,abs_tol=1e-15)
 return nll

def verify_terminal(handles_path):
 handles=read(handles_path);assert handles['schema']=='sepalith.sft11.cpt706-runtime-handles.v1' and handles['status']=='terminal_success'
 assert {key:handles.get(key) for key in EXPECTED_PIDS}==EXPECTED_PIDS
 assert sha(ACTIVE_LAUNCH)==ACTIVE_LAUNCH_SHA and sha(GUARD_LAUNCH)==GUARD_LAUNCH_SHA and sha(ALLOCATOR)==ALLOCATOR_SHA
 launch=read(ACTIVE_LAUNCH);guard_launch=read(GUARD_LAUNCH);allocator=read(ALLOCATOR)
 assert (launch['controller_pid'],launch['guard_pid'])==(EXPECTED_PIDS['controller_pid'],EXPECTED_PIDS['guard_pid'])
 assert (guard_launch['guard_pid'],guard_launch['child_pid'])==(EXPECTED_PIDS['guard_pid'],EXPECTED_PIDS['attestation_pid'])
 assert allocator['pid']==EXPECTED_PIDS['model_child_pid'] and allocator['trainer_sha256']=='5184354d974cc9bedfbc055f1d6b4c7091582e50a6d2b68c677c6a43ddfc79ee'
 assert Path(handles['controller_terminal'])==ACTIVE/'terminal.json' and Path(handles['guard_terminal'])==GUARD_LAUNCH.parent/'terminal.json'
 for key in EXPECTED_PIDS:assert not Path(f'/proc/{handles[key]}').exists(),f'live runtime handle:{key}'
 guard=read(handles['guard_terminal']);assert guard.get('status')=='completed' and guard.get('child_exit_code')==0
 controller=read(handles['controller_terminal']);assert controller.get('exit_code')==0
 assert sha(RECIPE)==RECIPE_SHA
 recipe=read(RECIPE);assert recipe['runtime_source']['manifest_sha256']==SOURCE_SHA
 validate_transition(recipe)
 source=LEAD/'r2-cpt450-cadence64-continuation-preparation-v2/source/experiments/training/full_weight_cpt_trainer.py'
 spec=importlib.util.spec_from_file_location('review706',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify_source(recipe)
 result=read(ARCHIVE/'run-result.json');assert Path(result['terminal_checkpoint'])==MODEL
 validate_accounting(result,[json.loads(x) for x in (ARCHIVE/'telemetry.jsonl').open()])
 manifest=read(MODEL/'campaign-manifest.json');validate_manifest_contract(manifest,DESTINATION_IDENTITY)
 assert manifest['identity']==module.identity(recipe)
 actual={str(p.relative_to(MODEL)) for p in MODEL.rglob('*') if p.is_file()};assert actual==FULL|{'campaign-manifest.json'}
 for name,expected in manifest['files'].items():
  path=MODEL/name;assert not path.is_symlink() and path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'],name
 state=read(MODEL/'campaign-state.json');sampler=state['sampler'];lineage=result['ordinary_canary_resume_lineage']
 assert state['identity']==manifest['identity'] and state['step']==706 and state['full'] is True and state['checkpoint_kind']=='full_weights'
 assert sampler['cursor']==sampler['stage_cursor']==10240 and sampler['global_step']==706 and sampler['global_optimizer_step_offset']==66
 assert sampler['effective_batch']==16 and sampler['draw_schedule_sha256']==recipe['cohort']['draw_schedule']['sha256'] and sampler['resume_lineage']==lineage
 assert read(MODEL/'trainer_state.json')['global_step']==706
 assert manifest['files']['tokenizer.json']['sha256']==recipe['parent']['files']['tokenizer.json']=='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
 assert manifest['files']['tokenizer_config.json']['sha256']==recipe['parent']['files']['tokenizer_config.json']
 precision=result['saved_precision_audit'];assert precision['exact_saved_values_verified'] is True and precision['fp32_tensors_restored']==85 and len(precision['tensors'])==85
 token=result['tokenizer_contract'];assert (token['bos'],token['eos'],token['pad'],token['vocab_entries'])==(0,1,1,130560)
 native_manifest=read(NATIVE/'campaign-manifest.json');assert native_manifest==manifest
 native_actual={str(p.relative_to(NATIVE)) for p in NATIVE.rglob('*') if p.is_file()};assert native_actual==actual
 for name,expected in manifest['files'].items():
  path=NATIVE/name;assert not path.is_symlink() and path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'],name
 internal=read(ARCHIVE/'evaluations/step-706.json');internal_nll=validate_internal(internal,recipe)
 return recipe,manifest,result,internal_nll

def prepare(handles_path):
 recipe,manifest,result,internal_nll=verify_terminal(handles_path)
 binding=read(LEAD/'r2-cpt90-long-eval-root-v1/binding.json')
 binding.update(status='ROOT_MUST_SET_admitted',model_path=str(NATIVE),checkpoint_step=706,runner_sha256=sha(PACKET/'evaluate_matched.py'),dtype_restoration_sha256=sha(PACKET/'saved_precision.py'),resource_condition='checkpoint706 terminal, internal anchor and native+durable full-state verification passed; fresh exclusive CUDA guard required')
 binding.pop('at',None);binding['model_files']={name:manifest['files'][name]['sha256'] for name in binding['model_files']}
 command=read(LEAD/'r2-cpt90-long-eval-root-v1/command.json');command[2]=str(PACKET/'evaluate_matched.py');command[command.index('--binding')+1]=str(PACKET/'binding.admitted.json');command[command.index('--output')+1]='/mnt/e/sepalith/campaign-20260915/evaluations/SFT11-cpt706-from450-matched-root-v1'
 review={'schema':'sepalith.sft11.cpt706-from450-terminal-review.v1','status':'checkpoint_verified_evaluation_not_admitted','step':706,'source_step':450,'updates_verified':256,'draws_verified':4096,'cursor':10240,'gradient_tensors_each_update':381,'source_identity_sha256':SOURCE_IDENTITY,'destination_identity_sha256':DESTINATION_IDENTITY,'checkpoint_manifest_sha256':sha(MODEL/'campaign-manifest.json'),'native_and_durable_payloads_verified':True,'internal_anchor2k':{'rows':499,'loss_tokens':661360,'mean_causal_nll':internal_nll},'external_evaluation_panels':['anchor2k','8k','16k'],'expected_denominators':{'anchor2k':{'rows':499,'loss_tokens':661360},'8k':{'rows':20,'loss_tokens':126464},'16k':{'rows':6,'loss_tokens':69137}},'cuda_launched':False}
 write(PACKET/'binding.review.json',binding);write(PACKET/'command.json',command);write(PACKET/'checkpoint-review.json',review);print(json.dumps(review,sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime-handles',required=True);a=p.parse_args();prepare(a.runtime_handles)
