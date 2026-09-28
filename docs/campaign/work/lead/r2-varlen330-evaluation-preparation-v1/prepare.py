#!/usr/bin/env python3
"""After both canaries finish, verify both complete payloads and prepare eval bindings."""
import hashlib,importlib.util,json,math,os,sys
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');LEAD=PLAN/'docs/campaign/work/lead';PACKET=Path(__file__).resolve().parent
CANARY=LEAD/'r2-native-varlen322-root-v1';SOURCE=LEAD/'r2-native-varlen-canary-preparation-v1';RUNNER=LEAD/'r2-cpt-long-eval-root-v2/evaluate.py';PRECISION=RUNNER.with_name('saved_precision.py');PANELS=LEAD/'r2-cpt-long-holdout-preparation-v1/evaluator-configs.json';ROOT_RECEIPT=PLAN/'docs/campaign/receipts/SFT-11-varlen322-root-launch.json'
PINS={'ordinary_reference':{'recipe':'52072ce7f58db1ffc714e3ea6f7373831a294c2db6c67211fd9b8e95221bfdf6','admission':'eae19e6bfc5ffca37efbd38ddb356acb1d7cd76c4abc0604bcee087bdf69354d'},'varlen_candidate':{'recipe':'12d09ed0158070b256b0976e6f80fc0e46e1f552c1556b44dbc7a6cf360d5997','admission':'df7d7c621da48028cb39bee332a33f9e7e2c4c9983bf76be8d7bacd8314a92b5'},'source_manifest':'faa7dd822910c2ad81154f6b5f9b7b8165952adcd7a38e41d3c3a90210a8b2b0','runner':'bd7cc1de187b7a17c65b9d743bf26f0c7a4177fdb04654ddeee55f05ce011d5b','precision':'3a46d3f301d2f6ade8332b0426edd182a16a7b52c44f28aee60a7b27af144629','panels':'c20e6261515451b5e6f4dbd84c06117a9b946cb80d7c874dff869a70d978863f','migration':'15da9983d473f1d9bc2e5328b0b9c1590dba0b1288a0ffc47834734a859ef1f6','root_launch':'49d11a406e2af7f63cf634cc74de66ea368c336d3040a5c46d6f88b8950f29f1'}
FILES={'campaign-state.json','chat_template.jinja','config.json','generation_config.json','model.safetensors','optimizer.pt','rng_state.pth','scheduler.pt','tokenizer.json','tokenizer_config.json','trainer_state.json','training_args.bin'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def verify_bound_file(path,record,label):
 if Path(path).is_symlink() or Path(path).stat().st_size!=record['bytes'] or sha(Path(path))!=record['sha256']:raise ValueError('payload differs:'+label)
def verify_pair(native,durable,expected_identity):
 nm=read(native/'campaign-manifest.json');dm=read(durable/'campaign-manifest.json')
 if nm!=dm or sha(native/'campaign-manifest.json')!=sha(durable/'campaign-manifest.json'):raise ValueError('native durable manifest differs')
 if nm.get('full') is not True or nm.get('checkpoint_kind')!='full_weights' or nm.get('step')!=330 or nm.get('identity')!=expected_identity or set(nm.get('files',{}))!=FILES:raise ValueError('checkpoint manifest contract')
 expected_actual=FILES|{'campaign-manifest.json'}
 for root in (native,durable):
  actual={str(x.relative_to(root)) for x in root.rglob('*') if x.is_file()}
  if actual!=expected_actual:raise ValueError('checkpoint inventory differs:'+str(root))
  for name in sorted(FILES):verify_bound_file(root/name,nm['files'][name],str(root)+':'+name)
  state=read(root/'campaign-state.json');trainer=read(root/'trainer_state.json')
  sampler=state.get('sampler',{})
  if state.get('identity')!=expected_identity or state.get('step')!=330 or trainer.get('global_step')!=330:raise ValueError('checkpoint state identity')
  if (sampler.get('cursor'),sampler.get('stage_cursor'),sampler.get('global_step'),sampler.get('global_optimizer_step_offset'),sampler.get('effective_batch'))!=(4224,4224,330,66,16):raise ValueError('sampler cursor identity')
 return nm
def validate_report(report,recipe,arm,durable):
 admission=read(CANARY/arm/'admission.json')
 expected_scientific={'tokenizer_sha256':recipe['parent']['files']['tokenizer.json'],'rows_sha256':recipe['cohort']['rows']['sha256'],'cache_manifest_sha256':recipe['cohort']['streaming_cache']['manifest_sha256'],'schedule_sha256':recipe['cohort']['draw_schedule']['sha256'],'optimizer_sha256':hashlib.sha256(json.dumps(recipe['runtime']['optimizer'],sort_keys=True,separators=(',',':')).encode()).hexdigest()}
 if report.get('status')!='root_admitted_execution_stopped' or report.get('canary_arm')!=arm or Path(report.get('terminal_checkpoint',''))!=durable:raise ValueError('terminal report status/arm/path')
 if (report.get('source_global_step'),report.get('global_step'),report.get('global_optimizer_step_offset'),report.get('initial_cursor'),report.get('observed_draws'),report.get('last_draw_position'),report.get('logical_updates'),report.get('logical_rows'))!=(322,330,66,4096,128,4223,8,128):raise ValueError('report step/cursor/update identity')
 if report.get('source_checkpoint_manifest_sha256')!='81dfdfb4af6d0690ebb114f68910fdbe270d20ad3393a84c47c1c337ce2dbbcf' or report.get('scientific_bindings')!=expected_scientific:raise ValueError('scientific binding differs')
 if report.get('canary_admission')!={'path':str((CANARY/arm/'admission.json').resolve()),'sha256':PINS[arm]['admission'],'arm':arm,'source_step':322,'target_step':330,'source_cursor':4096,'target_cursor':4224}:raise ValueError('canary admission binding differs')
 if report.get('execution_stop_admission')!={'stop_at_global_step':330}:raise ValueError('stop admission differs')
 if len(report.get('logical_update_losses',[]))!=8 or not all(math.isfinite(x) for x in report['logical_update_losses']) or len(report.get('loss_denominators',[]))!=8 or not all(type(x)is int and x>0 for x in report['loss_denominators']):raise ValueError('loss accounting differs')
 if len(report.get('warmup_update_seconds',[]))!=2 or len(report.get('timed_update_seconds',[]))!=6 or not all(math.isfinite(x) and x>0 for x in report['warmup_update_seconds']+report['timed_update_seconds']):raise ValueError('timing accounting differs')
 expected_backend='sdpa_ordinary' if arm=='ordinary_reference' else 'xformers'
 if report.get('attention_backend')!=expected_backend or report.get('loss_reduction')!='global_supervised_token_mean':raise ValueError('arm execution semantics differs')
 return {'draw_ids_sha256':report['draw_ids_sha256'],'loss_denominators':report['loss_denominators'],'scientific_bindings':expected_scientific}
def guard_path(arm):return Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training')/f"SFT11-varlen322-{arm}-root-v1-host-supervision/terminal.json"
def main():
 # Small immutable inputs only until the two terminal checkpoints exist.
 if sha(SOURCE/'source-manifest.json')!=PINS['source_manifest'] or sha(RUNNER)!=PINS['runner'] or sha(PRECISION)!=PINS['precision'] or sha(PANELS)!=PINS['panels'] or sha(CANARY/'migration.json')!=PINS['migration'] or sha(ROOT_RECEIPT)!=PINS['root_launch']:raise ValueError('shared source/evaluator pin')
 source_manifest=read(SOURCE/'source-manifest.json')
 for x in source_manifest['files']:
  q=SOURCE/x['path']
  if q.stat().st_size!=x['bytes'] or sha(q)!=x['sha256']:raise ValueError('canary source differs:'+x['path'])
 if not (CANARY/'terminal.json').is_file():raise ValueError('both canaries not terminal')
 terminal=read(CANARY/'terminal.json')
 if terminal.get('status')!='both_commands_completed_requires_root_payload_metric_review' or terminal.get('promotion') is not False:raise ValueError('both canaries not terminal')
 spec=importlib.util.spec_from_file_location('frozen_varlen_canary',SOURCE/'source/experiments/training/native_varlen_canary.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 prepared={}
 for arm in ('ordinary_reference','varlen_candidate'):
  rp=CANARY/arm/'recipe.json';ap=CANARY/arm/'admission.json'
  if sha(rp)!=PINS[arm]['recipe'] or sha(ap)!=PINS[arm]['admission']:raise ValueError('arm recipe/admission differs:'+arm)
  recipe=read(rp)
  if recipe['varlen_canary']['arm']!=arm or (recipe['varlen_canary']['source_global_step'],recipe['varlen_canary']['target_global_step'],recipe['varlen_canary']['source_cursor'],recipe['varlen_canary']['target_cursor'],recipe['varlen_canary']['updates'])!=(322,330,4096,4224,8):raise ValueError('recipe arm identity differs:'+arm)
  gt=read(guard_path(arm))
  if gt.get('status')!='completed' or gt.get('child_exit_code')!=0 or gt.get('reason') is not None:raise ValueError('guard terminal differs:'+arm)
  native=Path(recipe['outputs']['trainer'])/'checkpoint-330';durable=Path(recipe['outputs']['archive'])/'full/checkpoint-330';report_path=Path(recipe['outputs']['archive'])/'run-result.json'
  manifest=verify_pair(native,durable,module.identity(recipe));report=read(report_path);scientific=validate_report(report,recipe,arm,durable)
  model_names=('model.safetensors','config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json')
  binding={'schema':'sepalith.cpt.long-eval-root-binding.v1','status':'prepared_requires_root_admission','training_authorized':False,'runner_sha256':PINS['runner'],'model_path':str(native),'model_files':{name:manifest['files'][name]['sha256'] for name in model_names},'checkpoint_step':330,'resource_condition':'Both canary guard terminals and native+durable full payloads verified; fresh exclusive CUDA guard and root admission required','dtype_restoration_sha256':PINS['precision'],'canary_arm':arm,'canary_recipe_sha256':PINS[arm]['recipe'],'checkpoint_manifest_sha256':sha(native/'campaign-manifest.json')}
  d=PACKET/arm;d.mkdir()
  with (d/'binding.review.json').open('x') as f:json.dump(binding,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  review={'arm':arm,'status':'payload_verified_evaluation_requires_root_admission','guard_terminal_sha256':sha(guard_path(arm)),'run_result_sha256':sha(report_path),'checkpoint_manifest_sha256':binding['checkpoint_manifest_sha256'],'files_verified_each_copy':12,'native_and_durable_hashes_verified':True,**scientific}
  with (d/'checkpoint-review.json').open('x') as f:json.dump(review,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  prepared[arm]={'binding_review_sha256':sha(d/'binding.review.json'),'checkpoint_review_sha256':sha(d/'checkpoint-review.json'),**review}
 if prepared['ordinary_reference']['draw_ids_sha256']!=prepared['varlen_candidate']['draw_ids_sha256'] or prepared['ordinary_reference']['loss_denominators']!=prepared['varlen_candidate']['loss_denominators'] or prepared['ordinary_reference']['scientific_bindings']!=prepared['varlen_candidate']['scientific_bindings']:raise ValueError('paired canary cohort differs')
 with (PACKET/'payload-review.json').open('x') as f:json.dump({'schema':'sepalith.sft11.varlen330.payload_review.v1','status':'complete_requires_root_eval_admission','arms':prepared,'paired_draws_denominators_and_scientific_bindings':True,'promotion':False},f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'complete_requires_root_eval_admission','arms':list(prepared),'paired':True}))
if __name__=='__main__':main()
