#!/usr/bin/env python3
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PREP=PLAN/'docs/campaign/work/lead/r2-edit-context322-root-preparation-v1';RUN=PLAN/'docs/campaign/work/lead/r2-edit-context322-root-run-v2'
CHECKPOINT='/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-CPT-prefix-extension-v3-from194/runtime/checkpoint-322';MANIFEST='81dfdfb4af6d0690ebb114f68910fdbe270d20ad3393a84c47c1c337ce2dbbcf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
def guard(cap):
 suffix='' if cap==2048 else '-b';return Path(f'/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-edit-context322-{cap}-root-v1-host-supervision{suffix}')
def active_refs(root):
 root=Path(root).resolve();hits=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  refs=[]
  for name in ('cwd','root','exe'):
   try:refs.append((name,(proc/name).resolve()))
   except OSError:pass
  try:
   for fd in (proc/'fd').iterdir():
    try:refs.append(('fd',fd.resolve()))
    except OSError:pass
  except OSError:pass
  for kind,ref in refs:
   try:ref.relative_to(root)
   except ValueError:continue
   hits.append({'pid':int(proc.name),'kind':kind,'path':str(ref)});break
 return hits
rows=[]
for cap in (2048,4096,8192,16384):
 rp=PREP/str(cap)/'report.json';report=json.loads(rp.read_text());tp=guard(cap)/'terminal.json';terminal=json.loads(tp.read_text())
 need(report['status']=='completed_no_optimizer_update' and terminal['status']=='completed' and terminal['child_exit_code']==0,f'{cap}:terminal')
 need(report['checkpoint']==CHECKPOINT and report['checkpoint_manifest_sha256']==MANIFEST,f'{cap}:checkpoint')
 need(report['trainable_parameter_tensors']==report['finite_gradient_tensors']==report['nonzero_gradient_tensors']==381,f'{cap}:gradients')
 need(report['optimizer_state']=={'bytes':12206817280,'finite':True,'floating_fp32':True,'tensors':468},f'{cap}:optimizer')
 need(report['optimizer_step_called']is False and report['sampled_parameters_unchanged']is True,f'{cap}:update')
 need(report['rng_before']==report['rng_restored']==report['rng_after_profile'],f'{cap}:rng')
 need(report['scheduler_state_loaded_only']=={'last_epoch':322,'step_count':323},f'{cap}:scheduler')
 rows.append({'cap':cap,'report_sha256':sha(rp),'guard_terminal_sha256':sha(tp),'sequence_tokens':report['sequence_tokens'],'supervised_tokens':report['supervised_tokens'],'row_id':report['row_id'],'forward_backward_seconds':report['forward_backward_seconds'],'elapsed_seconds':report['elapsed_seconds'],'peak_allocated_bytes':report['memory']['peak_allocated'],'peak_reserved_bytes':report['memory']['peak_reserved'],'optimizer_state_bytes':report['optimizer_state']['bytes'],'optimizer_state_tensors':468,'gradient_tensors_finite_nonzero':381,'rng_before_restored_after_equal':True,'optimizer_step_called':False,'sampled_parameters_unchanged':True})
cap=32768;g=guard(cap);terminal=json.loads((g/'terminal.json').read_text());log=(g/'process.log').read_text(errors='replace');samples=[json.loads(x) for x in (g/'host-memory.jsonl').read_text().splitlines()]
need(terminal['status']=='stopped_or_failed' and terminal['reason']=='host_free_memory_below_soft_floor_persisted' and terminal['child_exit_code']==-15,'32K terminal differs');need(not (PREP/'32768/report.json').exists(),'unexpected 32K report')
need('out of memory' not in log.lower() and 'cuda error' not in log.lower() and 'traceback' not in log.lower(),'32K log contains CUDA/runtime error')
cache=[]
for path in ('/tmp/ry-target-uncertainty','/tmp/ry-target-stub'):
 p=Path(path);need((p/'CACHEDIR.TAG').is_file(),'candidate lacks CACHEDIR.TAG');cache.append({'path':path,'du_bytes':int(os.popen(f"du -s -B1 {path}").read().split()[0]),'cachedir_tag_sha256':sha(p/'CACHEDIR.TAG'),'active_proc_refs':active_refs(p),'action_taken':False})
free=os.statvfs('/').f_bavail*os.statvfs('/').f_frsize;reserve=19327352832
output={'schema':'sepalith.sft11.edit-context322-terminal-review.v1','status':'complete_cpu_read_only_review','created_at':datetime.now(timezone.utc).isoformat(),'checkpoint':{'path':CHECKPOINT,'manifest_sha256':MANIFEST},'successful_profiles':rows,'successful_profile_contract':{'optimizer_state':'468 finite FP32 tensors resident, 12,206,817,280 bytes','gradients':'381/381 finite and nonzero','rng':'before == restored == after profile','scheduler':'loaded only; last_epoch322 step_count323','updates':'optimizer_step_called=false and sampled parameters unchanged','quality_evaluation':False},'profile_32768':{'status':'guard_terminated_before_profile_report','guard_terminal':str(g/'terminal.json'),'guard_terminal_sha256':sha(g/'terminal.json'),'reason':terminal['reason'],'child_exit_code':-15,'signal':'SIGTERM','seconds':terminal['seconds'],'host_available_mib_samples':[x['AvailableMBytes'] for x in samples],'two_terminal_below_soft_floor_mib':[samples[-2]['AvailableMBytes'],samples[-1]['AvailableMBytes']],'soft_floor_mib':6144,'process_log_sha256':sha(g/'process.log'),'process_log_last_completed_phase':'381 model weight tensors loaded','profile_report_absent':True,'optimizer_state_residency_unproven':True,'gradient_execution_unproven':True,'cuda_oom_observed':False,'interpretation':'host guard termination; not evidence of CUDA OOM or 32K feasibility'},'recommendation':{'continue_selected_checkpoint_first':True,'supported_context_for_current_planning':16384,'reprobe_32768':'worthwhile only later under a fresh root admission and stable host free memory, after the selected continuation milestone; it is not launch-blocking now','why':'32K was killed during startup before a report, so neither rejection nor feasibility was measured'},'storage':{'filesystem_free_bytes_observed':free,'filesystem_free_gib_observed':free/(2**30),'recipe_full_checkpoint_reserve_bytes':reserve,'projected_free_after_reserve_bytes':free-reserve,'projected_free_after_reserve_gib':(free-reserve)/(2**30),'minimum_free_floor_gib':70,'floor_passes_after_reserve':free-reserve>=70*(2**30),'reclaim_observation':{'path':'/tmp/ry-target-sync','previous_observed_bytes':6681067520,'currently_absent':not Path('/tmp/ry-target-sync').exists(),'worker_deleted':False},'remaining_reversible_candidates':cache,'secondary':'evacuate inactive native ordinary330 only after retaining its independently verified durable E copy; preserve selected packed330 and production322','action_taken_by_worker':False},'resources':{'cpu_threads_max':2,'gpu_used':False,'model_loaded':False,'payload_read':False,'state_changed':False}}
Path(__file__).with_name('review.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','successful_caps':[x['cap'] for x in rows],'failed_cap':32768,'failure_reason':terminal['reason'],'storage_floor_passes_after_reserve':output['storage']['floor_passes_after_reserve']},sort_keys=True))
