#!/usr/bin/env python3
"""Bind a root-admitted new CPT corpus to a preserved full-state checkpoint."""
import argparse,json,os,tempfile
from pathlib import Path
from stage_transition_contract import inspect_transition,pin,require,sha256

def write_new(path,value):
 path=Path(path);require(not path.exists(),'bound recipe output must be fresh');path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def bind(template_path,admission_path,output):
 template_path,admission_path=Path(template_path),Path(admission_path);t=json.loads(template_path.read_text());a=json.loads(admission_path.read_text())
 require(t.get('schema')=='sepalith.sft11.full-weight-cpt-stage-transition-template.v1' and t.get('launch_authorized')is False,'stage-transition template differs')
 require(a.get('schema')=='sepalith.sft11.full-weight-cpt-stage-transition-root-admission.v1' and a.get('status')=='admitted' and a.get('launch_authorized')is True,'root transition admission missing')
 require(a.get('template_sha256')==sha256(template_path),'admission refers to another template')
 for key in ('data_admission','corpus_manifest','dtype_audit'):pin(a[key],key)
 attention=a.get('attention_backend',{});require(attention.get('selected') in ('xformers','flash_varlen'),'varlen attention backend is not root-selected')
 pin(attention.get('kernel_probe',{}),'varlen kernel probe');pin(attention.get('actual_model_parity',{}),'varlen actual-model parity')
 if attention['selected']=='xformers':
  require(Path(attention.get('overlay_root','')).is_dir(),'xformers overlay missing')
  require(len(attention.get('extension_sha256',''))==64 and len(attention.get('wheel_sha256',''))==64,'xformers build identity differs')
 cohort=a['cohort'];required={'id','scope','rows','draw_schedule','manifest','streaming_cache','unique_rows','documents','packages','input_tokens','payload_tokens','loss_tokens','max_sequence_tokens','named_replays','updates'}
 require(set(cohort)==required and cohort['scope']=='complete_admitted_train_corpus','destination corpus fields/scope differ')
 require(cohort['updates']*16==cohort['unique_rows']+cohort['named_replays'] and 0<=cohort['named_replays']<=15,'destination coverage arithmetic differs')
 require(cohort['max_sequence_tokens'] in (2048,4096,8192,16384,32768),'destination context is not profiled')
 pin(cohort['manifest'],'destination cohort manifest');cache=Path(cohort['streaming_cache']['path'])/'manifest.json';require(cache.is_file() and sha256(cache)==cohort['streaming_cache']['manifest_sha256'],'destination streaming cache differs')
 selected=a['selected'];keys={'optimizer','micro_batch','gradient_accumulation','learning_rate','scheduler','warmup_steps','checkpoint_every','mandatory_stop_stage_step','evaluation_stage_steps','selected_stage_milestones','telemetry_every'};require(set(selected)==keys,'selected transition fields differ')
 require(selected['micro_batch']==1 and selected['gradient_accumulation']==16,
         'varlen logical update requires reviewed micro1/acc16 membership')
 require(selected['optimizer'].get('arm')=='aurora_mix' and selected['optimizer'].get('hidden_lr')==selected['learning_rate'],'optimizer/rate differs')
 require(selected['scheduler']=='constant_with_warmup' and type(selected['warmup_steps'])is int and selected['warmup_steps']>=0,'transition scheduler differs')
 require(type(selected['checkpoint_every'])is int and selected['checkpoint_every']>0 and type(selected['telemetry_every'])is int and selected['telemetry_every']>0,'checkpoint/telemetry cadence differs')
 require(0<selected['mandatory_stop_stage_step']<cohort['updates'] and selected['mandatory_stop_stage_step']%selected['checkpoint_every']==0,'mandatory stage stop differs')
 for key in ('evaluation_stage_steps','selected_stage_milestones'):
  require(isinstance(selected[key],list) and selected[key]==sorted(set(selected[key])) and all(type(x)is int and 0<x<=cohort['updates'] for x in selected[key]),key+' differs')
 require(selected['mandatory_stop_stage_step'] in selected['evaluation_stage_steps'] and cohort['updates'] in selected['evaluation_stage_steps'],'stage evaluations incomplete')
 require(selected['mandatory_stop_stage_step'] in selected['selected_stage_milestones'] and cohort['updates'] in selected['selected_stage_milestones'],'preserved stage milestones incomplete')
 transition=a['transition'];require(transition.get('destination_sampler') in ('fresh_cursor_zero','preserve_verified_prefix') and transition.get('scheduler_state')=='load_source_without_rewarm','transition policy differs');step=transition['global_optimizer_step_offset']
 precision=a['saved_precision'];require(type(precision.get('fp32_tensors'))is int and precision['fp32_tensors']>=0 and type(precision.get('fp32_elements'))is int and precision['fp32_elements']>=0,'source checkpoint saved precision differs');require(precision['fp32_tensors']>0 or precision['fp32_elements']==0,'source checkpoint FP32 count/elements disagree')
 runtime={**selected,'effective_batch':16,'max_steps':step+cohort['updates'],'mandatory_stop_step':step+selected['mandatory_stop_stage_step'],'evaluation_steps':[step+x for x in selected['evaluation_stage_steps']],'selected_milestones':[step+x for x in selected['selected_stage_milestones']]}
 runtime.pop('mandatory_stop_stage_step');runtime.pop('evaluation_stage_steps');runtime.pop('selected_stage_milestones')
 checkpoint=Path(transition['source_checkpoint']['path']);manifest=json.loads((checkpoint/'campaign-manifest.json').read_text());files=manifest['files']
 parent={'kind':'full_weight_stage_checkpoint','candidate_id':transition['source_checkpoint']['candidate_id'],'path':str(checkpoint.resolve()),'files':{n:files[n]['sha256'] for n in ('model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json') if n in files},'saved_precision':a['saved_precision']}
 bound=dict(t);bound.update({'schema':'sepalith.sft11.full-weight-cpt-stage-transition-bound.v1','status':'root_admitted_not_launched','launch_authorized':True,'root_admission':{'path':str(admission_path.resolve()),'sha256':sha256(admission_path)},'data_admission':a['data_admission'],'corpus_manifest':a['corpus_manifest'],'dtype_audit':a['dtype_audit'],'attention_backend':attention,'cohort':cohort,'parent':parent,'transition':transition,'runtime':runtime})
 inspection=inspect_transition(bound,verify_payload=False)
 import full_weight_cpt_trainer as trainer
 trainer.validate_context_and_storage(bound);trainer._validated_cohort(bound,inspection['destination_initial_cursor']);write_new(output,bound);return bound

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--admission',required=True);p.add_argument('--output',required=True);a=p.parse_args();v=bind(a.template,a.admission,a.output);print(json.dumps({'status':v['status'],'sha256':sha256(a.output),'global_optimizer_step_offset':v['transition']['global_optimizer_step_offset'],'destination_initial_cursor':v['transition'].get('source_cursor',0)},sort_keys=True))
