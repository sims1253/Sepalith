#!/usr/bin/env python3
"""Bind a complete admitted CPT corpus/cache and root scientific choices."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def req(v,m):
 if not v:raise ValueError(m)
def pin(r,name):
 p=Path(r['path']);req(p.is_file() and sha(p)==r['sha256'],name+' differs')
def write_new(path,value):
 path=Path(path);req(not path.exists(),'bound recipe output must be fresh');path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def validate_parent(parent):
 req(parent.get('kind')=='merged_cpt_parent','parent kind differs');req(isinstance(parent.get('candidate_id'),str) and parent['candidate_id'],'parent identity missing');req(Path(parent.get('path','')).is_dir(),'parent path missing');req(isinstance(parent.get('files'),dict) and 'model.safetensors' in parent['files'] and 'tokenizer.json' in parent['files'],'parent file identity incomplete')
 if parent.get('candidate_evidence') is not None:pin(parent['candidate_evidence'],'parent evidence')
 precision=parent.get('saved_precision',{});req(type(precision.get('fp32_tensors'))is int and precision['fp32_tensors']>=0,'parent FP32 count missing');req(type(precision.get('fp32_elements'))is int and precision['fp32_elements']>=0,'parent FP32 elements missing');req(isinstance(precision.get('identity_source'),str) and precision['identity_source'],'precision identity missing');req(precision['fp32_tensors']>0 or precision['fp32_elements']==0,'zero FP32 count/elements disagree')
def bind(template_path,admission_path,output):
 template_path,admission_path=Path(template_path),Path(admission_path);t=json.loads(template_path.read_text());a=json.loads(admission_path.read_text());req(t.get('schema')=='sepalith.sft11.full-weight-cpt-full-corpus-template.v1' and t.get('launch_authorized')is False,'template differs');req(a.get('schema')=='sepalith.sft11.full-weight-cpt-full-corpus-root-admission.v1' and a.get('status')=='admitted' and a.get('launch_authorized')is True,'root admission missing');req(a.get('template_sha256')==sha(template_path),'admission refers to another template')
 for key in ('data_admission','corpus_manifest'):pin(a[key],key)
 cohort=a.get('cohort');expected={'id','scope','rows','draw_schedule','manifest','streaming_cache','unique_rows','documents','packages','input_tokens','payload_tokens','loss_tokens','max_sequence_tokens','named_replays','updates'};req(isinstance(cohort,dict) and set(cohort)==expected,'cohort fields differ');req(cohort['scope']=='complete_admitted_train_corpus','cohort does not claim exact admitted full corpus');req(all(type(cohort[k])is int and cohort[k]>0 for k in ('unique_rows','documents','packages','input_tokens','payload_tokens','loss_tokens','max_sequence_tokens','updates')),'cohort denominator invalid');req(type(cohort['named_replays'])is int and 0<=cohort['named_replays']<=15,'alignment replay denominator invalid');req(cohort['updates']*16==cohort['unique_rows']+cohort['named_replays'],'coverage/update arithmetic differs');req(cohort['max_sequence_tokens'] in (2048,4096,8192,16384,32768),'context bound unsupported')
 pin(cohort['manifest'],'cohort manifest');cache=cohort['streaming_cache'];req(set(cache)=={'path','manifest_sha256'},'streaming cache binding differs');cache_manifest=Path(cache['path'])/'manifest.json';req(cache_manifest.is_file() and sha(cache_manifest)==cache['manifest_sha256'],'streaming cache manifest differs');cm=json.loads(cache_manifest.read_text());req(cm['source']['rows']['sha256']==cohort['rows']['sha256'] and cm['source']['draw_schedule']['sha256']==cohort['draw_schedule']['sha256'],'cache source identity differs');req(cm['counts']['rows']==cohort['unique_rows'] and cm['counts']['documents']==cohort['documents'] and cm['counts']['draws']==cohort['updates']*16 and cm['counts']['named_replays']==cohort['named_replays'],'cache denominators differ')
 parent=a.get('parent');validate_parent(parent);selected=a.get('selected',{});required={'optimizer','micro_batch','gradient_accumulation','learning_rate','scheduler','warmup_steps','checkpoint_every','mandatory_stop_step','evaluation_steps','selected_milestones','telemetry_every'};req(set(selected)==required,'scientific fields differ');req(selected['micro_batch'] in (1,2) and selected['micro_batch']*selected['gradient_accumulation']==16,'effective batch differs');req(selected['optimizer'].get('arm')=='aurora_mix' and selected['optimizer'].get('hidden_lr')==selected['learning_rate'],'optimizer/rate differs');req(selected['scheduler'] in ('cosine','constant_with_warmup') and type(selected['warmup_steps'])is int and 0<=selected['warmup_steps']<cohort['updates'],'scheduler/warmup differs');stop=selected['mandatory_stop_step'];req(type(stop)is int and 0<stop<cohort['updates'],'mandatory stop must be intermediate');req(type(selected['checkpoint_every'])is int and selected['checkpoint_every']>0 and stop%selected['checkpoint_every']==0,'mandatory stop lacks checkpoint');req(stop in selected['evaluation_steps'] and stop in selected['selected_milestones'],'mandatory stop lacks evaluation/preservation');req(cohort['updates'] in selected['evaluation_steps'] and cohort['updates'] in selected['selected_milestones'],'terminal checkpoint/evaluation missing');req(type(selected['telemetry_every'])is int and selected['telemetry_every']>0,'telemetry cadence invalid')
 bound=dict(t);bound.update({'schema':'sepalith.sft11.full-weight-cpt-full-corpus-bound.v1','status':'root_admitted_not_launched','launch_authorized':True,'root_admission':{'path':str(admission_path.resolve()),'sha256':sha(admission_path)},'data_admission':a['data_admission'],'corpus_manifest':a['corpus_manifest'],'cohort':cohort,'parent':parent,'runtime':{**selected,'warmup_ratio':selected['warmup_steps']/cohort['updates'],'effective_batch':16,'max_steps':cohort['updates']}})
 import full_weight_cpt_trainer as trainer;trainer.validate_context_and_storage(bound);trainer._validated_cohort(bound);write_new(output,bound);return bound
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--admission',required=True);p.add_argument('--output',required=True);a=p.parse_args();v=bind(a.template,a.admission,a.output);print(json.dumps({'status':v['status'],'sha256':sha(a.output)}))
