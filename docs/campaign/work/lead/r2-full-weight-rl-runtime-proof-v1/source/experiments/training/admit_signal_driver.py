#!/usr/bin/env python3
"""Convert one exact pending model binding plus root admission into a runnable pilot config."""
import argparse,json,os,tempfile
from pathlib import Path
from signal_pilot_driver import canonical,load_config,require,sha256,verify_model

def write_new(path,value):
 path=Path(path);require(path.is_absolute() and not path.exists(),'fresh absolute admitted config output required');path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(json.dumps(value,indent=2,sort_keys=True).encode()+b'\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path);d=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0));os.fsync(d);os.close(d)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def admit(binding_path,admission_path,output):
 binding_path,admission_path=Path(binding_path).resolve(),Path(admission_path).resolve();config=load_config(binding_path);require(config.get('status')=='root_binding_prepared_admission_required' and config.get('model') is not None,'pending model binding differs');identity=verify_model(config['model']);admission=json.loads(admission_path.read_text())
 require(admission.get('schema')=='sepalith.rl11.expanded-signal-driver-root-admission.v1' and admission.get('status')=='admitted' and admission.get('launch_authorized') is True,'root signal admission missing')
 require(admission.get('binding_config_sha256')==sha256(binding_path),'admission refers to another pending binding');require(admission.get('source_manifest_sha256')==config['source']['manifest_sha256'] and admission.get('pilot_spec_sha256')==config['pilot_spec']['sha256'],'admission source/spec identity differs')
 require(admission.get('model_manifest_sha256')==identity['manifest_sha256'] and admission.get('merged_weights_sha256')==identity['merged_weights_sha256'] and admission.get('tokenizer_json_sha256')==identity['tokenizer_json_sha256'],'admission model/tokenizer identity differs')
 require(admission.get('rows_sha256')==config['training_data_binding']['rows_sha256'] and admission.get('reward_buffer_sha256')==config['training_data_binding']['reward_buffer_sha256'],'admission TRAIN/reward identity differs');require(admission.get('output_path')==config['output_path'],'admission output path differs')
 require(admission.get('model_stage')=='selected_full_weight_edit_sft' and admission.get('train_only_rewards') is True and admission.get('dev_or_final_access') is False and admission.get('optimizer_updates')==0,'admission scope differs')
 selected=admission.get('selected_model_evidence',{});sp=Path(str(selected.get('path','')));require(sp.is_file() and sha256(sp)==selected.get('sha256'),'selected full-weight SFT evidence differs')
 admitted=dict(config);admitted['status']='root_admitted_signal_only';admitted['binding_config']={'path':str(binding_path),'sha256':sha256(binding_path)};admitted['root_admission']={'path':str(admission_path),'sha256':sha256(admission_path)};write_new(output,admitted);return admitted

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--admission',required=True);p.add_argument('--output',required=True);a=p.parse_args();v=admit(a.binding,a.admission,a.output);print(json.dumps({'status':v['status'],'sha256':sha256(Path(a.output))},sort_keys=True))
