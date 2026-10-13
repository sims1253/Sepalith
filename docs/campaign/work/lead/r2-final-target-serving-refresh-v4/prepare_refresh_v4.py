#!/usr/bin/env python3
"""Validate an admitted dense target and emit conversion/quant/spec commands."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
HEX=set('0123456789abcdef');HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def require(x,msg):
 if not x:raise ValueError(msg)
def pin(item,name,large=True):
 require(isinstance(item,dict) and set(item)>={'path','sha256'},f'{name} binding missing');p=Path(item['path']);require(p.is_file() and not p.is_symlink(),f'{name} missing/symlink');require(len(item['sha256'])==64 and set(item['sha256'])<=HEX,f'{name} digest invalid')
 if large:require(sha(p)==item['sha256'],f'{name} hash mismatch')
 return p
def artifact(item,name):return pin(item,name,True)
def write_new(path,value):
 p=Path(path);require(p.is_absolute() and not p.exists(),'fresh absolute output required');p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,p);d=os.open(p.parent,os.O_DIRECTORY);os.fsync(d);os.close(d)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def validate_converter(c):
 m=pin(c['manifest'],'converter closure manifest');v=json.loads(m.read_text());require(v.get('schema')=='sepalith.llamacpp.converter-source-closure.v1' and v.get('git_commit')==c['git_commit'],'converter closure identity differs');root=Path(c['root']);require(str(root)==v['root'],'converter root differs')
 files=v.get('files');require(isinstance(files,list) and len(files)>=100,'converter closure incomplete')
 for x in files:
  p=root/x['path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],f'converter source differs: {x["path"]}')
 require(hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==v['identity_sha256']==c['identity_sha256'],'converter closure digest differs')
def validate_panel(x):
 p=pin({'path':x['path'],'sha256':x['sha256']},x['id']+' panel');m=pin({'path':x['manifest_path'],'sha256':x['manifest_sha256']},x['id']+' manifest');v=json.loads(m.read_text());require(v.get('schema_version')=='sepalith.r2.serving.spec-train-panel.v2' and v.get('profile')=={k:x[k] for k in ('id','context','cap','repetitions','expected_requests')},'panel profile differs');require(v.get('policy',{}).get('split')=='train' and v['policy'].get('contains_dev_or_final') is False,'panel not TRAIN-only');require(v['panel']['sha256']==sha(p) and v['panel']['row_count']*x['repetitions']==x['expected_requests'],'panel count differs')
def validate_target(t,check_large,serving_light=False):
 require(t.get('expected_tensor_count')==381 and t.get('expected_norm_count')==85 and type(t.get('expected_fp32_norm_count')) is int and 0<=t['expected_fp32_norm_count']<=85,'target tensor/dtype expectations differ')
 hf=Path(t['hf_dir']);require(hf.is_dir() and not hf.is_symlink(),'HF target missing');mp=pin({'path':t['campaign_manifest_path'],'sha256':t['campaign_manifest_sha256']},'campaign manifest');m=json.loads(mp.read_text());require(m.get('schema_version')==1 and m.get('full') is True and m.get('checkpoint_kind')=='full_weights','not a full dense checkpoint manifest');ident=hashlib.sha256(json.dumps(m['identity'],sort_keys=True,separators=(',',':')).encode()).hexdigest();require(ident==t['campaign_identity_sha256'],'campaign identity differs');files=m.get('files');require(isinstance(files,dict) and 'model.safetensors' in files,'dense weights absent');require(set(files['model.safetensors'])=={'bytes','sha256'} and files['model.safetensors']['sha256']==t['weights_sha256'],'weights manifest binding differs');require(not (hf/'model.safetensors.index.json').exists(),'sharded/unbound weights unsupported')
 checked = t['required_hf_files'].keys() if serving_light else files.keys()
 for name in checked:
  x=files[name]
  p=hf/name;require(p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'],f'checkpoint file differs: {name}')
  if check_large:require(sha(p)==x['sha256'],f'checkpoint hash differs: {name}')
 require({'model.safetensors','config.json','tokenizer.json','tokenizer_config.json','generation_config.json','chat_template.jinja'}<=set(t['required_hf_files']),'required HF closure incomplete')
 for name,digest in t['required_hf_files'].items():require(name in files and files[name]['sha256']==digest,f'HF companion binding differs: {name}')
 sel=pin({'path':t['selection_receipt_path'],'sha256':t['selection_receipt_sha256']},'selection receipt');s=json.loads(sel.read_text());require(s=={'schema':'sepalith.run06.final-target-selection.v1','status':'selected_for_serving_refresh','target_identity':t['identity'],'campaign_manifest_sha256':t['campaign_manifest_sha256'],'campaign_identity_sha256':t['campaign_identity_sha256']},'selection receipt differs')
def validate_binding(b,require_admission=True,check_large_payloads=False,serving_light=False):
 require(not (check_large_payloads and serving_light),'large and serving-light checks are mutually exclusive');require(b.get('schema')=='sepalith.r2.final-target-serving-refresh.v4' and b.get('launch_authorized') is True,'binding not launch-authorized');require(b.get('packet_source_manifest_sha256')==sha(HERE/'source-manifest.json'),'packet source binding differs');require(isinstance(b.get('target'),dict),'target remains null');validate_target(b['target'],check_large_payloads,serving_light);validate_converter(b['converter_source'])
 require(b['target']['required_hf_files']['tokenizer.json']==b['tokenizer_contract']['tokenizer_json_sha256'],'tokenizer binding differs');require(b['tokenizer_contract']=={'tokenizer_json_sha256':b['tokenizer_contract']['tokenizer_json_sha256'],'vocab_size':130560,'bos_id':0,'pad_id':1,'canonical_eos_id':1,'native_eog_ids':[1,130073]},'tokenizer contract differs')
 cal=b['calibration'];text=pin({'path':cal['text_path'],'sha256':cal['text_sha256']},'calibration');cm=pin({'path':cal['manifest_path'],'sha256':cal['manifest_sha256']},'calibration manifest');v=json.loads(cm.read_text());require(v.get('status')=='TRAIN_calibration_only' and v.get('rows')==84 and v.get('packages')==80 and v.get('calibration_sha256')==sha(text) and len(v.get('row_ids',[]))==84 and sum(v.get('families',{}).values())==84,'calibration semantics differ')
 for x in b['spec_profiles'].values():validate_panel(x)
 require(not ({r['row_id'] for r in json.loads(Path(b['spec_profiles']['ngram']['manifest_path']).read_text())['panel']['records']} & {r['row_id'] for r in json.loads(Path(b['spec_profiles']['draft']['manifest_path']).read_text())['panel']['records']}),'SPEC panels not fresh/disjoint')
 for x in b['tools'].values():pin(x,'tool')
 acceptance=b.get('acceptance',{});require(acceptance=={'point_speedup_min':1.4,'bootstrap_speedup_lower_95_min':1.0,'bootstrap_iterations':5000,'bootstrap_seed':260914,'candidate_p95_must_not_exceed_baseline':True,'exact_output_parity':True,'honest_no_winner':True,'bootstrap_unit':'paired_trace_cluster_stratified_by_2k_8k'},'SPEC acceptance differs')
 selected=b.get('selected_quant_artifact_key');require(selected in {'q8','q4_k_m','iq3_m'} and isinstance(b['outputs'].get(selected),dict),'selected quant remains unbound/unsupported')
 lifecycle=b.get('server_lifecycle',{});require(lifecycle.get('host')=='127.0.0.1' and type(lifecycle.get('port')) is int and 1024<=lifecycle['port']<=65535,'server host/port differs');require(lifecycle.get('health_path')=='/health' and lifecycle.get('health_timeout_seconds')==60 and 0<lifecycle.get('deadline_seconds',0)<=900 and lifecycle.get('terminate_grace_seconds')==15 and lifecycle.get('root_owned') is True,'server lifecycle differs')
 if require_admission:
  a=pin(b['root_admission'],'root admission');x=json.loads(a.read_text());require(x.get('schema')=='sepalith.run06.final-target-serving-refresh-admission.v4' and x.get('status')=='admitted' and x.get('launch_authorized') is True and x.get('target_identity')==b['target']['identity'] and x.get('campaign_manifest_sha256')==b['target']['campaign_manifest_sha256'] and x.get('converter_identity_sha256')==b['converter_source']['identity_sha256'] and x.get('packet_source_manifest_sha256')==b['packet_source_manifest_sha256'] and x.get('selected_quant_artifact_key')==selected,'root admission differs')
 return True
def validate_norm_audit(b,key):
 item=b['outputs'][key];receipt=pin({'path':item['norm_audit_path'],'sha256':item['norm_audit_sha256']},key+' norm audit');v=json.loads(receipt.read_text());require(v.get('schema')=='sepalith.run06.gguf-saved-norm-audit.v1' and v.get('status')=='pass' and v.get('target_identity')==b['target']['identity'] and v.get('artifact_key')==key and v.get('artifact_sha256')==item['sha256'] and v.get('saved_weights_sha256')==b['target']['weights_sha256'] and v.get('norms')==b['target']['expected_norm_count'] and v.get('saved_fp32_norms')==b['target']['expected_fp32_norm_count'] and v.get('bit_exact') is True,key+' norm audit differs')
def validate_imatrix_completion(b):
 item=b['outputs']['imatrix'];artifact(item,'imatrix');receipt=pin({'path':item['completion_receipt_path'],'sha256':item['completion_receipt_sha256']},'imatrix completion');v=json.loads(receipt.read_text());e=v.get('artifact_evidence',{});require(v.get('schema')=='sepalith.run06.imatrix-completion.v2' and v.get('status')=='pass' and v.get('exit_code')==0 and v.get('processed_chunks')==64 and v.get('non_embedding_coverage_mismatches')==0 and e.get('chunk_count')==64 and e.get('chunk_size')==2048 and e.get('missing_non_embedding_matrices')==[] and e.get('partial_non_embedding_matrices')==[],'imatrix execution incomplete');require(v.get('artifact_sha256')==item['sha256'] and v.get('f16_sha256')==b['outputs']['f16']['sha256'] and v.get('calibration_sha256')==b['calibration']['text_sha256'] and v.get('tool_sha256')==b['tools']['imatrix']['sha256'],'imatrix completion identity differs')
def imatrix_argv(b):
 return [b['tools']['imatrix']['path'],'-m',b['outputs']['f16']['path'],'-f',b['calibration']['text_path'],'-o',b['outputs']['imatrix']['path'],'-c','2048','-b','256','-ub','256','-t','6','-ngl','99','--chunks','64','--no-ppl','--process-output','--output-frequency','4']

def commands(b,stage):
 validate_binding(b,True,True)
 out=b['outputs'];tools=b['tools'];f16=Path(out['f16']['path']);im=Path(out['imatrix']['path'])
 if stage=='export':require(not f16.exists(),'F16 exists');return [['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',tools['convert']['path'],b['target']['hf_dir'],'--outfile',str(f16),'--outtype','f16']]
 artifact(out['f16'],'f16')
 if stage=='audit-f16':return [['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(HERE/'audit_gguf_norms.py'),'--binding','$BINDING','--artifact-key','f16','--out','$RUN/audit-f16.json']]
 if stage=='imatrix':validate_norm_audit(b,'f16');require(not im.exists(),'imatrix exists');return [imatrix_argv(b)]
 validate_imatrix_completion(b)
 if stage=='quantize':
  validate_norm_audit(b,'f16');result=[]
  for key,q in [('q8','Q8_0'),('q4_k_m','Q4_K_M'),('iq3_m','IQ3_M')]:require(not Path(out[key]['path']).exists(),key+' exists');result.append([tools['quantize']['path'],'--imatrix',str(im),str(f16),out[key]['path'],q,'4'])
  return result
 if stage=='audit-quants':
  return [['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(HERE/'audit_gguf_norms.py'),'--binding','$BINDING','--artifact-key',k,'--out',f'$RUN/audit-{k}.json'] for k in ('q8','q4_k_m','iq3_m')]
 if stage=='serving':
  key=b['selected_quant_artifact_key'];validate_norm_audit(b,key);model=artifact(out[key],key);common=[tools['server']['path'],'-m',str(model),*b['runtime']['common_server_argv']];result=[common,[*common,*b['runtime']['ngram_argv']]]
  for key in ('released_dspark','existing_trained_dspark'):
   d=b['drafts'][key]
   if d.get('available'):
    artifact(d,key);artifact({'path':d['header_receipt_path'],'sha256':d['header_receipt_sha256']},key+' header');result.append([*common,'-md',d['path'],*b['runtime']['dspark_argv']])
  return result
 raise ValueError('unknown stage')
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--stage',choices=('export','audit-f16','imatrix','quantize','audit-quants','serving'),required=True);p.add_argument('--out',required=True);a=p.parse_args();b=json.loads(Path(a.binding).read_text());write_new(a.out,{'schema':'sepalith.r2.final-target-serving-refresh.commands.v4','stage':a.stage,'commands':commands(b,a.stage),'execution_authorized_by_this_receipt':False})
if __name__=='__main__':main()
