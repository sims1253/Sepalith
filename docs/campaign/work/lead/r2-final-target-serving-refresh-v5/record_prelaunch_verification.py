#!/usr/bin/env python3
"""Hash the complete admitted target once, before a serving process is started."""
import argparse,json,stat
from pathlib import Path
from prepare_refresh_v5 import artifact,require,sha,validate_binding,validate_norm_audit,write_new

def file_seal(path):
 p=Path(path);s=p.stat(follow_symlinks=False);require(p.is_file() and not p.is_symlink() and stat.S_ISREG(s.st_mode),'sealed file missing/symlink/nonregular: '+str(p));return {'path':str(p.resolve()),'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def tracked_files(b):
 hf=Path(b['target']['hf_dir']);manifest=json.loads(Path(b['target']['campaign_manifest_path']).read_text());out={'selected_quant':Path(b['outputs'][b['selected_quant_artifact_key']]['path'])}
 for name in sorted(manifest['files']):out['checkpoint:'+name]=hf/name
 for name,t in sorted(b['tools'].items()):out['tool:'+name]=Path(t['path'])
 closure=json.loads(Path(b['converter_source']['manifest']['path']).read_text());converter=Path(b['converter_source']['root'])
 for item in closure['files']:out['converter:'+item['path']]=converter/item['path']
 for name,d in sorted(b['drafts'].items()):
  if name in {'released_dspark','existing_trained_dspark'} and d.get('available'):out['draft:'+name]=Path(d['path']);out['draft_header:'+name]=Path(d['header_receipt_path'])
 return out
def sealed_snapshot(paths):return {name:file_seal(path) for name,path in sorted(paths.items())}
def require_seals(expected):
 require(isinstance(expected,dict) and expected,'prelaunch file seals missing')
 for name,want in expected.items():require(file_seal(want['path'])==want,'sealed file changed after prelaunch verification: '+name)

def record(binding_path,out):
 bp=Path(binding_path).resolve();b=json.loads(bp.read_text());paths=tracked_files(b);before=sealed_snapshot(paths);validate_binding(b,True,True)
 key=b['selected_quant_artifact_key'];q=artifact(b['outputs'][key],key);validate_norm_audit(b,key)
 hf=Path(b['target']['hf_dir']);files={}
 for name,digest in sorted(b['target']['required_hf_files'].items()):
  p=hf/name;files[name]={'sha256':digest,'bytes':p.stat().st_size}
 drafts={}
 for name,d in sorted(b['drafts'].items()):
  if name in {'released_dspark','existing_trained_dspark'} and d.get('available'):
   p=artifact(d,name);header=artifact({'path':d['header_receipt_path'],'sha256':d['header_receipt_sha256']},name+' header');drafts[name]={'sha256':d['sha256'],'bytes':p.stat().st_size,'header_receipt_sha256':d['header_receipt_sha256'],'header_receipt_bytes':header.stat().st_size}
 after=sealed_snapshot(paths);require(before==after,'sealed file identity changed during prelaunch hashing')
 value={'schema':'sepalith.run06.serving-prelaunch-verification.v2','status':'pass','binding_path':str(bp),'binding_sha256':sha(bp),'packet_source_manifest_sha256':b['packet_source_manifest_sha256'],'target_identity':b['target']['identity'],'campaign_manifest_sha256':b['target']['campaign_manifest_sha256'],'campaign_identity_sha256':b['target']['campaign_identity_sha256'],'weights_sha256':b['target']['weights_sha256'],'required_hf_files':files,'selected_quant_artifact_key':key,'selected_quant_sha256':b['outputs'][key]['sha256'],'selected_quant_bytes':q.stat().st_size,'norm_audit_sha256':b['outputs'][key]['norm_audit_sha256'],'drafts':drafts,'file_seals':after,'large_payload_hashes_verified_before_server_start':True,'server_started_by_this_receipt':False}
 write_new(out,value);return value

def validate(b,binding_path,verification_path,verification_sha256):
 vp=Path(verification_path);require(vp.is_file() and not vp.is_symlink() and sha(vp)==verification_sha256,'prelaunch verification receipt differs');v=json.loads(vp.read_text());key=b['selected_quant_artifact_key'];expected={'schema':'sepalith.run06.serving-prelaunch-verification.v2','status':'pass','binding_path':str(Path(binding_path).resolve()),'binding_sha256':sha(binding_path),'packet_source_manifest_sha256':b['packet_source_manifest_sha256'],'target_identity':b['target']['identity'],'campaign_manifest_sha256':b['target']['campaign_manifest_sha256'],'campaign_identity_sha256':b['target']['campaign_identity_sha256'],'weights_sha256':b['target']['weights_sha256'],'selected_quant_artifact_key':key,'selected_quant_sha256':b['outputs'][key]['sha256'],'norm_audit_sha256':b['outputs'][key]['norm_audit_sha256'],'large_payload_hashes_verified_before_server_start':True,'server_started_by_this_receipt':False}
 for k,x in expected.items():require(v.get(k)==x,'prelaunch verification identity differs: '+k)
 require_seals(v.get('file_seals'))
 q=Path(b['outputs'][key]['path']);require(q.is_file() and not q.is_symlink() and q.stat().st_size==v['selected_quant_bytes'],'selected quant size changed after prelaunch verification')
 hf=Path(b['target']['hf_dir']);require(v.get('required_hf_files') and set(v['required_hf_files'])==set(b['target']['required_hf_files']),'prelaunch HF closure differs')
 for name,digest in b['target']['required_hf_files'].items():
  x=v['required_hf_files'][name];p=hf/name;require(x['sha256']==digest and p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'],'required HF file changed after prelaunch verification: '+name);require(name=='model.safetensors' or sha(p)==digest,'required HF metadata hash changed after prelaunch verification: '+name)
 for name,x in v.get('drafts',{}).items():
  d=b['drafts'][name];p=Path(d['path']);h=Path(d['header_receipt_path']);require(d.get('available') and x['sha256']==d['sha256'] and p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'],'draft changed after prelaunch verification: '+name);require(x['header_receipt_sha256']==d['header_receipt_sha256'] and h.is_file() and not h.is_symlink() and h.stat().st_size==x['header_receipt_bytes'] and sha(h)==x['header_receipt_sha256'],'draft header changed after prelaunch verification: '+name)
 return v

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(record(a.binding,a.out),sort_keys=True))
