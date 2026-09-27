#!/usr/bin/env python3
"""Create an immutable review envelope, then bind it to an explicit root admission."""
import argparse,hashlib,json
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''): h.update(x)
 return h.hexdigest()
def canon(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def parent():
 root=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-CPT-global-a-250-merged')
 vals={
 'model.safetensors':(5033557128,'5f12810692a90ebb32f589a25e5adcedac53d9fb6edd9166b8245fdec6e3dd1c'),
 'config.json':(748,'f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991'),
 'generation_config.json':(213,'7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269'),
 'tokenizer.json':(9894271,'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'),
 'tokenizer_config.json':(94391,'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b'),
 'chat_template.jinja':(9060,'cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c')}
 return {'path':str(root),'weights_sha256':vals['model.safetensors'][1],'manifest_sha256':'c5ddf086518e66a6709d8e4a16b88f7399d4e87ea477e24f3988ffcdecb1366e','files':{n:{'bytes':z,'sha256':s} for n,(z,s) in vals.items()}}
def review(registry,step):
 c=registry['candidates'][str(step)]
 assert c['status']=='verified_readback',f'candidate {step} has no verified readback: {c["status"]}'
 assert c['step']==step and c['full'] is True
 out={'schema':'sepalith.cloud-cpt-parent-merge.review.v1','status':'awaiting_root_admission','selected_checkpoint_step':step,'candidate':c,'parent':parent(),'tokenizer':{'native_tokenizer_json_sha256':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81','cloud_checkpoint_tokenizer_json_sha256':c['checkpoint_tokenizer_json_sha256'],'native_eog_ids':[1,130073],'bos_id':0,'eos_id':1,'pad_id':1,'policy':'load/copy native parent tokenizer only; checkpoint tokenizer is verified evidence and ignored'},'merge':{'device':'cpu','threads':2,'method':'streaming_per_matrix_fp32_add_one_bf16_cast_then_unload','estimated_peak_rss_gib':12,'minimum_available_host_ram_gib':18,'expected_output_weight_bytes_approx':5033557128},'output':f'/mnt/e/sepalith/campaign-20260915/models/SFT11-CPT-cloud-{step}-merged-candidate-v1','merge_source_sha256':digest(Path(__file__).with_name('merge_cloud_cpt_parent.py')),'candidate_registry_sha256':digest(Path(__file__).parents[1]/'candidates.json'),'candidate_selection_pending':True}
 out['binding_review_sha256']=canon(out)
 return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--candidate',type=int,required=True); ap.add_argument('--review-output',type=Path,required=True); ap.add_argument('--root-admission',type=Path); ap.add_argument('--binding-output',type=Path); a=ap.parse_args()
 reg=json.loads(a.registry.read_text()); assert reg['schema']=='sepalith.cloud-cpt-parent-merge.candidates.v1'
 r=review(reg,a.candidate); assert not a.review_output.exists(); a.review_output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 if a.root_admission:
  assert a.binding_output and not a.binding_output.exists()
  adm=json.loads(a.root_admission.read_text())
  assert adm['schema']=='sepalith.cloud-cpt-parent-merge.root-admission.v1' and adm['status']=='admitted'
  assert adm['action']=='merge_cloud_cpt_parent_candidate' and adm['selected_checkpoint_step']==a.candidate
  assert adm['binding_review_sha256']==r['binding_review_sha256'] and adm['authorized_output']==r['output']
  assert adm['candidate_remains_unselected_after_merge'] is True and adm['cuda_authorized'] is False and adm['final_promotion_authorized'] is False
  b=dict(r); b['schema']='sepalith.cloud-cpt-parent-merge.binding.v1'; b['status']='root_admitted_candidate_merge'; b['root_admission']={'path':str(a.root_admission.resolve()),'sha256':digest(a.root_admission)}
  a.binding_output.write_text(json.dumps(b,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'review':str(a.review_output),'binding_review_sha256':r['binding_review_sha256'],'binding':str(a.binding_output) if a.binding_output else None}))
if __name__=='__main__': main()
