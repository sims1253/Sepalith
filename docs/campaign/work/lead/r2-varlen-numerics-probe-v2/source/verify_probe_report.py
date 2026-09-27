#!/usr/bin/env python3
"""Model-independent verification and predeclared decision for probe output."""
import argparse,hashlib,json,math
from pathlib import Path

MANIFEST="6ae1d9cef615d179d4e55b7da7f68cc67b92239ffce5daeb8852c547c6dfc15d"
WEIGHTS="aac456d2481869d1cec9c6e5e693c8807068ecb8a981d78c1384dbb0d4e39701"
ROWS="96c5e875e473e53f941318bfd8ba1b2dff8196590e87eb1bfb9c35ae0dc77c4b"
SOURCE_SCHEMA="sepalith.sft11.varlen-numerics-probe-source.v2"
def req(v,m):
 if not v:raise ValueError(m)
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def verify(value):
 req(value.get('schema')=='sepalith.sft11.varlen-numerics-probe.v1' and value.get('status')=='measurement_complete_root_parity_decision_required','terminal report differs')
 req(value.get('model_manifest_sha256')==MANIFEST and value.get('model_weights_sha256')==WEIGHTS and value.get('rows_sha256')==ROWS,'input identity differs')
 closure=value.get('source_closure',{});req(closure.get('schema')==SOURCE_SCHEMA and type(closure.get('files')) is int and closure['files']>0 and len(closure.get('sha256',''))==64,'source closure differs')
 req(value.get('model_file_identity_before')==value.get('model_file_identity_after'),'model file changed')
 req(value.get('no_optimizer_created') is True and value.get('no_optimizer_step') is True,'optimizer scope differs')
 req(value.get('contract')=={'loss_denominator':9681,'physical_groups':1,'rows':16,'tokens':9700},'data contract differs')
 req(value.get('saved_precision',{}).get('fp32_tensors_restored')==85,'saved precision differs')
 peak=value.get('cuda_peak',{});req(type(peak.get('allocated_bytes')) is int and type(peak.get('reserved_bytes')) is int and 0<peak['allocated_bytes']<=peak['reserved_bytes']<=32641751449,'CUDA resource ceiling differs')
 r=value['result']; req(r.get('backend')=='xformers','attention backend differs');req(r['cross_document_isolation']['max_abs']==0.0,'attention isolation failed')
 req(set(r['layer0_attribution'])=={'input_rmsnorm','q_proj','k_proj','v_proj','gate_proj','up_proj','q_proj_fp32'},'layer0 stages differ')
 req(all(finite(r[name]) for name in ('standalone_loss','standalone_repeat_loss','packed_loss','ordinary_padded_batch_loss')),'loss metric differs')
 summaries=r['gradient_summaries']; repeat=summaries['same_mode_repeat']; packed=summaries['packed']; padded=summaries['ordinary_padded_batch']
 req(all(len(summary.get('per_tensor',{}))==21 for summary in summaries.values()),'gradient inventory differs')
 metrics=[repeat['max'],packed['median'],packed['max'],padded['median'],padded['max'],r['layer0_attribution']['q_proj_fp32']['relative_l2'],r['held_qkv_attention']['relative_l2']]
 req(all(finite(x) for x in metrics),'nonfinite decision metric')
 ceilings={'repeat_max':1e-7,'q_proj_fp32_max':1e-6,
  'packed_gradient_median':max(.01,1.25*padded['median']),
  'packed_gradient_max':max(.05,1.25*padded['max'])}
 conditions={'isolation_exact':True,'same_mode_repeat':repeat['max']<=ceilings['repeat_max'],
  'fp32_projection_control':r['layer0_attribution']['q_proj_fp32']['relative_l2']<=ceilings['q_proj_fp32_max'],
  'packed_gradient_median':packed['median']<=ceilings['packed_gradient_median'],
  'packed_gradient_max':packed['max']<=ceilings['packed_gradient_max']}
 return {'schema':'sepalith.sft11.varlen-numerics-decision.v1','production_varlen_pass':all(conditions.values()),'conditions':conditions,'ceilings':ceilings,
  'attribution':{'layer0':r['layer0_attribution'],'held_qkv_attention':r['held_qkv_attention']},
  'note':'Passing is a mechanics gate only; it is not a model-quality promotion.'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=verify(json.loads(a.report.read_text()));a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
