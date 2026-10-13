#!/usr/bin/env python3
"""Compare two completed canary arms without reading their large payloads."""
import argparse,hashlib,json,math,statistics
from pathlib import Path

def require(v,m):
 if not v:raise ValueError(m)
def load(root):
 root=Path(root);result=json.loads((root/'run-result.json').read_text());checkpoint=Path(result['terminal_checkpoint']);manifest=json.loads((checkpoint/'campaign-manifest.json').read_text());events=[json.loads(x)for x in (root/'telemetry.jsonl').read_text().splitlines()if x.strip()];pre=[x for x in events if x.get('event')=='pre_optimizer'];timing=[x for x in events if x.get('event')=='canary_update_timing']
 require(len(pre)==result['logical_updates']==len(timing),'canary optimizer/timing event count differs');require(all(x['finite_gradient_tensors']==x['nonzero_gradient_tensors']==381 for x in pre),'full gradient tensor gate fails');require(manifest.get('full')is True and manifest.get('checkpoint_kind')=='full_weights','canary checkpoint is not full state');require({'model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','campaign-state.json','trainer_state.json'}<=set(manifest['files']),'canary checkpoint state incomplete')
 return result,manifest,timing
def compare(reference,candidate,output):
 r,rm,rt=load(reference);c,cm,ct=load(candidate);require(r['canary_arm']=='ordinary_reference'and c['canary_arm']=='varlen_candidate','canary arms differ')
 keys=('source_global_step','source_checkpoint_manifest_sha256','global_optimizer_step_offset','global_step','initial_cursor','observed_draws','last_draw_position','loss_denominators','logical_rows','logical_updates','draw_ids_sha256','optimizer_dispatch_sha256','scientific_bindings')
 require(all(r[k]==c[k]for k in keys),'source, logical draws, denominator or scientific binding differs')
 updates=r['logical_updates'];require(updates in (1,8)and r['observed_draws']==r['logical_rows']==16*updates and r['physical_forward_backward_calls']==16*updates,'ordinary physical accounting differs');require(0<c['physical_forward_backward_calls']<=16*updates,'candidate physical accounting differs')
 require(rm['step']==cm['step']==r['global_step']==r['source_global_step']+updates,'canary checkpoint step differs')
 require(len(r['logical_update_losses'])==len(c['logical_update_losses'])==updates,'logical loss count differs');relative=[]
 for a,b in zip(r['logical_update_losses'],c['logical_update_losses']):
  require(math.isfinite(a)and math.isfinite(b)and a>0,'canary logical loss invalid');relative.append(abs(b-a)/a)
 require(max(relative)<=0.001,'packed logical-update loss differs by more than predeclared 0.1% bound')
 fields=('full_step_seconds','forward_backward_seconds','gradient_audit_seconds','optimizer_scheduler_to_step_end_seconds')
 require(all(x.get('synchronized')is True and x.get('excludes_save_and_evaluation')is True and all(math.isfinite(float(x.get(k,math.nan)))and float(x[k])>=0 for k in fields) for x in rt+ct),'synchronized component timing differs')
 warmup=2 if updates==8 else updates;ordinary_timed=[x for x in rt if x['ordinal']>warmup];packed_timed=[x for x in ct if x['ordinal']>warmup];require(len(ordinary_timed)==len(packed_timed)==(6 if updates==8 else 0),'timed update window differs')
 component_medians={k:{'ordinary':statistics.median([x[k]for x in ordinary_timed]),'packed':statistics.median([x[k]for x in packed_timed])}for k in fields} if ordinary_timed else None
 ordinary_seconds=[x['full_step_seconds']for x in ordinary_timed];packed_seconds=[x['full_step_seconds']for x in packed_timed]
 timing={'warmup_updates':warmup,'timed_updates':len(ordinary_seconds),'ordinary_seconds':ordinary_seconds,'packed_seconds':packed_seconds,'component_medians':component_medians,'median_speedup':(statistics.median(ordinary_seconds)/statistics.median(packed_seconds)if ordinary_seconds else None),'synchronized':True,'excludes_save_and_evaluation':True}
 result={'schema':'sepalith.sft11.native-varlen-canary-comparison.v1','status':'mechanical_canary_pass_quality_decision_pending','source_global_step':r['source_global_step'],'target_global_step':r['global_step'],'source_checkpoint_manifest_sha256':r['source_checkpoint_manifest_sha256'],'source_cursor':r['initial_cursor'],'target_cursor':r['last_draw_position']+1,'logical_updates':updates,'logical_rows':16*updates,'draw_ids_sha256':r['draw_ids_sha256'],'loss_denominators':r['loss_denominators'],'max_logical_loss_relative_difference':max(relative),'ordinary_forward_backward_calls':r['physical_forward_backward_calls'],'packed_forward_backward_calls':c['physical_forward_backward_calls'],'physical_call_reduction_ratio':r['physical_forward_backward_calls']/c['physical_forward_backward_calls'],'timing':timing,'full_gradient_tensors_each_update':381,'full_state_checkpoint_inventory':True,'production_admitted':False,'required_next':['matched 2K/8K/16K causal comparison between arms after 8 updates','root review of warm timing and peak GPU/host memory','root scientific decision']}
 Path(output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ordinary-archive',required=True);p.add_argument('--packed-archive',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(compare(a.ordinary_archive,a.packed_archive,a.output),sort_keys=True))
