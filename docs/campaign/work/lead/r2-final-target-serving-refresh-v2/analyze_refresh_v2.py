#!/usr/bin/env python3
"""Fail-closed SPEC comparison with complete coverage and deterministic paired bootstrap."""
from __future__ import annotations
import argparse,hashlib,json,math,random,tempfile,os
from pathlib import Path
from statistics import median
HEX=set('0123456789abcdef')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,msg):
 if not x:raise ValueError(msg)
def pct(xs,p):
 s=sorted(xs);x=(len(s)-1)*p;lo=int(x);hi=math.ceil(x);return s[lo]+(s[hi]-s[lo])*(x-lo)
def load_json(p):return json.loads(Path(p).read_text())
def expected(profile):
 manifest=load_json(profile['manifest_path']);records=manifest['panel']['records'];reps=profile['repetitions']
 return {(r['row_id'],'measurement',rep) for r in records for rep in range(1,reps+1)}, {r['row_id']:r['band'] for r in records}, manifest
def index_result(v,binding,bsha,profile,mode):
 require(v.get('schema_version')=='sepalith.r2.serving.spec-train-panel-probe.v2' and v.get('status') in {'completed','partial'},'result incomplete')
 x=v.get('artifact_binding');require(isinstance(x,dict),'artifact binding missing');require(x.get('binding_sha256')==bsha,'binding hash mismatch');require(x.get('target_identity')==binding['target']['identity'],'target mismatch');require(x.get('tokenizer_json_sha256')==binding['tokenizer_contract']['tokenizer_json_sha256'],'tokenizer mismatch');require(x.get('artifact_key')=='q8' and x.get('artifact_sha256')==binding['outputs']['q8']['sha256'],'Q8 binding mismatch');require(x.get('mode')==mode and x.get('profile_id')==profile['id'],'mode/profile mismatch')
 if mode in {'ordinary','ngram_mod'}:require(x.get('draft') is None,'model-free/ordinary result has draft binding')
 else:
  d=x.get('draft');expected_d=binding['drafts'][mode];require(isinstance(d,dict) and d.get('key')==mode and d.get('sha256')==expected_d['sha256'] and d.get('target_relation')==expected_d['target_relation'],'draft identity/relation mismatch')
 keys,bands,manifest=expected(profile);panel=v.get('panel',{});require(panel.get('manifest_sha256')==sha(profile['manifest_path']) and panel.get('source_sha256')==manifest['panel']['sha256'] and panel.get('profile')==manifest['profile'],'panel binding mismatch')
 require(v.get('native_contract')=={'bos_id':0,'canonical_eos_id':1,'native_eog_ids':[1,130073],'pad_id':1,'context_size':10240,'cap':64,'greedy':True,'tokenize_flags':{'add_special':False,'parse_special':False,'with_pieces':False},'no_authored_target_tail':True},'native contract mismatch')
 out={}
 for r in v.get('requests',[]):
  k=(r.get('row_id'),r.get('phase'),r.get('rep'));require(k not in out,'duplicate request key');out[k]=r
 require(set(out)==keys,f'request key coverage differs: expected {len(keys)} observed {len(out)}')
 for k,r in out.items():
  require(isinstance(r.get('protocol_status'),str),'protocol status missing');require(isinstance(r.get('combined_case_wall_ms'),(int,float)) and math.isfinite(r['combined_case_wall_ms']) and r['combined_case_wall_ms']>0,'latency missing/nonfinite');term=r.get('termination');require(isinstance(term,dict) and type(term.get('cap_hit')) is bool and term.get('stop_type') in {'eos','limit'},'termination evidence missing');require(isinstance(r.get('returned_token_ids'),list) and isinstance(r.get('raw_text'),str),'output evidence missing')
 return out,bands,x
def bootstrap_lb(base,cand,iterations=5000,seed=260914):
 ratios=[];rng=random.Random(seed);n=len(base)
 for _ in range(iterations):
  ix=[rng.randrange(n) for _ in range(n)];ratios.append(median([base[i] for i in ix])/median([cand[i] for i in ix]))
 return pct(ratios,.025)
def compare(binding,bsha,profile,base,cand,candidate_mode):
 a,bands,_=index_result(base,binding,bsha,profile,'ordinary');b,_,bx=index_result(cand,binding,bsha,profile,candidate_mode);keys=sorted(a)
 mismatch=[]
 for k in keys:
  x,y=a[k],b[k]
  if x['returned_token_ids']!=y['returned_token_ids'] or x['raw_text']!=y['raw_text']:mismatch.append({'key':list(k),'reason':'greedy_output'})
  elif x['termination']!=y['termination']:mismatch.append({'key':list(k),'reason':'termination'})
 protocol_ok=all(a[k]['protocol_status']=='accepted' and b[k]['protocol_status']=='accepted' for k in keys)
 bw=[float(a[k]['combined_case_wall_ms']) for k in keys];cw=[float(b[k]['combined_case_wall_ms']) for k in keys]
 med=median(bw)/median(cw);lb=bootstrap_lb(bw,cw);bp=pct(bw,.95);cp=pct(cw,.95)
 draft=[]
 for k in keys:
  r=b[k];require(type(r.get('draft_n')) is int and type(r.get('draft_n_accepted')) is int and 0<=r['draft_n_accepted']<=r['draft_n'],'candidate draft counters incomplete');draft.append((k,r['draft_n'],r['draft_n_accepted']))
 def group(band):
  z=[x for x in draft if bands[x[0][0]]==band];d=sum(x[1] for x in z);q=sum(x[2] for x in z);return {'requests':len(z),'drafted':d,'accepted':q,'acceptance_rate':None if d==0 else q/d}
 caps=lambda rows:{'denominator':len(rows),'hits':sum(bool(r['termination']['cap_hit']) for r in rows)}
 def caps_by_band(rows):return {band:caps([r for k,r in rows.items() if bands[k[0]]==band]) for band in ('2k','8k')}
 gates={'all_protocol_accepted':protocol_ok,'complete_exact_parity':not mismatch,'bootstrap_speedup_lower_95_ge_1_4':lb>=1.4,'candidate_p95_no_regression':cp<=bp,'draft_counters_complete':True}
 winner=candidate_mode if all(gates.values()) else None
 return {'schema':'sepalith.r2.spec-comparison.v2','status':'complete','profile_id':profile['id'],'candidate_mode':candidate_mode,'requests':len(keys),'mismatches':mismatch,'latency_ms':{'baseline_median':median(bw),'candidate_median':median(cw),'median_speedup':med,'bootstrap_iterations':5000,'bootstrap_seed':260914,'bootstrap_speedup_lower_95':lb,'baseline_p95':bp,'candidate_p95':cp},'cap_hits':{'baseline':{**caps(list(a.values())),'by_band':caps_by_band(a)},'candidate':{**caps(list(b.values())),'by_band':caps_by_band(b)}},'draft_acceptance':{'all':group('2k')|{} if False else {'requests':len(draft),'drafted':sum(x[1] for x in draft),'accepted':sum(x[2] for x in draft),'acceptance_rate':None if sum(x[1] for x in draft)==0 else sum(x[2] for x in draft)/sum(x[1] for x in draft)},'by_band':{'2k':group('2k'),'8k':group('8k')}},'gates':gates,'promotion_winner':winner,'promotion_status':'candidate_meets_all_gates' if winner else 'no_winner'}
def write_new(path,value):
 p=Path(path);require(not p.exists(),'fresh output required');p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,p);d=os.open(p.parent,os.O_DIRECTORY);os.fsync(d);os.close(d)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);ap.add_argument('--profile',choices=('ngram','draft'),required=True);ap.add_argument('--baseline',required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--candidate-mode',choices=('ngram_mod','released_dspark','existing_trained_dspark'),required=True);ap.add_argument('--out',required=True);a=ap.parse_args();binding=load_json(a.binding);profile=binding['spec_profiles'][a.profile];result=compare(binding,sha(a.binding),profile,load_json(a.baseline),load_json(a.candidate),a.candidate_mode);write_new(a.out,result);print(json.dumps({'status':'complete','promotion_status':result['promotion_status']}))
if __name__=='__main__':main()
