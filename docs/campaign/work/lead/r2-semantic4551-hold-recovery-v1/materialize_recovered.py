#!/usr/bin/env python3
import collections, hashlib, importlib.util, json, os, sys
from pathlib import Path

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
BASE=PLAN/'docs/campaign/work/lead/r2-semantic763-context-admission-v1/source/base_materializer.py'
INTEGRATE=PLAN/'docs/campaign/work/lead/r2-semantic763-dedup-integration-v1/integrate.py'
PREVIOUS=PLAN/'docs/campaign/work/lead/r2-semantic4551-provider-materialization-v1/materialize4551.py'
E=Path('/mnt/e/sepalith/campaign-20260915/data-work')
RID='407f72b3ac171a7e9f5e8e5f';RESERVE=2048
PINS={'render':'e8cb5030d2b7d8e6f9746aad7f129d8fe7a0a55069a7a48fa29dfd2a7aa63e60','input':'2d7fe0508568a9c27bb0e9ac26112dd308b87f3f1051b1b25e6d982ca7b98228','sidecar':'e31cc3c910de6fb426a6bda8c35382cc4dd50514c4ea5fd196513943cd0d1da6','tokenizer':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81','rows4435':'4c3d7898c803d565a001c14f56bab24a1ad24019d1bfb4f0e79e644a516133cc','prov4435':'138f7de823e23442ca79a3e9016718e2d9c791d7b875e094f16eafdbbedf9acd'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
B=loadmod('recover_base',BASE);I=loadmod('recover_integrate',INTEGRATE);M=loadmod('recover_previous',PREVIOUS)
def one(path):
 xs=[json.loads(l) for l in Path(path).open() if json.loads(l).get('row_id')==RID]
 if len(xs)!=1:raise ValueError('exact row missing/duplicate')
 return xs[0]
def main():
 root=E/'Semantic4551-hold-recovery-v1';render=root/'render-2048.jsonl';inp=root/'prediction-input.jsonl';side=E/'Semantic4554-root-preparation-v1/combined/training-sidecar.jsonl';tok=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
 if sha(inp)!=PINS['input'] or sha(render)!=PINS['render'] or sha(side)!=PINS['sidecar'] or sha(tok)!=PINS['tokenizer']:raise ValueError('input pin')
 r=json.loads(render.read_text());b=json.loads(inp.read_text());s=one(side);assert r['row_id']==b['row_id']==s['row_id']==RID and r['status']=='supported' and r['generation_reserve']==RESERVE and r['selection_target_or_gold_used'] is False
 tokenizer=B.TokenizerAdapter(tok);ctx=B.PROTOCOL.PromptContext.from_mapping(r['selected_context']);prompt=B.PROTOCOL.render_prompt(ctx);assert I.digest_text(prompt)==r['prompt_sha256'] and len(tokenizer.encode(prompt))==r['prompt_tokens']
 row=B.PROTOCOL.build_training_row(ctx,operation='replace',region_new=s['target_lines'],tokenizer=tokenizer,row_id=RID,family='roxygen_drafting',package_id=s['identity']['package_id'],split='train');assert not B.STRICT.validate_token_row(row,full_text=True);assert row['target_token_count']==1675 and row['target_token_count']<=RESERVE
 applied=B._apply_global(b['preedit_text'],ctx,s['target_lines']);assert applied is not None and I.digest_text(applied)==s['postedit_source_sha256'];assert row['target_body_text'] not in row['prompt_text'] and row['target_body_text'] not in b['preedit_text']
 existing,by_pair,by_prompt,by_source_target,by_path_target=I.load_existing(E/'DAT10-finish-source-repair-v3/train-token-rows.jsonl',E/'RL11-15006-context-v1/context-sidecar.jsonl')
 M.augment(E/'Semantic763-dedup-integration-v1/final-02/candidate-tokenrows.jsonl',E/'Semantic763-dedup-integration-v1/final-02/candidate-provenance.jsonl',M.PINS['rows616'],M.PINS['prov616'],existing,by_pair,by_prompt,by_source_target,by_path_target,tokenizer)
 M.augment(E/'Semantic-provider-integration-v1/final-01/candidate-tokenrows.jsonl',E/'Semantic-provider-integration-v1/final-01/candidate-provenance.jsonl',M.PINS['rows133'],M.PINS['prov133'],existing,by_pair,by_prompt,by_source_target,by_path_target,tokenizer)
 M.augment(E/'Semantic4551-provider-materialization-v1/final-01/candidate-tokenrows.jsonl',E/'Semantic4551-provider-materialization-v1/final-01/candidate-provenance.jsonl',PINS['rows4435'],PINS['prov4435'],existing,by_pair,by_prompt,by_source_target,by_path_target,tokenizer);assert len(existing)==20190
 pair=I.pair_key(row);promptkey=I.digest_text(row['prompt_text']);target=I.target_key(row);assert RID not in existing and pair not in by_pair and not [old for other,old in by_prompt.get(promptkey,set()) if other!=target]
 identity=s['identity'];assert not set().union(*(by_source_target.get((h,target),set()) for h in I.source_hashes(identity)|{identity['source_sha256']}))
 out=root/'final-01';out.mkdir();prov={'row_id':RID,'source_identity':identity,'mode':r['mode'],'context_size':r['context_size'],'generation_reserve':RESERVE,'prompt_sha256':r['prompt_sha256'],'preedit_sha256':b['preedit_sha256'],'postedit_source_sha256':s['postedit_source_sha256'],'target_truncated':False,'selection_target_or_gold_used':False}
 for name,value in [('candidate-tokenrows.jsonl',row),('candidate-provenance.jsonl',prov)]:
  with (out/name).open('x') as f:f.write(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
 manifest={'schema':'sepalith.dat10.semantic4551.hold_recovery.v1','status':'complete_review_only_root_admission_required','row_id':RID,'candidate_rows':1,'fixed_prediction_profile':{'context_size':16384,'generation_reserve':RESERVE,'prompt_tokens':r['prompt_tokens'],'total_reserved_tokens':1+r['prompt_tokens']+RESERVE,'selection_target_or_gold_used':False},'target':{'token_count':row['target_token_count'],'minimum_exact_fixed_reserve':row['target_token_count'],'truncated':False},'dedup_denominator':20190,'outputs':{name:{'sha256':sha(out/name),'bytes':(out/name).stat().st_size,'rows':1} for name in ('candidate-tokenrows.jsonl','candidate-provenance.jsonl')},'checks':{'prompt_rerender_exact':True,'strict_protocol':True,'source_reapplication':True,'dedup_against_20190':True},'training_admission':False}
 B.atomic_json(out/'manifest.json',manifest);print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__':main()
