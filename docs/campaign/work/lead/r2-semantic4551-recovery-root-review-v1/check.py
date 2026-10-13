import hashlib,json,importlib.util,sys,datetime
from pathlib import Path
P=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb'); E=Path('/mnt/e/sepalith/campaign-20260915/data-work'); R=E/'Semantic4551-hold-recovery-v1'; rid='407f72b3ac171a7e9f5e8e5f'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
bp=P/'docs/campaign/work/lead/r2-semantic763-context-admission-v1/source/base_materializer.py';spec=importlib.util.spec_from_file_location('root_recovery_base',bp);B=importlib.util.module_from_spec(spec);sys.modules[spec.name]=B;spec.loader.exec_module(B)
row=json.loads((R/'final-01/candidate-tokenrows.jsonl').read_text());before=json.loads((R/'prediction-input.jsonl').read_text());render=json.loads((R/'render-2048.jsonl').read_text());side=[json.loads(l) for l in (E/'Semantic4554-root-preparation-v1/combined/training-sidecar.jsonl').open() if json.loads(l)['row_id']==rid];assert len(side)==1;side=side[0]
ctx=B.PROTOCOL.PromptContext.from_mapping(render['selected_context']);applied=B._apply_global(before['preedit_text'],ctx,side['target_lines']);assert hashlib.sha256(applied.encode()).hexdigest()==side['postedit_source_sha256'];assert B.PROTOCOL.render_prompt(ctx)==row['prompt_text'];assert row['target_body_text'] not in before['preedit_text']
counts={};matches=[]
for name,file in [('original15006','DAT10-finish-source-repair-v3/train-token-rows.jsonl'),('candidate616','Semantic763-dedup-integration-v1/final-02/candidate-tokenrows.jsonl'),('candidate133','Semantic-provider-integration-v1/final-01/candidate-tokenrows.jsonl'),('candidate4435','Semantic4551-provider-materialization-v1/final-01/candidate-tokenrows.jsonl')]:
 n=0;h=hashlib.sha256()
 for line in (E/file).open('rb'):
  h.update(line);r=json.loads(line);n+=1
  if r['id']==row['id'] or r['prompt_text']==row['prompt_text'] or r['target_body_text']==row['target_body_text']:matches.append({'source':name,'id':r['id'],'same_prompt':r['prompt_text']==row['prompt_text'],'same_target':r['target_body_text']==row['target_body_text']})
 counts[name]={'rows':n,'sha256':h.hexdigest()}
assert [x['rows'] for x in counts.values()]==[15006,616,133,4435];assert not matches,matches
out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'row_id':rid,'independent_full_source_reapplication':True,'exact_prompt_rerender':True,'existing_pool':counts,'rows_checked':20190,'same_id_prompt_or_target_matches':matches,'candidate_sha256':sha(R/'final-01/candidate-tokenrows.jsonl'),'training_admitted':False,'remaining':'Independent full token/sequence audit and stage admission'}
Path(__file__).with_name('result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
