from pathlib import Path
import json,hashlib,importlib.util,statistics,datetime
C=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
R=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-05-theta0-long-context-a')
p=C/'work/theta0-q8-cuda-dev-review/review_theta0_q8_cuda_dev.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='6f69669f77fb02fdc85024b8256ad72c53067d8f07bbe163b0dc13501fa9af7d'
s=importlib.util.spec_from_file_location('prior_review',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);t,tok=m.load_tokenizer()
f=json.loads((C/'work/serving-transition-long-panel/long-transition-fixture.json').read_text());events={e['eventId']:e for e in f['events']}
rows=[];identity={}
for context in [2048,4096,8192]:
 d=R/str(context);o=json.loads((d/'trace.json').read_text());assert len(o['rows'])==6
 assert o['serverProps']['default_generation_settings']['n_ctx']==context
 term=json.loads((d/'terminal.json').read_text());assert term['server']['exit_code']==term['client']['exit_code']==0
 assert 'offloaded 43/43 layers to GPU' in (d/'server.log').read_text()
 for envfile in d.glob('server-env-*.json'):assert json.loads(envfile.read_text())==['GGML_CUDA_GRAPH_OPT=0']
 for r in o['rows']:
  e=events[r['eventId']];prompt=e['after']['promptText'];hf=[0]+t.encode(prompt,add_special_tokens=False,split_special_tokens=True)
  token_call=next(c for c in r['transportCalls'] if c['path']=='/tokenize');native=[0]+token_call['response']['tokens'];assert hf==native
  completions=[c for c in r['transportCalls'] if c['path']=='/completion']
  base={'context':context,'event':r['eventId'],'prompt_tokens':len(hf),'elapsed_ms':r['elapsedMs'],'duplicate_control':r['duplicateWarmControl']}
  if context==2048:
   assert len(hf)+192>context and r['contextOverflowRejected'] and not completions and not r['timeout']
   base['status']='explicit_input_budget_rejection_no_generation'
  else:
   assert len(completions)==1;c=completions[0];assert c['body']['prompt']==hf and c['body']['n_predict']==192
   response=c['response'];ids=response['tokens'];assert ids[-1]==1 and response['stop_type']=='eos' and not response['truncated']
   assert r['parserStatus']=='accepted' and r['responseApplicability']=='fresh' and not r['timeout']
   text=t.decode(ids[:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False);assert text==response['content']
   sig=(hf,ids,text,r['operation'],r['planHash'])
   if r['eventId'] in identity:assert sig==identity[r['eventId']]
   else:identity[r['eventId']]=sig
   base.update(status='fresh_response',output_tokens=len(ids),operation=r['operation'],raw_text_sha256=hashlib.sha256(text.encode()).hexdigest())
  rows.append(base)
metrics={}
for context in [4096,8192]:
 times=[r['elapsed_ms'] for r in rows if r['context']==context and not r['duplicate_control']]
 metrics[str(context)]={'fresh_rows':len(times),'median_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times)}
o={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'RUN-05','tokenizer':tok,'rows':rows,'metrics':metrics,'context2048_input_rejections':6,'context2048_generated_responses':0,'actual_4k_8k_output_and_plan_parity':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'run_seconds':json.loads((R/'terminal.json').read_text())['seconds'],'limits':['The stored protocolRejected count for2K includes input budget rejections; it is not six malformed model outputs.','5 fresh events and1 duplicate control per profile. No stable p95 or model-quality estimate.','Same1927-1975-token prompts at4K/8K; this is not true8K-token stress.','Native request latency only; no editor-visible measurement.']}
p=C/'work/lead/theta0-long-context-a/root-review.json';p.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps({'metrics':metrics,'root_review_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'counts':{'budget_rejections':6,'responses':12}}))
