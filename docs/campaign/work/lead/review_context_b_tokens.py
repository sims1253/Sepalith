import hashlib, importlib.util, json, statistics
from pathlib import Path
P=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
R=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training')
s=P/'work/theta0-q8-cuda-dev-review/review_theta0_q8_cuda_dev.py'
assert hashlib.sha256(s.read_bytes()).hexdigest()=='6f69669f77fb02fdc85024b8256ad72c53067d8f07bbe163b0dc13501fa9af7d'
sp=importlib.util.spec_from_file_location('review',s); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
t,tokenizer_audit=m.load_tokenizer()
results=[]; signatures={}; profiles={}
for run in ('a','b'):
 for context in (2048,4096,8192):
  p=R/f'RUN-05-theta0-cuda-context-{run}'/str(context)
  trace=json.loads((p/'trace.json').read_text());assert len(trace['rows'])==9
  assert trace['serverProps']['default_generation_settings']['n_ctx']==context
  assert trace['deadlineMs']==5000 and trace['nativeProfile']['maxOutputTokens']==192
  assert trace['denominator']['qualityEligibleFreshResponses']==8
  latencies=[];tokens=[]
  for row in trace['rows']:
   calls=row['transportCalls'];a=next(x for x in calls if x['path']=='/tokenize');c=next(x for x in calls if x['path']=='/completion')
   assert a['status']==200 and c['status']==200
   prompt=a['body']['content'];ids=[0]+t.encode(prompt,add_special_tokens=False,split_special_tokens=True)
   assert ids==c['body']['prompt']==row['promptTokenIds']==[0]+a['response']['tokens']
   assert c['body']['n_predict']==192 and c['body']['temperature']==0 and c['body']['cache_prompt'] is True
   response=c['response'];g=response['tokens'];assert g==row['generatedTokenIds']
   assert g[-1]==1 and response['stop_type']=='eos' and response['truncated'] is False
   decoded=t.decode(g[:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
   assert decoded==response['content']
   assert row['parserStatus']=='accepted' and row['responseApplicability']=='fresh'
   assert not row['timeout'] and not row['cancelled'] and row['elapsedMs']<5000
   assert decoded.endswith('>>>>>>> UPDATED')
   sig=(prompt,ids,g,decoded,row['operation'],row['planHash'])
   key=row['eventId']
   if key in signatures:assert signatures[key]==sig, (run,context,key)
   else:signatures[key]=sig
   tokens.append(len(ids))
   if row['qualityDenominatorEligible']:latencies.append(row['elapsedMs'])
  log=(p/'server.log').read_text()
  assert 'offloaded 43/43 layers to GPU' in log
  term=json.loads((p/'terminal.json').read_text())
  if run=='b':
   assert term['server']['exit_code']==term['client']['exit_code']==0
   assert not any(x['exists'] for x in term['server']['children'])
   assert not any(s in log for s in ['Received second interrupt','dumped core','terminate called without'])
  results.append({'run':run,'context':context,'raw_rows':9,'fresh_nonrepeat_rows':len(latencies),'repeat_controls':1,'prompt_tokens_min':min(tokens),'prompt_tokens_max':max(tokens),'fresh_elapsed_ms':latencies,'median_ms':statistics.median(latencies),'min_ms':min(latencies),'max_ms':max(latencies),'server_exit':term['server']['exit_code'],'trace_sha256':hashlib.sha256((p/'trace.json').read_bytes()).hexdigest()})
o={'rows_verified':54,'exact_all_six_cells_prompt_token_output_plan_parity':True,'tokenizer_audit':tokenizer_audit,'cells':results,'limits':['Short synthetic fixture only,148-300 prompt tokens including BOS depending on event.','No independent semantic correctness or useful-suggestion quality estimate.','2K/4K/8K native allocations;8K does not establish long-prompt support.','Single ordered pass per cell; no stable p95 or first-visible editor latency claim.','Run-a shutdown failed; run-b corrected teardown is clean.']}
f=P/'work/lead/theta0-context-b-root-token-review.json';f.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'cells':results}))
