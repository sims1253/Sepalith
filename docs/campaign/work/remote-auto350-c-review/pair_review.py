from pathlib import Path
import json,hashlib,re,collections
P=Path(__file__).resolve().parent;BASE=P.parent

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(P/n).write_text(json.dumps(x,indent=2)+'\n')
arms={};sources={}
for arm in (1500,350):
 p=BASE/f'remote-auto{arm}-c-review';r=load(p/'review.json');e=rows(p/'remote/harness-events.jsonl');g=rows(p/'desktop/gateway-events.jsonl');native=(p/'desktop/native.log').read_text();start={x['requestId']:x.get('method','')+' '+x['path'] for x in g if x['event']=='request_started'};disp=[x for x in g if x['event']=='backend_dispatched' and x.get('path')=='/completion'];body=collections.defaultdict(list)
 for x in disp:body[x['requestBodySha256']].append({'dispatch_sequence':x['dispatchSequence'],'request_id':x['requestId'],'bytes':x['requestBytes']})
 slot={}
 for line in native.splitlines():
  match=re.search(r'task (\d+) \| (.*)',line)
  if not match:continue
  task,data=match.groups();v=slot.setdefault(task,{'task_id':int(task)})
  for name,pattern,typ in [('prompt_tokens',r'task.n_tokens = (\d+)',int),('cached_prefix_tokens',r'cached n_tokens = (\d+)',int),('prompt_eval_ms',r'prompt eval time =\s*([\d.]+) ms',float),('evaluated_prompt_tokens',r'prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens',int),('generation_eval_ms',r'\s+eval time =\s*([\d.]+) ms',float),('native_total_ms',r'total time =\s*([\d.]+) ms',float)]:
   m=re.search(pattern,data)
   if m:v[name]=typ(m.group(1))
  if 'stop processing:' in data:v['released']=True
 slot=[x for x in slot.values() if 'prompt_tokens' in x]
 arms[str(arm)]={'frames':r['frame_count'],'dropped':r['dropped'],'controls':r['controls'],'control_cases':r['control_cases'],'key_count':r['keyboard']['valid'],'event_kind_counts':dict(collections.Counter(x['kind'] for x in e)),'gateway_client_requests_by_route':dict(collections.Counter(start.values())),'gateway_backend_dispatches_by_route':dict(collections.Counter(x.get('method','')+' '+x['path'] for x in g if x['event']=='backend_dispatched')),'gateway_client_cancellations_by_route':dict(collections.Counter(start.get(x['requestId'],'unknown') for x in g if x['event']=='request_cancelled')),'completion_dispatch_counts_by_case':{c['id']:c['completion_dispatch_count'] for c in r['cases']},'visible_events':[{'case':c['id'],'input_id':k['input_id'],'interval_ms':k['keypress_to_first_visible_interval_ms'],'before_configured_timer_deadline':k.get('before_current_timer_deadline'),'text':k.get('observed_ghost_text')} for c in r['cases'] for k in c['keys'] if k.get('keypress_to_first_visible_interval_ms')],'exact_repeated_completion_body_sha_groups':[{'sha256':h,'requests':a} for h,a in body.items() if len(a)>1],'native_tasks':slot,'native_task_count':len(slot),'native_prompt_eval_token_sum':sum(x.get('evaluated_prompt_tokens',0) for x in slot),'native_total_ms_sum':sum(x.get('native_total_ms',0) for x in slot),'last_frame_after_last_case_ms':r['last_frame_after_last_case_ms'],'buffer_outcomes':load(p/'buffer-parse.json')['rows'],'input_sequence':[(x['id'],x['input_id'],x['key'],x['before_version']) for x in e if x['kind']=='auto_input_requested'],'end_buffer_hashes':{x['id']:x['after_sha256'] for x in e if x['kind']=='auto_case_end'}}
 sources[str(arm)]={'review_sha256':sha(p/'review.json'),'native_log_sha256':sha(p/'desktop/native.log'),'gateway_sha256':sha(p/'desktop/gateway-events.jsonl'),'harness_events_sha256':sha(p/'remote/harness-events.jsonl')}
# Task specification is metadata; retain only RUN-04, not unrelated task content.
tasks=load(BASE.parent/'tasks.json')
def find(x):
 if isinstance(x,dict):
  if x.get('id')=='RUN-04':return x
  for y in x.values():
   z=find(y)
   if z:return z
 if isinstance(x,list):
  for y in x:
   z=find(y)
   if z:return z
spec=find(tasks);save('RUN-04-task-spec.json',{'source':'docs/campaign/tasks.json','source_sha256':sha(BASE.parent/'tasks.json'),'task':spec})
coverage=[{'class':'typing','coverage':'covered in bounded synthetic workload','evidence':'13 sequential trusted CDP keys per arm; matching document versions, hashes and actual text changes; four typing scenarios.','gap':'Not physical typing, representative workload, or broad latency distribution.'},{'class':'cursor movement','coverage':'incidental only; dedicated stratum absent','evidence':'18 selection events per arm include setup positioning and movement following typing/acceptance.','gap':'No isolated cursor-only move scenario or request/cache outcome tied to such a move.'},{'class':'history changes','coverage':'incidental edits only; controlled history stratum absent','evidence':'Document mutations are retained. The unchanged production provider may maintain history, but these traces expose no exact rendered history boundary/append/eviction proof.','gap':'No history-only mutation or event-stratified prompt/cache correctness test.'},{'class':'diagnostics refresh','coverage':'not exercised','evidence':'No diagnostic change case or diagnostics refresh event in retained harness traces. Renderer diagnostic DOM metadata is unrelated to code diagnostics.','gap':'No diagnostic-triggered context/recompute/correctness evidence.'},{'class':'anchor moves','coverage':'not exercised','evidence':'Short expression fixtures and no explicit truncation/context-anchor mutation scenario.','gap':'No proof that anchor movement rebuilds required context without stale source.'},{'class':'file switch','coverage':'covered narrowly','evidence':'R fixture switched to plaintext target after pending typing; zero gateway completion dispatch delta and no observed ghost after switch in each arm.','gap':'No dispatched in-flight request at switch; no R-to-R context switch or long-context anchor interaction.'}]
a=BASE/'lead/remote-auto1500-c/notebook-capsule';b=BASE/'lead/remote-auto350-c/notebook-capsule';checked=['observe_renderer.mjs','startup_gate.mjs','analyze_auto.mjs','analyze_renderer.mjs','primary-editor-harness/acceptance-v2.js','primary-editor-harness/extension.js','candidate.vsix'];eq={n:{'sha256':sha(a/n),'equal':sha(a/n)==sha(b/n)} for n in checked}
result={'status':'bounded_paired_observations_only_root_admission_required','sources':sources,'arms':arms,'same_inputs':arms['1500']['input_sequence']==arms['350']['input_sequence'],'same_end_buffer_hashes':arms['1500']['end_buffer_hashes']==arms['350']['end_buffer_hashes'],'source_equalities':eq,'task_spec_event_coverage':coverage,'limits':['One sequential arm pair, no randomized order, repetition or representative p95.','No debounce causality: production Automatic requests and explicit debounce timers can both act; trigger-kind telemetry is absent.','Native prefix/evaluation token counts and timing are aggregates, not editor-side cache/index timings and not per-event causal effects.','Repeated body SHA proves repeated dispatched HTTP body bytes only; does not prove a needless request or identical response.','Gateway HTTP transport cancellation/release does not prove native task cancellation or saved work.','Exact request/publication correlation remains tentative; notebook renderer and desktop native clocks are not subtracted.','No task-spec completion: independent cursor/history/diagnostic/anchor strata and host-cache timing remain open.']}
save('paired-review.json',result);print(json.dumps({'same_inputs':result['same_inputs'],'same_end_buffers':result['same_end_buffer_hashes'],'source_equalities':all(x['equal'] for x in eq.values()),'arms':{k:{n:v[n] for n in ['native_task_count','native_prompt_eval_token_sum','native_total_ms_sum','exact_repeated_completion_body_sha_groups','last_frame_after_last_case_ms']} for k,v in arms.items()}},indent=2))
