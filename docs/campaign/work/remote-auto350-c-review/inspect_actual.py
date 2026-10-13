from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;R=P/'remote'
def load(n):return json.loads((R/n).read_text())
def rows(n):return [json.loads(s) for s in (R/n).read_text().splitlines() if s.strip()]
f=rows('renderer-frames.jsonl');ev=rows('harness-events.jsonl');inputs=rows('renderer-inputs.jsonl');obs=rows('gateway-observations.jsonl');analysis=load('auto-analysis.json');observer=load('renderer-observer.json')
def event(k,id=None):return next(x for x in ev if x['kind']==k and (id is None or x.get('id')==id))
start=event('case_trigger','control-multiline')['epoch_ms'];end=event('case_commit','control-multiline')['epoch_ms'];mf=[x for x in f if start<=x['epoch_ms']<=end]
expected=['OBSERVER_MULTI_FIRST','OBSERVER_MULTI_SECOND','OBSERVER_MULTI_THIRD'];full=[];geometry=[];ordinary=[]
for frame in mf:
 seen=set()
 for g in frame['ghosts']:
  if g.get('geometry',{}).get('visible'):seen.add(g['text'])
  for line in (g.get('view_zone') or {}).get('lines',[]):
   geom=line.get('geometry',{})
   if geom.get('visible') and line.get('visible') and not line.get('occluded'):seen.add(line['text'])
   if geom.get('measurement')=='text_node_ranges':
    geometry.append({'text':line['text'],'geometry':geom})
    if geom.get('ownership',{}).get('kind')!='renderer_ghost_continuation' or geom.get('ownership',{}).get('owner_class')!='suggest-preview-text':ordinary.append(line)
 if frame['focused'] and frame['visibility']=='visible' and all(any(e in t for t in seen) for e in expected):full.append(frame)
requested=[x for x in ev if x['kind']=='auto_input_requested'];keyboard=[]
for req in requested:
 captured=[x for x in inputs if x['input_id']==req['input_id']];applied=[x for x in ev if x['kind']=='auto_input_applied' and x['input_id']==req['input_id']];startcase=event('auto_case_start',req['id']);a=applied[0] if len(applied)==1 else {};changes=[x for x in ev if x['kind']=='text_change' and x['path']==startcase['path'] and x['version']==a.get('version') and x.get('content_sha256')==a.get('after_sha256') and x.get('changes')]
 ok=len(captured)==1 and len(applied)==1 and captured[0]['trusted'] and captured[0]['focused'] and captured[0]['key']==req['key'] and a['version']==req['before_version']+1 and bool(changes)
 keyboard.append({'input_id':req['input_id'],'valid':bool(ok),'capture_count':len(captured),'document_change_evidence':bool(changes)})
gaps=[{'after_epoch_ms':a['epoch_ms'],'before_epoch_ms':b['epoch_ms'],'gap_ms':b['monotonic_ms']-a['monotonic_ms']} for a,b in zip(f,f[1:]) if not 0<=b['monotonic_ms']-a['monotonic_ms']<=100]
unique_geometry={json.dumps(x,sort_keys=True):x for x in geometry};lastcase=max(x['epoch_ms'] for x in ev if x['kind'] in ['case_end','auto_case_end']);identity=event('remote_extension_ready_dom')['identity'];settings=event('application_settings_observed')
report={'status':'independent_review_pending_image','controls':{k:analysis[k] for k in ['observer_positive_control','observer_multiline_control','observer_cancellation_control','visibility_evidence_admissible']},'control_cases':analysis['control_cases'],'frame_count':len(f),'dropped':observer.get('dropped'),'all_recorded_frames_focused_visible':all(x['focused'] and x['visibility']=='visible' for x in f),'gaps_over100ms':gaps,'last_frame_after_last_case_ms':f[-1]['epoch_ms']-lastcase,'multiline':{'frames_trigger_to_commit':len(mf),'all_three_visible_frames':len(full),'first_all_three_frame':full[0] if full else None,'unique_line_geometry':list(unique_geometry.values()),'invalid_owned_line_metadata_count':len(ordinary),'style_scope':'Pinned observer evaluates each text-node parent and ancestors; per-ancestor raw CSS values are not retained. Text-range visible=true requires style checks pass.','ordinary_document_lines':'Excluded by exact ghost continuation ownership selector; diagnostic siblings remain unscored.'},'keyboard':{'requested':len(requested),'captured':len(inputs),'valid':sum(x['valid'] for x in keyboard),'details':keyboard,'scope':'Synthetic trusted CDP renderer key events, not physical keyboard latency.'},'cases':analysis['cases'],'gateway_observation_rows':len(obs),'gateway_observation_errors':sum(bool(x.get('error')) for x in obs),'observer_terminal':observer,'guard_terminal':json.loads((P/'supervision/terminal.json').read_text()),'runtime_identity':identity,'settings':settings,'native_request_attribution':'Not established by renderer; root owns gateway/native/output identity review.','time_scope':'Notebook renderer monotonic and epoch milliseconds; gateway dispatch bounds use local notebook polling send/receive only. No cross-host clock subtraction.'}
(P/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'controls':report['controls'],'multiline_full_frames':len(full),'geometry_rows':len(unique_geometry),'keyboard_valid':report['keyboard']['valid'],'frames':len(f),'gaps':gaps,'observer_status':observer.get('status'),'cases':[{'id':x['id'],'dispatches':x.get('completion_dispatch_count'),'key_statuses':[k['status'] for k in x.get('keys',[])],'key_coverage':[k['coverage'] for k in x.get('keys',[])]} for x in analysis['cases']]},indent=2))
