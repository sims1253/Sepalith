#!/usr/bin/env python3
"""CPU-only independent review of the completed DAT-08 v3-a DEV rehearsal."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
RUN = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/DAT-08-native-dev-rehearsal-v3-a')
HIST = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-05-theta0-cuda-dev192-a/Q8_0-b256')
PANEL = PLAN/'docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl'
PROTOCOL = EXEC/'packages/sepalith/src/sepalith/campaign_protocol.py'
TOKENIZER = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0')
ACCEPTED_HELPER = PLAN/'docs/campaign/work/final-binding-integration-v2/final_quality_evaluator.py'
REVIEW_REFERENCE = PLAN/'docs/campaign/work/theta0-q8-cuda-dev-review/review_theta0_q8_cuda_dev.py'
EXPECTED = {
    'model_sha256':'22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559',
    'server_sha256':'e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee',
    'panel_sha256':'7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035',
    'protocol_sha256':'5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156',
    'tokenizer_json_sha256':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
    'tokenizer_config_sha256':'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b',
    'tokenizer_revision':'8dc5f6055b90fe4b9422340810b270b9569f37f3',
    'profile_sha256':'15f8e5a9f018cf3f2e7bdd80ea24bde331a53136e329e872af39ac1cd131a2d4',
    'cap':192,'context':4096,'eos':1,'native_eog':[1,130073],
}

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def sha256_text(value: str) -> str: return hashlib.sha256(value.encode('utf-8')).hexdigest()
def canonical(value: object) -> bytes:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def canonical_sha(value: object) -> str: return hashlib.sha256(canonical(value)).hexdigest()

def load_protocol():
    if sha256_file(PROTOCOL)!=EXPECTED['protocol_sha256']: raise RuntimeError('protocol pin mismatch')
    sys.path.insert(0,str(PROTOCOL.parent.parent))
    import sepalith.campaign_protocol as protocol
    if Path(protocol.__file__).resolve()!=PROTOCOL.resolve(): raise RuntimeError('protocol module path mismatch')
    return protocol

def load_panel():
    if sha256_file(PANEL)!=EXPECTED['panel_sha256']: raise RuntimeError('panel pin mismatch')
    rows=[];by_id={}
    for line_number,line in enumerate(PANEL.read_text().splitlines(),1):
        row=json.loads(line)
        if row.get('split')!='dev' or not isinstance(row.get('id'),str): raise RuntimeError(f'bad panel row {line_number}')
        if row['id'] in by_id: raise RuntimeError('duplicate panel id')
        rows.append(row);by_id[row['id']]=row
    if len(rows)!=75: raise RuntimeError(f'panel rows {len(rows)}')
    return rows,by_id

def load_tokenizer():
    import transformers
    tj=TOKENIZER/'tokenizer.json';tc=TOKENIZER/'tokenizer_config.json'
    if sha256_file(tj)!=EXPECTED['tokenizer_json_sha256']: raise RuntimeError('tokenizer json pin mismatch')
    if sha256_file(tc)!=EXPECTED['tokenizer_config_sha256']: raise RuntimeError('tokenizer config pin mismatch')
    tok=transformers.AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True,use_fast=True,trust_remote_code=False)
    ident={'vocab_size':len(tok),'bos_token_id':getattr(tok,'bos_token_id',None),'eos_token_id':getattr(tok,'eos_token_id',None),'pad_token_id':getattr(tok,'pad_token_id',None)}
    expected={'vocab_size':130560,'bos_token_id':0,'eos_token_id':1,'pad_token_id':1}
    if ident!=expected: raise RuntimeError(f'tokenizer identity {ident}')
    vocab=tok.get_vocab()
    return tok,{'path':str(TOKENIZER),'transformers_version':transformers.__version__,'files':{'tokenizer.json':sha256_file(tj),'tokenizer_config.json':sha256_file(tc)},'revision':EXPECTED['tokenizer_revision'],'identity':ident,'vocab_sha256':canonical_sha(sorted((str(k),int(v)) for k,v in vocab.items())),'vocab_size':len(vocab)}

def prepare_cases(protocol,tokenizer,rows):
    out=[]; checks=[]
    for row in rows:
        context=protocol.PromptContext.from_mapping(row['context'])
        prompt=protocol.render_prompt(context)
        ids=list(protocol.encode_prompt(context,tokenizer,include_bos=True))
        expected={'id':row['id'],'prompt_sha256':sha256_text(prompt),'target_sha256':sha256_text(canonical(row['region_new']).decode('utf-8')),'hf_prompt_ids':ids,'hf_prompt_ids_sha256':canonical_sha(ids),'hf_prompt_tokens_with_bos':len(ids)}
        checks.append(expected)
    return checks

def review_case(protocol,tokenizer,case,row,expected):
    issues=[]
    def eq(label,actual,want):
        if actual!=want: issues.append(f'{label}:{actual!r}!={want!r}')
    eq('id',case.get('id'),row['id']);eq('operation_label',case.get('operation_label'),row['operation']);eq('expected_noop',case.get('expected_noop'),row['operation']=='no_op')
    eq('prompt_sha256',case.get('prompt_sha256'),expected['prompt_sha256']);eq('target_sha256',case.get('target_sha256'),expected['target_sha256'])
    eq('hf_prompt_ids',case.get('hf_prompt_ids'),expected['hf_prompt_ids']);eq('hf_prompt_ids_sha256',case.get('hf_prompt_ids_sha256'),expected['hf_prompt_ids_sha256']);eq('hf_prompt_tokens_with_bos',case.get('hf_prompt_tokens_with_bos'),expected['hf_prompt_tokens_with_bos'])
    eq('source_provenance',case.get('source_provenance'),row.get('source_provenance'))
    ids=case.get('returned_token_ids')
    if not isinstance(ids,list) or any(type(x) is not int for x in ids): issues.append('returned_token_ids_not_integer_list');ids=[]
    raw=case.get('raw_text')
    if not isinstance(raw,str): issues.append('raw_text_not_string');raw=''
    eq('raw_text_sha256',case.get('raw_text_sha256'),sha256_text(raw));eq('returned_token_ids_sha256',case.get('returned_token_ids_sha256'),canonical_sha(ids))
    eq('response_received',case.get('response_received'),True);eq('response_complete',case.get('response_complete'),True)
    protocol_saved=case.get('protocol') if isinstance(case.get('protocol'),Mapping) else {}
    timing=case.get('server_timing') if isinstance(case.get('server_timing'),Mapping) else {}
    terminal=ids[-1] if ids else None
    canonical_tokens=bool(ids) and protocol.valid_generation_tokens(ids)
    body_ids=ids[:-1] if terminal==EXPECTED['eos'] else ids
    try:
        decoded=tokenizer.decode(body_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        wire_match=(decoded==raw)
        decoded_sha=sha256_text(decoded)
    except Exception:
        wire_match=False;decoded_sha=None
    parsed=protocol.parse_output(raw,protocol.PromptContext.from_mapping(row['context']))
    final_sse=protocol_saved.get('final_sse_tokens');malformed=protocol_saved.get('malformed_sse_frames')
    protocol_valid=bool(canonical_tokens and wire_match and len(ids)<=EXPECTED['cap'] and protocol_saved.get('saw_stop') is True and final_sse==0 and not malformed and timing.get('stop_type')=='eos' and parsed.status=='accepted')
    cap_hit=len(ids)>EXPECTED['cap'] or (terminal!=EXPECTED['eos'] and (timing.get('stop_type') in {'limit','length'} or timing.get('truncated') is True or (isinstance(timing.get('tokens_predicted'),int) and timing['tokens_predicted']>=EXPECTED['cap'])))
    if parsed.status=='accepted' and parsed.operation=='no_op': actual_region=list(protocol.PromptContext.from_mapping(row['context']).region_old)
    elif parsed.status=='accepted': actual_region=list(parsed.body)
    else: actual_region=[]
    exact=bool(protocol_valid and actual_region==list(row.get('region_new',[])))
    predicted_noop=bool(protocol_valid and parsed.operation=='no_op')
    eq('saved_protocol_valid',bool(case.get('quality',{}).get('protocol_valid')),protocol_valid)
    eq('saved_exact',bool(case.get('quality',{}).get('exact_edit')),exact)
    expected_noop_correct=bool(protocol_valid and row['operation']=='no_op' and predicted_noop)
    eq('saved_noop_correct',bool(case.get('quality',{}).get('strict_noop_correct')),expected_noop_correct)
    eq('saved_noop_false_positive',bool(case.get('quality',{}).get('noop_false_positive')),bool(protocol_valid and parsed.operation!='no_op' and row['operation']=='no_op'))
    eq('saved_cap_limit',case.get('cap',{}).get('limit'),EXPECTED['cap'])
    eq('saved_saw_stop',protocol_saved.get('saw_stop'),True);eq('saved_protocol_stop_type',protocol_saved.get('stop_type'),timing.get('stop_type'))
    return {'id':row['id'],'family':row.get('family'),'operation':row['operation'],'status':case.get('status'),'protocol_valid':protocol_valid,'exact':exact,'predicted_noop':predicted_noop,'cap_hit':bool(cap_hit),'returned_tokens':len(ids),'terminal_id':terminal,'stop_type':timing.get('stop_type'),'raw_text_sha256':sha256_text(raw),'returned_token_ids_sha256':canonical_sha(ids),'decoded_body_sha256':decoded_sha,'wire_hf_text_match_recomputed':wire_match,'hash_integrity_ok':not any(issue.startswith(('raw_text_sha256','returned_token_ids_sha256')) for issue in issues),'prompt_identity_ok':not any(issue.startswith(('prompt_sha256','target_sha256','hf_prompt_')) for issue in issues),'source_provenance_ok':not any(issue.startswith('source_provenance') for issue in issues),'issues':issues}

def percentile(xs,p):
    if not xs:return None
    xs=sorted(xs);i=(len(xs)-1)*p;lo=int(i);hi=min(len(xs)-1,lo+1);return xs[lo]+(xs[hi]-xs[lo])*(i-lo)

def compact_binding(run_data):
    adm=json.loads((run_data/'native-admission.json').read_text());before=json.loads((run_data/'dev-results/native-before.json').read_text());after=json.loads((run_data/'dev-results/native-after.json').read_text())
    props=before.get('props',{});generation=props.get('default_generation_settings',{})
    return {
      'admission_sha256':canonical_sha(adm),'admission_file_sha256':sha256_file(run_data/'native-admission.json'),
      'status':adm.get('status'),'purpose':adm.get('purpose'),'fresh_process':adm.get('fresh_process'),'exclusive_owner':adm.get('exclusive_owner'),'no_other_clients':adm.get('no_other_clients'),'root_verified_model_before_launch':adm.get('root_verified_model_before_launch'),'cuda_backend_observed':adm.get('cuda_backend_observed'),'all_layers_offloaded':adm.get('all_layers_offloaded'),'profile_sha256':adm.get('profile_sha256'),'model_sha256':adm.get('model',{}).get('sha256'),'pid':adm.get('pid'),'start_tick':adm.get('start_tick'),'endpoint':adm.get('endpoint'),'mapped_code_files':len(adm.get('mapped_code_files',[])),'bundle_files':len(adm.get('bundle_files',[])),
      'before_after':{'file_hashes_equal':sha256_file(run_data/'dev-results/native-before.json')==sha256_file(run_data/'dev-results/native-after.json'),'admission_sha_equal':before.get('admission_sha256')==after.get('admission_sha256'),'pid_equal':before.get('pid')==after.get('pid'),'start_tick_equal':before.get('start_tick')==after.get('start_tick'),'full_hash_before':before.get('full_hash_verified'),'full_hash_after':after.get('full_hash_verified'),'cuda_graph_opt_before':before.get('cuda_graph_opt'),'cuda_graph_opt_after':after.get('cuda_graph_opt'),'mapped_files_before':len(before.get('mapped_files',[])),'mapped_files_after':len(after.get('mapped_files',[])),'mapped_paths_equal':{x.get('path') for x in before.get('mapped_files',[])}=={x.get('path') for x in after.get('mapped_files',[])},'model_path':props.get('model_path'),'n_ctx':generation.get('n_ctx') if isinstance(generation,Mapping) else None,'total_slots':props.get('total_slots'),'build_info':props.get('build_info')},
    }

def run_review():
    rows,by_id=load_panel();protocol=load_protocol();tokenizer,tokenizer_audit=load_tokenizer();expected=prepare_cases(protocol,tokenizer,rows)
    result=json.loads((RUN/'dev-results/results.json').read_text()); hist=json.loads((HIST/'quality.json').read_text())
    cases=result.get('results');hcases=hist.get('cases')
    if not isinstance(cases,list) or len(cases)!=75: raise RuntimeError('v3 result rows incomplete')
    if not isinstance(hcases,list) or len(hcases)!=75: raise RuntimeError('historical rows incomplete')
    ids=[r['id'] for r in rows];case_ids=[c.get('id') for c in cases];hist_ids=[c.get('id') for c in hcases]
    checks=[review_case(protocol,tokenizer,c,by_id[c['id']],e) for c,e in zip(cases,expected)] if case_ids==ids else []
    hmap={c.get('id'):c for c in hcases}
    if hist_ids!=ids: raise RuntimeError('historical case IDs/order mismatch')
    independent_counts=Counter()
    issues=[]
    for x in checks:
        independent_counts['protocol_rows']+=int(x['protocol_valid']);independent_counts['protocol_error_rows']+=int(not x['protocol_valid']);independent_counts['cap_hit_rows']+=int(x['cap_hit'])
        independent_counts['exact_region_rows']+=int(x['exact'])
        if x['operation']=='no_op':
            independent_counts['strict_noop_correct_rows']+=int(x['protocol_valid'] and x['predicted_noop'])
            independent_counts['noop_false_positive_rows']+=int(x['protocol_valid'] and not x['predicted_noop'])
        else: independent_counts['edit_exact_rows']+=int(x['exact'])
        issues.extend({'id':x['id'],'issues':x['issues']} for _ in [0] if x['issues'])
    independent_counts=dict(sorted(independent_counts.items()))
    saved=result['counts'];saved_counts={'edit_exact_rows':saved.get('edit_exact'),'strict_noop_correct_rows':saved.get('strict_noop_correct'),'noop_false_positive_rows':saved.get('false_suggestions'),'protocol_rows':saved.get('protocol_valid'),'cap_hit_rows':saved.get('cap_hit')}
    hist_saved=hist['denominators'];hist_counts={'edit_exact_rows':hist_saved.get('edit_exact_rows'),'strict_noop_correct_rows':hist_saved.get('strict_noop_correct_rows'),'noop_false_positive_rows':hist_saved.get('noop_false_positive_rows'),'protocol_rows':hist_saved.get('protocol_rows'),'cap_hit_rows':hist_saved.get('cap_hit_rows')}
    v3_vs_hist=[]
    for c in cases:
        h=hmap[c['id']]
        changed={k:(c.get(k),h.get(k)) for k in ['raw_text_sha256','returned_token_ids_sha256','status','response_complete'] if c.get(k)!=h.get(k)}
        if changed:v3_vs_hist.append({'id':c['id'],'changes':changed})
    timing=[float(c.get('timing',{}).get('wall_ms')) for c in cases if isinstance(c.get('timing',{}).get('wall_ms'),(int,float))]
    hist_timing=[float(c.get('timing',{}).get('wall_ms')) for c in hcases if isinstance(c.get('timing',{}).get('wall_ms'),(int,float))]
    source=json.loads((RUN/'source-closure.json').read_text());adm=json.loads((RUN/'native-admission.json').read_text());before=json.loads((RUN/'dev-results/native-before.json').read_text());after=json.loads((RUN/'dev-results/native-after.json').read_text());controller=json.loads((RUN/'controller-terminal.json').read_text());host=json.loads((RUN/'host-terminal.json').read_text());launch=json.loads((RUN/'server-launch.json').read_text());client_launch=json.loads((RUN/'client-launch.json').read_text());tokenizer_saved=json.loads((RUN/'dev-results/tokenizer.json').read_text());
    absent=[p for p in source.get('runtime_policy',{}).get('must_remain_absent',[]) if not Path(p).exists()]
    source_graph_sha=client_launch.get('argv',[])[client_launch.get('argv',[]).index('--source-sha256')+1] if '--source-sha256' in client_launch.get('argv',[]) else None
    gate_source_sha=result.get('gate',{}).get('source_closure_sha256')
    source_summary={'path':str(RUN/'source-closure.json'),'sha256':sha256_file(RUN/'source-closure.json'),'bytes':(RUN/'source-closure.json').stat().st_size,'graph_sha256':source_graph_sha,'gate_graph_sha256':gate_source_sha,'graph_sha_matches_gate':source_graph_sha==gate_source_sha,'schema':source.get('schema'),'status':source.get('status'),'files':len(source.get('files',{})),'roots':source.get('roots'),'unresolved':source.get('unresolved'),'root_launch_approval_sha256':source.get('root_launch_approval_sha256'),'must_remain_absent_count':len(source.get('runtime_policy',{}).get('must_remain_absent',[])),'must_remain_absent_all_absent':len(absent)==len(source.get('runtime_policy',{}).get('must_remain_absent',[]))}
    binding=compact_binding(RUN)
    server_log=(RUN/'server.log').read_text(errors='replace');offload=re.findall(r'offloaded (\d+/\d+) layers',server_log);device=re.findall(r'using device CUDA0 \(([^)]+)\)',server_log);model_load=re.findall(r"loading model '([^']+)'",server_log)
    run_artifacts={n:{'path':str(RUN/n),'bytes':(RUN/n).stat().st_size,'sha256':sha256_file(RUN/n)} for n in ['dev-results/results.json','dev-results/native-before.json','dev-results/native-after.json','dev-results/tokenizer.json','native-admission.json','native-controller-before.json','native-controller-after.json','source-closure.json','server-launch.json','server.log','controller-terminal.json','host-terminal.json','native-maps-after-warmup.txt']}
    return {
      'schema_version':'dat08.native-dev-rehearsal-v3-a-independent-review.v1','task':'DAT-08 independent actual DEV75 rehearsal review','status':'independent_review_complete',
      'scope':{'cpu_only_reviewer':True,'max_cpu_threads':1,'http_or_native_client_started_by_reviewer':False,'gpu_or_server_started_by_reviewer':False,'ssh_or_cloud_used':False,'source_or_state_modified':False,'final_data_read':False,'quality_promotion':False},
      'run':{'path':str(RUN),'terminal_status':controller.get('failure') is None and host.get('controller_returncode')==0,'controller_terminal':controller,'host_terminal':host,'artifacts':run_artifacts,'quality_status':result.get('status'),'quality_elapsed_seconds':result.get('elapsed_seconds'),'quality_claim':result.get('quality_acceptance_claim')},
      'binding':{'native':binding,'source_closure':source_summary,'tokenizer_saved':tokenizer_saved,'profile_expected_sha256':EXPECTED['profile_sha256'],'server_launch':{'path':str(RUN/'server-launch.json'),'sha256':sha256_file(RUN/'server-launch.json'),'argv':launch.get('argv'),'pid':launch.get('pid'),'start_tick':launch.get('start_tick')},'server_log_observations':{'offload_last':offload[-1] if offload else None,'cuda_device_last':device[-1] if device else None,'model_path_last':model_load[-1] if model_load else None}},
      'panel':{'path':str(PANEL),'sha256':EXPECTED['panel_sha256'],'rows':len(rows),'edit_rows':sum(row['operation']!='no_op' for row in rows),'strict_noop_rows':sum(row['operation']=='no_op' for row in rows),'ordered_ids_sha256':canonical_sha(ids),'v3_case_ids_match':case_ids==ids,'historical_case_ids_match':hist_ids==ids},
      'tokenizer':tokenizer_audit,
      'prompt_reconstruction':{'all_75_reconstructed':len(expected)==75,'all_prompt_ids_and_hashes_match_saved':all(not any(x['issues'] for x in checks) for x in checks),'min_tokens_with_bos':min(x['hf_prompt_tokens_with_bos'] for x in expected),'max_tokens_with_bos':max(x['hf_prompt_tokens_with_bos'] for x in expected),'all_within_context':all(x['hf_prompt_tokens_with_bos']+EXPECTED['cap']<=EXPECTED['context'] for x in expected)},
      'independent_rows':{'rows':len(checks),'all_raw_and_token_hash_checks_pass':all(x['hash_integrity_ok'] for x in checks),'all_prompt_identity_checks_pass':all(x['prompt_identity_ok'] for x in checks),'all_source_provenance_checks_pass':all(x['source_provenance_ok'] for x in checks),'protocol_valid_rows':sum(x['protocol_valid'] for x in checks),'protocol_invalid_rows':sum(not x['protocol_valid'] for x in checks),'all_rows_have_stop_terminal':all(x['status'] in {'accepted','protocol_error'} and x['returned_tokens']>0 for x in checks),'saved_and_recomputed_fields_consistent':not issues,'issues':issues[:20],'issue_count':len(issues)},
      'independent_denominators':independent_counts,'saved_denominators':saved_counts,'denominator_comparison':{k:{'independent':independent_counts.get(k),'saved':saved_counts.get(k),'match':independent_counts.get(k)==saved_counts.get(k)} for k in sorted(saved_counts)},
      'historical_selected_q8_b256':{'path':str(HIST/'quality.json'),'quality_sha256':sha256_file(HIST/'quality.json'),'terminal_sha256':sha256_file(HIST/'terminal.json'),'panel_sha256':hist.get('panel',{}).get('sha256'),'protocol_sha256':hist.get('static_preflight',{}).get('protocol',{}).get('sha256'),'tokenizer_files':hist.get('tokenizer',{}).get('files'),'model_sha256':hist.get('model',{}).get('provenance'),'server_n_ctx':hist.get('server',{}).get('n_ctx'),'cap':hist.get('contract',{}).get('completion',{}).get('n_predict'),'temperature':hist.get('contract',{}).get('completion',{}).get('temperature'),'cache_prompt':hist.get('contract',{}).get('completion',{}).get('cache_prompt'),'counts':hist_counts,'timing_ms':{'total':sum(hist_timing),'median':percentile(hist_timing,.5),'p95':percentile(hist_timing,.95)}},
      'v3_vs_historical':{'same_source_panel':hist.get('panel',{}).get('sha256')==EXPECTED['panel_sha256'],'same_protocol':hist.get('static_preflight',{}).get('protocol',{}).get('sha256')==EXPECTED['protocol_sha256'],'same_tokenizer_files':hist.get('tokenizer',{}).get('files')==tokenizer_saved.get('identity',{}).get('files'),'same_model_hash':hist.get('model',{}).get('provenance')==adm.get('model',{}).get('sha256'),'same_profile_or_native_settings':'v3 admission profile '+str(adm.get('profile_sha256'))+'; historical contract renderer '+str(hist.get('contract',{}).get('renderer_id')),'same_context':hist.get('contract',{}).get('context_size')==EXPECTED['context'],'same_output_cap':hist.get('contract',{}).get('completion',{}).get('n_predict')==EXPECTED['cap'],'v3_counts':{'edit_exact_rows':saved_counts['edit_exact_rows'],'strict_noop_correct_rows':saved_counts['strict_noop_correct_rows'],'noop_false_positive_rows':saved_counts['noop_false_positive_rows'],'protocol_rows':saved_counts['protocol_rows'],'cap_hit_rows':saved_counts['cap_hit_rows']},'historical_counts':hist_counts,'deltas':{k:saved_counts[k]-hist_counts[k] for k in saved_counts},'changed_output_or_status_rows':v3_vs_hist,'changed_row_count':len(v3_vs_hist),'same_output_hash_rows':sum(1 for c in cases if c.get('raw_text_sha256')==hmap[c['id']].get('raw_text_sha256') and c.get('returned_token_ids_sha256')==hmap[c['id']].get('returned_token_ids_sha256')),'timing_ms':{'v3_total':sum(timing),'v3_median':percentile(timing,.5),'v3_p95':percentile(timing,.95),'historical_total':sum(hist_timing),'historical_median':percentile(hist_timing,.5),'historical_p95':percentile(hist_timing,.95)},'interpretation':'Descriptive paired backend comparison only; equal aggregate counts do not establish model equivalence or promotion.'},
      'source_and_runtime_gates':{'controller_failure':controller.get('failure'),'controller_terminal_complete':controller.get('failure') is None,'host_returncode':host.get('controller_returncode'),'all_owned_pids_absent':all(x.get('absent') is True for x in controller.get('cleanup',[])),'port_free':controller.get('port_free'),'native_before_after_consistent':binding['before_after'],'source_closure_status':source_summary,'native_admission_status':{k:adm.get(k) for k in ['schema','status','purpose','fresh_process','exclusive_owner','no_other_clients','root_verified_model_before_launch','cuda_backend_observed','all_layers_offloaded','profile_sha256']},'actual_model_sha256_recorded':adm.get('model',{}).get('sha256'),'no_direct_tensor_proof':before.get('binding_scope')},
      'limits':['The reviewer reads the completed client rows and saved native/source audits; it does not rerun HTTP/native work or hash model/native-library bytes.','The native endpoint reports no logits/NLL; the six finish cases are mechanically parsed and no R code is executed.','The historical comparison uses the selected Q8 b256 DEV192 quality artifacts with the same panel/protocol/tokenizer/model hash and output cap; source/profile provenance is reported separately.','The v3 run is DEV-only and quality_acceptance_claim remains false; a complete terminal establishes harness/runtime evidence, not a final quality promotion.'],
      'review_reference':{'path':str(REVIEW_REFERENCE),'sha256':sha256_file(REVIEW_REFERENCE),'accepted_helper_path':str(ACCEPTED_HELPER),'accepted_helper_sha256':sha256_file(ACCEPTED_HELPER)},
    }

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();result=run_review();args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':result['status'],'rows':result['independent_rows']['rows'],'denominators':result['independent_denominators'],'changed_row_count':result['v3_vs_historical']['changed_row_count']},sort_keys=True))
if __name__=='__main__':main()
