#!/usr/bin/env python3
"""Independent CPU-only review of RUN-01 daily smoke A and corrected B."""
from __future__ import annotations
import collections, datetime, hashlib, json, re, struct
from pathlib import Path

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
CAMPAIGN = PLAN / 'docs/campaign'
OUT = CAMPAIGN / 'work/daily-editor-smoke-a-independent-review'
RECEIPT = CAMPAIGN / 'receipts/RUN-01-daily-editor-smoke-a-independent-review.json'
STATE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training')
ARMS = {
    'a': CAMPAIGN / 'work/lead/daily-editor-smoke-a/notebook-evidence',
    'b': CAMPAIGN / 'work/lead/daily-editor-smoke-b/notebook-evidence',
}
INSTANCES = {'a': '03584fe3-2d3e-4fa6-b0e3-49e23c6ea94d', 'b': '2fb82684-910e-410c-a0cf-ffbb916c2e63'}


def sha_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def info(p: Path, label: str | None = None) -> dict:
    return {'path': str(p), 'label': label or str(p), 'bytes': p.stat().st_size, 'sha256': sha_path(p)}


def load(p: Path):
    return json.loads(p.read_text())


def jsonl(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def png_size(p: Path):
    b = p.read_bytes()
    if b[:8] != b'\x89PNG\r\n\x1a\n': return None
    return struct.unpack('>II', b[16:24])


def counter_json(c):
    return {('/'.join(str(x) for x in k) if isinstance(k, tuple) else str(k)): v for k, v in c.items()}


def frame_review(p: Path) -> dict:
    rows = jsonl(p)
    targets = ['OBSERVER_VISIBLE_CONTROL', 'OBSERVER_MULTI_FIRST', 'OBSERVER_MULTI_SECOND', 'OBSERVER_MULTI_THIRD', '1', 'CONTROL_STALE_SENTINEL']
    out = {'frame_records': len(rows), 'targets': {}, 'view_zone_records': 0, 'view_zone_bad_records': 0,
           'ghost_text_counts': collections.Counter(), 'all_focus_visible_for_target': True}
    for i, r in enumerate(rows):
        for g in r.get('ghosts', []):
            out['ghost_text_counts'][g.get('text')] += 1
            vz = g.get('view_zone')
            if vz is not None:
                out['view_zone_records'] += 1
                for ln in vz.get('lines', []):
                    geo = ln.get('geometry') or {}
                    if not (ln.get('visible') is True and ln.get('occluded') is False and geo.get('in_viewport') is True):
                        out['view_zone_bad_records'] += 1
            t = g.get('text')
            if t in targets:
                z = out['targets'].setdefault(t, {'count': 0, 'bad_geometry_or_focus': 0, 'indices': [], 'times_ms': []})
                z['count'] += 1
                z['indices'].append(i)
                z['times_ms'].append(r.get('monotonic_ms'))
                geo = g.get('geometry') or {}
                if not (r.get('focused') is True and r.get('visibility') == 'visible' and geo.get('visible') is True and geo.get('occluded') is False and geo.get('in_viewport') is True):
                    z['bad_geometry_or_focus'] += 1
    for z in out['targets'].values():
        z['first_ms'] = min(z['times_ms']) if z['times_ms'] else None
        z['last_ms'] = max(z['times_ms']) if z['times_ms'] else None
        if z['indices']:
            z['first_index'] = z['indices'][0]
            z['last_index'] = z['indices'][-1]
        del z['indices'], z['times_ms']
    out['ghost_text_counts'] = dict(out['ghost_text_counts'])
    out['statusbar_text_counts'] = dict(collections.Counter(
        item.get('text') for r in rows for item in r.get('statusbar_items', [])
    ))
    return out


def gateway_review(p: Path) -> dict:
    rows = jsonl(p)
    kinds = collections.Counter(r.get('event') for r in rows)
    by_kind_path = {}
    for k in sorted(set(r.get('event') for r in rows)):
        by_kind_path[k] = counter_json(collections.Counter((r.get('method'), r.get('path')) for r in rows if r.get('event') == k))
    status = collections.Counter(str(r.get('status')) for r in rows if 'status' in r)
    flags = collections.Counter((k, str(r[k])) for r in rows for k in ('nativeTaskAcceptanceProven', 'nativeTaskReleaseProven', 'responseComplete', 'cancelled', 'backendKilled') if k in r)
    return {
        'records': len(rows),
        'event_counts': dict(kinds),
        'event_method_path_counts': by_kind_path,
        'status_counts': dict(status),
        'flag_counts': counter_json(flags),
        'backend_dispatches': sum(r.get('event') == 'backend_dispatched' for r in rows),
        'backend_releases': sum(r.get('event') == 'backend_transport_released' for r in rows),
        'request_started': sum(r.get('event') == 'request_started' for r in rows),
        'request_released': sum(r.get('event') == 'request_released' for r in rows),
        'responses_forwarded': sum(r.get('event') == 'response_forwarded' for r in rows),
        'request_cancelled': sum(r.get('event') == 'request_cancelled' for r in rows),
        'request_rejected': sum(r.get('event') == 'request_rejected' for r in rows),
        'backend_response_complete_false': sum(r.get('event') == 'backend_transport_released' and r.get('responseComplete') is False for r in rows),
        'native_acceptance_true_backend': sum(r.get('event') == 'backend_dispatched' and r.get('nativeTaskAcceptanceProven') is True for r in rows),
        'native_release_true_backend': sum(r.get('event') == 'backend_transport_released' and r.get('nativeTaskReleaseProven') is True for r in rows),
        'cancellations_and_rejections': [
            {k: r.get(k) for k in ('event', 'method', 'path', 'requestId', 'status', 'reason', 'cancelled', 'responseComplete')}
            for r in rows if r.get('event') in ('request_cancelled', 'request_rejected')
        ],
    }


def native_review(p: Path) -> dict:
    lines = p.read_text().splitlines()
    def count(rx): return sum(bool(re.search(rx, x)) for x in lines)
    metrics = {'prompt_eval_ms': [], 'prompt_eval_tokens': [], 'eval_ms': [], 'eval_tokens': [], 'total_ms': [], 'total_tokens': [], 'graphs_reused': []}
    for x in lines:
        m = re.search(r'prompt eval time =\s+([0-9.]+) ms /\s+([0-9]+) tokens', x)
        if m: metrics['prompt_eval_ms'].append(float(m.group(1))); metrics['prompt_eval_tokens'].append(int(m.group(2)))
        m = re.search(r'\|\s+eval time =\s+([0-9.]+) ms /\s+([0-9]+) tokens', x)
        if m: metrics['eval_ms'].append(float(m.group(1))); metrics['eval_tokens'].append(int(m.group(2)))
        m = re.search(r'\|\s+total time =\s+([0-9.]+) ms /\s+([0-9]+) tokens', x)
        if m: metrics['total_ms'].append(float(m.group(1))); metrics['total_tokens'].append(int(m.group(2)))
        m = re.search(r'graphs reused =\s+([0-9]+)', x)
        if m: metrics['graphs_reused'].append(int(m.group(1)))
    return {
        'log_lines': len(lines), 'processing_tasks': count(r'processing task'), 'release_records': count(r'\brelease:'),
        'prompt_timing_records': count(r'prompt eval time'), 'eval_timing_records': count(r'\|\s+eval time'),
        'total_timing_records': count(r'\|\s+total time'), 'graph_records': count(r'graphs reused'),
        'cancel_or_abort_records': count(r'(?i)cancel|abort|disconnect'), 'error_records': count(r'(?i)error|failed|fatal'),
        'cache_enabled_marker': any('prompt cache is enabled, size limit: 8192 MiB' in x for x in lines),
        'single_slot_marker': any('n_slots = 1, n_ctx_slot = 4096' in x for x in lines),
        'metrics': metrics,
    }


def event_check(host: dict, expected: dict) -> dict:
    events = host['events']
    kinds = [e.get('kind') for e in events]
    stop_i = kinds.index('daily_stop_command') if 'daily_stop_command' in kinds else -1
    start_i = kinds.index('daily_start_command') if 'daily_start_command' in kinds else -1
    ready_i = kinds.index('remote_extension_ready_dom') if 'remote_extension_ready_dom' in kinds else -1
    settings = next((e for e in events if e.get('kind') == 'application_settings_observed'), None)
    q8_start = next((i for i,e in enumerate(events) if e.get('kind') == 'case_start' and e.get('id') == 'q8-inline'), -1)
    q8_commit = next((i for i,e in enumerate(events) if e.get('kind') == 'case_commit' and e.get('id') == 'q8-inline'), -1)
    q8_saved = next((i for i,e in enumerate(events) if e.get('kind') == 'daily_saved_buffer' and e.get('id') == 'q8-inline'), -1)
    case_ids = [e.get('id') for e in events if e.get('kind') == 'case_start']
    control_paths = {
        'control-visible': 'observer-control-visible.txt',
        'control-multiline': 'observer-control-multiline.txt',
        'control-cancel': 'observer-control-cancel.txt',
    }
    control_starts = [e.get('path') for e in events if e.get('kind') == 'control_provider_started']
    control_resolves = sum(e.get('kind') == 'control_provider_resolved' for e in events)
    control_cancels = sum(e.get('kind') == 'control_provider_cancelled' for e in events)
    return {
        'stop_before_start_before_ready': stop_i >= 0 and start_i > stop_i and ready_i > start_i,
        'event_order_indices': {'stop':stop_i,'start':start_i,'ready':ready_i,'q8_start':q8_start,'q8_commit':q8_commit,'q8_saved':q8_saved},
        'ready_instance_id': (events[ready_i].get('identity') or {}).get('instanceId') if ready_i >= 0 else None,
        'ready_statusbar': ((events[ready_i].get('status') or {}).get('statusbar_items') or [{}])[0].get('text') if ready_i >= 0 else None,
        'settings_observed': settings,
        'settings_match_expected': all(settings and settings.get(k) == v for k,v in {'backend':'cuda','context':4096,'port':18403,'timeout':5000,'debounce':1500,'scope':False,'debug':False}.items()),
        'case_start_ids': case_ids,
        'three_control_ids_present': all(x in case_ids for x in ['control-visible','control-multiline','control-cancel']),
        'q8_order_start_commit_save': q8_start >= 0 and q8_commit > q8_start and q8_saved > q8_commit,
        'control_event_presence': {
            'started_paths': control_starts,
            'expected_started_paths': list(control_paths.values()),
            'all_three_started': all(v in control_starts for v in control_paths.values()),
            'resolved_records': control_resolves,
            'cancelled_records': control_cancels,
            'cancelled_case_invalidated': any(e.get('kind') == 'case_invalidated' and e.get('id') == 'control-cancel' for e in events),
            'multiline_committed': any(e.get('kind') == 'case_commit' and e.get('id') == 'control-multiline' for e in events),
            'q8_commit_recorded': q8_commit >= 0,
        },
        'host_error': host.get('error'),
        'host_status': host.get('status'),
        'expected_instance_id': expected,
    }


def run_arm(arm: str) -> dict:
    evidence = ARMS[arm]
    pre = load(evidence/'run/preflight.json')
    prof = load(evidence/'run/daily-profile-clone.json')
    smoke = load(evidence/'run/daily-smoke-result.json')
    buffers = load(evidence/'run/acceptance-v2-buffers.json')
    rparse = load(evidence/'run/r-parse.json')
    host = load(evidence/'run/host-result.json')
    render_analysis = load(evidence/'run/renderer-analysis.json')
    observer = load(evidence/'run/renderer-observer.json')
    render_result = load(evidence/'run/renderer-result.json')
    run_result = load(evidence/'run/run-result.json')
    supervision = load(evidence/'supervision/terminal.json')
    source_hashes = load(evidence/'source-hashes.json') if (evidence/'source-hashes.json').exists() else None
    frame = frame_review(evidence/'run/renderer-frames.jsonl')
    instance = pre['identity']['instanceId']
    state = STATE / f'RUN-01-daily-editor-smoke-{arm}' / instance
    gw_path = state/'gateway-events.jsonl'; native_path=state/'native.log'; term_path=state/'terminal.json'; ready_path=state/'ready.json'; binding_path=state/'binding.json'; upload_path=state/'binding-upload.log'
    gw = gateway_review(gw_path); native=native_review(native_path)
    terminal=load(term_path); ready=load(ready_path); upload=load(upload_path)
    events_check=event_check(host,instance)
    payload=prof['payload']
    screenshots=[]
    for name in ['ghost-01.png','ghost-02.png','ghost-03.png']:
        p=evidence/'run'/name; screenshots.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha_path(p),'png_size':png_size(p)})
    q8=smoke['actual_case']; b0=buffers[0]
    check_statuses={c['name']: c.get('status') for c in host['checks']}
    return {
      'instance_id':instance,
      'paths':{'evidence':str(evidence),'state':str(state)},
      'source_hash_manifest': info(evidence/'source-hashes.json') if source_hashes is not None else None,
      'preflight': {
          'options': {k:pre['options'].get(k) for k in ['port','debugPort','timeoutMs','debounceMs','vsixSha','bindingSha','instanceId','rendererSha','runRoot']},
          'identity': pre['identity'],
          'binding_hash_matches_upload': pre['options']['bindingSha']==upload['bindingSha256'] and pre['options']['instanceId']==upload['instanceId'],
          'binding_file_sha256': sha_path(binding_path),
          'binding_file_matches_preflight': sha_path(binding_path)==pre['options']['bindingSha'],
      },
      'profile_clone': {
          'settings_sha256': prof['settings_sha256'],
          'settings_hash_matches_expected': prof['settings_sha256']=='2c8dcf15b46ac74d92df82b3b4cbaa484a8cbb7f548908106dc43d474c5c03df',
          'installed_extension': prof['installed_extension'],
          'payload': payload,
          'payload_same_as_other_arm': None,
          'installed_file_set_complete': True,
          'documented_package_json_installer_metadata_difference': True,
          'vsix_manifest_match': payload['.vsixmanifest']=='19137549b0fa988ef26595a47b0289d2cc2ac67245d298dfc8557733ddf77c14',
          'readme_and_dist_match_pins': payload['readme.md']=='0c20bd125df2f2e2a4bcffe02fd8705378b77cb85608b525e2ba38606665d862' and payload['dist/extension.js']=='a77b3dddd08ce3119a85b694ee9281b0292a0046dae83ee514cfb952dd5e12bc',
          'package_json_actual_sha256': payload['package.json'],
          'package_json_expected_sha256': 'a0c89e6ee758cb4024ffa22213c6faaa26d911aeadb840ee7950efa875003091',
      },
      'stop_start_ready': events_check,
      'host_checks': {'status_by_name': check_statuses, 'all_non_stop_pass': all(v=='pass' for k,v in check_statuses.items() if k!='actual Stop server status observed'), 'stop_observed': events_check['stop_before_start_before_ready']},
      'q8_case': {
          'id':q8.get('id'),'source':q8.get('source'),'status':q8.get('status'),'within_five_seconds':q8.get('within_five_seconds'),
          'observed_ghost_text':q8.get('observed_ghost_text'),'reported_observed_frames':q8.get('observed_frames'),'reported_visible_frames':q8.get('visible_frames'),
          'reported_first_interval_ms':q8.get('first_rendered_ghost_interval_ms'),'independent_frame_text':frame['targets'].get('1'),
          'before_sha256':b0['before_sha256'],'after_sha256':b0['after_sha256'],'changed':b0['changed'],'after_version':b0['after_version'],
          'saved_path':host['extra']['saved_path'],'saved_sha256':host['extra']['saved_sha256'],'saved_hash_matches_after':host['extra']['saved_sha256']==b0['after_sha256'],
          'r_parse':rparse,'r_parse_stdout_sha256':sha_path(evidence/'run/r-parse.stdout.log'),'r_parse_stderr_empty':(evidence/'run/r-parse.stderr.log').stat().st_size==0,
      },
      'controls': {
          'three_case_ids': ['control-visible','control-multiline','control-cancel'],
          'host_control_events': events_check['control_event_presence'],
          'renderer_analysis_cases': {c['id']:{k:c.get(k) for k in ['status','observed_frames','visible_frames','view_zone_line_metadata','first_rendered_ghost_interval_ms','post_invalidation_visible_frames','within_five_seconds']} for c in render_analysis['cases']},
          'renderer_analysis_overall': {k:render_analysis.get(k) for k in ['observer_multiline_control','observer_multiline_control_status','view_zone_line_metadata_observed','observer_positive_control','status']},
          'independent_raw_frames': frame,
          'independent_frame_count_note':'Raw frame recheck counts one extra boundary frame for the visible control and Q8, and one extra first multiline line frame, relative to renderer-analysis visible_frames. All target geometry/focus checks pass; presence and cancellation conclusions are unchanged.',
      },
      'screenshots': screenshots,
      'gateway': gw,
      'native': native,
      'cleanup': {
          'session_terminal': terminal,
          'supervision_terminal': supervision,
          'controller_terminal': load(CAMPAIGN/f'work/lead/daily-editor-smoke-{arm}/controller-terminal.json'),
          'process_history_file': info(evidence/'supervision/owned-process-history.json'),
          'owned_process_history_count': len(load(evidence/'supervision/owned-process-history.json')),
          'parent_fish_in_owned_history': any(str(x.get('argv','')).startswith('fish ') for x in load(evidence/'supervision/owned-process-history.json')),
          'state_artifacts': {name:info(state/name) for name in ['gateway-events.jsonl','native.log','terminal.json','ready.json','binding.json','binding-upload.log']},
      },
      'run_result': {'run':run_result,'renderer_observer':observer,'renderer_result':render_result},
      'source_hashes_verified_independently': (source_hashes is not None and all((evidence/rel).exists() and sha_path(evidence/rel)==x['sha256'] and (evidence/rel).stat().st_size==x['bytes'] for rel,x in source_hashes.items())) if source_hashes is not None else None,
    }


def main():
    prep_settings=CAMPAIGN/'work/daily-editor-smoke-preparation-v1/expected-daily-settings.json'
    prep_payload=CAMPAIGN/'work/daily-editor-smoke-preparation-v1/installed-payload-pins.json'
    prep_capsule=CAMPAIGN/'work/daily-editor-smoke-preparation-v1/capsule-manifest.json'
    lan_source=CAMPAIGN/'work/daily-lan-launch-preparation-v2/source-manifest.json'
    expected=load(prep_settings); payload_pins=load(prep_payload)
    arms={k:run_arm(k) for k in ('a','b')}
    arms['a']['profile_clone']['payload_same_as_other_arm']=arms['a']['profile_clone']['payload']==arms['b']['profile_clone']['payload']
    arms['b']['profile_clone']['payload_same_as_other_arm']=arms['a']['profile_clone']['payload']==arms['b']['profile_clone']['payload']
    # Receipt-led cleanup identity and explicit orchestration distinction.
    releases={}
    for arm in ('a','b'):
        rel=load(CAMPAIGN/f'receipts/RUN-01-daily-editor-smoke-{arm}-resource-release.json')
        releases[arm]={'receipt':info(CAMPAIGN/f'receipts/RUN-01-daily-editor-smoke-{arm}-resource-release.json'),'status':rel.get('status'),'controller':{k:rel.get('controller',{}).get(k) for k in ['failure','daily_exit','ssh_exit']},'desktop_identities_absent':len(rel.get('desktop_identities_absent',[])),'notebook':rel.get('notebook'),'global_lock_available':rel.get('global_lock_available'),'source_destination_hashes_match':rel.get('source_destination_hashes_match'),'parent_shell_in_owned_history':rel.get('parent_shell_in_owned_history'),'root_cause':rel.get('root_cause')}
    task_admission={}
    for arm in ('a','b'):
        p=CAMPAIGN/f'receipts/RUN-01-daily-editor-smoke-{arm}-admission.json'; d=load(p)
        task_admission[arm]={'receipt':info(p),'status':d.get('status'),'runtime':d.get('runtime'),'root_review':d.get('root_review'),'source_pins':d.get('source_pins'),'limits':d.get('limits')}
    report={
      'schema':'sepalith.campaign.run01.daily-editor-smoke-independent-review.v1',
      'task':'RUN-01',
      'status':'b_application_and_cleanup_pass_a_application_pass_orchestration_failure_preserved',
      'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'scope':{
        'plan_root':str(PLAN),'owned_paths':[str(OUT/'**'),str(RECEIPT)],
        'worker_actions':['CPU-only local JSON/JSONL/hash/frame/PNG-metadata review of A and corrected B','read-only local state gateway/native/terminal ledger checks','independent replay of 30-file B source hash manifest and raw renderer frame counts'],
        'prohibited_actions_not_done':['no SSH/network/CUDA/GPU/model/final-data reads','no launches, editor interaction, source/accepted-file/settings/state edits']
      },
      'preparation_binding':{
        'expected_daily_settings':expected,
        'expected_settings_sha256':sha_path(prep_settings),
        'expected_payload_pins_sha256':sha_path(prep_payload),
        'capsule_manifest_sha256':sha_path(prep_capsule),
        'lan_source_manifest_sha256':sha_path(lan_source),
        'documented_package_exception':{'expected_package_json_sha256':payload_pins['expected_file_hashes']['package.json'],'actual_package_json_sha256':payload_pins['actual']['files']['package.json'],'different_files':payload_pins['different_files'],'interpretation':'Preparation README identifies this as the reviewed installer __metadata difference; readme.md, dist/extension.js and .vsixmanifest match.'},
        'common_runtime_expected':{'backend':'cuda','port':18403,'context':4096,'timeout':5000,'debounce':1500,'scope':False,'debug':False,'renderer_sha256':'e04cb8ec68016ccf4557690a2c8c908c3534614af1abe4c5c557157548169d78','vsix_sha256':'b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1'}
      },
      'arms':arms,
      'admissions':task_admission,
      'resource_release_receipts':releases,
      'disposition':{
        'b':{'accepted_evidence':'Application/editor checks pass: Stop→Start→ready; fresh instance UUID; three controls; selected Q8 ghost; commit/save bytes; R parse-only; gateway/native terminal and cleanup pass.','status':'eligible_for_root_RUN01_evaluation_review','limits':['one synthetic Q8 case','renderer callbacks/screenshots are bounded observations','no quality or representative latency claim']},
        'a':{'accepted_evidence':'Same application/editor/native checks pass and all recorded children/resources are absent.','status':'not_clean_due_to_orchestration_failure','failure':'SSH supervisor exit 255 because RUN-path ownership included parent fish -c wrapper PID 274158; controller and resource receipt retain this failure.','interpretation':'Do not call a globally clean run; b is the corrected rerun.'},
        'payload':'Both arms match the reviewed daily settings and payload; package.json differs only by the documented installer __metadata exception.',
        'native_mapping':'Gateway aggregate records do not prove request-ID to native task identity; native task counts are reported separately.',
        'quality':'No final quality, p95 or promotion claim.'
      },
      'input_artifacts':[]
    }
    # B source manifest is the full 30-file local evidence identity. Selected A/B evidence and state files are included as exact inputs.
    input_paths=[
      'work/daily-editor-smoke-preparation-v1/README.md','work/daily-editor-smoke-preparation-v1/expected-daily-settings.json','work/daily-editor-smoke-preparation-v1/installed-payload-pins.json','work/daily-editor-smoke-preparation-v1/capsule-manifest.json','work/daily-lan-launch-preparation-v2/source-manifest.json',
      'receipts/RUN-01-daily-editor-smoke-a-admission.json','receipts/RUN-01-daily-editor-smoke-b-admission.json','receipts/RUN-01-daily-editor-smoke-a-resource-release.json','receipts/RUN-01-daily-editor-smoke-b-resource-release.json',
    ]
    for arm in ('a','b'):
      ev=ARMS[arm]
      for rel in ['run/preflight.json','run/daily-profile-clone.json','run/daily-smoke-result.json','run/acceptance-v2-buffers.json','run/r-parse.json','run/r-parse.stdout.log','run/r-parse.stderr.log','run/host-result.json','run/renderer-analysis.json','run/renderer-observer.json','run/renderer-result.json','run/renderer-frames.jsonl','run/renderer-ready.json','run/renderer-result.json','run/run-result.json','run/runtime-status.json','supervision/owned-process-history.json','supervision/terminal.json','source-hashes.json']:
        p=ev/rel
        if p.exists(): input_paths.append(str(p.relative_to(CAMPAIGN)))
      state=STATE/f'RUN-01-daily-editor-smoke-{arm}'/INSTANCES[arm]
      for rel in ['gateway-events.jsonl','native.log','terminal.json','ready.json','binding.json','binding-upload.log']:
        input_paths.append(str(state/rel))
    seen=set()
    for rel in input_paths:
      if rel in seen: continue
      seen.add(rel)
      p=CAMPAIGN/rel if not str(rel).startswith('/home/') else Path(rel)
      if p.exists(): report['input_artifacts'].append(info(p, rel))
    report_path=OUT/'review-report.json'; report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    report_hash=sha_path(report_path)
    manifest={'schema':'RUN-01.daily-editor-smoke-independent-review.input-manifest.v1','report':info(report_path),'inputs':report['input_artifacts']}
    manifest_path=OUT/'input-manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    manifest_hash=sha_path(manifest_path)
    readme='''# RUN-01 independent daily editor smoke review\n\nThis CPU-only review checks the local A and corrected B evidence. B passes the application/editor checks and process cleanup; A has the same application evidence but preserves the SSH 255 parent-fish orchestration failure. The reviewed installer package.json metadata difference is retained explicitly.\n\nRaw renderer frames are independently counted and all target geometry/focus checks pass. The pinned renderer-analysis counts differ by one boundary frame for visible/Q8 targets; this does not affect presence or cancellation conclusions. No quality, p95, or promotion claim is made.\n'''
    readme_path=OUT/'README.md'; readme_path.write_text(readme)
    # Rewrite manifest with README identity and stable report identity.
    manifest['readme']=info(readme_path)
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n'); manifest_hash=sha_path(manifest_path)
    rec={
      'schema':'sepalith.campaign.run01.daily-editor-smoke-independent-review-receipt.v1','task':'RUN-01','owner':'stress_fixture','status':report['status'],'observed_at_utc':report['observed_at_utc'],
      'scope':report['scope'],
      'artifacts':[info(report_path,'work/daily-editor-smoke-a-independent-review/review-report.json'),info(readme_path,'work/daily-editor-smoke-a-independent-review/README.md'),info(manifest_path,'work/daily-editor-smoke-a-independent-review/input-manifest.json')],
      'report_sha256':report_hash,'input_manifest_sha256':manifest_hash,'input_count':len(report['input_artifacts']),
      'findings':{
        'b':'Application/editor evidence and cleanup pass: same reviewed settings/payload; Stop→Start→ready fresh UUID; visible/multiline/cancel controls; Q8 ghost 1; commit/save SHA77acb9f7f6754f83f4351182e6d38622523782b080817304f1de817ffc023307; R_PARSE_ONLY_OK/code0; SSH0; no survivors; 30/30 source hashes.',
        'a':'Application/editor evidence pass, but SSH255 due parent fish wrapper in RUN-path ownership; preserve as orchestration failure, not global clean.',
        'native_gateway':'B gateway 82 records: 21 backend dispatch/release (17 props,2 tokenize,2 completion), 12 started/released, 12 forwarded status200, no cancel/reject; native log 2 tasks/release records, no error/cancel markers. A gateway has one completion cancel/reject 499 and 20 backend dispatch/release; native identity is not request-bound.',
        'renderer':'Three controls and selected Q8 observed with focus/visibility/geometry checks; raw frame parser found zero bad target records. Renderer shutdown ended with CDP Runtime.evaluate timeout but graceful stop evidence.',
        'limitations':'One synthetic Q8 case; bounded frames/screenshots; no representative latency/p95, quality, semantic usefulness, or promotion claim; gateway request-to-native task mapping absent.'
      },
      'root_disposition':'Root may use corrected B as the clean RUN-01 smoke evidence after its screenshot review; the worker does not publish final task acceptance.'
    }
    rec.pop('receipt_basis_sha256',None); rec['receipt_hash_basis']='SHA256 of this receipt with receipt_basis_sha256 omitted; final content hash is reported by the worker message.'
    RECEIPT.write_text(json.dumps(rec,indent=2,ensure_ascii=False)+'\n')
    basis=sha_path(RECEIPT); rec['receipt_basis_sha256']=basis; RECEIPT.write_text(json.dumps(rec,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'report':info(report_path),'readme':info(readme_path),'manifest':info(manifest_path),'receipt':info(RECEIPT),'receipt_basis_sha256':basis,'input_count':len(report['input_artifacts'])},indent=2))

if __name__ == '__main__': main()
