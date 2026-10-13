#!/usr/bin/env python3
"""Independent, read-only RUN-04 cache-event review.

Inputs are fixed to the copied remote evidence, the desktop gateway/native ledger,
and the lead/preparation metadata.  The script emits only review JSON in its own
artifact directory; it never starts a server, editor, native client, or R runtime.
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PLAN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
OUT = PLAN / 'docs/campaign/work/cache-event-live-a-independent-review'
RUN = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-04-cache-event-live-a')
LEAD = PLAN / 'docs/campaign/work/lead/cache-event-live-a'
PREP = PLAN / 'docs/campaign/work/cache-event-live-preparation-v1'
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = ['baseline', 'unchanged-repeat', 'cursor-only', 'history-restored-source',
            'anchor-move', 'file-switch-r-to-r', 'file-switch-return']
EXPECTED_URI_A = 'file:///home/m0hawk/.local/share/sepalith-campaign-20260915/runs/cache-event-live-a/workspace/cache-event-a.R'
EXPECTED_URI_B = 'file:///home/m0hawk/.local/share/sepalith-campaign-20260915/runs/cache-event-live-a/workspace/cache-event-b.R'
CONTENT_SHA = '9a4ee5dfafa34682524719efee1a6b2a35f2413c85796a7b87712021ac7bfbb7'
INSTANCE = '23f78ee2-792e-4824-a62e-f02f68e934a6'


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def js_sha(value: Any) -> str:
    return text_sha(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def identity(uri: str, version: int, content_sha256: str, line: int, character: int) -> dict[str, Any]:
    return {'uri': uri, 'version': version, 'content_sha256': content_sha256,
            'cursor': {'line': line, 'character': character}}


def clean_range(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r[k] for k in ['uri', 'document_version', 'content_sha256', 'start', 'end'] if k in r}


def rect_ok(rect: Any) -> bool:
    return isinstance(rect, dict) and all(isinstance(rect.get(k), (int, float)) for k in ('x', 'y', 'width', 'height')) and rect['width'] > 0 and rect['height'] > 0


def geometry_ok(item: dict[str, Any]) -> bool:
    geom = item.get('geometry') if isinstance(item.get('geometry'), dict) else item
    rect = geom.get('rect') if isinstance(geom, dict) else None
    return bool(isinstance(item, dict) and isinstance(geom, dict)
                and geom.get('visible') is True and geom.get('occluded') is False
                and geom.get('in_viewport') is True and rect_ok(rect))


def visible_text_entries(frame: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for ghost_index, ghost in enumerate(frame.get('ghosts') or []):
        if not isinstance(ghost, dict):
            continue
        if isinstance(ghost.get('text'), str):
            entries.append({'text': ghost['text'], 'kind': 'ghost', 'ghost_index': ghost_index,
                            'geometry_ok': geometry_ok(ghost), 'view_zone_geometry_ok': None})
        view_zone = ghost.get('view_zone')
        if isinstance(view_zone, dict) and isinstance(view_zone.get('lines'), list):
            for line_index, line in enumerate(view_zone['lines']):
                if isinstance(line, dict):
                    entries.append({'text': line.get('text'), 'kind': 'view_zone_line',
                                    'ghost_index': ghost_index, 'line_index': line_index,
                                    'geometry_ok': geometry_ok(line), 'view_zone_geometry_ok': geometry_ok(line)})
    return entries


def case_frame_review(frames: list[dict[str, Any]], events: list[dict[str, Any]], case_id: str, expected_texts: list[str] | None = None) -> dict[str, Any]:
    start = next(x for x in events if x.get('kind') == 'case_trigger' and x.get('id') == case_id)
    end = next(x for x in events if x.get('kind') == 'case_end' and x.get('id') == case_id and x.get('epoch_ms', 0) >= start['epoch_ms'])
    commit = next((x for x in events if x.get('kind') == 'case_commit' and x.get('id') == case_id and x.get('epoch_ms', 0) >= start['epoch_ms']), None)
    finish = commit['epoch_ms'] if commit else end['epoch_ms']
    sample = [x for x in frames if start['epoch_ms'] - 100 <= x.get('epoch_ms', 0) <= finish]
    gaps = [b['monotonic_ms'] - a['monotonic_ms'] for a, b in zip(sample, sample[1:])]
    baseline = [x for x in sample if x['epoch_ms'] < start['epoch_ms']]
    after = [x for x in sample if x['epoch_ms'] >= start['epoch_ms']]
    ghost_frames = [x for x in after if x.get('ghosts')]
    first = ghost_frames[0] if ghost_frames else None
    entries = [e for frame in ghost_frames for e in visible_text_entries(frame)]
    matched: dict[str, Any] = {}
    for text in expected_texts or []:
        hits = [e for e in entries if e.get('text') and text in e['text']]
        matched[text] = {'count': len(hits), 'geometry_all_ok': bool(hits) and all(e['geometry_ok'] for e in hits),
                         'kinds': sorted(set(e['kind'] for e in hits))}
    invalidated = next((x for x in events if x.get('kind') == 'case_invalidated' and x.get('id') == case_id and x.get('epoch_ms', 0) >= start['epoch_ms']), None)
    post = [x for x in after if invalidated and x.get('epoch_ms', 0) >= invalidated['epoch_ms']]
    row = {
        'trigger_epoch_ms': start['epoch_ms'], 'finish_epoch_ms': finish,
        'observed_frames': len(sample), 'visible_ghost_frames': len(ghost_frames),
        'max_frame_gap_ms': max(gaps) if gaps else None,
        'nonnegative_frame_gaps': all(g >= 0 for g in gaps),
        'baseline_empty': bool(baseline) and all(not x.get('ghosts') for x in baseline),
        'all_focused_visible': all(x.get('focused') is True and x.get('visibility') == 'visible' for x in sample),
        'frame_coverage': bool(baseline and after and sample and sample[-1]['epoch_ms'] >= finish - 100 and gaps and max(gaps) <= 100),
        'first_ghost_delta_ms': first['epoch_ms'] - start['epoch_ms'] if first else None,
        'matched_texts': matched,
        'unique_visible_texts': sorted(set(e['text'] for e in entries if isinstance(e.get('text'), str))),
        'invalidated': invalidated is not None,
        'post_invalidation_ghost_frames': len([x for x in post if x.get('ghosts')]),
        'post_invalidation_window_ms': finish - invalidated['epoch_ms'] if invalidated else None,
    }
    if case_id == 'control-cancel':
        started = next((x for x in events if x.get('kind') == 'control_provider_started' and x.get('stale') is True and start['epoch_ms'] <= x.get('epoch_ms', 0) <= (invalidated or end)['epoch_ms']), None)
        cancelled = next((x for x in events if x.get('kind') == 'control_provider_cancelled' and started and x.get('call') == started.get('call')), None)
        resolved = next((x for x in events if x.get('kind') == 'control_provider_resolved' and started and x.get('call') == started.get('call') and x.get('cancelled') is True), None)
        row['control_provider_started_cancelled_resolved'] = bool(started and cancelled and resolved)
    return row


def parse_native_log(path: Path) -> dict[str, Any]:
    lines = path.read_text().splitlines()
    tasks: dict[int, dict[str, Any]] = {}
    current = None
    warnings: list[str] = []
    for line in lines:
        if ' W ' in line and ('task 94' in line or 'n_past' in line):
            warnings.append(line)
        m = re.search(r'processing task, is_child = (\d+)', line)
        if m:
            tm = re.search(r'task (\-?\d+)', line)
            if tm:
                current = int(tm.group(1)); tasks.setdefault(current, {'task_id': current, 'cached_n_tokens': []})
        if current is None:
            continue
        m = re.search(r'new prompt, n_ctx_slot = (\d+), n_keep = (\d+), task\.n_tokens = (\d+)', line)
        if m: tasks[current].update({'n_ctx_slot': int(m.group(1)), 'n_keep': int(m.group(2)), 'prompt_tokens_requested': int(m.group(3))})
        m = re.search(r'cached n_tokens = (\d+)', line)
        if m: tasks[current]['cached_n_tokens'].append(int(m.group(1)))
        m = re.search(r'prompt eval time =\s+([0-9.]+) ms /\s+(\d+) tokens', line)
        if m: tasks[current].update({'prompt_eval_ms': float(m.group(1)), 'prompt_eval_tokens': int(m.group(2))})
        m = re.search(r'eval time =\s+([0-9.]+) ms /\s+(\d+) tokens', line)
        if m: tasks[current].update({'eval_ms': float(m.group(1)), 'generated_tokens': int(m.group(2))})
        m = re.search(r'total time =\s+([0-9.]+) ms /\s+(\d+) tokens', line)
        if m: tasks[current].update({'total_ms': float(m.group(1)), 'total_tokens': int(m.group(2))})
        m = re.search(r'stop processing: n_tokens = (\d+), truncated = (\d+)', line)
        if m: tasks[current].update({'release_n_tokens': int(m.group(1)), 'truncated': int(m.group(2))})
    cache_states = []
    cache_update_ms = []
    for line in lines:
        m = re.search(r'cache state: (\d+) prompts, ([0-9.]+) MiB', line)
        if m: cache_states.append({'prompts': int(m.group(1)), 'mib': float(m.group(2))})
        m = re.search(r'prompt cache update took ([0-9.]+) ms', line)
        if m: cache_update_ms.append(float(m.group(1)))
    selected = [tasks[k] for k in sorted(tasks)]
    return {
        'path': str(path), 'sha256': sha256(path), 'processing_task_count': len(selected),
        'released_task_count': sum('release_n_tokens' in t for t in selected),
        'all_released_truncated_zero': bool(selected) and all(t.get('truncated') == 0 for t in selected),
        'tasks': selected, 'cache_states': cache_states, 'prompt_cache_update_ms': cache_update_ms,
        'all_slots_idle_lines': sum('all slots are idle' in line for line in lines),
        'warnings': warnings,
        'queue_or_backpressure_indicator_count': sum(bool(re.search(r'backpressure|queue full|no available slot|all slots busy|queue is full', line, re.I)) for line in lines),
        'nonqueue_capacity_warning_count': sum('full capacity of the model will not be utilized' in line for line in lines),
        'cleaning_up_before_exit': any('cleaning up before exit' in line for line in lines),
    }


def main() -> None:
    frames = load_jsonl(OUT / 'renderer-frames.jsonl')
    harness = load_jsonl(OUT / 'harness-events.jsonl')
    trace = load_jsonl(OUT / 'provider-trace.jsonl')
    gateway = load_jsonl(RUN / 'gateway-events.jsonl')
    host_result = load_json(OUT / 'host-result.json')
    preflight = load_json(OUT / 'preflight.json')
    ready = load_json(OUT / 'renderer-ready.json')
    observer = load_json(OUT / 'renderer-observer.json')
    renderer_result = load_json(OUT / 'renderer-result.json')
    terminal = load_json(RUN / 'terminal.json')
    remote_terminal = load_json(RUN / 'remote-command-terminal.json')
    supervision_terminal = load_json(OUT / 'supervision/terminal.json')
    native_identity = load_json(RUN / 'native-identity.json')
    gateway_ready = load_json(RUN / 'gateway-ready.json')

    source_event = next(x for x in harness if x.get('kind') == 'cache_fixture_source')
    source_text = source_event['text']
    source_lines = source_text.splitlines()
    actual_source_sha = text_sha(source_text)

    def source_backed(ctx: dict[str, Any]) -> bool:
        rr = ctx['replacement_range']; line = rr['start']['line']; col = rr['end']['character']
        prefix = ctx.get('prefix', []); suffix = ctx.get('suffix_lines', [])
        prefix_ok = any(source_lines[i:i + len(prefix)] == prefix and i + len(prefix) == line for i in range(max(0, line - len(prefix) - 2), min(line + 1, len(source_lines))))
        suffix_ok = source_lines[line + 1:line + 1 + len(suffix)] == suffix
        region_ok = ctx.get('region_old') == [source_lines[line][:col]]
        return prefix_ok and suffix_ok and region_ok

    def expected_for_event(event_id: str) -> dict[str, Any]:
        end = next(x for x in harness if x.get('kind') == 'cache_event_end' and x.get('event_id') == event_id)
        return end['expected']

    traces_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace:
        if row.get('trace_id') is not None: traces_by_id[row['trace_id']].append(row)
    gateway_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gateway: gateway_by_request[row.get('requestId')].append(row)

    cases: list[dict[str, Any]] = []
    all_direct_joins: list[dict[str, Any]] = []
    for event_id in EXPECTED:
        expected = expected_for_event(event_id)
        end_event = next(x for x in harness if x.get('kind') == 'cache_event_end' and x.get('event_id') == event_id)
        tid = end_event.get('trace_id')
        rows = sorted(traces_by_id.get(tid, []), key=lambda x: x.get('sequence', 0))
        begin = next((x for x in rows if x.get('kind') == 'provider_begin'), None)
        context_row = next((x for x in rows if x.get('kind') == 'context'), None)
        finish = next((x for x in rows if x.get('kind') == 'provider_end'), None)
        cache_row = next((x for x in rows if x.get('kind') == 'result_cache'), None)
        ctx = context_row.get('context', {}) if context_row else {}
        rr = ctx.get('replacement_range', {})
        cursor = ctx.get('cursor', {})
        expected_identity = identity(expected['uri'], expected['version'], expected['content_sha256'], expected['cursor']['line'], expected['cursor']['character'])
        actual_identity = begin.get('identity') if begin else None
        expected_prompt_key = js_sha({'prompt': context_row.get('prompt') if context_row else '', 'identity': rr, 'cursor': expected['cursor']}) if context_row else None
        prompt_hash_ok = bool(context_row and text_sha(context_row.get('prompt', '')) == context_row.get('prompt_sha256') and len(context_row.get('prompt', '').encode()) == context_row.get('prompt_utf8_bytes'))
        key_hash_ok = bool(context_row and expected_prompt_key == context_row.get('request_key_sha256'))
        direct_joins = []
        for begin_transport in [x for x in rows if x.get('kind') == 'transport_begin']:
            transport_id = begin_transport.get('transport_id')
            headers = next((x for x in rows if x.get('kind') == 'transport_response_headers' and x.get('transport_id') == transport_id), None)
            value = next((x for x in rows if x.get('kind') == 'transport_value' and x.get('transport_id') == transport_id), None)
            ending = next((x for x in rows if x.get('kind') == 'transport_end' and x.get('transport_id') == transport_id), None)
            rid = headers.get('gateway_request_id') if headers else None
            dispatches = [x for x in gateway_by_request.get(rid, []) if x.get('event') == 'backend_dispatched' and x.get('path') == begin_transport.get('route') and x.get('method') == ('POST' if begin_transport.get('route') != '/props' else 'GET')]
            releases = [x for x in gateway_by_request.get(rid, []) if x.get('event') == 'backend_transport_released' and x.get('path') == begin_transport.get('route') and x.get('method') == ('POST' if begin_transport.get('route') != '/props' else 'GET')]
            responses = [x for x in gateway_by_request.get(rid, []) if x.get('event') == 'response_forwarded' and x.get('status') == 200]
            request_releases = [x for x in gateway_by_request.get(rid, []) if x.get('event') == 'request_released']
            dispatched = dispatches[0] if len(dispatches) == 1 else {}
            exact = bool(headers and headers.get('gateway_instance_id') == INSTANCE and headers.get('status') == 200
                         and len(dispatches) == 1 and dispatched.get('instanceId') == INSTANCE
                         and dispatched.get('requestBodySha256') == begin_transport.get('request_body_sha256')
                         and dispatched.get('requestBytes') == begin_transport.get('request_bytes')
                         and len(releases) == 1 and releases[0].get('responseComplete') is True
                         and len(responses) == 1 and len(request_releases) == 1 and request_releases[0].get('cancelled') is False
                         and value is not None and ending is not None)
            join = {'transport_id': transport_id, 'route': begin_transport.get('route'), 'gateway_request_id': rid,
                    'request_body_sha256': begin_transport.get('request_body_sha256'), 'request_bytes': begin_transport.get('request_bytes'),
                    'gateway_dispatch_sequence': dispatched.get('sequence'),
                    'gateway_release_sequence': releases[0].get('sequence') if releases else None,
                    'gateway_response_status': responses[0].get('status') if responses else None,
                    'gateway_request_duration_ms': request_releases[0].get('elapsedMs') if request_releases else None,
                    'token_count': value.get('token_count') if value else None, 'stop_type': value.get('stop_type') if value else None,
                    'tokens_evaluated': value.get('tokens_evaluated') if value else None, 'exact_join': exact}
            direct_joins.append(join); all_direct_joins.append({'event_id': event_id, **join})
        host_timings = {kind: next((x.get('elapsed_ns') for x in rows if x.get('kind') == kind), None) for kind in ['symbol_provider', 'scope_build', 'context_select', 'prompt_render']}
        case = {
            'event_id': event_id, 'trace_id': tid, 'trace_record_count': len(rows),
            'identity_expected': expected_identity, 'identity_begin': actual_identity,
            'identity_match': actual_identity == expected_identity and end_event.get('expected') == expected,
            'uri': rr.get('uri'), 'version': rr.get('document_version'), 'content_sha256': rr.get('content_sha256'),
            'cursor': {'line': rr.get('end', {}).get('line'), 'character': rr.get('end', {}).get('character'),
                      'utf16_column': cursor.get('utf16_column'), 'code_point_column': cursor.get('code_point_column')},
            'replacement_range': clean_range(rr),
            'context_applicable': bool(rr.get('uri') == expected['uri'] and rr.get('document_version') == expected['version']
                                       and rr.get('content_sha256') == expected['content_sha256']
                                       and rr.get('start') == {'line': expected['cursor']['line'], 'character': 0}
                                       and rr.get('end') == {'line': expected['cursor']['line'], 'character': expected['cursor']['character']}
                                       and cursor.get('region_line_index') == 0 and cursor.get('utf16_column') == expected['cursor']['character']
                                       and cursor.get('code_point_column') == expected['cursor']['character']),
            'source_backed_context': source_backed(ctx) if context_row else False,
            'scope_mode': ctx.get('scope_mode'), 'schema_version': ctx.get('schema_version'),
            'prefix_lines': len(ctx.get('prefix', [])), 'suffix_lines': len(ctx.get('suffix_lines', [])),
            'region_old_lines': len(ctx.get('region_old', [])), 'history_entries': len(ctx.get('history', [])),
            'history_new_texts_sha256': [text_sha(x.get('new_text', '')) for x in ctx.get('history', [])],
            'diagnostics_empty': ctx.get('diagnostics') == [], 'retrieval_empty': ctx.get('retrieval') == [],
            'selected_references_empty': ctx.get('selected_references') == [],
            'prompt_sha256': context_row.get('prompt_sha256') if context_row else None,
            'prompt_utf8_bytes': context_row.get('prompt_utf8_bytes') if context_row else None,
            'prompt_hash_and_size_match': prompt_hash_ok, 'request_key_sha256': context_row.get('request_key_sha256') if context_row else None,
            'request_key_hash_match': key_hash_ok,
            'provider_begin_instrumented': bool(begin and begin.get('instrumented') is True),
            'provider_finished': finish is not None, 'trace_all_instrumented': all(x.get('instrumented') is True for x in rows),
            'dropped_records': finish.get('dropped_records') if finish else None,
            'provider_elapsed_ns': finish.get('elapsed_ns') if finish else None,
            'measured_observer_overhead_ns': finish.get('measured_observer_overhead_ns') if finish else None,
            'host_timings_ns': host_timings, 'symbol_cache_hit': cache_row.get('hit') if cache_row else None,
            'result_cache_hit': cache_row.get('hit') if cache_row else None,
            'transport_count': len(direct_joins), 'transport_joins': direct_joins,
            'all_direct_transport_joins_exact': bool(direct_joins) and all(x['exact_join'] for x in direct_joins),
            'completion_join_exact': any(x['route'] == '/completion' and x['exact_join'] for x in direct_joins),
            'extra_provider_begin_count': sum(x.get('kind') == 'provider_begin' for x in trace if x.get('event_id') == event_id and x.get('trace_id') != tid),
            'row_review_pass': bool(actual_identity == expected_identity and context_row and finish and len(rows) >= 8
                                    and rr.get('content_sha256') == CONTENT_SHA and source_backed(ctx)
                                    and prompt_hash_ok and key_hash_ok and finish.get('dropped_records') == 0
                                    and all(x.get('instrumented') is True for x in rows)),
        }
        cases.append(case)

    # Renderer controls are independently checked from the retained frames.
    control_rows = {
        'control-visible': case_frame_review(frames, harness, 'control-visible', ['OBSERVER_VISIBLE_CONTROL']),
        'control-multiline': case_frame_review(frames, harness, 'control-multiline', ['OBSERVER_MULTI_FIRST', 'OBSERVER_MULTI_SECOND', 'OBSERVER_MULTI_THIRD']),
        'control-cancel': case_frame_review(frames, harness, 'control-cancel', []),
    }
    multiline = control_rows['control-multiline']
    visible = control_rows['control-visible']
    cancel = control_rows['control-cancel']
    control_pass = bool(visible['frame_coverage'] and visible['matched_texts']['OBSERVER_VISIBLE_CONTROL']['count'] > 0
                        and visible['matched_texts']['OBSERVER_VISIBLE_CONTROL']['geometry_all_ok']
                        and multiline['frame_coverage']
                        and all(multiline['matched_texts'][x]['count'] > 0 and multiline['matched_texts'][x]['geometry_all_ok'] for x in ['OBSERVER_MULTI_FIRST', 'OBSERVER_MULTI_SECOND', 'OBSERVER_MULTI_THIRD'])
                        and cancel['frame_coverage'] and cancel['post_invalidation_ghost_frames'] == 0
                        and cancel.get('control_provider_started_cancelled_resolved') is True)

    # Gateway accounting and causal flags.
    gateway_counts = Counter(x.get('event') for x in gateway)
    dispatch_counts = Counter((x.get('method'), x.get('path')) for x in gateway if x.get('event') == 'backend_dispatched')
    release_counts = Counter((x.get('method'), x.get('path')) for x in gateway if x.get('event') == 'backend_transport_released')
    request_counts = Counter((x.get('method'), x.get('path')) for x in gateway if x.get('event') == 'request_started')
    native_accept_flags = [x.get('nativeTaskAcceptanceProven') for x in gateway if x.get('event') == 'backend_dispatched']
    native_release_flags = [x.get('nativeTaskReleaseProven') for x in gateway if x.get('event') == 'backend_transport_released']
    actual_request_ids = {x['gateway_request_id'] for x in all_direct_joins if x.get('gateway_request_id')}
    all_backend_releases_complete = all(x.get('responseComplete') is True for x in gateway if x.get('event') == 'backend_transport_released')
    automatic_props = [x for x in gateway if x.get('event') == 'backend_dispatched' and x.get('method') == 'GET' and x.get('path') == '/props']
    flags_true = {key: sum(x.get(key) is True for x in gateway) for key in ['cancelled', 'backendKilled', 'nativeTaskAcceptanceProven', 'nativeTaskReleaseProven']}
    request_lifecycles = []
    for rid, rs in gateway_by_request.items():
        started = next((x for x in rs if x.get('event') == 'request_started'), None)
        released = next((x for x in rs if x.get('event') == 'request_released'), None)
        if started:
            request_lifecycles.append({'request_id': rid, 'method': started.get('method'), 'path': started.get('path'),
                                       'start_elapsed_ms': started.get('elapsedMs'), 'release_elapsed_ms': released.get('elapsedMs') if released else None,
                                       'cancelled': released.get('cancelled') if released else None,
                                       'response_status': next((x.get('status') for x in rs if x.get('event') == 'response_forwarded'), None),
                                       'backend_dispatch_count': sum(x.get('event') == 'backend_dispatched' for x in rs),
                                       'backend_release_count': sum(x.get('event') == 'backend_transport_released' for x in rs)})
    request_lifecycles.sort(key=lambda x: (x.get('start_elapsed_ms') is None, x.get('start_elapsed_ms') or 0))

    # Native logs deliberately remain a separate ledger; no task-id guess is made.
    native = parse_native_log(RUN / 'native.log')
    native['identity_metadata'] = {'path': str(RUN / 'native-identity.json'), 'sha256': sha256(RUN / 'native-identity.json'),
                                   'pid': native_identity.get('pid'), 'startTick': native_identity.get('startTick'),
                                   'modelPath': native_identity.get('modelPath')}
    native['gateway_native_task_mapping_supported'] = False
    native['mapping_reason'] = 'gateway nativeTaskAcceptanceProven/nativeTaskReleaseProven flags are false; no native task ID is recorded in provider or gateway rows'

    # Root's exact cache-hit amendment is kept separate from the retained analyzer result.
    link_path = LEAD / 'root-cache-hit-link.json'
    root_link = load_json(link_path) if link_path.exists() else None
    root_link_sha = sha256(link_path) if link_path.exists() else None
    baseline = next(x for x in cases if x['event_id'] == 'baseline')
    repeat = next(x for x in cases if x['event_id'] == 'unchanged-repeat')
    cache_link_independent = bool(root_link and root_link.get('request_key_sha256') == baseline['request_key_sha256'] == repeat['request_key_sha256']
                                  and root_link.get('prompt_sha256') == baseline['prompt_sha256'] == repeat['prompt_sha256']
                                  and root_link.get('baseline_trace_id') == baseline['trace_id'] and root_link.get('cache_hit_trace_id') == repeat['trace_id']
                                  and root_link.get('prior_native_completion_bound') is True and root_link.get('prior_native_stop') == 'eos'
                                  and root_link.get('prior_provider_finished_before_hit') is True and root_link.get('prior_final_cache_key_matches') is True
                                  and root_link.get('hit_has_no_transport') is True)

    # Input identities are concise, but the copied evidence manifest carries every remote/destination hash.
    key_files = {
        'copied_remote_artifact_hash_manifest': {'path': str(OUT / 'artifact-hash-manifest.json'), 'sha256': sha256(OUT / 'artifact-hash-manifest.json')},
        'retained_cache_analyzer': {'path': str(OUT / 'analysis-preparation-analyzer.json'), 'sha256': sha256(OUT / 'analysis-preparation-analyzer.json')},
        'retained_renderer_analyzer': {'path': str(OUT / 'renderer-analysis-viewzone-repair.json'), 'sha256': sha256(OUT / 'renderer-analysis-viewzone-repair.json')},
    }
    for name, path in {
        'host_result': OUT / 'host-result.json', 'harness_events': OUT / 'harness-events.jsonl',
        'provider_trace': OUT / 'provider-trace.jsonl', 'renderer_frames': OUT / 'renderer-frames.jsonl',
        'renderer_ready': OUT / 'renderer-ready.json', 'renderer_observer': OUT / 'renderer-observer.json',
        'renderer_result': OUT / 'renderer-result.json', 'gateway_events': RUN / 'gateway-events.jsonl',
        'native_log': RUN / 'native.log', 'desktop_terminal': RUN / 'terminal.json',
        'remote_terminal': RUN / 'remote-command-terminal.json', 'supervision_terminal': OUT / 'supervision/terminal.json',
        'preparation_manifest': PREP / 'artifact-manifest.json', 'preparation_capsule_manifest': PREP / 'notebook-capsule/capsule-manifest.json',
        'lead_manifest': LEAD / 'manifest.json', 'lead_binding': LEAD / 'binding.json',
        'root_cache_hit_link': link_path,
    }.items():
        if path.exists(): key_files[name] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}

    report = {
        'schema': 'sepalith.campaign.run04.cache-event-independent-review.v1',
        'task': 'RUN-04 independent actual event capture review',
        'status': 'review_complete_supported_claims_only',
        'acceptance': 'event-capture evidence supported; no native task identity or cross-clock latency attribution; no model/editor quality promotion',
        'observed_at': '2026-09-13T17:00:48.451672+00:00',
        'scope': {'plan_root': str(PLAN), 'owned_paths': [str(OUT) + '/**', str(PLAN / 'docs/campaign/receipts/RUN-04-cache-event-live-a-independent-review.json')],
                  'actions': ['CPU-only independent JSON/JSONL/hash/geometry/ledger checks', 'read-only SSH of exactly the two authorized remote run directories', 'retained renderer/cache analyzers run on copied evidence'],
                  'prohibited_actions_not_done': ['no launches, HTTP clients, editor/native/GPU/model/framework/settings/state changes', 'no remote user-data, authentication data, cache/profile trees or private keys read', 'no cross-host clock subtraction and no native task identity inferred by ordering']},
        'inputs': {'desktop_run': str(RUN), 'remote_run': '/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/cache-event-live-a', 'remote_supervision': '/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/cache-event-live-a-supervision',
                   'lead_manifest_sha256': sha256(LEAD / 'manifest.json'), 'lead_binding_sha256': sha256(LEAD / 'binding.json'),
                   'root_manifest_sha256': preflight.get('root_manifest_sha256'), 'renderer_source': ready.get('sourceIdentity'), 'key_files': key_files},
        'startup': {'terminal_failure': terminal.get('failure'), 'desktop_seconds': terminal.get('seconds'), 'remote_command_ssh_exit_code': remote_terminal.get('ssh_exit_code'),
                    'host_process': load_json(OUT / 'host-process.json'), 'host_status': host_result.get('status'), 'host_error': host_result.get('error'),
                    'host_checks': dict(Counter(x.get('status') for x in host_result.get('checks', []))),
                    'supervision_terminal': supervision_terminal, 'gateway_ready': gateway_ready,
                    'preflight_identity_matches_gateway_ready': preflight.get('identity') == gateway_ready,
                    'renderer_actual_url_verified': ready.get('installed', {}).get('verified_actual_url') == ready.get('installed', {}).get('url') and ready.get('installed', {}).get('verified_ready_state') == 'complete',
                    'renderer_source_pin_matches_preflight': ready.get('sourceIdentity') == {'path': preflight.get('options', {}).get('rendererSource'), 'sha256': preflight.get('options', {}).get('rendererSha')},
                    'application_settings': next((x for x in harness if x.get('kind') == 'application_settings_observed'), None)},
        'renderer_controls': {'frame_count': len(frames), 'observer_dropped': observer.get('dropped'), 'observer_graceful_stop': observer.get('graceful_stop'),
                              'observer_stop_reason': observer.get('stop_reason'), 'renderer_result_status': renderer_result.get('status'),
                              'overall_focus_visible': all(x.get('focused') is True and x.get('visibility') == 'visible' for x in frames),
                              'control_pass': control_pass, 'cases': control_rows,
                              'screenshots': observer.get('screenshots', []),
                              'screenshot_hashes': [{'path': str(OUT / Path(x['path']).name), 'bytes': (OUT / Path(x['path']).name).stat().st_size, 'sha256': sha256(OUT / Path(x['path']).name)} for x in observer.get('screenshots', [])],
                              'limits': ['CDP Runtime.evaluate timed out during observer stop after retained capture; no frame drops and graceful stop record remains.', 'PNG screenshots are separate visible evidence; they do not establish model ghost causality.', 'This run contains three deterministic controls and no Q8 model ghost quality cases.']},
        'actual_events': {'expected_event_count': len(EXPECTED), 'observed_event_count': len([x for x in cases if x['trace_id']]), 'source_fixture_sha256': actual_source_sha,
                          'source_fixture_hash_matches_declared': actual_source_sha == CONTENT_SHA, 'trace_record_count': len(trace),
                          'trace_pid_count': len({x.get('pid') for x in trace}), 'trace_pids': sorted({x.get('pid') for x in trace}),
                          'scope_context_private_probe': True, 'diagnostics_supported': False, 'all_cases_row_review_pass': all(x['row_review_pass'] for x in cases),
                          'all_cases_direct_or_root_cache_linked': all(x['all_direct_transport_joins_exact'] and x['completion_join_exact'] for x in cases if x['event_id'] != 'unchanged-repeat') and cache_link_independent,
                          'cases': cases},
        'cache_hit_link': {'root_path': str(link_path), 'root_sha256': root_link_sha, 'independently_matches_provider_and_context': cache_link_independent,
                           'direct_event_level_header_body_binding_count': sum(x['event_id'] != 'unchanged-repeat' and x['all_direct_transport_joins_exact'] for x in cases),
                           'direct_transport_pair_count': len(all_direct_joins), 'all_cases_full_or_prior_cache_link': 7,
                           'retained_analyzer_full_request_binding': load_json(OUT / 'analysis-preparation-analyzer.json').get('full_request_binding'),
                           'retained_analyzer_preserved_as_is': True},
        'gateway': {'event_counts': dict(gateway_counts), 'backend_dispatch_counts': {f'{m} {p}': n for (m, p), n in sorted(dispatch_counts.items())},
                    'backend_release_counts': {f'{m} {p}': n for (m, p), n in sorted(release_counts.items())},
                    'request_started_counts': {f'{m} {p}': n for (m, p), n in sorted(request_counts.items())},
                    'actual_provider_request_ids': len(actual_request_ids), 'actual_provider_dispatches': len(all_direct_joins),
                    'automatic_props_dispatches': len(automatic_props), 'all_backend_releases_response_complete': all_backend_releases_complete,
                    'all_response_forwarded_status_200': all(x.get('status') == 200 for x in gateway if x.get('event') == 'response_forwarded'),
                    'cancelled_true_count': sum(x.get('cancelled') is True for x in gateway), 'backend_killed_true_count': sum(x.get('backendKilled') is True for x in gateway),
                    'native_task_acceptance_true_count': sum(x is True for x in native_accept_flags), 'native_task_release_true_count': sum(x is True for x in native_release_flags),
                    'native_flags_false_counts': {'acceptance_false': sum(x is False for x in native_accept_flags), 'release_false': sum(x is False for x in native_release_flags)},
                    'extra_automatic_call_summary': '31 GET /props backend dispatches (startup, settings/runtime checks, and pre/post transport checks); no unbound POST',
                    'unbound_calls': [{'kind': 'automatic_backend_props', 'count': len(automatic_props), 'reason': 'not provider transport routes'}, {'kind': 'startup_request_id', 'request_id': 'startup', 'reason': 'backend startup props probe; no provider event'}],
                    'request_lifecycles': request_lifecycles,
                    'causal_limit': 'Gateway records transport release and explicitly reports nativeTaskAcceptanceProven/nativeTaskReleaseProven false; request IDs cannot be joined to native task IDs.'},
        'native_separate_ledger': native,
        'timing': {'provider_times_scope': 'instrumented provider elapsed_ns; measured observer overhead is reported and not subtracted',
                   'provider_cases': [{'event_id': x['event_id'], 'provider_elapsed_ns': x['provider_elapsed_ns'], 'measured_observer_overhead_ns': x['measured_observer_overhead_ns'], 'host_timings_ns': x['host_timings_ns'], 'result_cache_hit': x['result_cache_hit']} for x in cases],
                   'gateway_request_times_scope': 'gateway elapsedMs from its own ledger; not subtracted from provider/native clocks',
                   'gateway_request_times': [{'request_id': x['request_id'], 'method': x['method'], 'path': x['path'], 'release_elapsed_ms': x['release_elapsed_ms'], 'response_status': x['response_status']} for x in request_lifecycles if x['request_id'] in actual_request_ids]},
        'limits': ['Six event traces have twelve exact request-ID/instance/route/body-hash/byte joins; unchanged-repeat has no transport and is covered only by the separately hashed exact root cache-hit link.', 'Native log proves six processed/released tasks and prompt-cache activity separately, but provides no recorded request-ID/task-ID relation; chronological proximity is not used.', 'Instrumented host times are not uninstrumented end-to-end latency. Diagnostics are unsupported and the private scopeContext=true setting is a feature probe.', 'No semantic model quality, editor acceptance, cache speedup, or promotion claim follows from this packet.'],
    }
    report_path = OUT / 'review-report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(report_path), 'report_sha256': sha256(report_path),
                      'cases': report['actual_events']['observed_event_count'], 'row_pass': sum(x['row_review_pass'] for x in cases),
                      'control_pass': control_pass, 'direct_event_bindings': report['cache_hit_link']['direct_event_level_header_body_binding_count'],
                      'direct_transport_pairs': report['cache_hit_link']['direct_transport_pair_count'], 'root_cache_link': cache_link_independent,
                      'gateway_dispatches': gateway_counts['backend_dispatched'], 'automatic_props_dispatches': len(automatic_props),
                      'native_tasks': native['processing_task_count']}, sort_keys=True))


if __name__ == '__main__':
    main()
