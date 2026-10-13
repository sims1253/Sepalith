"""Read retained synthetic evidence; no model access or generated-code execution."""
import collections
import datetime
import hashlib
import json
import pathlib
import re

P = pathlib.Path(__file__).resolve().parent
def read(name):
    return json.loads((P / name).read_text())
def rows(name):
    return [json.loads(x) for x in (P / name).read_text().splitlines()]
def sha(b):
    return hashlib.sha256(b).hexdigest()
def emit(name, value):
    (P / name).write_text(json.dumps(value, indent=2) + '\n')

frames = rows('remote/renderer-frames.jsonl')
events = rows('remote/harness-events.jsonl')
inputs = rows('remote/renderer-inputs.jsonl')
gateway = rows('desktop/gateway-events.jsonl')
binding = read('desktop/binding.json')
ready = read('remote/renderer-ready.json')
preflight = read('remote/preflight.json')
startup = next(x for x in gateway if x['event'] == 'startup_verified')
assert sha((P / 'desktop/binding.json').read_bytes()) == preflight['binding_sha256']
assert sha(json.dumps(binding, separators=(',', ':'), ensure_ascii=False).encode()) == startup['bindingSha256']
assert binding == read('desktop/gateway-config.json')['binding']
assert preflight['identity'] == startup['runtime'] == read('desktop/ready.json')['gateway']
assert binding['instanceId'] == preflight['identity']['instanceId']
assert binding['manifest']['model']['sha256'] == preflight['identity']['modelSha256']
assert ready['installed']['url'] == ready['installed']['verified_actual_url'] == ready['target']['url']

starts = {x['id']: x for x in events if x['kind'] == 'auto_case_start'}
ends = {x['id']: x for x in events if x['kind'] == 'auto_case_end'}
requested = {x['input_id']: x for x in events if x['kind'] == 'auto_input_requested'}
applied = {x['input_id']: x for x in events if x['kind'] == 'auto_input_applied'}
assert len(inputs) == len(requested) == len(applied) == 13
assert len({x['input_id'] for x in inputs}) == 13
key_join = []
for key in inputs:
    req, app = requested[key['input_id']], applied[key['input_id']]
    assert key['trusted'] and key['focused'] and key['key'] == req['key']
    assert app['version'] == req['before_version'] + 1
    changes = [x for x in events if x['kind'] == 'text_change' and x['path'] == starts[req['id']]['path'] and x['version'] == app['version'] and x['changes']]
    assert len(changes) == 1
    change = changes[0]
    assert change['content_sha256'] == app['after_sha256']
    assert len(change['changes']) == 1 and change['changes'][0]['text_sha256'] == sha(key['key'].encode())
    assert req['epoch_ms'] <= key['epoch_ms'] <= change['epoch_ms'] <= app['epoch_ms']
    key_join.append({'input_id': key['input_id'], 'path': change['path'], 'version': app['version'], 'content_sha256': app['after_sha256'], 'trusted': True, 'focused': True})
emit('input-document-joins.json', key_join)

gaps = [b['monotonic_ms'] - a['monotonic_ms'] for a, b in zip(frames, frames[1:])]
continuations = []
for frame in frames:
    for g in frame['ghosts']:
        if not g.get('in_view_zone'): continue
        geom = g['geometry']
        assert geom['ownership']['kind'] == 'renderer_ghost_continuation'
        assert geom['ownership']['owner_class'] == 'suggest-preview-text'
        continuations.append((frame['epoch_ms'], g['text'], geom))
accepted = [(t, s, g) for t, s, g in continuations if g['visible']]
assert set(s for _, s, _ in accepted) == {'OBSERVER_MULTI_SECOND', 'OBSERVER_MULTI_THIRD'}
emit('exact-continuation-frame.json', next(f for f in frames if any(g.get('in_view_zone') for g in f['ghosts'])))
coverage = {'frames': len(frames), 'dropped': read('remote/renderer-result.json')['dropped'], 'focused_frames': sum(x['focused'] for x in frames), 'visible_document_frames': sum(x['visibility'] == 'visible' for x in frames), 'max_frame_gap_ms': max(gaps), 'gaps_over_100ms': sum(x > 100 for x in gaps), 'first_epoch_ms': frames[0]['epoch_ms'], 'last_epoch_ms': frames[-1]['epoch_ms'], 'last_frame_after_last_case_ms': frames[-1]['epoch_ms'] - max(x['epoch_ms'] for x in ends.values()), 'continuation_node_observations': len(continuations), 'accepted_continuation_node_observations': len(accepted), 'accepted_continuation_frames': len(set(t for t, _, _ in accepted)), 'ownership_gate': 'Only renderer_ghost_continuation owned by suggest-preview-text; diagnostic sibling ordinary line boxes excluded', 'stop': read('remote/renderer-observer.json')['stop_reason'], 'graceful_stop': read('remote/renderer-observer.json')['graceful_stop']}
emit('coverage-review.json', coverage)

by_id = collections.defaultdict(list)
for g in gateway:
    if 'requestId' in g: by_id[g['requestId']].append(g)
completion_rows = []
for g in gateway:
    if g['event'] != 'request_started' or g.get('path') != '/completion': continue
    chain = by_id[g['requestId']]
    dispatches = [x for x in chain if x['event'] == 'backend_dispatched' and x['path'] == '/completion']
    releases = [x for x in chain if x['event'] == 'backend_transport_released' and x['path'] == '/completion']
    terminal = [x for x in chain if x['event'] == 'request_released']
    assert len(terminal) == 1
    completion_rows.append({'request_id': g['requestId'], 'started_utc': g['utc'], 'dispatch': dispatches, 'native_transport_releases': releases, 'cancelled': any(x['event'] == 'request_cancelled' for x in chain), 'request_released': terminal[0], 'forwarded': [x for x in chain if x['event'] == 'response_forwarded'], 'rejected': [x for x in chain if x['event'] == 'request_rejected']})
emit('completion-request-joins.json', completion_rows)
c_id = '039c886e-522c-4fc0-a349-700265ac9dd2'
chain = by_id[c_id]
dispatch = next(x for x in chain if x['event'] == 'backend_dispatched' and x['path'] == '/completion')
cancel = next(x for x in chain if x['event'] == 'request_cancelled')
release = next(x for x in chain if x['event'] == 'backend_transport_released' and x['path'] == '/completion')
observed = next(x for x in events if x['kind'] == 'auto_cancel_dispatch_observed')
invalidated = next(x for x in events if x['kind'] == 'auto_invalidated')
emit('cancellation-join.json', {'request_id': c_id, 'desktop_chain': chain, 'notebook_observed_dispatch': observed, 'notebook_invalidation': invalidated, 'desktop_dispatch_to_disconnect_ms': (int(cancel['monotonicNs']) - int(dispatch['monotonicNs'])) / 1e6, 'desktop_disconnect_to_transport_release_ms': (int(release['monotonicNs']) - int(cancel['monotonicNs'])) / 1e6, 'notebook_observation_to_edit_ms': invalidated['epoch_ms'] - observed['epoch_ms'], 'scope': 'Causal response-before-edit plus same-desktop monotonic transport timing. No cross-host epoch subtraction; no native task UUID mapping.'})

native = (P / 'desktop/native.log').read_text()
native_started = re.findall(r'task (\d+) \| processing task,', native)
native_released = re.findall(r'task (\d+) \| stop processing:', native)
assert collections.Counter(native_started) == collections.Counter(native_released)
emit('native-aggregate-review.json', {'started_task_ids': native_started, 'released_task_ids': native_released, 'all_started_tasks_have_logged_release': True, 'explicit_cancel_lines': [x for x in native.splitlines() if 'cancel' in x.lower()], 'scope': 'Aggregate server task lifecycle; log lacks gateway UUID and does not prove a cancellation-specific native release latency.', 'native_timing_units': 'Prompt/eval/total print_timing values are milliseconds inside native processing, excluding editor debounce, SSH, HTTP and DOM observation.', 'offload': '43/43 layers to GPU', 'ctx': 4096, 'slots': 1})
summary = {'client_requests_by_route': dict(collections.Counter(x['path'] for x in gateway if x['event'] == 'request_started')), 'completion_started': len(completion_rows), 'completion_dispatched': sum(len(x['dispatch']) for x in completion_rows), 'completion_cancelled': sum(x['cancelled'] for x in completion_rows), 'completion_forwarded': sum(len(x['forwarded']) for x in completion_rows), 'completion_rejected': sum(len(x['rejected']) for x in completion_rows), 'all_completion_requests_released': True, 'case_dispatch_counts': {k: ends[k]['dispatch_after']['dispatchSequence'] - v['dispatch_before']['dispatchSequence'] for k, v in starts.items()}, 'native_started': len(native_started), 'native_released': len(native_released)}
emit('request-count-review.json', summary)

desktop_check = []
for entry in read('desktop/terminal.json')['cleanup']:
    ident = entry['identity']; proc = pathlib.Path('/proc') / str(ident['pid'])
    try:
        stat = (proc / 'stat').read_text(); tick = stat[stat.rfind(')') + 2:].split()[19]
        present = tick == ident['startTick']
        desktop_check.append({'role': entry['role'], 'identity': ident, 'owned_identity_present': present, 'observed_start_tick': tick})
    except FileNotFoundError:
        desktop_check.append({'role': entry['role'], 'identity': ident, 'owned_identity_present': False})
assert not any(x['owned_identity_present'] for x in desktop_check)
emit('desktop-process-recheck.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'identities': desktop_check, 'signals_sent': False, 'lock_acquisition_attempted': False})
print(json.dumps({'coverage': coverage, 'requests': summary, 'input_joins': len(key_join), 'identity_checks': 'PASS'}))
