"""Independent checks of saved editor frames, events, identities, and buffers."""
from pathlib import Path
import datetime
import hashlib
import json
import zipfile

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
CAPSULE = WORK / 'lead/theta0-editor-v2-capsule'
EVIDENCE = WORK / 'lead/theta0-editor-v2-evidence'
hashes = {}
checks = 0


def require(value, label):
    global checks
    checks += 1
    if not value:
        raise AssertionError(label)


def read(file):
    raw = file.read_bytes()
    hashes[str(file)] = hashlib.sha256(raw).hexdigest()
    return raw


def load(root, name):
    return json.loads(read(root / name))


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


capsule_manifest = load(CAPSULE, 'capsule-manifest.json')
for name, expected in capsule_manifest.items():
    require(hashlib.sha256(read(CAPSULE / name)).hexdigest() == expected, 'capsule source digest')
frames = [json.loads(line) for line in read(EVIDENCE / 'renderer-frames.jsonl').splitlines()]
events = [json.loads(line) for line in read(EVIDENCE / 'harness-events.jsonl').splitlines()]
analysis = load(HERE, 'renderer-analysis.json')
host = load(EVIDENCE, 'host-result.json')
observer = load(EVIDENCE, 'renderer-observer.json')
observer_result = load(EVIDENCE, 'renderer-result.json')
ready = load(EVIDENCE, 'renderer-ready.json')
preflight = load(EVIDENCE, 'preflight.json')
inventory = load(EVIDENCE, 'asset-inventory.json')
settings = load(EVIDENCE, 'settings.json')
guard = load(EVIDENCE, 'guard-terminal.json')
guard_launch = load(EVIDENCE, 'guard-launch.json')
run = load(EVIDENCE, 'run-result.json')
launch = load(EVIDENCE, 'launch.json')
buffers = load(EVIDENCE, 'acceptance-v2-buffers.json')
root_parse = load(EVIDENCE, 'root-r-parse.json')
require(len(root_parse) == len(buffers), 'root parse denominator')
require(host['events'] == events, 'host events and durable JSONL agree')
require(len(frames) == observer['frames'] == 2467 and observer['dropped'] == 0, 'recorded frame denominator')
require(len(events) == 59, 'recorded event denominator')
require(all(a['monotonic_ms'] < b['monotonic_ms'] and a['epoch_ms'] <= b['epoch_ms'] for a, b in zip(frames, frames[1:])), 'frame clock order')
require(all(f['focused'] and f['visibility'] == 'visible' for f in frames), 'focused visible frame samples')
host_start_ms = round(datetime.datetime.fromisoformat(host['started_at'].replace('Z', '+00:00')).timestamp() * 1000)
require(all(e['epoch_ms'] - host_start_ms == e['elapsed_ms'] for e in events), 'host epoch/elapsed time units')
require(ready['sourceIdentity'] == observer['sourceIdentity'], 'target renderer recorded identity')
require(ready['at'] < min(e['epoch_ms'] for e in events if e['kind'] == 'case_start'), 'observer attached before cases')
require(guard['child_exit_code'] == 0 and guard['survivors'] == [], 'saved guard completion only')
require(run['host']['code'] == 0 and not run['host']['timed_out'] and not run['host']['interrupted'], 'saved host completion only')
require(run['host_result']['checks'] == host['checks'], 'host summary checks agree')
require(preflight['inventory'] == inventory and inventory['ok'], 'inventory/preflight agreement')
require(inventory['model']['sha256'] == '22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559', 'declared selected Q8 identity')
require(inventory['model']['observed_bytes'] == inventory['model']['bytes'] == 2679710496, 'saved model size observation')
require(preflight['no_model_content_read'] is True, 'preflight model hash limitation preserved')
require(all(a['sha256'] == a['observed_sha256'] for a in inventory['assets']) and len(inventory['assets']) == 10, 'saved native asset hashes')
require(preflight['extension']['sha256'] == capsule_manifest['primary.vsix'] == 'a56d2e442ae7abf266ef5f508d25443d00acecb9a78f92787b7ac332e3e2836d', 'accepted VSIX identity')
require(settings['sepalith.modelPath'] == launch['model_path'] == inventory['model']['path'], 'configured model path identity')
require(settings['sepalith.backend'] == launch['backend'] == 'vulkan' and settings['sepalith.contextSize'] == 4096 and settings['sepalith.requestTimeoutMs'] == 5000, 'configured production route')
require(settings['sepalith.serverPath'] == '' and settings['sepalith.debounceMs'] == 0, 'managed route and manual triggers')
with zipfile.ZipFile(CAPSULE / 'primary.vsix') as archive:
    package = json.loads(archive.read('extension/package.json'))
    bundled = archive.read('extension/dist/extension.js').decode()
require(package['version'] == '0.0.7', 'VSIX package version')
require(any(e['kind'] == 'sepalith_extension_activated' and e['version'] == package['version'] for e in events), 'activated extension version')
require('checksum(modelOverride, signal) !== m.model.sha256' in bundled, 'pinned product source contains override hash guard')
require('registerInlineCompletionItemProvider({ language: "r" }, provider)' in bundled, 'production R provider route')

cases = []
for start in [e for e in events if e['kind'] == 'case_start']:
    cid = start['id']
    pick = lambda kind: next((e for e in events if e['kind'] == kind and e.get('id') == cid and e['epoch_ms'] >= start['epoch_ms']), None)
    trigger, end, commit, invalidated = [pick(kind) for kind in ('case_trigger', 'case_end', 'case_commit', 'case_invalidated')]
    require(trigger is not None and end is not None, 'case trigger/end pair')
    finish = (commit or end)['epoch_ms']
    before = [f for f in frames if trigger['epoch_ms'] - 100 <= f['epoch_ms'] < trigger['epoch_ms']]
    after = [f for f in frames if trigger['epoch_ms'] <= f['epoch_ms'] <= finish]
    samples = before + after
    gaps = [b['monotonic_ms'] - a['monotonic_ms'] for a, b in zip(samples, samples[1:])]
    covered = bool(before and not before[-1]['ghosts'] and after and after[-1]['epoch_ms'] >= finish - 100 and max(gaps) <= 100)
    expected = next(r for r in analysis['cases'] if r['id'] == cid)
    require(covered and expected['coverage'] == 'bounded_frame_observation', 'independent per-case frame coverage')
    require(len(samples) == expected['observed_frames'] and max(gaps) == expected['max_frame_gap_ms'], 'frame denominator and gap agreement')
    visible = [f for f in after if f['ghosts']]
    require(len(visible) == expected['visible_frames'], 'visible frame denominator')
    row = {'id': cid, 'source': start['source'], 'trigger_epoch_ms': trigger['epoch_ms'], 'end_epoch_ms': end['epoch_ms'], 'observed_frames': len(samples), 'visible_frames': len(visible), 'max_frame_gap_ms': max(gaps), 'coverage': 'bounded_for_configured_selectors'}
    if visible:
        first = visible[0]
        previous = samples[samples.index(first) - 1]
        interval = [max(0, previous['epoch_ms'] - trigger['epoch_ms']), first['epoch_ms'] - trigger['epoch_ms']]
        require(interval == expected['first_rendered_ghost_interval_ms'], 'first rendered interval units')
        row['first_rendered_ghost_interval_ms'] = interval
        row['observed_text'] = [g['text'] for g in first['ghosts']]
    if invalidated:
        row['edit_after_trigger_ms'] = invalidated['epoch_ms'] - trigger['epoch_ms']
        row['post_invalidation_window_ms'] = finish - invalidated['epoch_ms']
        row['post_invalidation_visible_frames'] = sum(bool(f['ghosts']) for f in after if f['epoch_ms'] >= invalidated['epoch_ms'])
    cases.append(row)
require(len(cases) == 6 and sum(c['source'] == 'deterministic_test_provider' for c in cases) == 2, 'two controls/four production-route cases')
last_case_end = max(c['end_epoch_ms'] for c in cases)
require(frames[-1]['epoch_ms'] > last_case_end, 'saved frames cover all completed cases despite observer terminal error')
control_start = next(e for e in events if e['kind'] == 'control_provider_started' and e['stale'])
control_cancel = next(e for e in events if e['kind'] == 'control_provider_cancelled' and e['call'] == control_start['call'])
control_resolved = next(e for e in events if e['kind'] == 'control_provider_resolved' and e['call'] == control_start['call'])
require(control_start['epoch_ms'] < control_cancel['epoch_ms'] < control_resolved['epoch_ms'] and control_resolved['cancelled'], 'actual deterministic provider cancellation chronology')

buffer_audit = []
for row in buffers:
    require(text_sha(row['before_text']) == row['before_sha256'] and text_sha(row['after_text']) == row['after_sha256'], 'exact accepted buffer hashes')
    require(row['changed'] and row['before_version'] == 1 and row['after_version'] == 2 and row['saved_to_disk'] is False, 'unsaved actual commit identity')
    require(row['after_text'].count('\n') - row['before_text'].count('\n') == row['added_line_count'], 'exact added line count')
    cid = row['id']
    commit = next(e for e in events if e['kind'] == 'case_commit' and e['id'] == cid)
    change = next(e for e in events if e['kind'] == 'text_change' and e['path'] == row['path'] and e['changes'] and e['content_sha256'] == row['after_sha256'])
    edit = change['changes'][0]
    lines = row['before_text'].splitlines(keepends=True)
    start = sum(len(x) for x in lines[:edit['range']['start']['line']]) + edit['range']['start']['character']
    stop = sum(len(x) for x in lines[:edit['range']['end']['line']]) + edit['range']['end']['character']
    prefix, suffix = row['before_text'][:start], row['before_text'][stop:]
    require(row['after_text'].startswith(prefix) and row['after_text'].endswith(suffix), 'buffer prefix/suffix preserved')
    inserted = row['after_text'][len(prefix):len(row['after_text']) - len(suffix) if suffix else None]
    require(text_sha(inserted) == edit['text_sha256'] and len(inserted) == edit['text_chars'], 'actual text-change insertion digest')
    require(prefix + inserted + suffix == row['after_text'] and commit['epoch_ms'] <= change['epoch_ms'], 'exact commit reconstruction')
    # These authored fixtures contain no brace literals or braces in comments.
    # This structural check is not a replacement for root's exact R parse.
    unmatched = row['after_text'].count('{') - row['after_text'].count('}')
    destination = HERE / (cid + '.after.R')
    destination.write_bytes(row['after_text'].encode())
    require(hashlib.sha256(destination.read_bytes()).hexdigest() == row['after_sha256'], 'unmodified parser handoff')
    parsed = next(p for p in root_parse if p['id'] == cid)
    require(parsed['buffer_sha256'] == row['after_sha256'], 'root R parse exact-byte binding')
    require(parsed['exit_code'] == (2 if unmatched else 0), 'root parse agrees with structural inspection')
    buffer_audit.append({'id': cid, 'path': str(destination), 'sha256': row['after_sha256'], 'added_line_count': row['added_line_count'], 'commit_to_change_ms': change['epoch_ms'] - commit['epoch_ms'], 'unmatched_open_braces': unmatched, 'R_parser_status': 'root_reported_pass' if parsed['exit_code'] == 0 else 'root_reported_fail', 'R_parser_evidence': parsed, 'accepted_scope': 'actual unsaved inline commit mechanics; root parse establishes syntax only'})

screenshots = []
for shot in observer['screenshots']:
    raw = read(EVIDENCE / shot['path'])
    matching = next(c for c in cases if c['trigger_epoch_ms'] <= shot['first_frame_epoch_ms'] <= c['end_epoch_ms'])
    screenshots.append({**shot, 'case_id': matching['id'], 'sha256': hashlib.sha256(raw).hexdigest(), 'requested_after_trigger_ms': shot['screenshot_requested_epoch_ms'] - matching['trigger_epoch_ms'], 'completed_after_trigger_ms': shot['screenshot_completed_epoch_ms'] - matching['trigger_epoch_ms']})
report = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'assertions_passed': checks, 'denominators': {'frames': len(frames), 'dropped_frames_reported': observer['dropped'], 'events': len(events), 'control_cases': 2, 'production_route_cases': 4, 'actual_committed_buffers': len(buffers), 'multiline_committed_buffers': 2, 'root_R_parse_passed': sum(p['exit_code'] == 0 for p in root_parse), 'root_R_parse_failed': sum(p['exit_code'] != 0 for p in root_parse)}, 'cases': cases, 'buffer_audit': buffer_audit, 'screenshots': screenshots, 'observer_terminal': observer_result, 'last_frame_after_last_case_ms': frames[-1]['epoch_ms'] - last_case_end, 'production_request_log_available': False, 'capsule_file_hashes_verified': len(capsule_manifest), 'target_renderer_identity_recorded_only': observer['sourceIdentity'], 'VSIX': {'sha256': capsule_manifest['primary.vsix'], 'extension_version': package['version'], 'bundle_sha256': text_sha(bundled)}, 'model_identity_scope': 'Selected model manifest hash/size and configured managed route agree; preflight observed size only. Product source contains override checksum guard. No model bytes or actual runtime load log inspected.', 'input_sha256': hashes, 'limitations': ['Production in-flight cancellation remains pending without a request start/cancel/response ledger correlated to the edit.', 'Observer positive control validates the selected single-line selectors; no positive multiline observer control was supplied.', 'Both multiline after-buffers lack the outer closing brace and fail the root exact-byte R parse with unexpected end of input. No repair was applied.', 'DOM frame intervals precede compositor presentation. Screenshot evidence bounds visibility later; this is not native request latency or representative p95.', 'Renderer observer terminated with Runtime.evaluate timeout; per-case coverage remains independently verified for saved frames only.', 'Two plaintext deterministic provider cases are instrumentation controls, not selected-Q8 results.', 'Multiline insertion through inlineSuggest.commit does not establish explicit WorkspaceEdit multiline replacement.', 'Headless Xvfb screenshots and saved process cleanup do not establish real-user interaction or current live process state.']}
(HERE / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'assertions_passed': checks, 'denominators': report['denominators'], 'buffers': buffer_audit, 'cases': cases}, indent=2))
