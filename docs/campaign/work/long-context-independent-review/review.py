"""Read saved RUN-05 traces and pinned tokenizer assets; never launch jobs."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import re
import statistics
import sys

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
NATIVE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training')
OUT = Path(__file__).resolve().parent
inputs = {}
checks = 0


def require(value, label):
    global checks
    checks += 1
    if not value:
        raise AssertionError(label)


def read(path):
    path = Path(path)
    require(path.suffix in {'.py', '.ts', '.json', '.log', '.txt'}, 'only explicit source/artifact/tokenizer files')
    raw = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def js(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def load(path):
    return json.loads(read(path))


tokenizer_path = Path('/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain')
for name, expected in {
    'tokenizer.json': '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
    'tokenizer_config.json': 'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b',
}.items():
    require(hashlib.sha256(read(tokenizer_path / name)).hexdigest() == expected, name)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=True, trust_remote_code=False)
require((len(tokenizer), tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (130560, 0, 1, 1), 'tokenizer identity')
protocol_path = EXEC / 'packages/sepalith/src/sepalith/campaign_protocol.py'
require(hashlib.sha256(read(protocol_path)).hexdigest() == '5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156', 'protocol source')
spec = importlib.util.spec_from_file_location('independent_long_protocol', protocol_path)
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)

fixture_path = PLAN / 'work/serving-transition-long-panel/long-transition-fixture.json'
fixture = load(fixture_path)
events = fixture['events']
expected = {}
for event in events:
    context = protocol.PromptContext.from_mapping(event['after']['context'])
    prompt = protocol.render_prompt(context)
    require(prompt == event['after']['promptText'], 'independent Python renderer parity')
    expected[event['eventId']] = {'text': prompt, 'tokens': [0] + tokenizer.encode(prompt, add_special_tokens=False, split_special_tokens=True)}
require(len(expected) == 6, 'six unique events')
manifest = load(PLAN / 'work/lead/theta0-notebook-long-a/manifest.json')
for name, digest in manifest.items():
    require(hashlib.sha256(read(name)).hexdigest() == digest, 'staged source manifest')
root_review = load(PLAN / 'work/lead/theta0-long-context-a/root-review.json')
read(PLAN / 'work/lead/theta0-long-context-a/review_saved.py')
profiles = []
all_rows = []
parity = {}


def log_time(line):
    # All saved lines are within the first minute. The final two groups are
    # milliseconds/microseconds; compare on the server's own relative clock.
    match = re.match(r'0\.(\d+)\.(\d+)\.(\d+) ', line)
    require(match is not None, 'short server log timestamp')
    second, milli, micro = map(int, match.groups())
    return second * 1000 + milli + micro / 1000


for backend, context, directory in [
    *[('cuda', ctx, NATIVE / 'RUN-05-theta0-long-context-a' / str(ctx)) for ctx in (2048, 4096, 8192)],
    ('notebook_vulkan', 4096, NATIVE / 'RUN-05-theta0-notebook-long-4k-a'),
    ('notebook_vulkan', 8192, NATIVE / 'RUN-05-theta0-notebook-long-8k-a'),
]:
    trace = load(directory / 'trace.json')
    terminal = load(directory / 'terminal.json')
    props = load(directory / ('props.json' if backend == 'cuda' else 'remote-props.json'))
    launch = load(directory / ('server-launch.json' if backend == 'cuda' else 'remote-launch.json'))
    log = read(directory / ('server.log' if backend == 'cuda' else 'remote-server.log')).decode()
    require(props == trace['serverProps'], 'saved/live trace props agreement')
    require(props['default_generation_settings']['n_ctx'] == context and props['total_slots'] == 1, 'context and single slot')
    require(props['build_info'] == 'b10453-3cb7ffb1a', 'reported native build')
    require(props['model_path'].endswith('/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf'), 'reported selected model path')
    require('offloaded 43/43 layers to GPU' in log, 'actual offload log')
    require(launch['argv'][launch['argv'].index('-c') + 1] == str(context), 'launch context')
    require(len(trace['rows']) == 6 and trace['mode'] == 'serial' and trace['deadlineMs'] == 5000, 'event/deadline mode')
    for name, digest in trace['source']['exactSourceFiles'].items():
        source = protocol_path if name.endswith('.py') else EXEC / 'extensions/vscode-sepalith/src' / name
        require(hashlib.sha256(read(source)).hexdigest() == digest, 'exact runtime source bytes')
    cancel_records, releases, prompts = {}, {}, []
    for line_number, line in enumerate(log.splitlines(), 1):
        match = re.search(r'cancel task, id_task = (\d+)', line)
        if match:
            cancel_records[int(match[1])] = {'line': line_number, 'relative_ms': log_time(line)}
        match = re.search(r'release:.*task (\d+) \| stop processing: n_tokens = (\d+)', line)
        if match:
            releases[int(match[1])] = {'line': line_number, 'relative_ms': log_time(line), 'tokens_at_release': int(match[2])}
        match = re.search(r'task (\d+) \| new prompt,.*task.n_tokens = (\d+)', line)
        if match:
            prompts.append({'task_id': int(match[1]), 'tokens': int(match[2]), 'line': line_number})
    rows = []
    for index, row in enumerate(trace['rows']):
        event_id = events[index]['eventId']
        require(row['eventId'] == event_id and row['eventIndex'] == index, 'event identity/order')
        wanted = expected[event_id]
        tokenize = [c for c in row['transportCalls'] if c['path'] == '/tokenize']
        completion = [c for c in row['transportCalls'] if c['path'] == '/completion']
        require(len(tokenize) == 1 and tokenize[0]['status'] == 200, 'one tokenizer response')
        require(tokenize[0]['body'] == {'content': wanted['text'], 'add_special': False, 'parse_special': False, 'with_pieces': False}, 'actual tokenizer request')
        require([0] + tokenize[0]['response']['tokens'] == wanted['tokens'], 'actual HF/native token identity')
        for call in row['transportCalls']:
            require(call['requestSha256'] == sha(js(call['body'])), 'transport request digest')
            if 'response' in call:
                require(call['responseSha256'] == sha(js(call['response'])), 'transport response digest')
            require(int(call['finishedMonotonicNs']) >= int(call['startedMonotonicNs']), 'transport monotonic order')
        duplicate = index == 5
        require(row['duplicateWarmControl'] == duplicate, 'duplicate control denominator')
        item = {'event': event_id, 'prompt_tokens': len(wanted['tokens']), 'prompt_sha256': sha(wanted['text']), 'elapsed_ms': row['elapsedMs'], 'duplicate_control': duplicate}
        if context == 2048:
            require(len(wanted['tokens']) + 192 > context and not completion and row['contextOverflowRejected'], 'input rejection without generation')
            require(row['error']['code'] == 'context_budget' and not row['timeout'] and row['parserStatus'] is None, 'input rejection category')
            item['status'] = 'input_budget_rejection_no_model_output'
        else:
            require(len(completion) == 1, 'one completion dispatch')
            call = completion[0]
            require(call['body'] == {'prompt': wanted['tokens'], 'n_predict': 192, 'temperature': 0, 'stream': False, 'cache_prompt': True, 'return_tokens': True}, 'actual completion body identity')
            if row['timeout']:
                require(backend == 'notebook_vulkan' and not duplicate and 'response' not in call and row['cancellationReason'] == 'deadline', 'natural deadline failure')
                require(row['completionDispatchStarted'] and not row['clientAbortRequested'] and row['serverCancellation'] == 'server_cancellation_unverified', 'deadline versus caller cancellation fields')
                require(5000 <= row['elapsedMs'] < 5020 and not row['qualityDenominatorEligible'], 'deadline tolerance and exclusion')
                item['status'] = 'deadline_timeout_no_client_model_output'
            else:
                response = call['response']; ids = response['tokens']
                require(ids[-1] == 1 and response['stop_type'] == 'eos' and not response['truncated'], 'actual EOS and nontruncation')
                require(len(ids) == response['tokens_predicted'] == response['timings']['predicted_n'], 'native output count agreement')
                require(response['tokens_evaluated'] == len(wanted['tokens']), 'native total prompt token count')
                text = tokenizer.decode(ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                require(text == response['content'] and ids == row['generatedTokenIds'], 'output decode and token parity')
                require(row['promptTokenIds'] == wanted['tokens'] and row['promptTextSha256'] == sha(wanted['text']), 'trace prompt fields')
                require(row['promptTokenIdsSha256'] == sha(js(wanted['tokens'])) and row['generatedTokenIdsSha256'] == sha(js(ids)) and row['rawResponseSha256'] == sha(text), 'trace output hashes')
                parsed = protocol.parse_output(text, protocol.PromptContext.from_mapping(events[index]['after']['context']))
                require(parsed.status == row['parserStatus'] == 'accepted' and parsed.operation == row['operation'], 'independent parser agreement')
                signature = (wanted['tokens'], ids, text, row['operation'], row['planHash'])
                if event_id in parity:
                    require(signature == parity[event_id], 'cross-profile actual output/recorded-plan identity')
                parity[event_id] = signature
                item.update(status='excluded_repeat_control_response' if duplicate else 'natural_event_response', output_tokens=len(ids), operation=row['operation'], server_timings=response['timings'])
        rows.append(item)
    if backend == 'cuda':
        require('95s' in launch['argv'] and '--foreground' in launch['argv'], 'recorded CUDA process deadline')
        require(terminal['client']['exit_code'] == terminal['server']['exit_code'] == 0 and not terminal['client']['pid_exists'] and not terminal['server']['pid_exists'], 'recorded CUDA terminal state')
        require(terminal['trace_sha256'] == inputs[str(directory / 'trace.json')], 'CUDA terminal trace digest')
        for child in terminal['server']['children']:
            require(child['exists'] is False, 'recorded child exit')
            env = load(directory / f"server-env-{child['pid']}.json")
            require(env == ['GGML_CUDA_GRAPH_OPT=0'], 'actual CUDA env snapshot')
            maps = read(directory / f"server-maps-{child['pid']}.txt").decode()
            require('libggml-cuda' in maps, 'actual CUDA backend mapping')
    else:
        remote_terminal = load(directory / 'remote-terminal.json')
        require(terminal['client_exit_code'] == terminal['ssh_exit_code'] == remote_terminal['server_exit_code'] == 0 and terminal['error'] is None, 'recorded notebook terminal exits')
        for name, digest in terminal['artifacts'].items():
            require(hashlib.sha256(read(directory / name)).hexdigest() == digest, 'notebook artifact receipt digest')
        device = load(directory / 'remote-live-device-audit.json')
        require(any(x['dri_fds'] and any('libggml-vulkan' in line for line in x['mapped_backend_libraries']) for x in device['server']), 'actual Vulkan mappings and device fd')
        require(launch['supervisor_sha256'] == manifest[str(PLAN / 'work/lead/theta0-notebook-long-a/notebook_long_server.py')], 'saved executed supervisor source pin')
        local_launch = load(directory / 'launch.json')
        require(local_launch['supervisor_sha256'] == manifest[str(PLAN / 'work/lead/theta0-notebook-long-a/run_notebook_long.py')], 'saved executed controller source pin')
        require('140s' in launch['argv'] and '180s' in local_launch['argv'][-1], 'notebook server and outer deadline arguments')
        require(launch['quality_deadlines'] == 'client65s; server140s; outer180s; no profiling', 'recorded deadline policy')
        require(launch['model_sha256'] == '22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559', 'root-reported model digest only')
        require(len(cancel_records) == 5 and len(prompts) == 6, 'server task/cancellation denominators')
        require([p['tokens'] for p in prompts] == [len(expected[e['eventId']]['tokens']) for e in events], 'server prompt counts/order')
        require([p['task_id'] for p in prompts[:5]] == list(cancel_records), 'serial cancelled task identities')
    natural_times = [r['elapsed_ms'] for r in rows if r['status'] == 'natural_event_response']
    cancellation = [{"task_id": task, 'cancel': observed, 'release': releases[task], 'cancel_to_release_ms': round(releases[task]['relative_ms'] - observed['relative_ms'], 3)} for task, observed in cancel_records.items()]
    require(all(row['cancel_to_release_ms'] >= 0 for row in cancellation), 'same-task cancellation then release')
    record = {'backend': backend, 'context': context, 'rows': rows, 'natural_events': 5, 'natural_responses': len(natural_times), 'duplicate_controls': 1, 'input_rejections': sum(r['status'] == 'input_budget_rejection_no_model_output' for r in rows), 'deadline_timeouts': sum(r['status'] == 'deadline_timeout_no_client_model_output' for r in rows), 'completed_client_outputs': sum('output_tokens' in r for r in rows), 'server_cancellation_evidence': cancellation, 'reported_native_build': props['build_info'], 'recorded_elapsed_seconds': terminal['seconds']}
    if natural_times:
        record['natural_request_ms'] = {'min': min(natural_times), 'median': statistics.median(natural_times), 'max': max(natural_times)}
        require(record['natural_request_ms'] == {k.replace('_ms', ''): v for k, v in root_review['metrics'][str(context)].items() if k.endswith('_ms')}, 'independent/root CUDA timing agreement')
    profiles.append(record)
    all_rows.extend(rows)

result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'assertions_passed': checks, 'profiles': profiles,
          'totals': {'rows': len(all_rows), 'independent_rendered_prompts': len(expected), 'HF_native_token_comparisons': len(all_rows), 'decoded_completed_outputs': sum('output_tokens' in r for r in all_rows), 'natural_events': 25, 'natural_responses': sum(r['status'] == 'natural_event_response' for r in all_rows), 'input_rejections': 6, 'deadline_timeouts': 10, 'excluded_repeat_controls': 5},
          'input_sha256': inputs,
          'limits': ['Input-budget rejections are not malformed model outputs.', 'Notebook natural requests: 0/5 within 5 seconds per profile; repeat controls excluded.', 'Client elapsedMs = local hrtime nanoseconds / 1e6, spanning tokenize/completion/protocol handling. Notebook client timing includes tunnel and queue delays, not editor display latency.', 'Server timings prompt_ms/predicted_ms are milliseconds, prompt_n is newly evaluated prompt tokens; repeat controls use prompt_n=1, not 1967 fresh prompt tokens.', 'Server same-task cancel/release logs support eventual stop, not immediate cancellation, editor nonpublication, or synchronized client-to-server cancellation latency.', 'Saved terminal records only; no live PID audit or process release performed.', 'Source pins cover enumerated sources, not a full transitive closure. Model hash is reported root evidence; no model bytes were read or hashed.', 'All profiles use 1927–1975 token prompts and output cap192; 8192 allocation does not establish 8K-token stress or comparative quality.']}
(OUT / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'assertions_passed': checks, 'totals': result['totals'], 'profiles': [{k:v for k,v in p.items() if k not in ['rows','server_cancellation_evidence']} for p in profiles]}, indent=2))
