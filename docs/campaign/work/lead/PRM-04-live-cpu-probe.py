"""Bounded native CPU transport/termination proof; no editing-quality claim."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
BINARY = Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server')
MODEL = Path('/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-bf16.gguf')
TOKENIZER = MODEL.with_name('minicpm5-2b-midtrain')
PORT = 18109
RECEIPT = ROOT / 'receipts/PRM-04-live-cpu-parity.json'
sys.path.insert(0, str(EXEC / 'packages/sepalith/src'))
from sepalith.campaign_protocol import encode_prompt, parse_output, render_prompt, serialize_target
from transformers import AutoTokenizer


def save(record):
    temporary = RECEIPT.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(RECEIPT)


def request(route, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/{route}', data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    if RECEIPT.exists():
        raise RuntimeError('Each probe attempt requires a fresh receipt path')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', PORT))
    if hashlib.sha256(BINARY.read_bytes()).hexdigest() != '123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804':
        raise RuntimeError('Pinned native server hash differs')
    export = json.loads((ROOT / 'receipts/PRE-05-cpu-conversion.json').read_text())
    assert export['status'] == 'verified_CPU_export' and MODEL.stat().st_size == export['output_bytes']
    module_path = EXEC / 'packages/sepalith/tests/test_campaign_protocol.py'
    spec = importlib.util.spec_from_file_location('prm04_fixture', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = module.context()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    prompt = render_prompt(context)
    expected_ids = encode_prompt(context, tokenizer)
    argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', '590s', str(BINARY),
            '-m', str(MODEL), '--alias', 'campaign-midtrain-transport-preflight',
            '--host', '127.0.0.1', '--port', str(PORT), '-t', '2', '-tb', '2',
            '-ngl', '0', '-c', '1024', '--parallel', '1', '-b', '256', '-ub', '128', '--temp', '0']
    log_path = ROOT / 'work/lead/PRM-04-native-server.log'
    record = {'task': 'PRM-04', 'owner': 'lead', 'status': 'launch_configured',
              'started_at': datetime.now(timezone.utc).isoformat(), 'argv': argv,
              'model_sha256': export['output_sha256'], 'model_bytes': export['output_bytes'],
              'binary_sha256': '123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804',
              'resource': 'CPU two threads; ngl0; current legacy CUDA owner continues',
              'scope': 'Pinned transport/token/EOG and constrained output fixtures; no model-quality or latency decision',
              'checks': [], 'CUDA_launches': 0}
    save(record)
    child = None
    try:
        with log_path.open('x') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                                     start_new_session=True)
            record.update(status='server_loading', supervisor_pid=child.pid, process_group=child.pid)
            save(record)
            ready_deadline = time.monotonic() + 240
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'Owned server exited during load: {child.returncode}')
                try:
                    result = request('completion', {'prompt': [0, 109], 'n_predict': 1,
                                                     'temperature': 0, 'return_tokens': True, 'cache_prompt': False}, 30)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= ready_deadline:
                        raise
                    time.sleep(1)
            record.update(status='server_ready_by_completion', launch_probe=result,
                          properties=request('props', timeout=30))
            save(record)
            for text in (prompt, "literal <- '</s>'", "literal <- '<s>'", "😀 café\n  "):
                native = request('tokenize', {'content': text, 'add_special': False, 'parse_special': False}, 30)['tokens']
                hf = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
                assert native == hf
                record['checks'].append({'kind': 'live_token_parity', 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                         'tokens': len(native), 'pass': True})
            shared = {'prompt': expected_ids, 'n_predict': 64, 'temperature': 0, 'return_tokens': True,
                      'cache_prompt': False, 'stop': [], 'seed': 3407}
            for token_id in (1, 130073):
                response = request('completion', {**shared, 'n_predict': 1, 'logit_bias': [[token_id, 1000.0]]})
                assert response['tokens'] == [token_id] and response['stop_type'] == 'eos'
                record['checks'].append({'kind': 'forced_EOG', 'forced_token': token_id, 'response': response,
                                         'primary_acceptance': token_id == 1, 'pass': True})
                save(record)
            for operation, new in (
                ('no_op', list(context.region_old)), ('delete', []),
                ('replace', ['', '  keep spaces  ', '']),
                ('replace', ["literal <- '</s>'", 'marker <- ">>>>>>> UPDATED"']),
            ):
                target = serialize_target(operation, new)
                grammar = 'root ::= ' + json.dumps(target, ensure_ascii=False)
                response = request('completion', {**shared, 'grammar': grammar, 'logit_bias': [[1, 1000.0]]})
                assert response['content'] == target, 'Constrained native output bytes differ'
                assert response['tokens'][-1] == 1 and response['stop_type'] == 'eos'
                parsed = parse_output(response['content'], context)
                assert parsed.status == 'accepted' and parsed.operation == operation
                actual = list(context.region_old) if parsed.operation == 'no_op' else list(parsed.body)
                assert actual == new
                record['checks'].append({'kind': 'constrained_protocol_roundtrip', 'operation': operation,
                                         'response': response, 'pass': True})
                save(record)
            record['status'] = 'verified_CPU_transport'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            record['server_supervisor_exit_code'] = child.returncode
        with socket.socket() as sock:
            record['port_released'] = sock.connect_ex(('127.0.0.1', PORT)) != 0
        record['ended_at'] = datetime.now(timezone.utc).isoformat()
        record['lease_released'] = record['port_released'] and child is not None and child.poll() is not None
        save(record)
    print(json.dumps({'status': record['status'], 'checks': len(record['checks']),
                      'lease_released': record['lease_released']}))


if __name__ == '__main__':
    main()
