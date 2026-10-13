"""Lead-owned, bounded CPU layout timing and cache/fresh token comparison."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'work/lead/PRM-06-v3-independent-report.json'
RECEIPT = ROOT / 'receipts/PRM-06-cpu-layout-timing.json'
SERVER = Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server')
MODEL = Path('/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-bf16.gguf')
PORT = 18110


def save(record):
    temporary = RECEIPT.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(RECEIPT)


def request(route, payload=None, timeout=90):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/{route}',
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    return result, time.monotonic() - started


def completion(ids, cache):
    response, wall = request('completion', {
        'prompt': [0, *ids], 'n_predict': 1, 'temperature': 0, 'seed': 3407,
        'return_tokens': True, 'cache_prompt': cache, 'stop': [],
    })
    if response.get('truncated') or response.get('tokens_evaluated') != len(ids) + 1:
        raise RuntimeError('Native server changed the declared complete prompt')
    tokens = response.get('tokens')
    if not isinstance(tokens, list) or len(tokens) != 1:
        raise RuntimeError('One-token timing request returned an unexpected denominator')
    return {'wall_seconds': wall, 'timings': response['timings'], 'tokens': tokens,
            'tokens_evaluated': response['tokens_evaluated'],
            'tokens_predicted': response['tokens_predicted'], 'stop_type': response['stop_type'],
            'cache_prompt': cache}


def main():
    if RECEIPT.exists():
        raise RuntimeError('Use a fresh receipt path for every timing attempt')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', PORT))
    report = json.loads(REPORT.read_text())
    if hashlib.sha256(REPORT.read_bytes()).hexdigest() != 'a4b7e55db0dd0e1e366699dc9ca5677ec3eddad1c6805ca3e47c6d3cef6030a3':
        raise RuntimeError('Reviewed synthetic fixture report changed')
    if hashlib.sha256(SERVER.read_bytes()).hexdigest() != '123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804':
        raise RuntimeError('Pinned native server changed')
    if MODEL.stat().st_size != 5039008768:
        raise RuntimeError('Pinned BF16 model size changed')
    jobs = [(case, layout) for case in report['per_case'] for layout in case['metrics']]
    random.Random(3407).shuffle(jobs)
    argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', '1190s', str(SERVER),
            '-m', str(MODEL), '--alias', 'campaign-prm06-cpu', '--host', '127.0.0.1', '--port', str(PORT),
            '-t', '2', '-tb', '2', '-ngl', '0', '-c', '2048', '--parallel', '1',
            '-b', '256', '-ub', '128', '--temp', '0']
    record = {'task': 'PRM-06', 'owner': 'lead', 'status': 'launch_configured',
              'started_at': datetime.now(timezone.utc).isoformat(), 'argv': argv,
              'model_sha256': '047060045f8e5e896ca44e8933775b9f41686375cefccc216171711534d7c715',
              'model_identity_evidence': 'PRE-05-cpu-conversion.json; immutable export, size rechecked',
              'fixture_report_sha256': hashlib.sha256(REPORT.read_bytes()).hexdigest(),
              'scope': 'Synthetic layout/cache diagnostic on desktop CPU, not target notebook latency or model quality',
              'contention': 'Legacy O2 training PID1700431 still owns CUDA; CPU workers may use two threads each',
              'expected_layout_pairs': len(jobs), 'order': [[case['id'], layout] for case, layout in jobs],
              'results': [], 'CUDA_launches': 0, 'paid_spend': 0}
    save(record)
    child = None
    deadline = time.monotonic() + 1130
    try:
        with (ROOT / 'work/lead/PRM-06-native-server.log').open('x') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
            record.update(status='loading', supervisor_pid=child.pid, process_group=child.pid)
            save(record)
            for _ in range(180):
                if child.poll() is not None:
                    raise RuntimeError('Owned native server exited during readiness')
                try:
                    ready, _ = request('completion', {'prompt': [0, 109], 'n_predict': 1,
                                                      'return_tokens': True, 'temperature': 0}, timeout=10)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            else:
                raise RuntimeError('Native server did not become ready')
            props, _ = request('props', timeout=10)
            record.update(status='running', readiness_tokens=ready.get('tokens'),
                          native_build=props.get('build_info'), context_tokens=props['default_generation_settings']['n_ctx'])
            save(record)
            for index, (case, layout) in enumerate(jobs):
                if time.monotonic() + 90 >= deadline:
                    record['status'] = 'partial_time_budget'
                    break
                metric = case['metrics'][layout]
                before = completion(metric['before_prompt_ids'], False)
                after_warm = completion(metric['after_prompt_ids'], True)
                after_fresh = completion(metric['after_prompt_ids'], False)
                result = {'id': case['id'], 'event_class': case['event_class'], 'layout': layout,
                          'changed_pair': case['changed_pair'],
                          'before_prompt_sha256': metric['before_prompt_sha256'],
                          'after_prompt_sha256': metric['after_prompt_sha256'],
                          'token_lcp': metric['token_lcp'], 'before_fresh': before,
                          'after_warm': after_warm, 'after_fresh': after_fresh,
                          'warm_fresh_greedy_token_equal': after_warm['tokens'] == after_fresh['tokens']}
                # A repeated fresh baseline on three predeclared positions estimates A/A noise.
                if index in (0, 15, 30):
                    result['after_fresh_repeat'] = completion(metric['after_prompt_ids'], False)
                record['results'].append(result)
                save(record)
                if not result['warm_fresh_greedy_token_equal']:
                    raise RuntimeError('Cached and fresh greedy outputs differ on the identical prompt')
            else:
                record['status'] = 'terminal_measurements_complete'
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
            record['supervisor_exit_code'] = child.returncode
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', PORT))
            record['port_bind_available'] = True
        except OSError:
            record['port_bind_available'] = False
        record['lease_released'] = bool(child is not None and child.poll() is not None and record['port_bind_available'])
        record['ended_at'] = datetime.now(timezone.utc).isoformat()
        save(record)
    print(json.dumps({'status': record['status'], 'pairs': len(record['results']),
                      'lease_released': record['lease_released']}))


if __name__ == '__main__':
    main()
