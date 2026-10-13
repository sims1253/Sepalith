"""Root-owned four-cell Q8 prewarm pilot; fresh native server for every cell."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path('/home/m0hawk/.local/share/sepalith-campaign-20260915')
PACK = Path(__file__).resolve().parent
RUN = ROOT / 'runs/cache-ram-ab-a/pair0-default'
BIN = ROOT / 'build-b10453-vulkan-avx2/bin/llama-server'
MODEL = ROOT / 'models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf'
PORT = 18403
STOP = False
START = time.monotonic()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def interrupted(*_):
    global STOP
    STOP = True

def check():
    if STOP or (RUN / 'stop').exists():
        raise RuntimeError('lead_stop')
    if time.monotonic() - START > 90:
        raise RuntimeError('total_90s_deadline')
    mem = dict(s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0]) < 2 * 1024 * 1024:
        raise RuntimeError('memory_floor_2GiB')

def free_port():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', PORT))

def reap(p):
    if p is None:
        return None
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=5)
    return {'pid': p.pid, 'exit_code': p.returncode, 'pid_exists': Path(f'/proc/{p.pid}').exists()}

def ready(server):
    until = time.monotonic() + 60
    while time.monotonic() < until:
        check()
        if server.poll() is not None:
            raise RuntimeError('server_early_exit')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=1) as f:
                health = json.load(f)
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/props', timeout=1) as f:
                props = json.load(f)
        except (OSError, ValueError):
            time.sleep(.2)
            continue
        if health.get('status') != 'ok':
            time.sleep(.2)
            continue
        assert props['default_generation_settings']['n_ctx'] == 4096
        assert props['model_path'] == str(MODEL)
        return {'health': health, 'props': props}
    raise RuntimeError('health_60s_deadline')

def main():
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, interrupted)
    manifest = json.loads((PACK / 'capsule-manifest.json').read_text())
    for name, expected in manifest.items():
        assert sha(PACK / name) == expected, name
    assert sha(BIN) == '92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f'
    assert sha(MODEL) == '22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559'
    free_port()
    RUN.mkdir(exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_') and k != 'SEPALITH_VK_TRACE'}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    save(RUN / 'launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'owner': 'lead', 'task': 'RUN-04', 'pid': os.getpid(), 'wall_seconds': 400, 'fresh_server_per_cell': True, 'warm_lead_ms': 1500, 'cells': [[0, 'off'], [0, 'on'], [3, 'on'], [3, 'off']], 'capsule_manifest_sha256': sha(PACK / 'capsule-manifest.json'), 'model_sha256': sha(MODEL), 'binary_sha256': sha(BIN)})
    summaries = []
    failure = None
    try:
        for index, arm in [(0, 'off'), (0, 'on'), (3, 'on'), (3, 'off')]:
            check()
            free_port()
            cell = RUN / f'case{index}-{arm}'
            cell.mkdir()
            server = client = None
            start = time.monotonic()
            summary = {'fixture_index': index, 'arm': arm}
            try:
                argv = [str(BIN), '-m', str(MODEL), '--host', '127.0.0.1', '--port', str(PORT), '-t', '6', '-tb', '6', '--threads-http', '2', '--parallel', '1', '-c', '4096', '-b', '256', '-ub', '256', '-ngl', '99', '-lv', '4']
                with (cell / 'server.log').open('xb') as log:
                    server = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                save(cell / 'server-launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'pid': server.pid, 'argv': argv})
                save(cell / 'health.json', ready(server))
                (cell / 'server-maps.txt').write_text(Path(f'/proc/{server.pid}/maps').read_text())
                argv = ['node', '--experimental-strip-types', str(PACK / 'harness/native_live_ab_harness.ts'), '--arm', arm, '--fixture-index', str(index), '--fixture', str(PACK / 'fixture.jsonl'), '--manifest', str(PACK / 'fixture-manifest.json'), '--context-capsule', str(PACK / 'context-capsule.json'), '--warm-lead-ms', '1500', '--base-url', f'http://127.0.0.1:{PORT}', '--run-id', 'pilot-a', '--output-dir', str(cell)]
                with (cell / 'client.log').open('xb') as log:
                    client = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                save(cell / 'client-launch.json', {'pid': client.pid, 'argv': argv})
                deadline = time.monotonic() + 25
                while client.poll() is None:
                    check()
                    if time.monotonic() > deadline:
                        raise RuntimeError('client_25s_deadline')
                    if server.poll() is not None:
                        raise RuntimeError('server_exited_during_client')
                    time.sleep(.1)
                result = cell / f'pilot-a-{arm}-case{index}.json'
                if not result.exists():
                    raise RuntimeError('client_missing_result')
                value = json.loads(result.read_text())
                summary.update(client_returncode=client.returncode, result_sha256=sha(result), result_status=value['status'], result_error=value.get('error'))
            finally:
                summary.update(client=reap(client), server=reap(server), seconds=time.monotonic() - start)
                free_port()
                save(cell / 'terminal.json', summary)
                summaries.append(summary)
    except Exception as e:
        failure = repr(e)
    save(RUN / 'terminal.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'seconds': time.monotonic() - START, 'failure': failure, 'cells': summaries, 'acceptance': 'Root must review native logs, requests, output identity and timing. Pilot is not editor validation or model quality evidence.'})
    return 1 if failure else 0

if __name__ == '__main__':
    raise SystemExit(main())
