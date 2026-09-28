"""Root-owned CUDA, gateway, SSH and isolated notebook editor experiment."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shlex
import signal
import socket
import ssl
import subprocess
import time
import urllib.request

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
WORK = PLAN / 'docs/campaign/work/lead/remote-auto350-b'
GATEWAY = PLAN / 'docs/campaign/work/remote-primary-gateway-v1/gateway.mjs'
NATIVE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
RUN = NATIVE / 'training/RUN-04-remote-auto350-b'
BIN = Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server')
MODEL = NATIVE / 'models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf'
REMOTE = '/home/m0hawk/.local/share/sepalith-campaign-20260915'
HOST = 'm0hawk@192.168.178.40'
ADMISSION = PLAN / 'docs/campaign/receipts/RUN-04-remote-auto350-b-admission.json'
stopped = False
started = None

def interrupt(*_):
    global stopped
    stopped = True

def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def save(name, value):
    with (RUN / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def check():
    if stopped or (RUN / 'stop').exists():
        raise RuntimeError('root_stop')
    if started is not None and time.monotonic() - started > 600:
        raise RuntimeError('wall600s')
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0]) < 8 * 1024 * 1024:
        raise RuntimeError('host_memory_floor8GiB')

def get(url, context=None):
    with urllib.request.urlopen(url, timeout=2, context=context) as response:
        return json.load(response)

def wait_ready(process, url, predicate):
    until = time.monotonic() + 35
    while time.monotonic() < until:
        check()
        if process.poll() is not None:
            raise RuntimeError('process_exited_before_ready')
        try:
            value = get(url)
        except (OSError, ValueError):
            time.sleep(.2)
            continue
        if not predicate(value):
            raise RuntimeError('ready_identity_mismatch')
        return value
    raise RuntimeError('readiness35s')

def reap(process, wrapper=False):
    if process is None:
        return None
    child_file = Path(f'/proc/{process.pid}/task/{process.pid}/children')
    children = [int(v) for v in child_file.read_text().split()] if child_file.exists() else []
    if process.poll() is None:
        # A foreground timeout forwards TERM once to its child.
        process.terminate()
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    return {'pid': process.pid, 'exit_code': process.returncode,
            'pid_exists': Path(f'/proc/{process.pid}').exists(), 'wrapper': wrapper,
            'children': [{'pid': p, 'exists': Path(f'/proc/{p}').exists()} for p in children]}

def main():
    global started
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, interrupt)
    pins = json.loads((WORK / 'manifest.json').read_text())
    assert json.loads(ADMISSION.read_text())['status'] == 'admitted'
    assert sha(WORK / 'manifest.json') == json.loads(ADMISSION.read_text())['manifest_sha256']
    for file, digest in pins.items():
        assert sha(file) == digest, file
    for port in [18403, 18423, 18443]:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
    RUN.mkdir(exist_ok=False)
    started = time.monotonic()
    save('launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'task': 'RUN-04',
         'owner': 'lead', 'supervisor_pid': os.getpid(), 'wall_seconds': 600,
         'admission_sha256': sha(ADMISSION), 'source_manifest_sha256': sha(WORK / 'manifest.json')})
    binding = json.loads((WORK / 'binding.json').read_text())
    server = gateway = assets = ssh = None
    failure = None
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_')}
    env.update(CUDA_VISIBLE_DEVICES='0', GGML_CUDA_GRAPH_OPT='0', OMP_NUM_THREADS='6',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               LD_LIBRARY_PATH=str(BIN.parent))
    def launch(role, argv):
        with (RUN / (role + '.log')).open('xb') as log:
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        save(role + '-launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
                                   'pid': process.pid, 'argv': argv})
        return process
    try:
        check()
        server = launch('native', ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s',
            '600s', str(BIN), '-m', str(MODEL), '--alias', 'sepalith', '--temp', '0',
            '--host', '127.0.0.1', '--port', '18403', '-c', '4096', '-b', '256', '-ub', '256',
            '--parallel', '1', '-t', '6', '-tb', '6', '--threads-http', '2', '-ngl', '99', '-lv', '4'])
        props = wait_ready(server, 'http://127.0.0.1:18403/props', lambda x:
                           x['model_path'] == str(MODEL) and
                           x['default_generation_settings']['n_ctx'] == 4096 and x['total_slots'] == 1)
        save('props.json', props)
        children = Path(f'/proc/{server.pid}/task/{server.pid}/children').read_text().split()
        assert len(children) == 1
        pid = int(children[0])
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        native_identity = {'pid': pid, 'startTick': stat[19], 'uid': Path(f'/proc/{pid}').stat().st_uid,
                           'modelPath': str(MODEL)}
        maps = Path(f'/proc/{pid}/maps').read_text()
        (RUN / 'native-maps.txt').write_text(maps)
        assert 'libggml-cuda.so' in maps
        assert 'offloaded 43/43 layers to GPU' in (RUN / 'native.log').read_text()
        save('native-identity.json', native_identity)
        config = {'schema': 1, 'binding': binding, 'native': native_identity,
                  'admission': {'leaseId': 'RUN-04-remote-auto350-b', 'preflightReceiptSha256': sha(ADMISSION)}}
        save('gateway-config.json', config)
        gateway = launch('gateway', ['node', '--experimental-strip-types', str(GATEWAY),
                         '--config', str(RUN / 'gateway-config.json'), '--receipt', str(RUN / 'gateway-events.jsonl')])
        identity = wait_ready(gateway, 'http://127.0.0.1:18423/sepalith/runtime', lambda x:
                              x['instanceId'] == binding['instanceId'] and x['modelSha256'] == binding['manifest']['model']['sha256'])
        save('gateway-ready.json', identity)
        assets = launch('assets', ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s',
                                  '600s', 'python3', str(WORK / 'serve_assets.py'), '--directory', str(WORK)])
        context = ssl.create_default_context(cafile=str(WORK / 'asset-ca.pem'))
        until = time.monotonic() + 10
        while True:
            check()
            try:
                live_manifest = get('https://127.0.0.1:18443/manifest.json', context)
                break
            except OSError:
                if time.monotonic() > until:
                    raise
                time.sleep(.2)
        assert live_manifest == binding['manifest']
        save('asset-service-ready.json', {'manifest_matches_binding': True, 'port': 18443})
        remote_command = shlex.join(['python3', REMOTE + '/remote-auto350-b-capsule/notebook_editor_supervisor.py'])
        ssh = launch('ssh', ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            '-R', '127.0.0.1:18403:127.0.0.1:18423', '-R', '127.0.0.1:18443:127.0.0.1:18443',
            HOST, remote_command])
        while ssh.poll() is None:
            check()
            assert server.poll() is None and gateway.poll() is None and assets.poll() is None
            time.sleep(.5)
        # Remote route quality failures are evidence; process cleanup must still be verified.
        save('remote-command-terminal.json', {'ssh_exit_code': ssh.returncode,
                                             'acceptance': 'Inspect remote guard, host, buffers and renderer evidence.'})
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        cleanup = {'ssh': reap(ssh), 'gateway': reap(gateway), 'assets': reap(assets, True),
                   'native': reap(server, True)}
        save('terminal.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
             'seconds': time.monotonic() - started, 'failure': failure, 'cleanup': cleanup,
             'acceptance': 'Terminal process state only; root must review model, editor, timing, cancellation and quality evidence.'})
    return 1 if failure else 0

if __name__ == '__main__':
    raise SystemExit(main())
