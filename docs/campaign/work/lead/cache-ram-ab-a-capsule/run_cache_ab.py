"""Three paired TRAIN recovery sequences; no model or runtime mutation."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

PACK = Path(__file__).resolve().parent
ROOT = Path('/home/m0hawk/.local/share/sepalith-campaign-20260915')
RUN = ROOT / 'runs/cache-ram-ab-a'
STOP = False

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

def stop(*_):
    global STOP
    STOP = True

def main():
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, stop)
    manifest = json.loads((PACK / 'manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha(PACK / name) == digest, name
    order = json.loads((PACK / 'order.json').read_text())
    RUN.mkdir(exist_ok=False)
    save(RUN / 'launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'pid': os.getpid(), 'task': 'RUN-04', 'owner': 'lead', 'wall_ceiling_seconds': 600, 'order': order, 'manifest_sha256': sha(PACK / 'manifest.json')})
    started = time.monotonic()
    cells = []
    failure = None
    try:
        for name in order:
            if STOP or (RUN / 'stop').exists() or time.monotonic() - started > 480:
                raise RuntimeError('stop_or_insufficient_remaining_time')
            with (RUN / (name + '-controller.log')).open('xb') as log:
                child = subprocess.Popen(['python3', str(PACK / name / 'run_recovery.py')], stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                save(RUN / (name + '-controller-launch.json'), {'pid': child.pid})
                while child.poll() is None:
                    if STOP or (RUN / 'stop').exists():
                        child.terminate()
                    time.sleep(.2)
                terminal = json.loads((RUN / name / 'terminal.json').read_text())
                assert child.returncode == 0 and terminal.get('result_status') == 'pass' and not terminal.get('failure'), name
                assert not terminal['server']['pid_exists'] and not terminal['client']['pid_exists'], name
                cells.append({'name': name, 'terminal_sha256': sha(RUN / name / 'terminal.json'), 'seconds': terminal['seconds']})
    except Exception as e:
        failure = repr(e)
    save(RUN / 'terminal.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'seconds': time.monotonic() - started, 'cells': cells, 'failure': failure, 'scope': 'Three TRAIN pairs; root must independently verify output parity, logs and timing before any runtime promotion.'})
    return 1 if failure else 0

if __name__ == '__main__':
    raise SystemExit(main())
