import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

packet = Path(__file__).resolve().parent
manifest = json.loads((packet / 'source-manifest.json').read_text())
for item in manifest['files']:
    path = packet / item['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError('Reviewed queue source differs: ' + str(path))
base = Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-v2')
base.mkdir(parents=True, exist_ok=True)
output = base / 'shards0to4-root-01'
if output.exists():
    raise RuntimeError('Fresh semantic output required')
env = os.environ.copy()
env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONPATH='/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/parser-dependencies-root-v1')
command = ['timeout', '--signal=TERM', '--kill-after=30s', '3600', 'ionice', '-c3',
           'nice', '-n', '10', 'taskset', '-c', '2', '/usr/bin/python3', '-B',
           str(packet / 'source/run_streaming_semantic_queue.py'),
           '--replay-root', '/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01',
           '--output', str(output), '--shards', '0,1,2,3,4', '--max-workers', '1']
start = time.monotonic()
with (base / 'shards0to4-root-01.log').open('xb') as log:
    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (base / 'shards0to4-root-01.launch.json').write_text(json.dumps({
        'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'controller_pid': os.getpid(), 'timeout_pid': child.pid, 'command': command,
        'scope': 'committed shards0to4 partial semantic review; no training admission; no global closure',
    }, indent=2) + '\n')
    result = child.wait()
(base / 'shards0to4-root-01.terminal.json').write_text(json.dumps({
    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'exit_code': result, 'elapsed_seconds': time.monotonic() - start,
}, indent=2) + '\n')
