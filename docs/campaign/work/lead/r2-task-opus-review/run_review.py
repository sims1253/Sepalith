"""Run a bounded subscription-only Opus patch request on an explicit capsule."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('packet', type=Path)
p.add_argument('--task', required=True)
p.add_argument('--seconds', type=int, default=1200)
a = p.parse_args()
assert 60 <= a.seconds <= 1800
packet = a.packet.resolve()
assert packet.suffix == '.txt' and packet.is_file()
run = packet.parent / (packet.stem + '-run')
run.mkdir(exist_ok=False)
env = {k:v for k,v in os.environ.items() if not k.startswith('ANTHROPIC_') and k not in
       ('CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY')}
env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
auth = subprocess.run(['claude', 'auth', 'status'], env=env, cwd=run,
                      capture_output=True, text=True, timeout=20)
status = json.loads(auth.stdout)
assert auth.returncode == 0 and status.get('loggedIn') is True
assert status.get('authMethod') == 'claude.ai' and status.get('apiProvider') == 'firstParty'
assert status.get('subscriptionType') in ('pro', 'max')
argv = ['timeout', '--signal=TERM', '--kill-after=15s', str(a.seconds)+'s', 'claude', '-p',
        '--model', 'claude-opus-5', '--effort', 'xhigh', '--tools', '', '--safe-mode',
        '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json',
        '--system-prompt', 'You are a senior training engineer reviewing a bounded source capsule for the supervising scientist. Follow the supplied review task. No tools are available. Never claim tests or measurements were executed. Return concrete defects and minimal reviewable fixes only.']
def save(name, value):
    with (run / name).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())

start = time.monotonic()
with packet.open('rb') as inp, (run/'response.json').open('xb') as out, (run/'stderr.log').open('xb') as err:
    proc = subprocess.Popen(argv, stdin=inp, stdout=out, stderr=err, env=env, cwd=run)
    save('launch.json', {'owner':'lead','task':a.task,'at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'supervisor_pid':os.getpid(),'timeout_pid':proc.pid,'model_requested':'claude-opus-5',
        'effort':'xhigh','tools':[],'input_sha256':hashlib.sha256(packet.read_bytes()).hexdigest(),
        'billing':'Verified claude.ai firstParty subscription; API and provider overrides removed.',
        'max_seconds':a.seconds,'scope':'Explicit source capsule only; output patch and tests; no file, network, GPU, model or benchmark authority.'})
    code = proc.wait()
save('terminal.json', {'exit_code':code,'seconds':time.monotonic()-start,
    'at':dt.datetime.now(dt.timezone.utc).isoformat(),
    'response_sha256':hashlib.sha256((run/'response.json').read_bytes()).hexdigest(),
    'response_bytes':(run/'response.json').stat().st_size})
