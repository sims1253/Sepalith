#!/usr/bin/env python3
"""Bounded subscription-only, tools-disabled external serving review."""
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
a = p.parse_args()
packet = a.packet.resolve()
assert packet.suffix == '.txt' and packet.is_file()
run = packet.parent / (packet.stem + '-run')
run.mkdir(exist_ok=False)
env = {k:v for k,v in os.environ.items() if not k.startswith('ANTHROPIC_') and k not in
       ('CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY')}
auth = subprocess.run(['claude', 'auth', 'status'], env=env, cwd=run,
                      capture_output=True, text=True, timeout=20)
status = json.loads(auth.stdout)
assert auth.returncode == 0 and status.get('loggedIn') is True
assert status.get('authMethod') == 'claude.ai' and status.get('apiProvider') == 'firstParty'
assert status.get('subscriptionType') in ('pro', 'max')
argv = ['timeout', '--signal=TERM', '--kill-after=15s', '720s', 'claude', '-p',
        '--model', 'claude-opus-5', '--effort', 'high', '--tools', '', '--safe-mode',
        '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json',
        '--system-prompt', 'You are a senior inference engineer advising a supervising scientist. Use only the supplied packet. Distinguish observations from hypotheses. Give concrete bounded experiments with rejection criteria. You have no tools or execution authority.']
def save(name, obj):
    with (run / name).open('x') as f:
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
started = time.monotonic()
with packet.open('rb') as inp, (run / 'response.json').open('xb') as out, (run / 'stderr.log').open('xb') as err:
    proc = subprocess.Popen(argv, stdin=inp, stdout=out, stderr=err, env=env, cwd=run)
    save('launch.json', {'owner':'lead', 'task':'RUN-06', 'at':dt.datetime.now(dt.timezone.utc).isoformat(),
         'supervisor_pid':os.getpid(), 'timeout_pid':proc.pid, 'model_requested':'claude-opus-5',
         'effort':'high', 'tools':[], 'input_sha256':hashlib.sha256(packet.read_bytes()).hexdigest(),
         'billing':'verified claude.ai firstParty subscription; API-key and third-party overrides removed',
         'max_seconds':720})
    code = proc.wait()
save('terminal.json', {'exit_code':code, 'seconds':time.monotonic()-started,
     'at':dt.datetime.now(dt.timezone.utc).isoformat(),
     'response_sha256':hashlib.sha256((run/'response.json').read_bytes()).hexdigest(),
     'response_bytes':(run/'response.json').stat().st_size})
