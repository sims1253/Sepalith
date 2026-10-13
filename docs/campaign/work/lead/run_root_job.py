#!/usr/bin/env python3
"""Dispatch exactly one root-admitted immutable recipe, then pause the runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
sys.path.insert(0, str(EXEC / 'packages/sepalith/src'))
from sepalith.runner import Runner

p = argparse.ArgumentParser()
p.add_argument('recipe', type=Path)
p.add_argument('admission', type=Path)
a = p.parse_args()
admission = json.loads(a.admission.read_text())
recipe = json.loads(a.recipe.read_text())
assert admission['owner'] == 'lead'
assert admission['launch_admitted'] is True
assert admission['runner_recipe_sha256'] == hashlib.sha256(a.recipe.read_bytes()).hexdigest()
assert admission['job'] == recipe['id']
for pid in admission.get('required_terminated_pids', []):
    assert not Path(f'/proc/{pid}').exists(), f'Prior owner PID {pid} still exists'
r = Runner('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state')
state = r.plan()
assert state['paused'], 'Runner must start paused'
assert not any(x['status'] == 'running' for x in state['attempts'])
assert not any(x['status'] in ('queued', 'running') for x in state['jobs'])
r.verify_snapshot(recipe['snapshot'])
r.enqueue(recipe)
r.resume()
print(json.dumps({'status': 'dispatching', 'pid': os.getpid(), 'job': recipe['id']}), flush=True)
try:
    result = r.run_next()
    print(json.dumps({'result': result}, default=str), flush=True)
finally:
    r.pause()
