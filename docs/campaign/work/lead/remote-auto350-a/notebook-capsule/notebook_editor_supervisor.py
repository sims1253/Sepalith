"""Root-owned bounded notebook editor smoke; retain evidence and reap owned children."""
from pathlib import Path
import datetime as dt
import json
import hashlib
import os
import signal
import socket
import subprocess
import time

ROOT = Path('/home/m0hawk/.local/share/sepalith-campaign-20260915')
PACKET = ROOT / 'remote-auto350-a-capsule'
RUN = ROOT / 'runs/remote-auto350-a'
GUARD = ROOT / 'runs/remote-auto350-a-supervision'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
for name, digest in json.loads((PACKET/'capsule-manifest.json').read_text()).items():
    assert sha(PACKET/name) == digest, name
# The model is root-owned on the desktop. The launcher verifies live remote binding.
assert not RUN.exists() and not GUARD.exists()
GUARD.mkdir(parents=True)
for port in [19403]:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', port))
def save(name, value):
    (GUARD/name).write_text(json.dumps(value, indent=2)+'\n')

def processes():
    found = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p/'stat').read_text().rsplit(')',1)[1].split()
            cmd = (p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            found[int(p.name)] = {'pid':int(p.name),'ppid':int(stat[1]),'pgrp':int(stat[2]),
                                  'start_ticks':stat[19],'state':stat[0],'argv':cmd[:1200]}
        except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):
            pass
    return found

owned = {}
child = None
interrupted = None
def interrupted_handler(sig, _frame):
    global interrupted
    interrupted = sig
signal.signal(signal.SIGTERM, interrupted_handler)
signal.signal(signal.SIGINT, interrupted_handler)

def observe():
    rows = processes()
    roots = {child.pid} if child else set()
    roots |= {pid for pid,r in rows.items() if str(RUN) in r['argv'] and pid != os.getpid()}
    roots |= {pid for pid,r in rows.items() if pid in owned and r['start_ticks']==owned[pid]['start_ticks']}
    changed = True
    while changed:
        nxt = {pid for pid,r in rows.items() if r['ppid'] in roots}
        changed = not nxt.issubset(roots)
        roots |= nxt
    for pid in roots:
        if pid in rows and pid != os.getpid():
            owned[pid] = rows[pid]
    return [rows[pid] for pid,r in owned.items() if pid in rows and rows[pid]['start_ticks']==r['start_ticks'] and rows[pid]['state']!='Z']

argv = ['xvfb-run','-a','--server-args=-screen 0 1280x800x24','node',
 str(PACKET/'run_remote_editor.mjs'),'--code','/usr/bin/code',
 '--vsix',str(PACKET/'candidate.vsix'),'--vsix-sha256','b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1',
 '--binding',str(PACKET/'binding.json'),'--binding-sha256','58174b4b3e75764553c545944d111da00b1bb40fd1ed0f0d85ef3ae56d1035b3',
 '--instance-id','f3882149-bf3a-4b02-ba01-ae15fed73165','--debug-port','19403',
 '--renderer-source','/usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js',
 '--renderer-sha256','e04cb8ec68016ccf4557690a2c8c908c3534614af1abe4c5c557157548169d78',
 '--run-root',str(RUN),'--timeout-ms','180000','--debounce-ms','350']
start = time.monotonic()
reason = None
with (GUARD/'stdout.log').open('wb') as out, (GUARD/'stderr.log').open('wb') as err:
    child = subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True)
    save('launch.json',{'at':dt.datetime.now(dt.timezone.utc).isoformat(),'owner':'lead','task':'RUN-04',
                       'supervisor_pid':os.getpid(),'child_pid':child.pid,'argv':argv,'wall_seconds':300,
                       'scope':'Exclusive temporary profile and synthetic workspace; all evidence retained.'})
    try:
        while child.poll() is None:
            current = observe()
            save('owned-processes.json',current)
            mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(mem['MemAvailable'].strip().split()[0])
            if interrupted is not None:
                reason = 'interrupted'; break
            if time.monotonic()-start > 300:
                reason = 'wall_deadline'; break
            if available < 2*1024*1024:
                reason = 'notebook_memory_floor_2GiB'; break
            time.sleep(1)
    finally:
        remaining = observe()
        for r in remaining:
            try: os.kill(r['pid'],signal.SIGTERM)
            except ProcessLookupError: pass
        until=time.monotonic()+10
        while observe() and time.monotonic()<until:
            time.sleep(.25)
        for r in observe():
            try: os.kill(r['pid'],signal.SIGKILL)
            except ProcessLookupError: pass
        try: child.wait(timeout=5)
        except subprocess.TimeoutExpired: pass
        time.sleep(.5)
        survivors=observe()
        save('owned-process-history.json',list(owned.values()))
        save('terminal.json',{'at':dt.datetime.now(dt.timezone.utc).isoformat(),'reason':reason,
             'child_exit_code':child.returncode,'seconds':time.monotonic()-start,'survivors':survivors,
             'status':'completed_process_cleanup_verified' if not survivors else 'owned_survivors_require_review',
             'experiment_acceptance':'Root must inspect run-result, host-result and native logs; process exit is not a pass.'})
assert not survivors, 'owned processes survived cleanup'
