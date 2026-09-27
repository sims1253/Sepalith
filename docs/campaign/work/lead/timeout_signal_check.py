"""Check one-signal teardown through GNU timeout without model or GPU access."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

CHILD = '''
import json, signal, sys, time
from pathlib import Path
p=Path(sys.argv[1]); count=0; deadline=None
def stop(sig, frame):
 global count, deadline
 count+=1; deadline=time.monotonic()+.2
 p.write_text(json.dumps({'pid':__import__('os').getpid(),'signals':count}))
signal.signal(signal.SIGTERM, stop)
p.write_text(json.dumps({'pid':__import__('os').getpid(),'signals':count}))
while deadline is None or time.monotonic()<deadline: time.sleep(.01)
'''

results = []
with tempfile.TemporaryDirectory(prefix='sepalith-timeout-signals-') as directory:
    for mode in ('group_default', 'wrapper_default', 'wrapper_foreground'):
        for repeat in range(3):
            output = Path(directory) / f'{mode}-{repeat}.json'
            argv = ['timeout']
            if mode == 'wrapper_foreground':
                argv.append('--foreground')
            argv += ['--signal=TERM', '--kill-after=2s', '5s', sys.executable, '-c', CHILD, str(output)]
            process = subprocess.Popen(argv, start_new_session=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 3
                while not output.exists():
                    assert process.poll() is None and time.monotonic() < deadline
                    time.sleep(.01)
                child_pid = json.loads(output.read_text())['pid']
                if mode == 'group_default':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=3)
                result = json.loads(output.read_text())
                result.update(mode=mode, repeat=repeat, wrapper_exit=process.returncode,
                              child_absent=not Path(f'/proc/{child_pid}').exists())
                results.append(result)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
                if child_pid is not None and Path(f'/proc/{child_pid}').exists():
                    os.kill(child_pid, signal.SIGKILL)
assert all(r['signals'] == 1 and r['child_absent'] for r in results
           if r['mode'] == 'wrapper_foreground')
print(json.dumps(results, indent=2))
