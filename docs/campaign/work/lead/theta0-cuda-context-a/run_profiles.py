"""Root-owned sequential CUDA context allocation probes on synthetic events."""
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
WORK = PLAN / 'docs/campaign/work/lead/theta0-cuda-context-a'
NATIVE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
BASE = NATIVE / 'training/RUN-05-theta0-cuda-context-a'
spec = importlib.util.spec_from_file_location('guard', PLAN / 'docs/campaign/work/lead/prewarm-recovery-a-capsule/pilot_control.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
g.BIN = Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server')
g.MODEL = NATIVE / 'models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf'
g.PORT = 18403

def check():
    g.check()
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0]) < 8 * 1024 * 1024:
        raise RuntimeError('local_memory_floor_8GiB')

def ready(server, context):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        check()
        assert server.poll() is None, 'server exited during load'
        try:
            with urllib.request.urlopen('http://127.0.0.1:18403/props', timeout=1) as f:
                props = json.load(f)
        except (OSError, ValueError):
            time.sleep(.2)
            continue
        assert props['default_generation_settings']['n_ctx'] == context
        assert props['model_path'] == str(g.MODEL)
        return props
    raise RuntimeError('health_30s_deadline')

def main():
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, g.interrupted)
    pins = json.loads((WORK / 'manifest.json').read_text())
    for name, digest in pins.items():
        assert g.sha(Path(name)) == digest, name
    BASE.mkdir(exist_ok=False)
    g.save(BASE / 'launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'owner': 'lead', 'task': 'RUN-05', 'pid': os.getpid(), 'contexts': [2048,4096,8192], 'wall_seconds': 300, 'manifest_sha256': g.sha(WORK / 'manifest.json')})
    start = time.monotonic()
    cells = []
    failure = None
    env = {k:v for k,v in os.environ.items() if not k.startswith('GGML_') and k != 'SEPALITH_VK_TRACE'}
    env.update(GGML_CUDA_GRAPH_OPT='0', OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    try:
        for context in [2048,4096,8192]:
            g.RUN = BASE / str(context)
            g.START = time.monotonic()
            check()
            g.free_port()
            g.RUN.mkdir(exist_ok=False)
            server = client = None
            cell = {'context': context}
            try:
                argv = ['timeout','--signal=TERM','--kill-after=10s','95s',str(g.BIN),'-m',str(g.MODEL),'--host','127.0.0.1','--port','18403','-t','6','-tb','6','--threads-http','2','--parallel','1','-c',str(context),'-b','256','-ub','256','-ngl','99','-lv','4']
                with (g.RUN / 'server.log').open('xb') as log:
                    server = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                g.save(g.RUN / 'server-launch.json', {'pid':server.pid,'argv':argv,'at':dt.datetime.now(dt.timezone.utc).isoformat()})
                g.save(g.RUN / 'props.json', ready(server, context))
                children = Path(f'/proc/{server.pid}/task/{server.pid}/children').read_text().split()
                g.save(g.RUN / 'server-children.json', children)
                for pid in children:
                    (g.RUN / ('server-maps-' + pid + '.txt')).write_text(Path(f'/proc/{pid}/maps').read_text())
                argv = ['node','--no-warnings=MODULE_TYPELESS_PACKAGE_JSON','--experimental-strip-types',str(WORK/'replay-context-trace.ts'),'--fixture',str(PLAN/'docs/campaign/work/serving-transition-panel/transition-fixture.json'),'--server','http://127.0.0.1:18403','--context',str(context),'--mode','serial','--out',str(g.RUN/'trace.json')]
                with (g.RUN / 'client.log').open('xb') as log:
                    client = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                g.save(g.RUN / 'client-launch.json', {'pid':client.pid,'argv':argv})
                deadline = time.monotonic() + 65
                while client.poll() is None:
                    check()
                    if time.monotonic() > deadline: raise RuntimeError('client_65s_deadline')
                    time.sleep(.1)
                assert client.returncode == 0, 'trace client failed'
                value = json.loads((g.RUN / 'trace.json').read_text())
                cell.update(trace_sha256=g.sha(g.RUN/'trace.json'),denominator=value['denominator'])
            finally:
                cell.update(client=g.reap(client),server=g.reap(server),seconds=time.monotonic()-g.START)
                g.free_port()
                g.save(g.RUN/'terminal.json',cell)
                cells.append(cell)
    except Exception as e:
        failure = repr(e)
    g.save(BASE/'terminal.json',{'at':dt.datetime.now(dt.timezone.utc).isoformat(),'seconds':time.monotonic()-start,'cells':cells,'failure':failure,'limits':'Same nine short synthetic events. 8192 is allocation stress only, not 8192-token prompt support or model quality. Root must review identities, raw outputs and timing.'})
    return 1 if failure else 0

if __name__ == '__main__':
    raise SystemExit(main())
