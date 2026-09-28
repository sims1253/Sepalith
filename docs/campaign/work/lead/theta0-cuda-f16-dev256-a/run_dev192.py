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
WORK = PLAN / 'docs/campaign/work/lead/theta0-cuda-f16-dev256-a'
NATIVE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
BASE = NATIVE / 'training/RUN-05-theta0-cuda-f16-dev256-a'
spec = importlib.util.spec_from_file_location('guard', PLAN / 'docs/campaign/work/lead/prewarm-recovery-a-capsule/pilot_control.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
g.BIN = Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server')
g.MODEL = NATIVE / 'models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf'
g.PORT = 18403


def reap_timeout(p):
    if p is None:
        return None
    path = Path(f'/proc/{p.pid}/task/{p.pid}/children')
    children = [int(x) for x in path.read_text().split()] if path.exists() else []
    action = 'already_exited'
    if p.poll() is None:
        action = 'SIGTERM_wrapper_only_timeout_foreground_forwards_once'
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            action += '_then_group_SIGKILL_after_10s'
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=5)
    return {'pid': p.pid, 'exit_code': p.returncode,
            'pid_exists': Path(f'/proc/{p.pid}').exists(), 'action': action,
            'children': [{'pid': n, 'exists': Path(f'/proc/{n}').exists()} for n in children]}

GRID_START = None

def check():
    if GRID_START is not None and time.monotonic()-GRID_START > 240: raise RuntimeError('grid_240s_wall_deadline')
    if g.STOP or (g.RUN / "stop").exists(): raise RuntimeError("lead_stop")
    if time.monotonic()-g.START > 185: raise RuntimeError("cell185sdeadline")
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
    global GRID_START
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, g.interrupted)
    pins = json.loads((WORK / 'manifest.json').read_text())
    for name, digest in pins.items():
        assert g.sha(Path(name)) == digest, name
    BASE.mkdir(exist_ok=False)
    g.save(BASE / 'launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'owner': 'lead', 'task': 'RUN-05', 'pid': os.getpid(), 'grid': [['F16',256]], 'wall_seconds': 240, 'manifest_sha256': g.sha(WORK / 'manifest.json')})
    start = time.monotonic()
    GRID_START = start
    cells = []
    failure = None
    env = {k:v for k,v in os.environ.items() if not k.startswith('GGML_') and k != 'SEPALITH_VK_TRACE'}
    env.update(GGML_CUDA_GRAPH_OPT='0', OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    try:
        for candidate, batch in [('F16',256)]:
            if time.monotonic()-start > 240: raise RuntimeError('grid_240s_wall_deadline')
            context = 4096
            g.MODEL = NATIVE / ('models/SFT-primary-step1000-runtime-gguf/model-F16.gguf' if candidate == 'F16' else 'models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf')
            g.RUN = BASE / (candidate+'-b'+str(batch))
            g.START = time.monotonic()
            check()
            g.free_port()
            g.RUN.mkdir(exist_ok=False)
            server = client = None
            cell = {'context': context, 'candidate':candidate, 'batch':batch, 'ubatch':batch}
            try:
                argv = ['timeout','--foreground','--signal=TERM','--kill-after=10s','200s',str(g.BIN),'-m',str(g.MODEL),'--host','127.0.0.1','--port','18403','-t','6','-tb','6','--threads-http','2','--parallel','1','-c',str(context),'-b',str(batch),'-ub',str(batch),'-ngl','99','-lv','4']
                with (g.RUN / 'server.log').open('xb') as log:
                    server = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                g.save(g.RUN / 'server-launch.json', {'pid':server.pid,'argv':argv,'at':dt.datetime.now(dt.timezone.utc).isoformat()})
                g.save(g.RUN / 'props.json', ready(server, context))
                children = Path(f'/proc/{server.pid}/task/{server.pid}/children').read_text().split()
                g.save(g.RUN / 'server-children.json', children)
                for pid in children:
                    (g.RUN / ('server-maps-' + pid + '.txt')).write_text(Path(f'/proc/{pid}/maps').read_text())
                    g.save(g.RUN / ('server-env-' + pid + '.json'), [v.decode() for v in Path(f'/proc/{pid}/environ').read_bytes().split(bytes([0])) if v.startswith(b'GGML_CUDA_GRAPH_OPT=')])
                argv = ['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(WORK/'run_dev192_client.py'),'--url','http://127.0.0.1:18403','--tokenizer-dir','/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain','--execution-root','/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912','--output',str(g.RUN/'quality.json'),'--model-label','theta0-F16-CUDA-b'+str(batch)+'-out192','--model-provenance','ee7ba2fd7e7b6446d74224e1e1f9e4724fbcfa54e56e60c4b623981d1e4f45ed','--case-deadline-seconds','30','--deadline-seconds','160','--reserve-seconds','15']
                with (g.RUN / 'client.log').open('xb') as log:
                    client = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                g.save(g.RUN / 'client-launch.json', {'pid':client.pid,'argv':argv})
                deadline = time.monotonic() + 170
                while client.poll() is None:
                    check()
                    if time.monotonic() > deadline: raise RuntimeError('client_170s_deadline')
                    time.sleep(.1)
                assert client.returncode == 0, 'trace client failed'
                value = json.loads((g.RUN / 'quality.json').read_text())
                cell.update(quality_sha256=g.sha(g.RUN/'quality.json'),quality_status=value.get('status'),quality_complete=value.get('complete'))
            finally:
                cell.update(client=g.reap(client),server=reap_timeout(server),seconds=time.monotonic()-g.START)
                g.free_port()
                g.save(g.RUN/'terminal.json',cell)
                cells.append(cell)
                if cell['server'] is not None and (cell['server']['exit_code'] != 0 or any(x['exists'] for x in cell['server']['children'])):
                    raise RuntimeError('server_teardown_not_clean')
    except Exception as e:
        failure = repr(e)
    g.save(BASE/'terminal.json',{'at':dt.datetime.now(dt.timezone.utc).isoformat(),'seconds':time.monotonic()-start,'cells':cells,'failure':failure,'limits':'F16 CUDA batch/ubatch256 compared with accepted Q8 batch256 baseline correctedDEV75,43edits32noops. Explicit output192 matches deliveredruntime. Historical512-cap results not conflated; no sealed final. Root must score/review everyrawrow before promotion.'})
    return 1 if failure else 0

if __name__ == '__main__':
    raise SystemExit(main())
