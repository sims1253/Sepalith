"""Root-owned single-server cancellation-recovery pilot; 90 seconds maximum."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import pilot_control as g

def main():
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, g.interrupted)
    pack = Path(__file__).resolve().parent
    for name, digest in json.loads((pack / 'capsule-manifest.json').read_text()).items():
        assert g.sha(pack / name) == digest, name
    assert g.sha(g.BIN) == '92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f'
    assert g.sha(g.MODEL) == '22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559'
    g.check()
    g.free_port()
    g.RUN.mkdir(exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_') and k != 'SEPALITH_VK_TRACE'}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    server = client = None
    terminal = {}
    try:
        argv = [str(g.BIN), '-m', str(g.MODEL), '--host', '127.0.0.1', '--port', str(g.PORT), '-t', '6', '-tb', '6', '--threads-http', '2', '--parallel', '1', '-c', '4096', '-b', '256', '-ub', '256', '-ngl', '99', '-lv', '4']
        with (g.RUN / 'server.log').open('xb') as log:
            server = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        g.save(g.RUN / 'launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'owner': 'lead', 'task': 'RUN-04', 'pid': os.getpid(), 'server_pid': server.pid, 'argv': argv, 'wall_seconds': 90, 'model_sha256': g.sha(g.MODEL), 'binary_sha256': g.sha(g.BIN), 'capsule_manifest_sha256': g.sha(pack / 'capsule-manifest.json')})
        g.save(g.RUN / 'health.json', g.ready(server))
        (g.RUN / 'server-maps.txt').write_text(Path(f'/proc/{server.pid}/maps').read_text())
        fixture = g.ROOT / 'prewarm-v3-pilot-a-capsule'
        argv = ['node', '--experimental-strip-types', str(pack / 'recovery_native_harness.ts'), '--fixture', str(fixture / 'fixture.jsonl'), '--manifest', str(fixture / 'fixture-manifest.json'), '--context-capsule', str(fixture / 'context-capsule.json'), '--base-url', f'http://127.0.0.1:{g.PORT}', '--run-id', 'recovery-a', '--output-dir', str(g.RUN), '--warm-lead-ms', '1500', '--long-cancel-ms', '5000', '--short-cancel-ms', '5000', '--drain-ms', '2000']
        with (g.RUN / 'client.log').open('xb') as log:
            client = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        g.save(g.RUN / 'client-launch.json', {'pid': client.pid, 'argv': argv})
        deadline = time.monotonic() + 25
        while client.poll() is None:
            g.check()
            if time.monotonic() > deadline:
                raise RuntimeError('client_25s_deadline')
            if server.poll() is not None:
                raise RuntimeError('server_early_exit')
            time.sleep(.1)
        result = g.RUN / 'recovery-a-recovery.json'
        value = json.loads(result.read_text())
        terminal.update(client_returncode=client.returncode, result_sha256=g.sha(result), result_status=value['status'])
    except Exception as e:
        terminal['failure'] = repr(e)
    finally:
        terminal.update(client=g.reap(client), server=g.reap(server), seconds=time.monotonic() - g.START, at=dt.datetime.now(dt.timezone.utc).isoformat())
        g.free_port()
        g.save(g.RUN / 'terminal.json', terminal)
    return 0 if terminal.get('client_returncode') == 0 and not terminal.get('failure') else 1

if __name__ == '__main__':
    raise SystemExit(main())
