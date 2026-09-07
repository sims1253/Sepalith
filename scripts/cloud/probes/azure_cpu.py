"""Finite standard-library Azure CPU compatibility probe; no scientific verdict."""
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import tempfile
import time


def run():
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('60-second probe limit')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    receipt = {'operational_status': 'failed', 'scientific_verdict': 'not_applicable',
               'python': sys.version, 'interpreter': sys.executable,
               'machine': platform.machine(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        samples = [(i / 32, 3 * (i / 32) + 2) for i in range(-32, 33)]
        weight, bias = 0., 0.
        for _ in range(1000):
            errors = [weight * x + bias - y for x, y in samples]
            weight -= 0.05 * 2 * sum(e*x for e, (x, _) in zip(errors, samples)) / len(samples)
            bias -= 0.05 * 2 * sum(errors) / len(samples)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'checkpoint.json'
            p.write_text(json.dumps({'weight': weight, 'bias': bias}, sort_keys=True))
            restored = json.loads(p.read_text())
            mse = sum((restored['weight']*x + restored['bias']-y)**2 for x,y in samples)/len(samples)
            if mse >= 1e-12:
                raise ValueError('Checkpoint regression gate failed')
            receipt.update(operational_status='succeeded', mse=mse, checkpoint=restored,
                           checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                           input_sha256=hashlib.sha256(json.dumps(samples).encode()).hexdigest())
        for name in ('memory.max', 'cpu.max'):
            p = Path('/sys/fs/cgroup') / name
            if p.exists(): receipt[name] = p.read_text().strip()
        receipt['cpu_affinity_count'] = len(os.sched_getaffinity(0))
    except Exception as error:
        receipt['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        signal.alarm(0)
    receipt['elapsed_seconds'] = time.monotonic() - started
    print(json.dumps(receipt, sort_keys=True), flush=True)
    # ACI may restart a nonzero exit even with Never. Failure is in the receipt.
    return receipt


if __name__ == '__main__':
    run()
