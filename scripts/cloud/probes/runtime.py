"""Receipt wrapper for finite capability probes, with no credential logging."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import signal
import sys
import time


def run_probe():
    def deadline(signum, frame):
        raise TimeoutError('Capability probe deadline exceeded')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(CODE_TIMEOUT)
    start = time.monotonic()
    report = {'probe': PROBE_NAME, 'checks': {}, 'stage': 'start'}
    try:
        probe(report)
        report['outcome'] = 'supported'
    except Exception as error:
        # The probes never handle credentials. Retain a bounded error summary.
        report.update(outcome='blocked', error_type=type(error).__name__, error=str(error)[:1000])
    report['elapsed_seconds'] = time.monotonic() - start
    versions = {}
    for name in ['torch', 'transformers', 'jax', 'jaxlib', 'libtpu', 'numpy']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    report.update(python=platform.python_version(), interpreter=sys.executable, packages=versions)
    Path('report.json').write_text(json.dumps(report, indent=2) + '\n')
    outputs = {}
    for path in sorted(p for p in Path('.').iterdir() if p.is_file() and p.suffix in ('.json', '.npy', '.safetensors')):
        if path.name == 'receipt.json': continue
        data = path.read_bytes()
        outputs[path.name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    receipt = {'job_id': JOB_ID, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'operational_status': 'succeeded', 'scientific_verdict': 'not_applicable', 'outputs': outputs}
    Path('receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    # Measuring an unsupported capability is a completed probe, not a success
    # claim about the capability. Failures of this wrapper still fail the job.


if __name__ == '__main__':
    run_probe()
