"""Exercise the real run_session duration wiring with no sockets or native processes."""
import contextlib
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'replay'))
import daily_lan as d

class FakeChild:
    pid = 123456
    returncode = 0
    def wait(self, timeout=None): return 0

class FakeOwner:
    instances = []
    def __init__(self, run, lockfd):
        self.run = run
        self.calls = []
        self.cleaned = 0
        self.__class__.instances.append(self)
    def launch(self, role, argv, env=None, stdin=None):
        self.calls.append((role, argv))
        if role == 'native':
            (self.run / 'native.log').write_text('offloaded 43/43 layers to GPU')
        return FakeChild()
    def check(self): self.guard()
    def complete(self, child): pass
    def cleanup(self):
        self.cleaned += 1
        return []

class DurationFlow(unittest.TestCase):
    def run_one(self, seconds, omit_argument=False):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            p = pathlib.Path(directory)
            model = p / 'synthetic-byte-placeholder'
            model.write_bytes(b'x')
            manifest = json.loads((ROOT / 'replay/primary-manifest.json').read_text())
            manifest['model'].update(bytes=1, sha256=hashlib.sha256(b'x').hexdigest())
            manifest['bundles'][0]['files'] = []
            (p / 'primary-manifest.json').write_text(json.dumps(manifest))
            (p / 'notebook_setup.py').write_text('# synthetic placeholder')
            admission = p / 'admission.json'; admission.write_text('{}')
            ledger = p / 'ledger'; ledger.write_text('| RTX 5090 | None | synthetic |\n')
            clock = [0.0]
            def sleep(value):
                clock[0] = float(seconds) if value == .5 else clock[0] + value
            def ready(owner, url, predicate, instance=None):
                if url.endswith('/props'):
                    value = {'model_path': str(model), 'default_generation_settings': {'n_ctx': 4096}, 'total_slots': 1}
                else:
                    value = {'schema': 1, 'instanceId': instance, 'modelSha256': manifest['model']['sha256'], 'serverSha256': manifest['bundles'][0]['launchProfile']['serverSha256'], 'backend': 'cuda', 'contextSize': 4096, 'maxOutputTokens': 192, 'renderer': manifest['modelProfile']['renderer'], 'cudaGraphOptimization': 0}
                self.assertTrue(predicate(value)); return value
            with contextlib.ExitStack() as stack:
                actual_identity = d.identity
                stack.enter_context(patch.object(d, 'identity', lambda pid: {'pid': pid, 'startTick': 'synthetic', 'uid': 1000} if pid == FakeChild.pid else actual_identity(pid)))
                for name, value in {'HERE':p, 'MODEL':model, 'LEDGER':ledger, 'Owned':FakeOwner, 'ready':ready, 'check_manifest':lambda:None, 'check_admission':lambda *_:None, 'free_port':lambda _:None, 'exclusive_lock':lambda:contextlib.nullcontext(-1)}.items():
                    stack.enter_context(patch.object(d, name, value))
                stack.enter_context(patch.object(d.time, 'monotonic', lambda:clock[0]))
                stack.enter_context(patch.object(d.time, 'sleep', sleep))
                stack.enter_context(patch.object(d.signal, 'signal'))
                result = d.run_session(p/'state', admission) if omit_argument else d.run_session(p/'state', admission, seconds)
            owner = FakeOwner.instances[-1]
            self.assertEqual(result, 0, (owner.run/'terminal.json').read_text())
            self.assertEqual(owner.cleaned, 1)
            metadata = json.loads((owner.run/'ready.json').read_text())
            self.assertEqual(metadata['sessionSeconds'], seconds)
            terminal = json.loads((owner.run/'terminal.json').read_text())
            self.assertEqual(terminal['seconds'], seconds)
            self.assertEqual(terminal['ledgerRelease']['status'], 'released')
            argv = next(argv for role, argv in owner.calls if role == 'gateway')
            self.assertEqual(argv[argv.index('--daily-max-run-ms')+1], str(max(1800000, seconds*1000)))
            self.assertIn('| RTX 5090 | None |', ledger.read_text())
    def test_duration_reaches_real_ready_loop_and_gateway(self):
        for seconds in [60, 120, 1800, 28800]:
            with self.subTest(seconds=seconds): self.run_one(seconds)
    def test_omitted_argument_uses_eight_hour_default(self):
        self.run_one(28800, omit_argument=True)

if __name__ == '__main__': unittest.main(verbosity=2)
