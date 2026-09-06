import json
from pathlib import Path
import tempfile
import unittest
import kaggle_job as job


class Jobs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'job'
        self.record = job.prepare(self.root, 'test-owner')
        self.calls = []

    def invoke(self, binary, *args):
        self.calls.append(args)
        if args[0] == 'quota':
            return json.dumps([dict(resource='GPU', remaining='29.25h')])
        if args[1] == 'status':
            return 'has status "KernelWorkerStatus.COMPLETE"'
        if args[1] == 'output':
            self.artifacts(Path(args[args.index('-p') + 1]))
        return ''

    def artifacts(self, path):
        (path / 'checkpoint.pt').write_bytes(b'fake checkpoint, never deserialized')
        (path / 'metrics.json').write_text(json.dumps(dict(losses=[1.] * 63 + [.001], restored_mse=.001)))
        outputs = {name: dict(sha256=job.digest(path / name), bytes=(path / name).stat().st_size) for name in ['checkpoint.pt', 'metrics.json']}
        (path / 'receipt.json').write_text(json.dumps(dict(job_id=self.record['job_id'], source_sha256=self.record['source_sha256'], operational_status='succeeded', scientific_verdict='not_applicable', outputs=outputs)))

    def test_complete_and_idempotent_collection(self):
        job.submit(self.root, self.record, 'fake', self.invoke)
        job.collect(self.root, self.record, 'fake', self.invoke)
        self.assertEqual(self.record['status'], 'verified')
        job.collect(self.root, self.record, 'fake', self.invoke)
        self.assertEqual(sum(c[1:2] == ('output',) for c in self.calls), 1)
        with self.assertRaises(ValueError):
            job.submit(self.root, self.record, 'fake', self.invoke)

    def test_uncertain_launch_survives_reload_and_never_repushes(self):
        def lost(binary, *args):
            if args[0] == 'quota': return self.invoke(binary, *args)
            raise TimeoutError('response lost after remote acceptance')
        with self.assertRaises(TimeoutError): job.submit(self.root, self.record, 'fake', lost)
        record = json.loads((self.root / 'state.json').read_text())
        self.assertEqual(record['status'], 'unknown')
        with self.assertRaises(ValueError): job.submit(self.root, record, 'fake', self.invoke)
        job.collect(self.root, record, 'fake', self.invoke)
        self.assertEqual(record['status'], 'verified')

    def test_crash_before_push_requires_investigation(self):
        self.record['status'] = 'submitting'
        job.save(self.root, self.record)
        with self.assertRaises(ValueError): job.submit(self.root, self.record, 'fake', self.invoke)
        self.assertEqual(self.calls, [])

    def test_source_mutation_blocks_submission(self):
        (self.root / 'source/kernel.py').write_text('changed')
        with self.assertRaises(ValueError): job.submit(self.root, self.record, 'fake', self.invoke)
        self.assertEqual(self.calls, [])

    def test_quota_blocks_before_submission(self):
        def low(*args): return '[{"resource":"GPU","remaining":"0.01h"}]'
        with self.assertRaises(ValueError): job.submit(self.root, self.record, 'fake', low)
        self.assertEqual(self.record['status'], 'prepared')

    def test_bad_artifacts_are_not_published(self):
        job.submit(self.root, self.record, 'fake', self.invoke)
        def corrupt(binary, *args):
            result = self.invoke(binary, *args)
            if args[1] == 'output': (Path(args[args.index('-p') + 1]) / 'checkpoint.pt').write_text('corrupt')
            return result
        with self.assertRaises(ValueError): job.collect(self.root, self.record, 'fake', corrupt)
        self.assertFalse((self.root / 'artifacts').exists())
        job.collect(self.root, self.record, 'fake', self.invoke)
        self.assertEqual(self.record['status'], 'verified')

    def test_error_and_unknown_status_do_not_allow_collection(self):
        self.record['status'] = 'submitted'
        for output in ['has status "error"', 'unrecognized']:
            with self.assertRaises(ValueError): job.collect(self.root, self.record, 'fake', lambda *a: output)
        self.assertFalse((self.root / 'artifacts').exists())

    def test_lock_excludes_second_controller(self):
        with job.locked(self.root):
            with self.assertRaises(BlockingIOError):
                with job.locked(self.root): self.fail('second controller acquired lock')


if __name__ == '__main__': unittest.main()
