import json
from pathlib import Path
import tempfile
import unittest
import kaggle_probe as probe
import kaggle_job as jobs


class Probes(unittest.TestCase):
    def test_proxy_redirects_never_forward_credentials(self):
        from probe_benchmark_proxy import NoRedirect
        import urllib.request
        handler = NoRedirect()
        request = urllib.request.Request('https://mp.kaggle.net/models', headers={'Authorization': 'Bearer fake'})
        for destination in ['https://other.example/collect', 'http://mp.kaggle.net/collect', 'https://mp.kaggle.net/new']:
            self.assertIsNone(handler.redirect_request(request, None, 302, 'redirect', {}, destination))

    def test_interrupted_profile_preparation_cannot_submit_default_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            record = jobs.prepare(root, 'tester', ready=False)
            self.assertEqual(record['status'], 'preparing')
            with self.assertRaises(ValueError): jobs.submit(root, record, 'fake', lambda *args: self.fail('No CLI call allowed'))

    def test_proxy_error_classification_preserves_sdk_http_code(self):
        from probe_benchmark_proxy import failure_code
        self.assertEqual(failure_code(ValueError('Authentication failed (403). Details')), 403)
        self.assertEqual(failure_code(ValueError('Endpoint not found (404). Details')), 404)
        self.assertIsNone(failure_code(ValueError('Other error')))

    def test_each_profile_freezes_source_and_requests_correct_resource(self):
        for name, (accelerator, resource, timeout, _) in probe.PROFILES.items():
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / 'run'
                record = probe.prepare(root, 'tester', name)
                source = (root / 'source/kernel.py').read_text()
                compile(source, 'kernel.py', 'exec')
                metadata = json.loads((root / 'source/kernel-metadata.json').read_text())
                self.assertEqual(metadata['enable_tpu'], resource == 'TPU')
                self.assertTrue(metadata['is_private'])
                calls = []
                def invoke(binary, *args):
                    calls.append(args)
                    if args[0] == 'quota': return json.dumps([dict(resource=resource, remaining='1.00h')])
                    return ''
                jobs.submit(root, record, 'fake', invoke)
                self.assertIn(str(timeout), calls[-1])
                self.assertIn(accelerator, calls[-1])
                with self.assertRaises(ValueError): jobs.submit(root, record, 'fake', invoke)

    def test_blocked_probe_is_verified_without_claiming_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            record = probe.prepare(root, 'tester', 'tpu')
            record['status'] = 'submitted'
            def invoke(binary, *args):
                if args[1] == 'status': return 'status "complete"'
                folder = Path(args[args.index('-p') + 1])
                p = folder / 'report.json'
                p.write_text(json.dumps(dict(probe='tpu', outcome='blocked', checks={}, error='CPU fallback rejected')))
                receipt = dict(job_id=record['job_id'], source_sha256=record['source_sha256'], operational_status='succeeded', scientific_verdict='not_applicable', outputs={'report.json': dict(sha256=jobs.digest(p), bytes=p.stat().st_size)})
                (folder / 'receipt.json').write_text(json.dumps(receipt))
                return ''
            probe.collect(root, record, 'fake', invoke)
            self.assertEqual(record['capability'], 'blocked')
            self.assertEqual(record['status'], 'verified')

    def test_supported_report_requires_all_gates_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            record = probe.prepare(root, 'tester', 'tpu')
            p = root / 'report.json'
            p.write_text(json.dumps(dict(probe='tpu', outcome='supported', checks={})))
            (root / 'receipt.json').write_text(json.dumps(dict(job_id=record['job_id'], source_sha256=record['source_sha256'], operational_status='succeeded', scientific_verdict='not_applicable', outputs={'report.json': dict(sha256=jobs.digest(p), bytes=p.stat().st_size)})))
            with self.assertRaises(ValueError): probe.verify(root, record)


if __name__ == '__main__': unittest.main()
