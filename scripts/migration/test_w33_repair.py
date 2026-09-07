"""Calibration gates preserve historical rows and reject incomplete repairs."""
import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import w33_repair as w33


class W33Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.run = self.root / 'run'
        self.run.mkdir()
        (self.assets / 'runtime').mkdir()
        (self.assets / 'runtime/llama-server').write_text('fake')
        (self.assets / 'runtime/llama-server').chmod(0o700)
        (self.assets / 'model.gguf').write_text('fake')
        examples = [dict(prompt=str(i), target='answer', package='pkg') for i in range(20)]
        saved = [dict(i=i, package='pkg', pred='answer', latency_s=1,
                      **w33.eval_ablation.score(['answer'], ['answer'])) for i in range(16)]
        for name, rows in [('examples', examples), ('saved', saved)]:
            (self.assets / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        w33.prepare(self.run, self.assets)

    def measure(self, prediction):
        class Server:
            def __init__(self, *args, **kwargs): pass
            def start(self): pass
            def stop(self): pass
        with patch.object(w33.spec_bench, 'SpecServer', Server), patch.object(
                w33.eval_ablation, 'complete', return_value=(prediction, 0.1)):
            w33.measure(self.run, self.assets)

    def test_matching_calibration_fills_only_missing_rows(self):
        before = (self.assets / 'saved.jsonl').read_bytes()
        self.measure('answer')
        w33.evaluate(self.run)
        rows = w33.read(self.run / 'predictions.jsonl')
        self.assertEqual([16, 17, 18, 19], [r['i'] for r in rows if r['phase'] == 'repair'])
        self.assertEqual(before, (self.assets / 'saved.jsonl').read_bytes())

    def test_mismatch_stops_before_repair_and_records_negative_verdict(self):
        self.measure('different')
        w33.evaluate(self.run)
        self.assertEqual(16, len(w33.read(self.run / 'predictions.jsonl')))
        self.assertEqual('CALIBRATION-MISMATCH', json.loads((self.run / 'verdict.json').read_text())['verdict'])

    def test_partial_output_cannot_be_a_successful_repair(self):
        self.measure('answer')
        path = self.run / 'predictions.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        with self.assertRaisesRegex(ValueError, 'incomplete repair'):
            w33.evaluate(self.run)

    def test_deadline_exception_is_not_retried_as_readiness_io(self):
        self.assertFalse(issubclass(w33.DeadlineExceeded, OSError))

    def test_fresh_scope_requires_all_921_rows_without_historical_alignment(self):
        target = self.root / 'fresh'
        target.mkdir()
        examples = [dict(prompt='prompt', target='answer', package='new-package') for _ in range(921)]
        (self.assets / 'examples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in examples))
        w33.prepare(target, self.assets, fresh=True)
        prepared = json.loads((target / 'prepared.json').read_text())
        self.assertEqual([], prepared['calibration'])
        self.assertEqual(list(range(921)), prepared['missing'])
        rows = [dict(i=i, phase='repair', pred='answer', latency_s=1,
                     **w33.eval_ablation.score(['answer'], ['answer'])) for i in range(921)]
        (target / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        w33.evaluate(target)
        self.assertEqual('FRESH-PLAIN-SPLIT-MEASURED', json.loads((target / 'verdict.json').read_text())['verdict'])

    def test_historical_input_mismatch_rejects_before_measurement(self):
        target = self.root / 'mismatch'
        target.mkdir()
        p = self.assets / 'examples.jsonl'
        p.write_text(p.read_text().replace('pkg', 'other'))
        with self.assertRaisesRegex(ValueError, 'does not align'):
            w33.prepare(target, self.assets)

    def test_fresh_output_limit_is_scored_and_keeps_completion_metadata(self):
        prepared = dict(fresh=True, calibration=[], missing=[0], max_seconds=30, max_tokens=640, context=16384)
        (self.run / 'prepared.json').write_text(json.dumps(prepared))
        class Server:
            def __init__(self, *a, **kw): pass
            def start(self): pass
            def stop(self): pass
        def response(request, **kwargs):
            result = {'tokens': [1, 2, 3]} if request.full_url.endswith('/tokenize') else {
                'choices': [{'text': 'partial', 'finish_reason': 'length'}],
                'usage': {'prompt_tokens': 3, 'completion_tokens': 640}}
            return io.BytesIO(json.dumps(result).encode())
        with patch.object(w33.spec_bench, 'SpecServer', Server), patch.object(
                w33.urllib.request, 'urlopen', side_effect=response):
            w33.measure(self.run, self.assets)
        w33.evaluate(self.run)
        row = w33.read(self.run / 'predictions.jsonl')[0]
        self.assertEqual('length', row['finish_reason'])
        self.assertEqual('partial', row['raw_text'])
        self.assertEqual(0, row['exact'])

    def test_cleanup_can_finish_after_measurement_deadline(self):
        import time
        prepared = json.loads((self.run / 'prepared.json').read_text())
        prepared['max_seconds'] = 1
        (self.run / 'prepared.json').write_text(json.dumps(prepared))
        completed = []
        class Server:
            def __init__(self, *args, **kwargs): pass
            def start(self): pass
            def stop(self):
                time.sleep(1.1)
                completed.append(True)
        with patch.object(w33.spec_bench, 'SpecServer', Server), patch.object(
                w33.eval_ablation, 'complete', return_value=('answer', 0.1)):
            w33.measure(self.run, self.assets)
        self.assertEqual([True], completed)

    def test_gpu_recipe_refuses_cpu_fallback(self):
        prepared = json.loads((self.run / 'prepared.json').read_text())
        prepared['gpu'] = True
        (self.run / 'prepared.json').write_text(json.dumps(prepared))
        stopped = []
        class Server:
            def __init__(self, *args, **kwargs):
                self.path = kwargs['log_path']
            def start(self):
                self.path.write_text('offloaded 0/25 layers to GPU\n')
            def stop(self):
                stopped.append(True)
        with patch.object(w33.spec_bench, 'SpecServer', Server), self.assertRaisesRegex(ValueError, 'GPU offload'):
            w33.measure(self.run, self.assets)
        self.assertEqual([True], stopped)
        self.assertFalse((self.run / 'predictions.jsonl').exists())

    def test_gpu_recipe_enables_offload_evidence_at_trace_verbosity(self):
        prepared = json.loads((self.run / 'prepared.json').read_text())
        prepared['gpu'] = True
        (self.run / 'prepared.json').write_text(json.dumps(prepared))
        class Server:
            def __init__(self, model, port, extra_flags, **kwargs):
                self.path = kwargs['log_path']
                self.flags = extra_flags
            def start(self):
                self.path.write_text('offloaded 25/25 layers to GPU\n' if '-lv' in self.flags else 'model loaded\n')
            def stop(self): pass
        with patch.object(w33.spec_bench, 'SpecServer', Server), patch.object(
                w33.eval_ablation, 'complete', return_value=('answer', 0.1)):
            w33.measure(self.run, self.assets)
        self.assertEqual(20, len(w33.read(self.run / 'predictions.jsonl')))
