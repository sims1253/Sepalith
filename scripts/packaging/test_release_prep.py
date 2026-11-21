"""CPU release checks: negative parity and streaming timing boundaries."""
import json
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/sepalith/src"))

from notebook_latency import measure, percentile, run
from release_parity import compare_prompts, validate_request
from sepalith.campaign_protocol import PromptContext, render_prompt


class ReleasePrepTests(unittest.TestCase):
    def test_render_mismatch_rejected(self):
        contexts = json.loads(Path('docs/training/plan-20260927/c17/prompts.json').read_text())
        prompts = [render_prompt(PromptContext.from_dict(c)) for c in contexts]
        self.assertEqual(len(compare_prompts(contexts, prompts)), 8)
        prompts[0] += '\n'
        with self.assertRaises(ValueError):
            compare_prompts(contexts, prompts)

    def test_chat_and_implicit_bos_rejected(self):
        for settings in ({'content': 'x', 'add_special': True, 'parse_special': False, 'with_pieces': False},
                         {'content': 'x', 'add_special': False, 'parse_special': True, 'with_pieces': False}):
            with self.assertRaises(ValueError):
                validate_request({'prompt': 'x', 'tokenize': settings, 'request': {}})
        with self.assertRaises(ValueError):
            validate_request({'prompt': 'x', 'tokenize': {'content': 'x', 'add_special': False,
                             'parse_special': False, 'with_pieces': False}, 'request': {'prompt': [0, 0, 42]}})

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile(list(range(1, 41)), .95), 38)
        self.assertEqual(percentile([4], .5), 4)
        for bad in ([], [float('nan')], [-1]):
            with self.assertRaises(ValueError):
                percentile(bad, .5)

    def test_stream_ignores_empty_events_and_counts_final_timing(self):
        events = [b': keepalive\n', b'data: {"content":""}\n',
                  b'data: {"content":"a"}\n',
                  b'data: {"stop":true,"timings":{"predicted_n":4,"predicted_ms":200,"prompt_n":32}}\n']
        ticks = iter([10.1, 10.3])
        result = measure(events, 10, lambda: next(ticks))
        self.assertAlmostEqual(result['ttft_ms'], 100)
        self.assertEqual(result['tokens_s'], 20)
        self.assertEqual(result['prompt_n'], 32)

    def test_partial_or_error_stream_fails(self):
        for events in ([b'data: {"content":"x"}\n'], [b'data: {"error":"failed"}\n'], []):
            with self.assertRaises(ValueError):
                measure(events, 0, lambda: 1)

    def test_run_excludes_warmup_and_sends_one_bos(self):
        context = json.loads(Path('docs/training/plan-20260927/c17/prompts.json').read_text())[0]
        calls = []
        def respond(url, endpoint, body):
            calls.append((endpoint, body))
            if endpoint == '/tokenize':
                return io.BytesIO(b'{"tokens":[42,43]}')
            return io.BytesIO(b'data: {"content":"x"}\n'
                                b'data: {"stop":true,"timings":{"predicted_n":4,"predicted_ms":200,"prompt_n":3}}\n')
        with patch('notebook_latency.request', side_effect=respond):
            result = run('http://unused', [context], 5)
        self.assertEqual(len(result['samples']), 5)
        self.assertEqual(len(calls), 7)
        for endpoint, body in calls[1:]:
            self.assertEqual(body['prompt'], [0, 42, 43])
            self.assertFalse(body['cache_prompt'])
            self.assertTrue(body['stream'])
        self.assertEqual(result['summary']['tokens_s']['p50'], 20)

    def test_invalid_protocol_inputs_fail_before_network(self):
        with self.assertRaises(ValueError):
            run('http://invalid', [], 5)
        with self.assertRaises(ValueError):
            run('http://invalid', [{}], 0)


if __name__ == '__main__':
    unittest.main()
