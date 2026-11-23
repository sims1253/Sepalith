"""CPU release checks: negative parity and streaming timing boundaries."""
import json
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/sepalith/src"))

from latency_receipt import backend_evidence, gate, probe_memory
from notebook_latency import measure, percentile, run
from release_parity import check_tiers, compare_prompts, validate_request
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
                  b'data: {"stop":true,"timings":{"predicted_n":4,"predicted_ms":200,"predicted_per_second":15.0,"prompt_n":32}}\n']
        ticks = iter([10.1, 10.3])
        result = measure(events, 10, lambda: next(ticks))
        self.assertAlmostEqual(result['ttft_ms'], 100)
        self.assertEqual(result['tokens_s'], 15.0)
        self.assertEqual(result['prompt_n'], 32)

    def test_partial_or_error_stream_fails(self):
        one_token = b'data: {"stop":true,"timings":{"predicted_n":1,"predicted_ms":5,"predicted_per_second":0,"prompt_n":3}}\n'
        for events in ([b'data: {"content":"x"}\n'], [b'data: {"error":"failed"}\n'], [],
                       [b'data: {"content":"x"}\n', one_token]):
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
                                b'data: {"stop":true,"timings":{"predicted_n":4,"predicted_ms":200,"predicted_per_second":15.0,"prompt_n":3}}\n')
        with patch('notebook_latency.request', side_effect=respond):
            result = run('http://unused', [context], 5)
        self.assertEqual(len(result['samples']), 5)
        self.assertEqual(len(calls), 7)
        for endpoint, body in calls[1:]:
            self.assertEqual(body['prompt'], [0, 42, 43])
            self.assertFalse(body['cache_prompt'])
            self.assertTrue(body['stream'])
        self.assertEqual(result['summary']['tokens_s']['p50'], 15.0)

    def test_tier_set_must_match_evidence(self):
        import tempfile, hashlib
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp, 'a.gguf'), Path(tmp, 'b.gguf')
            a.write_bytes(b'a'); b.write_bytes(b'b')
            ev = {'tiers': [{'gguf': {'sha256': hashlib.sha256(x).hexdigest()}} for x in (b'a', b'b')]}
            check_tiers([a, b], ev)
            for paths in ([a], [a, a], [a, b, b]):
                with self.assertRaises(ValueError):
                    check_tiers(paths, ev)

    def test_invalid_protocol_inputs_fail_before_network(self):
        with self.assertRaises(ValueError):
            run('http://invalid', [], 5)
        with self.assertRaises(ValueError):
            run('http://invalid', [{}], 0)


    def test_backend_evidence_rejects_silent_fallback(self):
        vk = ('0.0 I cmn  common_param:   - Vulkan0 : AMD Radeon Graphics (RADV RENOIR) (8156 MiB, 7284 MiB free)\n'
              '0.4 I load_tensors: offloaded 43/43 layers to GPU\n'
              '0.4 I load_tensors:      Vulkan0 model buffer size =  1821.99 MiB\n')
        cpu = '0.4 I load_tensors:   CPU_Mapped model buffer size =  2092.93 MiB\n'
        warn = 'warning: no usable GPU found, --gpu-layers option will be ignored\n'
        rise = probe_memory('before gpu_busy=0 vram_used=1 gtt_used=100\nloaded gpu_busy=9 vram_used=1 gtt_used=900\n')
        flat = probe_memory('before gpu_busy=0 vram_used=1 gtt_used=100\nloaded gpu_busy=0 vram_used=1 gtt_used=100\n')
        busy, idle = {'busy_nonzero': 5}, {'busy_nonzero': 0}
        self.assertTrue(backend_evidence('vulkan', '', vk, rise, busy)['confirmed'])
        self.assertTrue(backend_evidence('cpu', warn, cpu, flat)['confirmed'])
        self.assertTrue(backend_evidence('cpu', warn, cpu, flat, idle)['confirmed'])
        self.assertFalse(backend_evidence('vulkan', '', cpu, flat, busy)['confirmed'])
        self.assertFalse(backend_evidence('vulkan', '', vk.replace('43/43', '30/43'), rise, busy)['confirmed'])
        self.assertFalse(backend_evidence('vulkan', '', vk, flat, busy)['confirmed'])
        self.assertFalse(backend_evidence('vulkan', warn, vk, rise, busy)['confirmed'])
        self.assertFalse(backend_evidence('vulkan', '', vk, rise, idle)['confirmed'])
        self.assertFalse(backend_evidence('vulkan', '', vk, rise)['confirmed'])
        self.assertFalse(backend_evidence('cpu', warn, vk, rise)['confirmed'])
        self.assertFalse(backend_evidence('cpu', warn, cpu, flat, busy)['confirmed'])

    def test_receipt_gate_needs_full_clean_run(self):
        sample = {'ttft_ms': 5.0, 'tokens_s': 20.0, 'wall_ms': 9.0, 'predicted_n': 12, 'stop_type': 'eos'}
        result = {'samples': [{'prompt': p, 'repeat': r, 'prompt_n': 100 + p, **sample}
                              for p in range(8) for r in range(5)],
                  'tier': 'Q6_K', 'runtime_commit': '3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70',
                  'model_sha256': '6593785330b99076b80ae3ac679bda76c5fcff1ebd1873f46f944c1ec43a3261',
                  'prompts_sha256': 'c61cabc8cdb8a5d4eaf83cd90dc1633a1e20c858d3fbe85cbfd511ab2d2e36d2'}
        line = ('slot print_timing: id  0 | task 0 |        eval time =  600.00 ms /    12 tokens '
                '(   50.00 ms per token,    20.00 tokens per second)\n'
                'slot print_timing: id  0 | task 0 | prompt eval time =  1.00 ms /   1 tokens '
                '(    1.00 ms per token,    99.00 tokens per second)\n')
        log = (line + 'stop processing: n_tokens = 142, truncated = 0\n') * 48
        wrong_run = log.replace('20.00 tokens per second', '21.37 tokens per second')
        self.assertEqual(gate(result, {'confirmed': True}, log)['outcome'], 'pass')
        self.assertIsNone(gate(result, {'confirmed': True}, log)['latency_cutoff'])
        self.assertEqual(gate(result, {'confirmed': False}, log)['outcome'], 'fail')
        self.assertEqual(gate(result, {'confirmed': True}, '')['outcome'], 'fail')
        self.assertEqual(gate(result, {'confirmed': True}, log.split('stop processing', 1)[1])['outcome'], 'fail')
        self.assertEqual(gate(result, {'confirmed': True}, wrong_run)['outcome'], 'fail')
        truncated = log.replace('truncated = 0', 'truncated = 1', 1)
        self.assertEqual(gate(result, {'confirmed': True}, truncated)['outcome'], 'fail')
        short = {**result, 'samples': result['samples'][:39]}
        self.assertEqual(gate(short, {'confirmed': True}, log)['outcome'], 'fail')
        bad = {**result, 'samples': [*result['samples'][:39], {**result['samples'][39], 'tokens_s': float('nan')}]}
        self.assertEqual(gate(bad, {'confirmed': True}, log)['outcome'], 'fail')


if __name__ == '__main__':
    unittest.main()
