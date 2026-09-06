import importlib.util
import json
import os
import uuid
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('export_gguf.py')
spec = importlib.util.spec_from_file_location('corpus', SCRIPT.with_name('build_imatrix_corpus.py'))
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


class QuantTests(unittest.TestCase):
    def test_import_does_not_parse_args_or_load_training_stack(self):
        result = subprocess.run([sys.executable, '-I', '-c',
            "import runpy, sys; runpy.run_path(sys.argv[1], run_name='export_library'); "
            "assert 'unsloth' not in sys.modules; assert 'torch' not in sys.modules",
            str(SCRIPT.resolve())], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_control_requires_opt_in_before_model_loading(self):
        p = subprocess.run([sys.executable, str(SCRIPT), 'base', 'adapter', 'test', '--tiers', 'Q4_K_M'], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('require --imatrix', p.stderr)

    def test_tiers_pass_policy_and_publish_receipts(self):
        stem = 'test_quant_' + uuid.uuid4().hex
        models = SCRIPT.resolve().parents[1] / 'models'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake = d / 'quantize'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\nPath(sys.argv[-3]).write_text(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            source = d / 'source.gguf'; source.write_bytes(b'fake f16')
            matrix = d / 'imatrix.gguf'; matrix.write_bytes(b'fake matrix')
            try:
                result = subprocess.run([sys.executable, str(SCRIPT), 'unused', 'unused', stem,
                    '--f16', str(source), '--imatrix', str(matrix), '--tiers', 'Q8_0', 'Q4_K_M'],
                    env={**os.environ, 'LLAMA_QUANT': str(fake)}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                q8 = json.loads((models / f'{stem}-Q8_0.gguf').read_text())
                q4 = json.loads((models / f'{stem}-Q4_K_M.gguf').read_text())
                self.assertNotIn('--imatrix', q8)
                self.assertIn('--imatrix', q4)
                self.assertEqual(q4[q4.index('--output-tensor-type') + 1], 'q8_0')
                self.assertTrue(source.exists())
                self.assertTrue((models / f'{stem}-Q4_K_M.gguf.json').exists())
                self.assertFalse(list(models.glob(f'{stem}*.partial')))
            finally:
                for path in models.glob(f'{stem}*'):
                    path.unlink()

    def test_reject_matrix_from_another_model(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source = d / 'source.gguf'; source.write_bytes(b'full precision')
            matrix = d / 'matrix.gguf'; matrix.write_bytes(b'importance')
            matrix.with_suffix('.gguf.json').write_text(json.dumps({'model_sha256': '0'*64}))
            result = subprocess.run([sys.executable, str(SCRIPT), 'unused', 'unused', 'test_reject_matrix',
                '--f16', str(source), '--imatrix', str(matrix), '--tiers', 'Q4_K_M'],
                env={**os.environ, 'LLAMA_QUANT': sys.executable}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('does not match this source GGUF', result.stderr)

    def test_sampling_is_reproducible_and_balanced(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / 'train.jsonl'
            src.write_text(''.join(json.dumps(dict(text=f'{family} {i}', family=family))+'\n' for family in ['a', 'b'] for i in range(20)))
            a = corpus.build(src, d / 'one.txt', 3)
            b = corpus.build(src, d / 'two.txt', 3)
            self.assertEqual(a, b)
            self.assertEqual(a['selected'], {'a': 3, 'b': 3})
            self.assertEqual((d / 'one.txt').read_bytes(), (d / 'two.txt').read_bytes())
            with self.assertRaises(ValueError):
                corpus.build(d / 'eval.jsonl', d / 'out.txt')


if __name__ == '__main__':
    unittest.main()
