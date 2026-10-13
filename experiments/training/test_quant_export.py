import hashlib
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

# Fake llama.cpp tools: argument plumbing only, no real conversion or quantization.
FAKE_QUANT = """#!/usr/bin/env python3
import json, sys
from pathlib import Path
print('version: 0.1.0-dev (build 10453, commit 3cb7ffb1a)')
if '--dry-run' in sys.argv:
    print('fake quantize dry run', json.dumps(sys.argv[1:]))
    sys.exit(0)
Path(sys.argv[-3]).write_text(json.dumps(sys.argv[1:]))
"""
FAKE_CONVERT = """import json, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('--outfile') + 1])
(Path(sys.argv[0]).parent / 'converter-ran').write_text('1')
out.write_text(json.dumps(sys.argv[1:]))
"""


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fakes(d: Path) -> Path:
    quant = d / 'quantize'
    quant.write_text(FAKE_QUANT)
    quant.chmod(0o755)
    (d / 'convert.py').write_text(FAKE_CONVERT)
    return quant


def fake_hf_model(path: Path, arch: str) -> Path:
    path.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps({'architectures': [arch]}))
    (path / 'model.safetensors').write_bytes(b'weights')
    (path / 'tokenizer.json').write_text('{}')
    (path / 'optimizer.pt').write_bytes(b'training state, not a converter input')
    return path


def run_export(d: Path, args: list[str], **env: str) -> subprocess.CompletedProcess[str]:
    quant = write_fakes(d)
    full = {**os.environ, 'LLAMA_QUANT': str(quant), 'LLAMA_CONVERT': str(d / 'convert.py'),
            'LLAMA_CONVERT_PYTHON': sys.executable, **env}
    return subprocess.run([sys.executable, str(SCRIPT), *args], env=full, capture_output=True, text=True)


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

    def test_conversion_defaults_to_bf16_and_keeps_arch_gate(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for arch, outtype, nextn in (('Qwen3_5ForCausalLM', None, True), ('LlamaForCausalLM', 'f16', False)):
                model = fake_hf_model(d / arch, arch)
                out = d / ('out-' + arch)
                extra = ['--outtype', outtype] if outtype else []
                result = run_export(d, ['unused', str(model), 'conv', '--out-dir', str(out), '--keep-source',
                                        '--tiers', 'Q8_0', *extra], NO_LORA='1')
                self.assertEqual(result.returncode, 0, result.stderr)
                source = out / f'conv-{outtype or "bf16"}.gguf'
                argv = json.loads(source.read_text())
                self.assertEqual(argv[argv.index('--outtype') + 1], outtype or 'bf16')
                self.assertEqual('--no-nextn' in argv, nextn)
                self.assertFalse(list(out.glob('*.partial')))
                src_receipt = json.loads(source.with_suffix('.gguf.json').read_text())
                self.assertEqual(set(src_receipt['conversion']['hf_inputs']),
                                 {'config.json', 'model.safetensors', 'tokenizer.json'})
                self.assertEqual(src_receipt['conversion']['nextn_stripped'], nextn)
                q8 = json.loads((out / 'conv-Q8_0.gguf.json').read_text())
                self.assertEqual(q8['source']['sha256'], src_receipt['source']['sha256'])
                self.assertEqual(q8['conversion']['architectures'], [arch])
                self.assertEqual(q8['quantizer']['build'], {'number': 10453, 'commit': '3cb7ffb1a'})
                self.assertEqual(q8['output']['sha256'], sha(out / 'conv-Q8_0.gguf'))
                self.assertNotIn('--output-tensor-type', json.loads((out / 'conv-Q8_0.gguf').read_text()))

    def test_source_is_removed_without_keep_source(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = fake_hf_model(d / 'model', 'LlamaForCausalLM')
            result = run_export(d, ['unused', str(model), 'drop', '--out-dir', str(d / 'out')], NO_LORA='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(p.name for p in (d / 'out').iterdir()),
                             ['drop-Q8_0.gguf', 'drop-Q8_0.gguf.json'])

    def test_quantizer_flags_pass_through_and_are_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source = d / 'source.gguf'; source.write_bytes(b'bf16 source')
            matrix = d / 'imatrix.gguf'; matrix.write_bytes(b'importance')
            out = d / 'out'
            result = run_export(d, ['unused', 'unused', 'flags', '--source-gguf', str(source), '--out-dir', str(out),
                                    '--imatrix', str(matrix), '--tiers', 'Q5_K_M', 'IQ4_XS', 'Q6_K',
                                    '--tensor-type', 'attn_v=q6_k', '--tensor-type', r'blk\.0\..*=BF16',
                                    '--output-tensor-type', 'bf16', '--token-embedding-type', 'Q6_K',
                                    '--threads', '3'])
            self.assertEqual(result.returncode, 0, result.stderr)
            for tier in ('Q5_K_M', 'IQ4_XS', 'Q6_K'):
                argv = json.loads((out / f'flags-{tier}.gguf').read_text())
                self.assertEqual(argv[-2:], [tier, '3'])
                self.assertEqual(argv[argv.index('--output-tensor-type') + 1], 'bf16')
                self.assertEqual(argv[argv.index('--token-embedding-type') + 1], 'q6_K')
                overrides = [argv[i + 1] for i, a in enumerate(argv) if a == '--tensor-type']
                self.assertEqual(overrides, ['attn_v=q6_K', r'blk\.0\..*=bf16'])
                self.assertEqual(argv[argv.index('--imatrix') + 1], str(matrix.resolve()))
                receipt = json.loads((out / f'flags-{tier}.gguf.json').read_text())
                self.assertEqual(receipt['imatrix'], {'path': str(matrix.resolve()), 'sha256': sha(matrix)})
                self.assertEqual(receipt['source']['sha256'], sha(source))
                self.assertTrue(receipt['source']['reused'])
                self.assertIsNone(receipt['conversion'])
                self.assertEqual(receipt['flags']['tensor_types'], overrides)
                self.assertEqual(receipt['quantize_command'][-3:], [str(out / f'flags-{tier}.gguf'), tier, '3'])
                self.assertEqual(receipt['quantizer']['sha256'], sha(d / 'quantize'))
            self.assertTrue(source.exists())

    def test_invalid_combinations_fail_before_any_work(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source = d / 'source.gguf'; source.write_bytes(b'bf16 source')
            base = ['unused', 'unused', 'bad', '--source-gguf', str(source), '--out-dir', str(d / 'out')]
            cases = [
                (['--tiers', 'IQ4_XS', '--uncalibrated'], 'IQ types require --imatrix'),
                (['--tiers', 'Q6_K', '--token-embedding-type', 'iq4_nl'], 'IQ types require --imatrix'),
                (['--tensor-type', 'attn_v'], 'PATTERN=TYPE'),
                (['--tensor-type', '=q8_0'], 'PATTERN=TYPE'),
                (['--tensor-type', 'attn_v=q9_0'], 'invalid ggml type'),
                (['--output-tensor-type', 'Q4_K_M'], 'invalid ggml type'),
                (['--tiers', 'Q3_K_M'], 'invalid choice'),
                (['--outtype', 'f32'], 'invalid choice'),
            ]
            for extra, message in cases:
                result = run_export(d, base + extra)
                self.assertNotEqual(result.returncode, 0, extra)
                self.assertIn(message, result.stderr, extra)
            self.assertFalse((d / 'out').exists())

    def test_dry_run_plans_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = fake_hf_model(d / 'model', 'LlamaForCausalLM')
            out = d / 'out'
            result = run_export(d, ['unused', str(model), 'plan', '--out-dir', str(out), '--dry-run',
                                    '--tiers', 'Q8_0', 'Q4_K_M', '--uncalibrated'], NO_LORA='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PLAN convert:', result.stdout)
            self.assertIn('--outtype bf16', result.stdout)
            self.assertIn('PLAN quantize:', result.stdout)
            self.assertFalse(out.exists())
            self.assertFalse((d / 'converter-ran').exists())
            source = d / 'source.gguf'; source.write_bytes(b'bf16 source')
            result = run_export(d, ['unused', 'unused', 'plan', '--source-gguf', str(source), '--out-dir', str(out),
                                    '--dry-run', '--tiers', 'Q4_K_M', '--uncalibrated'])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('fake quantize dry run', result.stdout)
            self.assertEqual(source.read_bytes(), b'bf16 source')
            self.assertFalse(out.exists())

    def test_default_converter_must_be_the_pin(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            llama = d / 'llama.cpp'
            (llama / 'src').mkdir(parents=True)
            (llama / 'src' / 'convert_hf_to_gguf.py').write_text('')
            git = ['git', '-C', str(llama / 'src'), '-c', 'user.name=t', '-c', 'user.email=t@t',
                   '-c', 'commit.gpgsign=false']
            subprocess.run(git + ['init', '-q'], check=True)
            subprocess.run(git + ['add', '.'], check=True)
            subprocess.run(git + ['commit', '-qm', 'not the pin'], check=True)
            model = fake_hf_model(d / 'model', 'LlamaForCausalLM')
            env = {k: v for k, v in os.environ.items() if k not in ('LLAMA_CONVERT', 'LLAMA_QUANT')}
            env.update(SEPALITH_LLAMA_CPP=str(llama), LLAMA_QUANT=str(write_fakes(d)), NO_LORA='1')
            result = subprocess.run([sys.executable, str(SCRIPT), 'unused', str(model), 'pin',
                                     '--out-dir', str(d / 'out')], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must be pinned to', result.stderr)

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
