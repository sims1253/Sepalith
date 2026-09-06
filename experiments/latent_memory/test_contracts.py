"""Contract tests precede the learnability run; CPU fixtures use the pinned backend."""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from experiments.latent_memory.data import prepare_view, make_dataset, validate_dataset
from experiments.latent_memory.model import (ModuleEncoder, attach_adapters, embedding_forward,
    target_loss, token_embedding_contract, load_memory, save_memory, compatibility)


def tiny_decoder():
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=['linear_attention', 'full_attention'], linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': 1., 'mrope_section': [2, 2, 4]})
    config._attn_implementation = 'eager'
    model = Qwen3_5ForCausalLM(config).eval()
    model._latent_base_identity = 'tiny-fixture-not-a-pretrained-checkpoint'
    return model


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1273)
        torch.set_num_threads(1)

    def test_token_ids_equal_embeddings_both_layer_types(self):
        model = tiny_decoder()
        result = token_embedding_contract(model, torch.tensor([[1, 2, 3, 4]]), 1e-6, 1e-5)
        self.assertTrue(result['greedy_equal'])

    def test_target_only_shift_and_gradients(self):
        model = tiny_decoder()
        slots = torch.randn(1, 3, 32, requires_grad=True)
        prompt, target = torch.tensor([[2, 3]]), torch.tensor([[4, 5]])
        loss, details = target_loss(model, prompt, target, slots)
        embeds = torch.cat([slots, model.get_input_embeddings()(torch.cat([prompt, target], 1))], 1)
        logits = embedding_forward(model, embeds).logits
        labels = torch.tensor([[-100, -100, -100, -100, -100, 4, 5]])
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, 64), labels[:, 1:].reshape(-1))
        torch.testing.assert_close(loss, expected)
        self.assertEqual(details['scored_tokens'], 2)
        loss.backward()
        self.assertGreater(slots.grad.abs().sum().item(), 0)

    def test_reuse_rejected(self):
        model = tiny_decoder()
        with self.assertRaises(ValueError):
            embedding_forward(model, torch.zeros(1, 2, 32), past_key_values=object())
        self.assertIsNone(embedding_forward(model, torch.zeros(1, 2, 32)).past_key_values)

    def test_artifact_identity_freshness_integrity_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = compatibility('encoder-a', 'decoder-a', 'tokenizer-a', 3, 32, 'float32', 'module-a')
            hashes = {'R/a.R': 'a' * 64}
            manifest = save_memory(root, torch.ones(3, 32), expected, hashes)
            torch.testing.assert_close(load_memory(root, manifest, expected, hashes), torch.ones(3, 32))
            for changed in [dataclasses.replace(expected, decoder_revision='other'),
                            dataclasses.replace(expected, encoder_revision='other'),
                            dataclasses.replace(expected, layout_contract='other')]:
                with self.assertRaises(ValueError):
                    load_memory(root, manifest, changed, hashes)
            with self.assertRaises(ValueError):
                load_memory(root, manifest, expected, {'R/a.R': 'b' * 64})
            with self.assertRaises(ValueError):
                save_memory(root, torch.ones(4, 32), expected, hashes)
            (root / manifest.payload_path).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                load_memory(root, manifest, expected, hashes)

    def test_masked_answers_and_future_edits_excluded(self):
        parent = {'R/local.R': 'active answer', 'R/remote.R': 'abc SECRET xyz',
                  'R/copy.R': 'abc SECRET xyz', 'generated/future.R': 'FUTURE'}
        view = prepare_view(parent, 'parent', 'parent', 'R/local.R', ['R/remote.R'],
                            {'R/remote.R': [(4, 10)]}, 'query')
        self.assertNotIn('SECRET', view.modules[0].source)
        self.assertNotIn('FUTURE', repr(view))
        self.assertNotIn('answer', dataclasses.asdict(view))
        with self.assertRaises(ValueError):
            prepare_view(parent, 'future', 'parent', 'R/local.R', ['R/remote.R'], {}, 'q')
        with self.assertRaises(ValueError):
            prepare_view(parent, 'parent', 'parent', 'R/local.R', ['R/local.R'], {}, 'q')
        with self.assertRaises(ValueError):
            prepare_view(parent, 'parent', 'parent', 'R/local.R', ['R/copy.R'],
                         {'R/remote.R': [(4, 10)]}, 'q')

    def test_git_reader_uses_parent_not_edit_or_worktree(self):
        import subprocess
        from .data import git_parent_view
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)
            git('init')
            git('config', 'user.email', 'fixture@example.invalid')
            git('config', 'user.name', 'Fixture')
            (root / 'R').mkdir()
            (root / 'R/remote.R').write_text('parent value = 17')
            git('add', '.')
            git('commit', '-m', 'parent')
            (root / 'R/remote.R').write_text('future answer = 99')
            git('commit', '-am', 'edit')
            (root / 'R/remote.R').write_text('uncommitted future = 88')
            view = git_parent_view(root, 'HEAD', 'R/local.R', ['R/remote.R'], {}, 'query')
            self.assertEqual(view.modules[0].source, 'parent value = 17')
            self.assertNotIn('future answer = 99', repr(view))
            self.assertNotIn('uncommitted future = 88', repr(view))

    def test_input_view_is_independent_of_scoring_target(self):
        from .data import input_view
        row = make_dataset(1273, 2, 2)['train'][0]
        before = input_view(row)
        row['target'] = 'POST_EDIT_SECRET'
        row['future_edit'] = {'R/facts.R': 'POST_EDIT_SECRET'}
        self.assertEqual(input_view(row), before)

    def test_split_before_generation_and_deterministic(self):
        first = make_dataset(1273, 16, 8)
        self.assertEqual(first, make_dataset(1273, 16, 8))
        validate_dataset(first)
        broken = json.loads(json.dumps(first))
        broken['eval'][0]['package'] = broken['train'][0]['package']
        with self.assertRaises(ValueError):
            validate_dataset(broken)

    def test_encoder_pooler_and_explicit_adapters_receive_gradients(self):
        model = tiny_decoder()
        targets = attach_adapters(model, rank=2, alpha=4, dropout=0)
        self.assertEqual(len(targets), 9)
        encoder = ModuleEncoder(32, width=32, layers=1, heads=4, slots=3, max_bytes=128, dropout=0)
        memory = encoder(['R/a.R\nf_abc <- function(x = 73) x'])
        self.assertEqual(tuple(memory.shape), (1, 3, 32))
        loss, _ = target_loss(model, torch.tensor([[2, 3]]), torch.tensor([[4, 5]]), memory)
        loss.backward()
        self.assertGreater(encoder.queries.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad))
        for name in targets:
            self.assertGreater(model.get_submodule(name).B.grad.abs().sum().item(), 0, name)

class ExecutionContracts(unittest.TestCase):
    def test_full_tiny_training_checkpoint_and_all_evaluation_arms(self):
        from .experiment import train_arm, evaluate, verdict, restore_checkpoint
        from .provenance import Ledger
        class FixtureTokenizer:
            eos_token_id = 1
            def encode(self, text, add_special_tokens=False):
                return [2 + ord(c) % 60 for c in text]
            def decode(self, ids, skip_special_tokens=True):
                return ''.join(chr(32 + i) for i in ids if i != 1)
        torch.set_num_threads(1)
        recipe = json.loads(Path(__file__).with_name('recipe.json').read_text())
        recipe.update(steps_per_arm=1, effective_batch=2, max_prompt_tokens=256,
                      max_retrieval_tokens=512, bootstrap_samples=100)
        recipe['encoder'] = dict(width=32, layers=1, heads=4, slots=3, max_bytes=1024, dropout=0.)
        recipe['adapter'] = dict(rank=2, alpha=4, dropout=0.)
        dataset = make_dataset(1273, 4, 4)
        tokenizer = FixtureTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root, recipe)
            predictions = []
            identities = {'encoder': 'encoder', 'decoder': 'decoder', 'tokenizer': 'tokenizer'}
            predictions += evaluate(tiny_decoder(), tokenizer, None, dataset['eval'], 'base', recipe,
                                    root, ledger, identities, torch.device('cpu'))
            for arm in recipe['arms']:
                model = tiny_decoder()
                arm_root = root / arm
                arm_root.mkdir()
                encoder, hashes = train_arm(model, tokenizer, dataset['train'], arm, recipe,
                                            arm_root, ledger, torch.device('cpu'))
                self.assertIn('adapter.safetensors', hashes)
                self.assertTrue((arm_root / 'final/training-state.pt').is_file())
                restore_checkpoint(arm_root / 'final', model, encoder)
                original_identity = model._latent_base_identity
                model._latent_base_identity = 'different-base'
                with self.assertRaises(ValueError):
                    restore_checkpoint(arm_root / 'final', model, encoder)
                model._latent_base_identity = original_identity
                for eval_arm in (['latent', 'absent', 'shuffled'] if arm == 'latent' else [arm]):
                    predictions += evaluate(model, tokenizer, encoder, dataset['eval'], eval_arm,
                                            recipe, root, ledger, identities, torch.device('cpu'))
            result = verdict(predictions, recipe)
            self.assertEqual(len(predictions), 24)
            self.assertEqual(result['editing_improvement'], 'NOT TESTED')
            self.assertEqual(ledger.totals['scored_tokens'], 18)

    def test_absolute_worker_entry_and_unpinned_freeze_rejection(self):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / 'frozen'
            frozen.mkdir()
            (frozen / 'freeze.json').write_text(json.dumps({'weights_pinned': False}))
            (frozen / 'recipe.json').write_text(Path(__file__).with_name('recipe.json').read_text())
            output = root / 'output'
            output.mkdir()
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
            env.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('entry.py').resolve()),
                                     '_worker', '--frozen', str(frozen), '--output', str(output)],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Preparation only', result.stderr)
            self.assertEqual(json.loads((output / 'result.json').read_text())['status'], 'INCOMPLETE')

    def test_compute_cap_rejects_before_next_batch(self):
        from .provenance import Ledger
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(directory, {'max_wall_seconds': 60, 'max_scored_tokens': 3})
            ledger.reserve(scored_tokens=3)
            with self.assertRaises(RuntimeError):
                ledger.reserve(scored_tokens=1)
            self.assertEqual(ledger.totals['scored_tokens'], 3)

    def test_verdict_requires_relevant_gain_and_shuffle_drop(self):
        from .experiment import verdict
        recipe = json.loads(Path(__file__).with_name('recipe.json').read_text())
        recipe['bootstrap_samples'] = 200
        records = [{'id': str(i), 'arm': arm, 'exact': arm in ['latent', 'retrieval']}
                   for arm in recipe['eval_arms'] for i in range(40)]
        self.assertEqual(verdict(records, recipe)['status'], 'PASS')
        for row in records:
            if row['arm'] == 'shuffled':
                row['exact'] = True
        self.assertEqual(verdict(records, recipe)['status'], 'FAIL')
        with self.assertRaises(ValueError):
            verdict(records[:-1], recipe)


if __name__ == '__main__':
    unittest.main()
