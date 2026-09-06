"""CPU integration against frozen historical full-projection evaluators.

Extract only evaluator functions; do not import executable trainer entry points.
CUDA transfers are identities and autocast is a no-op: GPU precision is untested.
Historical function bodies are frozen from commit 73d0bfe, not rebuilt at runtime.
"""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from experiments.training.poc_twin.model import chunked_eval_ce

ROOT = Path(__file__).parent


def load_evaluator(filename, name, namespace):
    tree = ast.parse((ROOT / "ladder" / filename).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


@torch.no_grad()
def historical_quick_eval(path, n_blocks=64, bs=8):
    eb = np.load(path, mmap_mode='r')
    (nats, toks) = (0.0, 0)
    for i in range(0, min(n_blocks, len(eb)), bs):
        x = torch.from_numpy(eb[i:i + bs].astype(np.int64)).cuda()
        h = fwd_trunk(x[:, :-1], probe=False)
        logits = F.linear(h, model.embed.weight)
        lg = logits.view(-1, logits.size(-1))
        tg = x[:, 1:].reshape(-1)
        for c in range(0, lg.size(0), 4096):
            nats += F.cross_entropy(lg[c:c + 4096].float(), tg[c:c + 4096], reduction='sum').item()
        toks += tg.numel()
    return nats / toks

@torch.no_grad()
def historical_bpb(model, blocks, total_bytes, bs=8, chunk=4096):
    (nats, toks) = (0.0, 0)
    for i in range(0, len(blocks), bs):
        x = torch.from_numpy(blocks[i:i + bs].astype(np.int64)).cuda()
        h = model.trunk(x[:, :-1], probe=False)
        logits = F.linear(h, model.embed.weight)
        lg = logits.view(-1, logits.size(-1))
        tg = x[:, 1:].reshape(-1)
        for c in range(0, lg.size(0), chunk):
            nats += F.cross_entropy(lg[c:c + chunk].float(), tg[c:c + chunk], reduction='sum').item()
        toks += tg.numel()
    return dict(bpb=nats / (total_bytes * math.log(2)), loss_per_tok=nats / toks, tokens=toks)

@torch.no_grad()
def historical_exit_bpbs(model, blocks, bs=8, chunk=4096):
    """{exit_label: nats} over the held-out blocks + mtp nats."""
    nats = {}
    n_tok = 0
    for i in range(0, len(blocks), bs):
        x = torch.from_numpy(blocks[i:i + bs].astype(np.int64)).cuda()
        inp = x[:, :-1]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            (h_pre, taps) = model.trunk_taps(inp, probe=False)
            targets = x[:, 1:]
            all_h = {'top': model.ln_f(h_pre)}
            for (e, tap) in zip(model.exit_layers, taps.values()):
                all_h[f'exit_{e}'] = tap
            for (label, h) in all_h.items():
                logits = F.linear(h, model.embed.weight)
                lg = logits.view(-1, logits.size(-1))
                tg = targets.reshape(-1)
                tot = 0.0
                for c in range(0, lg.size(0), chunk):
                    tot += F.cross_entropy(lg[c:c + chunk].float(), tg[c:c + chunk], reduction='sum').item()
                nats[label] = nats.get(label, 0.0) + tot
            if model.use_mtp:
                Tm = inp.size(1) - 1
                e_next = model.embed(inp[:, 1:Tm + 1])
                mtp_in = model.mtp_proj(torch.cat([model.mtp_norm(h_pre)[:, :Tm], e_next], dim=-1))
                cos = model.rope_cos[:Tm].cuda()
                sin = model.rope_sin[:Tm].cuda()
                mh = model.mtp_block(mtp_in, cos, sin, probe=False)
                lg = F.linear(mh, model.embed.weight).view(-1, model.embed.weight.size(0))
                tg = x[:, 2:Tm + 2].reshape(-1)
                tot = 0.0
                for c in range(0, lg.size(0), chunk):
                    tot += F.cross_entropy(lg[c:c + chunk].float(), tg[c:c + chunk], reduction='sum').item()
                nats['mtp'] = nats.get('mtp', 0.0) + tot
        n_tok += targets.numel()
    return (nats, n_tok)

class LadderEvalIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.blocks = np.arange(18).reshape(3, 6) % 17
        embed = torch.nn.Embedding(17, 5)
        self.model = SimpleNamespace(
            embed=embed, trunk=lambda x, probe=False: embed(x),
            trunk_taps=lambda x, probe=False: (embed(x), {1: embed(x) * 0.7}),
            ln_f=lambda h: h * 1.2, exit_layers=[1], use_mtp=True,
            mtp_norm=lambda h: h * 0.8,
            mtp_proj=lambda h: h[..., :5] + h[..., 5:] * 0.3,
            mtp_block=lambda h, cos, sin, probe=False: h * 1.1,
            rope_cos=torch.ones(6, 1), rope_sin=torch.zeros(6, 1))
        self.namespace = dict(torch=torch, np=np, math=math, F=F,
                              chunked_eval_ce=chunked_eval_ce,
                              model=self.model, fwd_trunk=self.model.trunk)
        self.transfers = patch.object(torch.Tensor, "cuda", lambda tensor, *a, **k: tensor)
        self.transfers.start()
        self.addCleanup(self.transfers.stop)

    def test_bpb_multi_batch_and_short_chunk(self):
        actual_fn = load_evaluator("bpb_eval.py", "bpb", self.namespace)
        expected = historical_bpb(self.model, self.blocks, 37, bs=2, chunk=4)
        actual = actual_fn(self.model, self.blocks, 37, bs=2, chunk=4)
        self.assertEqual(actual["tokens"], 15)
        for key in ("bpb", "loss_per_tok"):
            self.assertAlmostEqual(actual[key], expected[key], delta=2e-6)

    def test_quick_eval_preserves_batch_and_token_reduction(self):
        actual_fn = load_evaluator("train_ladder.py", "quick_eval", self.namespace)
        with patch.dict(historical_quick_eval.__wrapped__.__globals__,
                        model=self.model, fwd_trunk=self.model.trunk), \
                patch.object(np, "load", return_value=self.blocks):
            expected = historical_quick_eval("unused.npy", bs=2)
            actual = actual_fn("unused.npy", bs=2)
        self.assertAlmostEqual(actual, expected, delta=2e-6)

    def test_exits_and_mtp_preserve_alignment_and_autocast_scope(self):
        actual_fn = load_evaluator("eval_a2_exits.py", "exit_bpbs", self.namespace)
        for use_mtp in (False, True):
            self.model.use_mtp = use_mtp
            with self.subTest(use_mtp=use_mtp), patch.object(
                    torch, "autocast", side_effect=lambda *a, **k: nullcontext()) as cast:
                expected, expected_count = historical_exit_bpbs(
                    self.model, self.blocks, bs=2, chunk=4)
                actual, count = actual_fn(self.model, self.blocks, bs=2, chunk=4)
                self.assertEqual(count, expected_count)
                self.assertEqual(count, 15)
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    self.assertAlmostEqual(actual[key], expected[key], delta=2e-5)
                self.assertEqual(cast.call_count, 4)
                for call in cast.call_args_list:
                    self.assertEqual(call.args, ("cuda",))
                    self.assertEqual(call.kwargs, {"dtype": torch.bfloat16})


if __name__ == "__main__":
    unittest.main()
