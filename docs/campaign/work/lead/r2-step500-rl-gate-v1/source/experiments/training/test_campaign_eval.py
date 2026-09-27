"""Check development accounting and protocol outcomes without a model launch."""
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from campaign_checkpoint import digest
from campaign_eval import classify, development_evaluator, loss_sums
from sepalith.campaign_protocol import (
    Cursor, Position, PromptContext, ReplacementRange, RENDERER_ID, serialize_target,
)

torch.set_num_threads(1)


def context():
    return PromptContext(
        path="R/example.R", prefix=(), selected_references=(), history=(), diagnostics=(), retrieval=(),
        scope_mode="off", scope_lines=(), suffix_lines=(), region_old=("x <- 1",),
        cursor=Cursor(0, 6, 6), replacement_range=ReplacementRange(
            "file:///R/example.R", 1, "0" * 64, Position(0, 0), Position(0, 6)),
    )


class TextTokenizer:
    """Lossless fixture tokenizer, not a claim about the pinned production model."""
    def encode(self, text, **kwargs):
        return [ord(char) + 512 for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token - 512) for token in ids)


class ControlledModel:
    def __init__(self, continuations):
        self.continuations = iter(continuations)
        self.parameter = torch.zeros(1)

    def parameters(self):
        yield self.parameter

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 1024))

    def generate(self, input_ids, **kwargs):
        assert kwargs['eos_token_id'] == [1, 130073]
        continuation = torch.tensor([next(self.continuations)])
        return torch.cat([input_ids, continuation], dim=1)


class EvaluationTests(unittest.TestCase):
    def test_prompt_target_loss_partition_matches_full_next_token_loss(self):
        ids = torch.tensor([[0, 3, 5, 6, 1]])
        logits = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) / 7
        result = loss_sums(logits, ids, target_start=3)
        expected = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item()
        self.assertAlmostEqual(result["prompt_nll_sum"] + result["target_nll_sum"], expected, places=5)
        self.assertEqual((result["prompt_tokens"], result["target_tokens"]), (2, 2))

    def test_noop_copy_delete_and_missing_eos_remain_distinct(self):
        ctx = context()
        for raw in (serialize_target("no_op", ctx.region_old), serialize_target("replace", ctx.region_old)):
            result = classify(raw, ctx, list(ctx.region_old), [42, 1])
            self.assertTrue(result["exact_region"] and result["predicted_noop"])
        result = classify(serialize_target("delete", []), ctx, [], [42, 1])
        self.assertTrue(result["exact_region"] and result["suggestion"])
        self.assertFalse(classify(serialize_target("delete", []), ctx, [], [42])["protocol_valid"])
        for tokens in ([42, 130073], [42, 2, 1], [42, 130082, 1], [42, 130073, 1]):
            self.assertFalse(classify(serialize_target("delete", []), ctx, [], tokens)["protocol_valid"])

    def test_panel_denominators_partial_output_and_final_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [{"id": "noop", "split": "dev", "family": "no_op", "package_id": "pkg-a",
                      "operation": "no_op", "region_new": ["x <- 1"], "context": context().to_dict()},
                     {"id": "delete", "split": "dev", "family": "delete", "package_id": "pkg-b",
                      "operation": "delete", "region_new": [], "context": context().to_dict()}]
            panel = root / "panel.jsonl"
            panel.write_text("".join(json.dumps(case) + "\n" for case in cases))
            recipe = {"development_panel": {"path": str(panel), "sha256": digest(panel)},
                      "renderer_id": RENDERER_ID, "development_case_ids": ["noop", "delete"],
                      "development_max_new_tokens": 64, "parameters": {"max_sequence_tokens": 4096}}
            tok = TextTokenizer()
            outputs = [tok.encode(serialize_target(case["operation"], case["region_new"])) + [1] for case in cases]
            evaluate = development_evaluator(recipe)
            result = evaluate(ControlledModel(outputs), tok, root / "archive/full/checkpoint-2", 2)
            self.assertEqual(result["denominators"]["cases"], 2)
            self.assertEqual(result["denominators"]["strict_noop"], 1)
            self.assertEqual(result["counts"]["exact_region"], 2)
            self.assertEqual(result["counts"]["strict_noop_false_suggestions"], 0)
            self.assertAlmostEqual(result["target_nll"], math.log(1024), places=5)
            self.assertEqual(json.loads(Path(result["cases_path"]).read_text())["status"], "complete")
            with self.assertRaises(StopIteration):
                evaluate(ControlledModel(outputs[:1]), tok, root / "archive/full/checkpoint-3", 3)
            partial = json.loads((root / "archive/evaluations/cases-step-3.json").read_text())
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(len(partial["results"]), 1)
            cases[0]["split"] = "final"
            panel.write_text("".join(json.dumps(case) + "\n" for case in cases))
            recipe["development_panel"]["sha256"] = digest(panel)
            with self.assertRaisesRegex(ValueError, "Only development"):
                development_evaluator(recipe)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
