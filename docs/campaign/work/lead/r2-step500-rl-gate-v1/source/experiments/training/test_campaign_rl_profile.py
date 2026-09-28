"""CPU checks for the bounded PRM-07/RL-01 mechanism probe."""

from __future__ import annotations

import ast
from contextlib import ExitStack, nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = Path(__file__).with_name("campaign_profile.py")
PROFILE_SPEC = importlib.util.spec_from_file_location("campaign_profile", PROFILE_PATH)
assert PROFILE_SPEC is not None and PROFILE_SPEC.loader is not None
profile = importlib.util.module_from_spec(PROFILE_SPEC)
sys.modules["campaign_profile"] = profile
PROFILE_SPEC.loader.exec_module(profile)

RL_PATH = Path(__file__).with_name("campaign_rl_profile.py")
RL_SPEC = importlib.util.spec_from_file_location("campaign_rl_profile", RL_PATH)
assert RL_SPEC is not None and RL_SPEC.loader is not None
rl = importlib.util.module_from_spec(RL_SPEC)
sys.modules["campaign_rl_profile"] = rl
RL_SPEC.loader.exec_module(rl)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(row_id: str, length: int, *, split: str = "train") -> dict[str, object]:
    if length < 8:
        raise ValueError(length)
    prompt_count = length - 5
    body_ids = [101, 102]
    terminal_ids = [103]
    ids = [rl.BOS_ID, *([100] * prompt_count), *body_ids, *terminal_ids, rl.EOS_ID]
    assert len(ids) == length
    return {
        "id": row_id,
        "input_ids": ids,
        "target_start": 1 + prompt_count,
        "target_body_tokens": body_ids,
        "target_terminal_tokens": terminal_ids,
        "target_body_token_count": len(body_ids),
        "target_terminal_token_count": len(terminal_ids),
        "family": "fixture",
        "package_id": "fixture-package",
        "renderer_id": "zeta2-prm03-v1",
        "prompt_text": "fixture prompt\n",
        "target_text": "x\n>>>>>>> UPDATED",
        "target_body_text": "x",
        "target_operation": "replace",
        "prompt_token_count": prompt_count,
        "target_token_count": len(body_ids) + len(terminal_ids),
        "bos_token_id": rl.BOS_ID,
        "eos_token_id": rl.EOS_ID,
        "tokenizer_revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
        "tokenizer_json_sha256": profile.EXPECTED_TOKENIZER_JSON_SHA256,
        "tokenization_policy": "hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0_terminal_eos1_final_lf_v1",
        "split": split,
    }


class CampaignRLProfileTests(unittest.TestCase):
    def test_larger_arm_failure_preserves_completed_smaller_arm(self):
        events = []
        model = mock.Mock()
        fast = mock.Mock()
        deadline = mock.Mock()
        prompts = [{"id": "p", "prompt_tokens": [0, 100]}]

        def generate(_model, _torch, _prompts, count, **_kwargs):
            events.append(("generation", count))
            if count == 4:
                raise RuntimeError("simulated larger-arm OOM")
            return [{"generated_tokens": [1]}] * count, {"seconds": 1.0}

        def policy(_model, _torch, _prompts, records, **_kwargs):
            events.append(("policy", len(records)))
            return {"finite": True}

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory) / "partial.json"
            replacements = {
                "load_profile_rows": ([], [], {}),
                "model_identity": {"revision": "pinned"},
                "prepare_prompt_records": prompts,
                "live_resource_guard": {"status": "clear"},
                "_load_rl_components": (model, None, None, fast, None, {}, {}),
                "_device_of": "cpu",
                "generation_mode": nullcontext(),
                "account_generation_records": {"records": 2},
                "load_pinned_reference_tokenizer": None,
                "restore_pinned_tokenizer_contract": {"verified": True},
            }
            for name, value in replacements.items():
                stack.enter_context(mock.patch.object(rl, name, return_value=value))
            stack.enter_context(mock.patch.object(rl, "_generate_for_prompts", side_effect=generate))
            stack.enter_context(mock.patch.object(rl, "_run_policy_phase", side_effect=policy))
            with self.assertRaisesRegex(RuntimeError, "larger-arm OOM"):
                rl.run_live_rl_profile(
                    mock.Mock(), prompt_count=1, prompt_ids=None,
                    candidate_counts=(2, 4), max_existing_vram_mib=0,
                    deadline=deadline, progress_path=path,
                )
            saved = json.loads(path.read_text())
            self.assertEqual(events, [("generation", 2), ("policy", 2), ("generation", 4)])
            self.assertEqual(saved["stage"], "policy_candidates_2")
            self.assertEqual(saved["generation_results"][0]["candidate_count"], 2)
            self.assertEqual(saved["policy_results"][0]["policy_batch_rows"], 2)
            self.assertFalse(saved["quality_claim"])

    def test_module_import_has_no_model_framework_import(self):
        tree = ast.parse(RL_PATH.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any(
            name == "torch" or name.startswith(("torch.", "unsloth", "transformers", "trl"))
            for name in imported
        ))

    def test_stored_prompt_selection_requires_manual_bos_and_cap(self):
        short = _row("short", 12)
        too_long_prompt = _row("long-prompt", 2_100)
        records = rl.prepare_prompt_records([short, too_long_prompt], prompt_count=1)
        self.assertEqual(records[0]["id"], "short")
        self.assertEqual(records[0]["prompt_tokens"], short["input_ids"][:short["target_start"]])
        broken = dict(short)
        broken["input_ids"] = [99, *short["input_ids"][1:]]
        with self.assertRaisesRegex(rl.RLInputError, "manual BOS"):
            rl.prepare_prompt_records([broken])
        with self.assertRaisesRegex(rl.RLInputError, "fit"):
            rl.prepare_prompt_records([too_long_prompt], prompt_count=1)

    def test_generation_trim_and_terminal_accounting(self):
        canonical = rl.trim_generated_sequence([0, 100, 100, 1, 1, 1], 2)
        noncanonical = rl.trim_generated_sequence([0, 100, 100, 130073, 1], 2)
        capped = rl.trim_generated_sequence([0, 100, *([100] * 192)], 2)
        controlled = rl.trim_generated_sequence([0, 100, 130072, 1], 2)
        result = rl.account_generation_records([
            {"prompt_tokens": [0, 100], **canonical},
            {"prompt_tokens": [0, 100], **noncanonical},
            {"prompt_tokens": [0, 100], **capped},
            {"prompt_tokens": [0, 100], **controlled},
        ])
        self.assertEqual(result["records"], 4)
        self.assertEqual(result["canonical_eos"], 2)
        self.assertEqual(result["noncanonical_eog"], 1)
        self.assertEqual(result["cap_hits"], 1)
        self.assertEqual(result["control_before_terminal"], 1)
        self.assertEqual(result["padded_after_terminal"], 3)
        self.assertEqual(result["valid_canonical_records"], 1)

    def test_policy_mask_uses_exact_prompt_response_geometry(self):
        import torch

        batch = rl.build_policy_tensors(
            [[0, 10], [0, 20, 21]], [[1], [5, 1]], torch, torch.device("cpu"),
        )
        self.assertEqual(batch["input_ids"].tolist(), [[0, 10, 1, 1, 1], [0, 20, 21, 5, 1]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        self.assertEqual(batch["response_mask"].tolist(), [[False, True, False, False], [False, False, True, True]])

    def test_fake_policy_logprob_backward_is_finite_and_nonzero(self):
        import torch

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(rl.VOCAB_SIZE))
                self.calls = []

            def forward(self, input_ids, attention_mask, use_cache=False, **kwargs):
                del attention_mask, use_cache
                self.calls.append(kwargs)
                logits = self.bias.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
                return type("Output", (), {"logits": logits})()

        model = FakeModel()
        prompts = [[rl.BOS_ID, 100], [rl.BOS_ID, 101, 102]]
        generated = [[7, rl.EOS_ID], [8, rl.EOS_ID]]
        loss, stats = rl.recompute_policy_logprob(model, torch, prompts, generated, torch.device("cpu"))
        self.assertEqual(stats["generated_token_denominator"], 4)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(model.calls, [{"labels": None, "return_dict": True}])
        loss.backward()
        gradients = rl.check_finite_nonzero_gradients(model, torch)
        self.assertTrue(gradients["finite"])
        self.assertTrue(gradients["nonzero"])

    def test_policy_logsoftmax_receives_only_masked_response_rows(self):
        import torch

        class ShapeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(rl.VOCAB_SIZE))

            def forward(self, input_ids, **kwargs):
                del kwargs
                return type("Output", (), {
                    "logits": self.bias.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1),
                })()

        model = ShapeModel()
        prompts = [[rl.BOS_ID, 100], [rl.BOS_ID, 101, 102]]
        generated = [[7, rl.EOS_ID], [8, rl.EOS_ID]]
        original = torch.log_softmax
        seen = []

        def capture(values, dim=None, **kwargs):
            seen.append(tuple(values.shape))
            return original(values, dim=dim, **kwargs)

        with mock.patch.object(torch, "log_softmax", side_effect=capture):
            loss, stats = rl.recompute_policy_logprob(
                model, torch, prompts, generated, torch.device("cpu"),
            )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(seen, [(4, rl.VOCAB_SIZE)])
        self.assertEqual(stats["full_logits_shape"], [2, 5, rl.VOCAB_SIZE])
        self.assertEqual(stats["response_logits_shape"], [4, rl.VOCAB_SIZE])
        self.assertTrue(stats["logits_selected_before_fp32"])

    def test_policy_phase_reports_prompt_times_candidate_batch_rows(self):
        import torch

        class BatchModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(rl.VOCAB_SIZE))

            def forward(self, input_ids, **kwargs):
                del kwargs
                logits = self.bias.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
                return type("Output", (), {"logits": logits})()

        prompts = [
            {"id": "p1", "prompt_tokens": [rl.BOS_ID, 100]},
            {"id": "p2", "prompt_tokens": [rl.BOS_ID, 101, 102]},
        ]
        records = [
            {"prompt_id": "p1", "generated_tokens": [7, rl.EOS_ID]},
            {"prompt_id": "p1", "generated_tokens": [8, rl.EOS_ID]},
            {"prompt_id": "p2", "generated_tokens": [9, rl.EOS_ID]},
            {"prompt_id": "p2", "generated_tokens": [10, rl.EOS_ID]},
        ]
        result = rl._run_policy_phase(
            BatchModel(), torch, prompts, records, deadline=type("Deadline", (), {"check": lambda self: None})(),
        )
        self.assertEqual(result["policy_batch_rows"], 4)
        self.assertEqual(result["policy_prompt_count"], 2)
        self.assertEqual(result["loss"]["generated_token_denominator"], 8)

    def test_profile_receipt_assigns_live_owner_and_lease(self):
        args = type("Args", (), {
            "preflight": False, "prompt_count": 1, "wall_timeout_seconds": 900.0,
            "max_existing_vram_mib": 1024.0,
        })()
        spec = profile.ProfileInput(
            candidate_file=Path("/tmp/candidates.jsonl"), candidate_sha256="0" * 64,
            selected_ids_file=Path("/tmp/selected.json"), selected_ids_sha256="1" * 64,
        )
        receipt = rl._base_receipt(args, spec, (2, 4))
        self.assertEqual(receipt["owner"], "lead")
        self.assertIn("lead-owned", receipt["resource_lease"])

    def test_candidate_count_parser_is_bounded_and_ordered(self):
        self.assertEqual(rl.parse_candidate_counts("2,4"), (2, 4))
        with self.assertRaises(rl.RLInputError):
            rl.parse_candidate_counts("2,5")

    def test_framework_free_preflight_checks_explicit_hashed_inputs(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            candidate = directory / "candidate.jsonl"
            candidate.write_text(json.dumps(_row("short", 12)) + "\n", encoding="utf-8")
            selected = directory / "selected.json"
            selected.write_text(json.dumps({
                "schema_version": profile.SELECTED_IDS_SCHEMA_VERSION,
                "split": "train",
                "row_ids": ["short"],
            }, sort_keys=True) + "\n", encoding="utf-8")
            spec = profile.ProfileInput(
                candidate_file=candidate,
                candidate_sha256=_sha(candidate),
                selected_ids_file=selected,
                selected_ids_sha256=_sha(selected),
                model_path=profile.EXPECTED_MODEL_PATH,
            )
            with mock.patch.object(rl, "_frameworks_imported", return_value=[]), \
                    mock.patch.object(rl, "model_identity", return_value={"path": str(profile.EXPECTED_MODEL_PATH)}):
                result = rl.preflight_rl(spec, prompt_count=1, candidate_counts=(2, 4))
            self.assertFalse(result["CUDA_started"])
            self.assertTrue(result["framework_free"])
            self.assertEqual(result["prompt_policy"]["prompt_ids"], ["short"])
            self.assertEqual(result["generation_policy"]["candidate_counts"], [2, 4])

    def test_main_writes_durable_input_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            candidate = directory / "candidate.jsonl"
            candidate.write_text(json.dumps(_row("short", 12)) + "\n", encoding="utf-8")
            selected = directory / "selected.json"
            selected.write_text(json.dumps({
                "schema_version": profile.SELECTED_IDS_SCHEMA_VERSION,
                "split": "train",
                "row_ids": ["short"],
            }, sort_keys=True) + "\n", encoding="utf-8")
            receipt = directory / "receipt.json"
            with mock.patch.object(rl, "_frameworks_imported", return_value=[]):
                code = rl.main([
                    "--preflight",
                    "--candidate-file", str(candidate), "--candidate-sha256", "0" * 64,
                    "--selected-train-ids", str(selected), "--selected-ids-sha256", _sha(selected),
                    "--receipt", str(receipt),
                ])
            self.assertEqual(code, 1)
            value = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "input_failed")
            self.assertEqual(value["error"]["type"], "ProfileInputError")


if __name__ == "__main__":
    unittest.main()
