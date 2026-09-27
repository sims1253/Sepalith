"""CPU metadata tests for the expanded step-250 merge/native selection packet."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import unittest

H = Path(__file__).resolve().parent
P = H / "packet"
sys.path[:0] = [str(P), str(P / "native_evaluator")]
from bind_native_profile import build  # noqa: E402
from selection_contract import (PARENT, SOURCE, TOK, TOKCFG, check_task_recipe,
                                export_commands)  # noqa: E402


class ExpandedSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((H.parent / "r2-expanded-sft-b/recipe.json").read_text())

    def test_actual_expanded_recipe_is_supported(self):
        parent = check_task_recipe(self.recipe)
        self.assertEqual(parent["weights_sha256"], PARENT)
        self.assertEqual(self.recipe["identity"]["source"], SOURCE)

    def test_old_stage_parent_source_and_resume_reject(self):
        mutations = [
            ("stage", "task_sft_prm03_v1"),
            ("source", "1" * 64),
            ("kind", "cpt_merged"),
            ("weights", "2" * 64),
            ("resume", "/synthetic/old-checkpoint"),
        ]
        for name, value in mutations:
            candidate = copy.deepcopy(self.recipe)
            if name == "stage": candidate["stage"] = value
            elif name == "source": candidate["identity"]["source"] = value
            elif name == "kind": candidate["identity"]["parent"]["kind"] = value
            elif name == "weights": candidate["identity"]["parent"]["weights_sha256"] = value
            else: candidate["resume_from"] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_task_recipe(candidate)

    def test_merge_requires_exact_full250_state_and_sampler(self):
        tree = ast.parse((P / "merge_expanded_sft_cpu.py").read_text())
        text = ast.unparse(tree)
        for required in (
            "checkpoint['step'] == 250",
            "checkpoint['full'] is True",
            "campaign_state['full'] is True",
            "trainer_state['global_step']",
            "campaign_state['sampler']['consumed_draws']",
            "campaign_state['sampler']['schedule_sha256']",
            "campaign_state['sampler']['split_id']",
            "verify_checkpoint(args.checkpoint, recipe['identity'], require_full=True)",
        ):
            self.assertIn(required, text)

    def test_merge_restores_original_tokenizer_and_fp32_accumulation_order(self):
        tree = ast.parse((P / "merge_expanded_sft_cpu.py").read_text())
        calls = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "model"
                    and node.func.attr in ("float", "merge_and_unload", "to")):
                calls[node.func.attr] = node.lineno
        self.assertLess(calls["float"], calls["merge_and_unload"])
        self.assertLess(calls["merge_and_unload"], calls["to"])
        text = (P / "merge_expanded_sft_cpu.py").read_text()
        self.assertIn("shutil.copyfile(base / name, temporary / name)", text)
        self.assertIn("'accumulation_dtype': 'float32'", text)
        self.assertIn("'rounding': 'one final BF16 cast'", text)

    def test_export_commands_are_fresh_f16_then_q8_on_e(self):
        commands = export_commands(
            "/mnt/e/sepalith/campaign-20260915/models/SFT11-expanded-postsft500-b-250-merged",
            "/mnt/e/sepalith/campaign-20260915/models/SFT11-expanded-postsft500-b-250-quant",
        )
        self.assertEqual(commands[0][-2:], ["--outtype", "f16"])
        self.assertEqual(commands[1][-2:], ["Q8_0", "2"])
        self.assertIn("model-Q8_0.gguf", [Path(value).name for value in commands[1]])

    def test_expanded_profile_binds_same_native_dev_contract(self):
        identity = self.recipe["identity"]
        checkpoint = {"identity": identity, "full": True, "step": 250}
        parent = {
            "schema_version": "sepalith.r2-task-sft.parent-manifest.v1",
            "kind": "merged_task_sft", "tokenizer_original_bytes_restored": True,
            "sft_identity": identity, "sft_checkpoint_manifest": checkpoint,
            "source_cursor": 4000, "recipe_sha256": "3" * 64,
            "checkpoint_manifest_sha256": "4" * 64,
            "merged_model_path": "/mnt/e/synthetic-expanded-merged",
            "merge_verification": {"accumulation_dtype": "float32",
                "rounding": "one final BF16 cast", "independent_lora_matrix_merges_exact": 294},
        }
        integrity = {"status": "Q8_integrity_verified_pending_native_quality",
            "parent_manifest_sha256": "5" * 64,
            "q8": {"path": "/mnt/e/synthetic-expanded-quant/model-Q8_0.gguf",
                   "bytes": 2679710496, "sha256": "6" * 64}}
        profile = build(parent, integrity, "5" * 64, "7" * 64)
        self.assertEqual((profile["backend"], profile["build"], profile["context"],
                          profile["batch"], profile["ubatch"], profile["parallel"],
                          profile["output"], profile["cuda_graph_opt"]),
                         ("cuda", "b10453", 4096, 256, 256, 1, 192, 0))
        self.assertEqual(profile["selection"]["task_source"], SOURCE)
        self.assertEqual(profile["selection"]["parent_kind"], "sft_merged")

    def test_reused_harness_changes_only_expanded_selection_binding(self):
        old = H.parent / "r2-task-global500-native-selection/packet"
        for relative in (
            "native_controller/root_controller.py", "native_controller/guarded_dev_client.py",
            "native_controller/client_source_policy.py", "native_controller/origin_snapshot.py",
            "native_controller/observe_client.py", "native_controller/build_closure.py",
            "native_controller/native-mapped-code-allowlist.json",
            "native_controller/elf-loader-metadata.json",
            "native_evaluator/native_identity.py", "native_evaluator/native_transport.py",
            "native_evaluator/native_evaluator.py", "native_evaluator/rehearse_dev.py",
            "native_evaluator/final_binding.py", "native_evaluator/final_row_gate.py",
        ):
            self.assertEqual((old / relative).read_bytes(), (P / relative).read_bytes(), relative)

    def test_no_model_framework_imported(self):
        self.assertFalse(set(sys.modules) & {"torch", "transformers", "peft", "gguf"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
