import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("contract", HERE / "full_weight_rl_update_contract.py")
contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)


def digest(char): return char * 64


def binding():
    value = {
        "schema": "sepalith.rl11.full-weight-update-binding.v1",
        "optimizer_updates": True, "objective": "trl-0.24-grpo-bnpo-group-beta0",
        "candidate_count": 4, "policy_step": 7, "optimizer_global_step": 7,
        "rollout_cursor": 7, "rollout_complete": True,
        "same_process_live_policy_logprobs": True,
        "full_weight_checkpoint_kind": "full_weights",
        "sampler_boundary": "one_generation_buffer_per_optimizer_update",
    }
    for i, key in enumerate(("source_manifest_sha256", "model_manifest_sha256", "model_weights_sha256",
                              "tokenizer_json_sha256", "reward_buffer_manifest_sha256",
                              "prompt_context_manifest_sha256", "generation_policy_sha256",
                              "optimizer_dispatch_sha256"), 1): value[key] = digest(hex(i)[2:])
    return value


def group(index=3, row="r1", same=False):
    prompt = [0, 44, 45]
    generations=[]; rewards=[]
    for candidate in range(4):
        ids = [100 + candidate, 1]
        h = contract.ids_sha256(ids)
        generations.append({"group_index": index, "row_id": row, "candidate_index": candidate,
                            "prompt_ids_sha256": contract.ids_sha256(prompt), "generated_ids": ids,
                            "generated_ids_sha256": h, "cap_hit": False})
        rewards.append({"group_index": index, "row_id": row, "candidate_index": candidate,
                        "output_ids_sha256": h, "reward": 1.0 if same else float(candidate),
                        "parser_infrastructure_failure": False})
    return {"group_index": index, "row_id": row, "generations": generations, "rewards": rewards}


class ContractTests(unittest.TestCase):
    def test_valid_batch_has_exact_masks_and_cursor(self):
        value = contract.prepare_update(binding=binding(), groups=[group()],
            prompt_ids_by_row={"r1": [0,44,45]}, expected_first_group=3)
        self.assertEqual((value["completion_rows"], value["next_group"]), (4,4))
        self.assertTrue(all(row["completion_loss_mask"] == [1,1] for row in value["rows"]))
        self.assertEqual(len(value["batch_sha256"]), 64)

    def test_prompt_mismatch_rejected(self):
        bad=group(); bad["generations"][1]["prompt_ids_sha256"] = digest("f")
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            contract.prepare_update(binding=binding(), groups=[bad], prompt_ids_by_row={"r1":[0,44,45]}, expected_first_group=3)

    def test_reward_ids_mismatch_rejected(self):
        bad=group(); bad["rewards"][2]["output_ids_sha256"] = digest("f")
        with self.assertRaisesRegex(ValueError, "reward IDs"):
            contract.prepare_update(binding=binding(), groups=[bad], prompt_ids_by_row={"r1":[0,44,45]}, expected_first_group=3)

    def test_partial_or_gapped_resume_rejected(self):
        bad=group(4); bad["generations"].pop()
        with self.assertRaisesRegex(ValueError, "cursor|incomplete"):
            contract.prepare_update(binding=binding(), groups=[bad], prompt_ids_by_row={"r1":[0,44,45]}, expected_first_group=3)

    def test_no_signal_rejected_for_optimizer_update(self):
        with self.assertRaisesRegex(ValueError, "no within-group reward signal"):
            contract.prepare_update(binding=binding(), groups=[group(same=True)], prompt_ids_by_row={"r1":[0,44,45]}, expected_first_group=3)

    def test_cap_and_terminal_geometry(self):
        valid=group(); valid["generations"][0]["generated_ids"]=[77,78]
        valid["generations"][0]["generated_ids_sha256"]=contract.ids_sha256([77,78]); valid["generations"][0]["cap_hit"]=True
        valid["rewards"][0]["output_ids_sha256"]=valid["generations"][0]["generated_ids_sha256"]
        contract.prepare_update(binding=binding(), groups=[valid], prompt_ids_by_row={"r1":[0,44,45]}, expected_first_group=3)
        invalid=copy.deepcopy(valid); invalid["generations"][0]["generated_ids"]=[77,1,78]
        invalid["generations"][0]["generated_ids_sha256"]=contract.ids_sha256([77,1,78]); invalid["rewards"][0]["output_ids_sha256"]=invalid["generations"][0]["generated_ids_sha256"]
        with self.assertRaisesRegex(ValueError, "terminal EOG"):
            contract.prepare_update(binding=binding(), groups=[invalid], prompt_ids_by_row={"r1":[0,44,45]}, expected_first_group=3)

    def test_diagnostic_or_stale_policy_rejected(self):
        bad=binding(); bad["optimizer_updates"]=False
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            contract.validate_update_binding(bad)
        bad=binding(); bad["policy_step"]=6
        with self.assertRaisesRegex(ValueError, "policy and optimizer"):
            contract.validate_update_binding(bad)

if __name__ == "__main__": unittest.main()
