"""Card C13: the bound RFT and DPO recipes on arm A 536 and their night jobs.

The recipes pre-register the C04 optimizer and LR schedule with the card's LR
ratios and gates at halves, bind the starting model's own gate 536 as the
reference, and the data built from the C12 census. The 5 GB parent is replaced
by a stand-in holding its real campaign manifest; cohort, schedules, panels
and the gate-536 results are verified for real.
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from typing import cast

from sepalith.ops import night_queue
from sepalith.training.offline.dpo import PairDataset
from sepalith.training.offline import data as offline_data
from sepalith.training.offline import recipe as offline_recipe
from sepalith.training.offline import trainer as offline_trainer
from sepalith.training.sft import recipe_v2

REPO = Path(__file__).resolve().parents[4]
DEPLOY = Path("/home/m0hawk/sandbox/Sepalith")
PATHS = recipe_v2.PathMap(DEPLOY, REPO)
C13 = REPO / "docs/training/plan-20260927/c13"
RFT = C13 / "bound/rft-1-arm-a-536.json"
DPO = C13 / "bound/dpo-1-arm-a-536.json"
ARM_A = REPO / "docs/training/plan-20260927/c04/bound/sft-v2-lr1e-5.json"
DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work/C13-offline-v1/sepalith-2b-edit-sft-arm-a")
PARENT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536")
DECISION = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates/SFT-v2-r4-lr1e-5/decisions/decision-step-536-1.json")
HAVE_DATA = DATA.is_dir() and PARENT.is_dir() and DECISION.is_file()


def localized(path):
    return PATHS.localize(json.loads(Path(path).read_text()))


class BoundRecipeTests(unittest.TestCase):
    def setUp(self):
        self.rft, self.dpo, self.arm_a = localized(RFT), localized(DPO), localized(ARM_A)

    def test_pre_registration(self):
        rft, dpo = self.rft, self.dpo
        for value in (rft, dpo):
            offline_recipe.validate(value)
            offline_recipe.verify_source(value["source"])
            self.assertEqual(value["runtime"]["optimizer"] | {"hidden_lr": 0, "side_lr": 0},
                             self.arm_a["runtime"]["optimizer"] | {"hidden_lr": 0, "side_lr": 0})
            for key in ("scheduler", "lr_scheduler_kwargs", "warmup_steps", "weight_decay", "micro_batch", "gradient_accumulation",
                        "effective_batch"):
                self.assertEqual(value["runtime"][key], self.arm_a["runtime"][key], key)
            self.assertEqual(value["dev_gate"]["panels"], self.arm_a["dev_gate"]["panels"])
            self.assertEqual(value["dev_gate"]["generation_batching"], self.arm_a["dev_gate"]["generation_batching"])
            self.assertEqual(value["retention"]["policy"], "retain_all")
            self.assertEqual(value["parent"]["files"]["model.safetensors"],
                             "befe587a0d4969e33b0daaa8fe66247371ec32f331616db74724616878ad6030")  # = public arm-a
            self.assertEqual(value["parent_reference"]["metrics"],
                             {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 117, "noedit": 1, "edit_total": 106})
        self.assertEqual((rft["runtime"]["learning_rate"], rft["runtime"]["optimizer"]["side_lr"]), (3.33333e-06, 3.33333e-07))
        self.assertEqual((dpo["runtime"]["learning_rate"], dpo["runtime"]["optimizer"]["side_lr"]), (1e-06, 1e-07))
        self.assertEqual((rft["cohort"]["updates"], rft["dev_gate"]["mandatory_steps"]), (435, [218, 435]))
        self.assertEqual((dpo["cohort"]["updates"], dpo["dev_gate"]["mandatory_steps"]), (1142, [571, 1142]))
        self.assertEqual((rft["cohort"]["unique_rows"], dpo["cohort"]["unique_rows"]), (6304, 18268))
        self.assertEqual(dpo["pre_registered"]["beta"], 0.1)
        self.assertNotEqual(rft["outputs"]["archive"], self.arm_a["outputs"]["archive"])

    def test_validate_refuses_changes(self):
        for change in (lambda r: r["runtime"].update(learning_rate=1e-5), lambda r: r["runtime"].update(warmup_steps=16),
                       lambda r: r["dev_gate"].update(mandatory_steps=[109, 218, 326, 435]),
                       lambda r: r["runtime"]["optimizer"].update(rms_scale=0.3), lambda r: r.update(retention={"policy": "x"}),
                       lambda r: r["pre_registered"].update(sft_learning_rate=2e-5)):
            value = copy.deepcopy(self.rft)
            change(value)
            with self.assertRaises(ValueError):
                offline_recipe.validate(value)
        value = copy.deepcopy(self.dpo)
        value["pre_registered"]["beta"] = 0.05
        with self.assertRaisesRegex(ValueError, "DPO loss"):
            offline_recipe.validate(value)

    @unittest.skipUnless(HAVE_DATA, "campaign data is not mounted")
    def test_load_and_preflight_with_a_stand_in_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "SFT-v2-r4-lr1e-5" / "full" / "checkpoint-536"
            parent.mkdir(parents=True)
            for kind, value in (("rft", self.rft), ("dpo", self.dpo)):
                recipe = copy.deepcopy(value)
                for name in recipe["parent"]["files"]:
                    if name == "campaign-manifest.json":
                        shutil.copyfile(PARENT / name, parent / name)
                    elif not (parent / name).exists():
                        (parent / name).write_text(name)
                    recipe["parent"]["files"][name] = recipe_v2.sha256(parent / name)
                recipe["parent"]["path"] = str(parent)
                path = Path(tmp) / f"{kind}.json"
                path.write_text(json.dumps(recipe))
                loaded = offline_recipe.load_offline(path, require_reference=False)
                self.assertEqual(loaded["kind"], kind)
                if kind == "rft":
                    report = offline_trainer.rft_dataset(loaded)[1]
                    self.assertEqual((report["unique_rows"], report["draws"]), (6304, 6960))
                    identity = json.loads(json.dumps(offline_recipe.identity(loaded)))
                    self.assertEqual(identity["renderer"]["objective"], "exact_target_body_terminal_eos_only")
                else:
                    dataset = cast(PairDataset, offline_trainer.load_pairs(loaded, with_reference=False))
                    self.assertEqual(len(dataset), 1142 * 16)
                    first = json.loads(json.dumps(dataset[0]))
                    self.assertEqual(first["chosen_ids"][:first["target_start"]], first["rejected_ids"][:first["target_start"]])
                    # Before the reference job: missing; after it: bound to the real parent, not the stand-in.
                    with self.assertRaisesRegex(ValueError, "run the reference job first|reference manifest is for another model"):
                        offline_recipe.load_offline(path)
            # A changed gate reference is refused.
            broken = copy.deepcopy(json.loads((Path(tmp) / "rft.json").read_text()))
            broken["parent_reference"]["metrics"]["score250"] = 120
            (Path(tmp) / "broken.json").write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "gate reference differs"):
                offline_recipe.load_offline(Path(tmp) / "broken.json", require_reference=False)

    @unittest.skipUnless(HAVE_DATA, "campaign data is not mounted")
    def test_data_manifest_matches_the_user_decision(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        self.assertEqual(manifest["rft"]["selected"], 4413)
        self.assertEqual(manifest["rft"]["selected_by_family"],
                         {"finish_block": 1181, "format_propagation": 810, "na_rm_propagation": 68, "no_op": 548,
                          "pipe_rewrite": 994, "rename_propagation": 791, "roxygen_drafting": 21})
        self.assertEqual((manifest["cohort"]["train_mix_rows"], manifest["schedule"]["noop_share"]), (1891, 0.1875))
        self.assertLessEqual(max(int(k) for k in manifest["schedule"]["exposure_histogram_by_pool"]["noop"]), 3)
        self.assertEqual((manifest["dpo"]["pairs"], manifest["dpo"]["chosen_sources"]), (18268, {"gold": 15210, "model": 3058}))
        self.assertEqual(self.rft["cohort"]["manifest"]["sha256"], recipe_v2.sha256(DATA / "manifest.json"))


RFT2 = C13 / "bound/rft-2-arm-a-536-t12.json"
DPO2 = C13 / "bound/dpo-2-arm-a-536-t12.json"
DATA2 = Path("/mnt/e/sepalith/campaign-20260915/data-work/C13-offline-v1/sepalith-2b-edit-sft-arm-a-t12")
HAVE_DATA2 = DATA2.is_dir() and PARENT.is_dir() and DECISION.is_file()


class SecondIterationTests(unittest.TestCase):
    """Iteration 2 (R4 step 4, user decision 2026-10-22): only the census temperature changes (1.0 -> 1.2)."""

    def setUp(self):
        self.rft, self.dpo = localized(RFT2), localized(DPO2)
        self.rft1, self.dpo1 = localized(RFT), localized(DPO)

    def test_same_pre_registration_as_iteration_1(self):
        for new, old in ((self.rft, self.rft1), (self.dpo, self.dpo1)):
            offline_recipe.validate(new)
            offline_recipe.verify_source(new["source"])
            self.assertEqual(new["source"], old["source"])
            for key in ("pre_registered", "development", "dev_panels", "expected_model", "optimizer_dispatch", "seed", "retention",
                        "parent", "parent_reference", "parent_recipe"):
                expected = old[key]
                if key == "pre_registered":
                    expected = {**old[key], "gate_updates": new[key]["gate_updates"]}
                if key == "parent":  # the same checkpoint; the tag names the iteration
                    expected = {**old[key], "tag": "arm-a-536-t12"}
                self.assertEqual(new[key], expected, key)
            runtime = {k: v for k, v in new["runtime"].items() if k not in ("max_steps", "checkpoint_every", "evaluation_steps",
                                                                            "selected_milestones", "mandatory_dev_gate_steps",
                                                                            "gate_evidence_directory")}
            self.assertEqual(runtime, {k: old["runtime"][k] for k in runtime})
            updates = new["cohort"]["updates"]
            self.assertEqual(new["dev_gate"]["mandatory_steps"], [round(updates / 2), updates])
            self.assertTrue(new["arm"]["name"].endswith("-2-arm-a-536-t12"))
            self.assertNotEqual(new["outputs"]["archive"], old["outputs"]["archive"])
            self.assertNotEqual(new["cohort"]["rows"]["sha256"], old["cohort"]["rows"]["sha256"])
        self.assertEqual(self.rft["arm"]["name"], "C13-rft-2-arm-a-536-t12")
        self.assertEqual(self.dpo["arm"]["name"], "C13-dpo-2-arm-a-536-t12")
        self.assertNotEqual(self.dpo["reference"]["logps"], self.dpo1["reference"]["logps"])

    @unittest.skipUnless(HAVE_DATA2, "campaign data is not mounted")
    def test_data_follows_the_iteration_1_rules_on_the_t12_census(self):
        manifest = json.loads((DATA2 / "manifest.json").read_text())
        census = json.loads(Path(manifest["inputs"]["census_summary"]["path"]).read_text())
        self.assertEqual(census["sampling"]["temperature"], 1.2)
        self.assertEqual(manifest["model_id"], "sepalith-2b-edit-sft-arm-a-t12")
        self.assertEqual((manifest["split"], manifest["final_set_access"]), ("train", False))
        rft = manifest["rft"]
        pool, train = rft["distinct_pool_max2_per_prompt"], rft["train_rows_by_family"]
        self.assertEqual(rft["caps_by_family"], {f: min(rft["supply_by_family"].get(f, 0), train[f] * pool // 15006) for f in train})
        self.assertEqual(rft["selected"], sum(rft["caps_by_family"].values()))
        self.assertEqual(manifest["cohort"]["train_mix_rows"], offline_data.train_mix_count(rft["selected"]))
        self.assertEqual(manifest["schedule"]["noop_share"], 0.1875)
        self.assertLessEqual(max(int(k) for k in manifest["schedule"]["exposure_histogram_by_pool"]["noop"]), 3)
        self.assertEqual(manifest["dpo"]["pairs"], census["dpo"]["pairs"])
        self.assertEqual(self.rft["cohort"]["manifest"]["sha256"], recipe_v2.sha256(DATA2 / "manifest.json"))
        self.assertEqual(self.dpo["cohort"]["unique_rows"], manifest["dpo"]["pairs"])

    @unittest.skipUnless(HAVE_DATA2, "campaign data is not mounted")
    def test_load_and_preflight_with_a_stand_in_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "SFT-v2-r4-lr1e-5" / "full" / "checkpoint-536"
            parent.mkdir(parents=True)
            for kind, value in (("rft", self.rft), ("dpo", self.dpo)):
                recipe = copy.deepcopy(value)
                for name in recipe["parent"]["files"]:
                    if name == "campaign-manifest.json":
                        shutil.copyfile(PARENT / name, parent / name)
                    elif not (parent / name).exists():
                        (parent / name).write_text(name)
                    recipe["parent"]["files"][name] = recipe_v2.sha256(parent / name)
                recipe["parent"]["path"] = str(parent)
                path = Path(tmp) / f"{kind}.json"
                path.write_text(json.dumps(recipe))
                loaded = offline_recipe.load_offline(path, require_reference=False)
                cohort = cast(dict, loaded["cohort"])
                if kind == "rft":
                    report = offline_trainer.rft_dataset(loaded)[1]
                    self.assertEqual(report["unique_rows"], cohort["unique_rows"])
                else:
                    dataset = cast(PairDataset, offline_trainer.load_pairs(loaded, with_reference=False))
                    self.assertEqual(len(dataset), cohort["updates"] * 16)
                    with self.assertRaisesRegex(ValueError, "run the reference job first|reference manifest is for another model"):
                        offline_recipe.load_offline(path)


class JobTests(unittest.TestCase):
    def test_jobs_are_valid_and_point_at_the_bound_recipes(self):
        jobs = {path.stem: json.loads(path.read_text()) for path in sorted((C13 / "jobs").glob("*.json"))}
        self.assertEqual(sorted(jobs), ["C13-dpo-1-arm-a-536", "C13-dpo-1-arm-a-536-reference", "C13-dpo-2-arm-a-536-t12",
                                        "C13-dpo-2-arm-a-536-t12-reference", "C13-rft-1-arm-a-536", "C13-rft-2-arm-a-536-t12"])
        for name, job in jobs.items():
            self.assertEqual(job["id"], name)
            night_queue.validate_job(copy.deepcopy(job))
            command = job["command"]
            self.assertIn("sepalith.training.offline.night", command)
            self.assertEqual(command[command.index("sepalith.training.offline.night") + 1], "reference" if name.endswith("-reference") else "run")
            recipe = Path(command[command.index("--recipe") + 1])
            self.assertEqual(recipe, DEPLOY / "docs/training/plan-20260927/c13/bound" / f"{name.removesuffix('-reference').removeprefix('C13-')}.json")
            self.assertTrue(job["resumable"])
            self.assertEqual(job["cwd"], str(DEPLOY))
        self.assertEqual(jobs["C13-dpo-1-arm-a-536"]["requires"], ["C13-dpo-1-arm-a-536-reference"])
        self.assertEqual(jobs["C13-rft-1-arm-a-536"]["requires"], [])
        self.assertEqual(jobs["C13-dpo-2-arm-a-536-t12"]["requires"], ["C13-dpo-2-arm-a-536-t12-reference"])
        self.assertEqual(jobs["C13-rft-2-arm-a-536-t12"]["requires"], [])


if __name__ == "__main__":
    unittest.main()
