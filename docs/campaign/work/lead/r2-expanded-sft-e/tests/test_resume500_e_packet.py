import hashlib
import json
import os
from pathlib import Path
import unittest

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
E = PLAN / "docs/campaign/work/lead/r2-expanded-sft-e"
D = PLAN / "docs/campaign/work/lead/r2-expanded-sft-d"
CHECKPOINT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-d/full/checkpoint-500")
SOURCE_ID = "bb2f9fdc1d016679533beab7d9906c9b996c9ce59d36e54ff98101adbaeed384"
SCHEDULE_SHA = "2ad2074d194b99bdef6826eb74543669fd8a7cf3490aedd692844b6402bed498"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Resume500PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((E / "recipe.json").read_text())
        cls.d_recipe = json.loads((D / "recipe.json").read_text())
        cls.manifest = json.loads((CHECKPOINT / "campaign-manifest.json").read_text())
        cls.state = json.loads((CHECKPOINT / "campaign-state.json").read_text())
        cls.trainer = json.loads((CHECKPOINT / "trainer_state.json").read_text())

    def test_actual_full500_optimizer_rng_sampler_checkpoint(self):
        self.assertTrue(self.manifest["full"])
        self.assertEqual(self.manifest["step"], 500)
        self.assertEqual(self.state["step"], self.trainer["global_step"])
        self.assertEqual(self.state["sampler"]["consumed_draws"], 8000)
        self.assertEqual(self.state["sampler"]["schedule_sha256"], SCHEDULE_SHA)
        self.assertEqual(self.manifest["identity"], self.state["identity"])
        self.assertEqual(self.manifest["identity"], self.recipe["identity"])
        required = {"adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "campaign-state.json"}
        self.assertLessEqual(required, set(self.manifest["files"]))
        for name, record in self.manifest["files"].items():
            path = CHECKPOINT / name
            self.assertEqual(path.stat().st_size, record["bytes"], name)
            self.assertEqual(sha(path), record["sha256"], name)

    def test_source_is_exact_d_snapshot_without_migration(self):
        source_manifest_path = D / "source-manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text())
        self.assertEqual(source_manifest["id"], SOURCE_ID)
        self.assertEqual(self.recipe["identity"]["source"], SOURCE_ID)
        self.assertEqual(self.manifest["identity"]["source"], SOURCE_ID)
        self.assertNotIn("resume_identity_compatibility", self.recipe)
        self.assertIn("ordinary exact-identity resume", self.recipe["recovery_binding"]["resume_semantics"])
        for record in source_manifest["files"]:
            path = D / "source" / record["path"]
            self.assertEqual(sha(path), record["sha256"], record["path"])
            self.assertEqual(os.stat(path).st_mode & 0o777, record["mode"], record["path"])

    def test_training_identity_data_and_full_schedule_are_unchanged(self):
        self.assertEqual(self.recipe["identity"], self.d_recipe["identity"])
        self.assertEqual(self.recipe["parameters"], self.d_recipe["parameters"])
        self.assertEqual(self.recipe["parameters"]["max_steps"], 1000)
        self.assertEqual(self.recipe["schedule_binding"], self.d_recipe["schedule_binding"])
        self.assertEqual(self.recipe["draw_schedule"], self.d_recipe["draw_schedule"])
        self.assertEqual(self.recipe["token_rows"], self.d_recipe["token_rows"])
        self.assertEqual(self.recipe["expanded_data"], self.d_recipe["expanded_data"])
        self.assertEqual(self.recipe["expanded_data"]["candidate_rows"], 11505)
        self.assertEqual(self.recipe["schedule_binding"]["draws"], 16000)

    def test_750_is_evaluation_only_and_terminal_is1000(self):
        self.assertEqual(self.recipe["checkpoint"]["evaluation_steps"], [250, 500, 750, 1000])
        self.assertEqual(self.recipe["mandatory_stop_steps"], [])
        self.assertEqual(self.recipe["decision_steps"], [])
        self.assertEqual(self.recipe["milestones"], [250, 500, 1000])
        self.assertIn("cannot declare750", self.recipe["recovery_binding"]["displaced_750_gate"])

    def test_paths_are_fresh_and_d_checkpoint_is_read_only_parent(self):
        self.assertEqual(Path(self.recipe["resume_from"]), CHECKPOINT)
        self.assertFalse(Path(self.recipe["output_dir"]).exists())
        self.assertFalse(Path(self.recipe["archive_dir"]).exists())
        self.assertFalse(Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-e-host-supervision").exists())
        runner_root = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-sft-e-v1")
        if runner_root.exists():
            snapshot = runner_root / "snapshots" / SOURCE_ID / "manifest.json"
            self.assertTrue(snapshot.is_file())
            self.assertEqual(json.loads(snapshot.read_text())["id"], SOURCE_ID)
        self.assertTrue(CHECKPOINT.is_dir())
        declared = {(row["path"], row["sha256"]) for row in self.recipe["inputs"]}
        self.assertIn((str(CHECKPOINT / "campaign-manifest.json"), sha(CHECKPOINT / "campaign-manifest.json")), declared)
        self.assertIn((str(D / "source-manifest.json"), sha(D / "source-manifest.json")), declared)

    def test_runner_uses_fresh_state_and_exact_d_source(self):
        runner = json.loads((E / "runner-recipe.json").read_text())
        commands = json.loads((E / "commands.json").read_text())
        self.assertEqual(runner["snapshot"], SOURCE_ID)
        self.assertIn(str(D / "source"), commands["snapshot"])
        joined = json.dumps([runner, commands], sort_keys=True)
        self.assertIn("runner-expanded-sft-e-v1", joined)
        self.assertIn("SFT11-expanded-postsft500-e", joined)
        self.assertNotIn("runner-expanded-sft-d-v1", joined)
        guard_argv = runner["steps"][1]["argv"]
        self.assertEqual(guard_argv[4:6], ["--output", "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-e-host-supervision"])
        self.assertNotIn("SFT11-expanded-postsft500-d-host-supervision", guard_argv)


if __name__ == "__main__":
    unittest.main()
