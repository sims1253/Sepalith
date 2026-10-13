"""CPU-only tests for the narrow expanded-SFT step-150 recovery seam."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve()
TRAINING = HERE.parents[1] / "source" / "experiments" / "training"
sys.path.insert(0, str(TRAINING))

import campaign_checkpoint  # noqa: E402
import campaign_expanded_sft as expanded  # noqa: E402
import campaign_sft  # noqa: E402
from test_campaign_expanded_sft import recipe as base_recipe, _rows  # noqa: E402


OLD_SOURCE = "1" * 64
NEW_SOURCE = "2" * 64


def write_resume(root: Path, identity: dict, *, step: int = 150, full: bool = True,
                 consumed_draws: int = 2400) -> Path:
    resume = root / "resume"
    resume.mkdir()
    state = {
        "step": step,
        "full": full,
        "identity": identity,
        "sampler": {
            "split_id": identity["data"]["split_id"],
            "schedule_sha256": identity["data"]["draw_schedule_sha256"],
            "consumed_draws": consumed_draws,
        },
    }
    (resume / "campaign-state.json").write_text(json.dumps(state))
    (resume / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    manifest = {"schema_version": 1, "step": step, "full": full, "identity": identity, "files": {}}
    (resume / "campaign-manifest.json").write_text(json.dumps(manifest))
    return resume


def bind(candidate: dict, resume: Path) -> None:
    candidate["resume_from"] = str(resume)
    candidate["identity"]["source"] = NEW_SOURCE
    candidate["target_gate_start_steps"] = [0, 150, 250, 500]
    candidate["resume_milestone"] = 150
    candidate["checkpoint"]["full_every"] = 50
    candidate["mandatory_stop_steps"] = [250]
    candidate["resume_identity_compatibility"] = {
        "schema": "sepalith.sft.resume-source-compatibility.v1",
        "mode": "exact_source_only",
        "reason": "external_interruption",
        "checkpoint_step": 150,
        "checkpoint_manifest_sha256": hashlib.sha256((resume / "campaign-manifest.json").read_bytes()).hexdigest(),
        "predecessor_source": OLD_SOURCE,
        "current_source": NEW_SOURCE,
    }


def bind_same_source(candidate: dict, resume: Path, step: int) -> None:
    candidate["resume_from"] = str(resume)
    candidate["identity"]["source"] = OLD_SOURCE
    candidate["target_gate_start_steps"] = [0, step, 250, 500]
    candidate["resume_milestone"] = step
    candidate["mandatory_stop_steps"] = [250]
    candidate["checkpoint"]["full_every"] = 50
    candidate["resume_identity_compatibility"] = {
        "schema": "sepalith.sft.resume-source-compatibility.v1",
        "mode": "exact_same_source",
        "reason": "external_interruption",
        "checkpoint_step": step,
        "checkpoint_manifest_sha256": hashlib.sha256((resume / "campaign-manifest.json").read_bytes()).hexdigest(),
        "predecessor_source": OLD_SOURCE,
        "current_source": OLD_SOURCE,
    }


class Resume150RecoveryTests(unittest.TestCase):
    def test_exact_source_only_migration_accepts_full_150_identity(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            root = Path(temp)
            candidate = base_recipe(root)
            predecessor = copy.deepcopy(candidate["identity"])
            predecessor["source"] = OLD_SOURCE
            resume = write_resume(root, predecessor)
            bind(candidate, resume)
            self.assertEqual(campaign_sft.resume_checkpoint_identity(candidate, candidate["identity"]), predecessor)
            self.assertTrue(expanded.expanded_target_only_policy(candidate))

    def test_non_source_identity_change_rejects(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            root = Path(temp)
            candidate = base_recipe(root)
            predecessor = copy.deepcopy(candidate["identity"])
            predecessor["source"] = OLD_SOURCE
            predecessor["schedule"]["learning_rate"] = 1e-5
            resume = write_resume(root, predecessor)
            bind(candidate, resume)
            with self.assertRaisesRegex(ValueError, "only the source identity"):
                campaign_sft.resume_checkpoint_identity(candidate, candidate["identity"])

    def test_manifest_hash_and_cursor_tampering_reject(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            root = Path(temp)
            candidate = base_recipe(root)
            predecessor = copy.deepcopy(candidate["identity"])
            predecessor["source"] = OLD_SOURCE
            resume = write_resume(root, predecessor, consumed_draws=2399)
            bind(candidate, resume)
            with self.assertRaisesRegex(ValueError, "cursor"):
                expanded.validate_expanded_admission(candidate, _rows())
            candidate["resume_identity_compatibility"]["checkpoint_manifest_sha256"] = "3" * 64
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                campaign_sft.resume_checkpoint_identity(candidate, candidate["identity"])

    def test_adapter_only_checkpoint_rejects_full_resume(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            root = Path(temp)
            identity = base_recipe(root)["identity"]
            resume = write_resume(root, identity, full=False)
            with self.assertRaisesRegex(ValueError, "not a full resume checkpoint"):
                campaign_checkpoint.verify_checkpoint(resume, identity, require_full=True)

    def test_same_source_full_200_interruption_is_admitted(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            root = Path(temp)
            candidate = base_recipe(root)
            predecessor = copy.deepcopy(candidate["identity"])
            predecessor["source"] = OLD_SOURCE
            resume = write_resume(root, predecessor, step=200, consumed_draws=3200)
            bind_same_source(candidate, resume, 200)
            self.assertEqual(campaign_sft.resume_checkpoint_identity(candidate, candidate["identity"]), predecessor)
            self.assertTrue(expanded.expanded_target_only_policy(candidate))

    def test_non_cadence_175_interruption_rejects(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as temp:
            root = Path(temp)
            candidate = base_recipe(root)
            predecessor = copy.deepcopy(candidate["identity"])
            predecessor["source"] = OLD_SOURCE
            resume = write_resume(root, predecessor, step=175, consumed_draws=2800)
            bind_same_source(candidate, resume, 175)
            with self.assertRaisesRegex(ValueError, "full-cadence interruption"):
                expanded.expanded_target_only_policy(candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
