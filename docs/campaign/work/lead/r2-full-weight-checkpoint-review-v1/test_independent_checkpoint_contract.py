"""Independent CPU review of the frozen full-weight checkpoint helper."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
SOURCE = PLAN / "docs/campaign/work/lead/r2-full-weight-checkpoint-v1"
EXPECTED = {
    "model_layout.py": "e67663c513e297a3aa2fd32a864b7bb522a08a27651cfe095bced113635c4e25",
    "campaign_checkpoint.py": "89e1c02f66815eb3f362a3bcc820d99f453d56315caba7a89776f19e48e7628f",
}
sys.path.insert(0, str(SOURCE))

from campaign_checkpoint import FULL_STATE_FILES, seal_checkpoint, verify_checkpoint
from model_layout import validate_dense_weights


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    return {
        name: {"id": f"review-{name}"}
        for name in ("parent", "tokenizer", "renderer", "data", "source", "policy", "schedule")
    }


class IndependentCheckpointReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, expected in EXPECTED.items():
            observed = sha256(SOURCE / name)
            if observed != expected:
                raise AssertionError(f"Frozen source hash differs for {name}: {observed}")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "config.json").write_text("{}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def dense(self, filename="model.safetensors", key="model.weight"):
        save_file({key: np.ones((2, 3), dtype=np.float32)}, str(self.root / filename))

    def full_state(self, step=7):
        for name in FULL_STATE_FILES:
            content = json.dumps({"global_step": step}) if name == "trainer_state.json" else "fixture"
            (self.root / name).write_text(content)

    def test_malformed_index_json_is_rejected(self):
        (self.root / "model.safetensors.index.json").write_text('{"weight_map":')
        with self.assertRaises(json.JSONDecodeError):
            validate_dense_weights(self.root)

    def test_malformed_indexed_shard_header_is_rejected(self):
        (self.root / "part.safetensors").write_bytes(b"not-a-safetensors-header")
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.weight": "part.safetensors"}})
        )
        with self.assertRaises(Exception):
            validate_dense_weights(self.root)

    def test_malformed_single_header_is_rejected(self):
        (self.root / "model.safetensors").write_bytes(b"not-a-safetensors-header")
        with self.assertRaises(Exception):
            validate_dense_weights(self.root)

    def test_missing_optimizer_rejects_full_dense_seal(self):
        self.dense()
        for name in FULL_STATE_FILES:
            if name != "optimizer.pt":
                content = json.dumps({"global_step": 7}) if name == "trainer_state.json" else "fixture"
                (self.root / name).write_text(content)
        with self.assertRaisesRegex(ValueError, "optimizer.pt"):
            seal_checkpoint(
                self.root, identity(), 7, full=True, sampler={"cursor": 7},
                checkpoint_kind="full_weights",
            )

    def test_full_adapter_is_currently_accepted_by_require_full(self):
        """Documents the blocking missing expected-kind binding."""
        save_file({"layer.lora_A.default.weight": np.ones((1, 1), dtype=np.float32)},
                  str(self.root / "adapter_model.safetensors"))
        (self.root / "adapter_config.json").write_text("{}\n")
        self.full_state()
        seal_checkpoint(
            self.root, identity(), 7, full=True, sampler={"cursor": 7},
            checkpoint_kind="adapter",
        )
        manifest = verify_checkpoint(self.root, identity(), require_full=True)
        self.assertEqual(manifest["checkpoint_kind"], "adapter")

    def test_manifest_kind_is_not_bound_to_sealed_campaign_state(self):
        """The excluded manifest can change kind while byte inventory remains valid."""
        self.dense()
        self.full_state()
        seal_checkpoint(
            self.root, identity(), 7, full=True, sampler={"cursor": 7},
            checkpoint_kind="full_weights",
        )
        manifest_path = self.root / "campaign-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["checkpoint_kind"] = "adapter"
        manifest_path.write_text(json.dumps(manifest))
        observed = verify_checkpoint(self.root, identity(), require_full=True)
        self.assertEqual(observed["checkpoint_kind"], "adapter")
        state = json.loads((self.root / "campaign-state.json").read_text())
        self.assertEqual(state["checkpoint_kind"], "full_weights")

    def test_duplicate_weight_map_key_is_currently_collapsed(self):
        """Python's default JSON decoder accepts a duplicate tensor key."""
        self.dense("part.safetensors", "model.weight")
        (self.root / "model.safetensors.index.json").write_text(
            '{"weight_map":{"model.weight":"unused.safetensors",'
            '"model.weight":"part.safetensors"}}'
        )
        observed = validate_dense_weights(self.root)
        self.assertEqual(observed["tensor_count"], 1)

    def test_valid_single_and_indexed_layouts_still_pass(self):
        self.dense()
        self.assertEqual(validate_dense_weights(self.root)["kind"], "full_weights")
        (self.root / "model.safetensors").unlink()
        self.dense("part.safetensors", "a")
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "part.safetensors"}})
        )
        self.assertEqual(validate_dense_weights(self.root)["tensor_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
