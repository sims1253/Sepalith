"""Independent regression replay against the root's frozen v2 helper."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
SOURCE = PLAN / "docs/campaign/work/lead/r2-full-weight-checkpoint-v2"
EXPECTED = {
    "model_layout.py": "e67663c513e297a3aa2fd32a864b7bb522a08a27651cfe095bced113635c4e25",
    "campaign_checkpoint.py": "c7ca01f7f6ba6f4a8bbb23dd1cf6c29432d6f04853b6967f614df78190921fd5",
    "test_model_layout.py": "22b7ea75bba6b39a2f39d3d30543f0785c93f74e267da0043adea16c5d4a1dd6",
}
for name, expected in EXPECTED.items():
    observed = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(f"Frozen v2 source hash differs for {name}: {observed}")
sys.path.insert(0, str(SOURCE))

from campaign_checkpoint import FULL_STATE_FILES, seal_checkpoint, verify_checkpoint
from model_layout import validate_dense_weights


def identity():
    return {
        name: {"id": f"review-{name}"}
        for name in ("parent", "tokenizer", "renderer", "data", "source", "policy", "schedule")
    }


class V2CheckpointContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "config.json").write_text("{}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def dense(self):
        save_file({"model.weight": np.ones((2, 3), dtype=np.float32)},
                  str(self.root / "model.safetensors"))

    def state(self, step=7):
        for name in FULL_STATE_FILES:
            content = json.dumps({"global_step": step}) if name == "trainer_state.json" else "fixture"
            (self.root / name).write_text(content)

    def test_adapter_full_is_rejected_for_full_weight_resume(self):
        save_file({"layer.lora_A.default.weight": np.ones((1, 1), dtype=np.float32)},
                  str(self.root / "adapter_model.safetensors"))
        (self.root / "adapter_config.json").write_text("{}\n")
        self.state()
        seal_checkpoint(self.root, identity(), 7, full=True, sampler={"cursor": 7}, checkpoint_kind="adapter")
        with self.assertRaisesRegex(ValueError, "kind differs"):
            verify_checkpoint(
                self.root, identity(), require_full=True,
                expected_checkpoint_kind="full_weights",
            )

    def test_manifest_kind_tamper_is_rejected_by_state_binding(self):
        self.dense()
        self.state()
        seal_checkpoint(self.root, identity(), 7, full=True, sampler={"cursor": 7}, checkpoint_kind="full_weights")
        path = self.root / "campaign-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["checkpoint_kind"] = "adapter"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "kind differs"):
            verify_checkpoint(self.root, identity(), require_full=True)

    def test_dense_header_is_revalidated_at_resume(self):
        self.dense()
        self.state()
        seal_checkpoint(self.root, identity(), 7, full=True, sampler={"cursor": 7}, checkpoint_kind="full_weights")
        # Replace bytes and update the self-contained manifest inventory to isolate
        # the semantic header validation from the ordinary digest guard.
        (self.root / "model.safetensors").write_bytes(b"malformed")
        import campaign_checkpoint as checkpoint
        path = self.root / "campaign-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"] = checkpoint.inventory(self.root)
        path.write_text(json.dumps(manifest))
        with self.assertRaises(Exception):
            verify_checkpoint(
                self.root, identity(), require_full=True,
                expected_checkpoint_kind="full_weights",
            )

    def test_missing_optimizer_is_rejected_at_seal(self):
        self.dense()
        self.state()
        (self.root / "optimizer.pt").unlink()
        with self.assertRaisesRegex(ValueError, "optimizer.pt"):
            seal_checkpoint(
                self.root, identity(), 7, full=True, sampler={"cursor": 7},
                checkpoint_kind="full_weights",
            )

    def test_malformed_single_and_indexed_headers_rejected(self):
        (self.root / "model.safetensors").write_bytes(b"malformed")
        with self.assertRaises(Exception):
            validate_dense_weights(self.root)
        (self.root / "model.safetensors").unlink()
        (self.root / "part.safetensors").write_bytes(b"malformed")
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.weight": "part.safetensors"}})
        )
        with self.assertRaises(Exception):
            validate_dense_weights(self.root)

    def test_manifest_step_and_full_tampering_are_rejected(self):
        self.dense()
        self.state()
        seal_checkpoint(self.root, identity(), 7, full=True, sampler={"cursor": 7}, checkpoint_kind="full_weights")
        path = self.root / "campaign-manifest.json"
        original = json.loads(path.read_text())
        for key, value, message in (
            ("step", 8, "state and manifest disagree"),
            ("full", False, "not a full resume checkpoint"),
        ):
            tampered = dict(original)
            tampered[key] = value
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, message):
                verify_checkpoint(
                    self.root, identity(), require_full=True,
                    expected_checkpoint_kind="full_weights",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
