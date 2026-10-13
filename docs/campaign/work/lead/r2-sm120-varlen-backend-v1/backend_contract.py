#!/usr/bin/env python3
"""Static admission checks for the isolated sm120 varlen diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

WHEEL = Path("/mnt/e/sepalith/campaign-20260915/build-work/sm120-varlen-v1/cu130-wheel/xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl")
OVERLAY = Path("/mnt/e/sepalith/campaign-20260915/build-work/sm120-varlen-v1/cu130-overlay")
WHEEL_SHA256 = "962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d"
OVERLAY_C_SHA256 = "bb2a59af5ed03aa28ea0e1ed705384fcb6d5d0e6ab8bd825b004ffbb3af9d669"
CANONICAL = Path("/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages")
OVERLAY_MANIFEST = Path(__file__).with_name("overlay-manifest.json")
SOURCE_PINS = {
    "unsloth/utils/attention_dispatch.py": "9f0b160da7e3105586c252a9f6384218ce00736ecca2b0a72b741412934b5a31",
    "unsloth/models/llama.py": "8a79d94ab5f7f76c241d49da9bb1af9a90a9491977d3d08bbf356334dd007f2d",
    "unsloth/models/_utils.py": "a297eb31e25dea0ae06aa3b7d6c9108f67f469010969f196e69fc9acb84e3ef7",
    "unsloth_zoo/loss_utils.py": "c49e8623999d2410f2bccb5b15236aae73daabd91cd62eb6b7705d3f264f0c71",
    "transformers/trainer.py": "17e212935057f58b52efe79c3f952bf044bbada8afd4f13c8e905142b1d2ef0a",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate() -> dict:
    if sha256(WHEEL) != WHEEL_SHA256:
        raise ValueError("isolated official cu130 wheel bytes changed")
    extension = OVERLAY / "xformers/_C.so"
    if sha256(extension) != OVERLAY_C_SHA256:
        raise ValueError("isolated overlay extension bytes changed")
    overlay_manifest = json.loads(OVERLAY_MANIFEST.read_text())
    expected_files = {entry["path"]: entry for entry in overlay_manifest["files"]}
    actual_files = {path.relative_to(OVERLAY).as_posix(): path for path in OVERLAY.rglob("*") if path.is_file()}
    if set(actual_files) != set(expected_files):
        raise ValueError("isolated overlay file inventory changed")
    for relative, path in actual_files.items():
        expected = expected_files[relative]
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"isolated overlay file changed: {relative}")
    for relative, expected in SOURCE_PINS.items():
        if sha256(CANONICAL / relative) != expected:
            raise ValueError(f"canonical source changed: {relative}")
    trainer = (CANONICAL / "transformers/trainer.py").read_text()
    zoo = (CANONICAL / "unsloth_zoo/loss_utils.py").read_text()
    attention = (CANONICAL / "unsloth/utils/attention_dispatch.py").read_text()
    model_utils = (CANONICAL / "unsloth/models/_utils.py").read_text()
    required = {
        "trainer_prefetches_accumulation_window": "batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches" in trainer,
        "trainer_passes_common_denominator": "self.training_step(model, inputs, num_items_in_batch)" in trainer,
        "unsloth_counts_shifted_labels": 'labels[..., 1:] != -100' in zoo,
        "unsloth_masks_attention": 'attention_mask[..., 1:] != 0' in zoo,
        "unsloth_subtracts_packed_boundaries": "count -= torch.count_nonzero(seq_lengths > 0).item() - 1" in zoo,
        "sm120_runtime_probe": "_xformers_disabled_for_capability" in attention,
        "xformers_block_diagonal_route": "build_xformers_block_causal_mask" in attention,
        "unsloth_marks_compiled_model_loss_kwargs": '_shadow_accepts_loss_kwargs(model, True)' in model_utils,
    }
    if not all(required.values()):
        raise ValueError(f"source-defined runtime contract changed: {required}")
    return {
        "schema": "sepalith.sft11.sm120_varlen_static_audit.v1",
        "wheel": {"path": str(WHEEL), "sha256": WHEEL_SHA256},
        "overlay": {"path": str(OVERLAY), "extension_sha256": OVERLAY_C_SHA256,
                    "manifest_path": str(OVERLAY_MANIFEST), "files": len(expected_files)},
        "source_pins": SOURCE_PINS,
        "checks": required,
        "loss_denominator": {
            "scope": "one Transformers gradient-accumulation window",
            "formula": "sum over physical microbatches of labels[...,1:] != -100 and attention_mask[...,1:] != 0; for packed_seq_lengths the installed patch additionally subtracts member_count-1",
            "effect": "the same aggregate num_items_in_batch is passed to every forward; accumulated gradients implement a global supervised-token mean when the model accepts loss kwargs",
            "production_guard": "record model_accepts_loss_kwargs=true and the exact denominator at the first optimizer update before admitting packed training",
        },
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
