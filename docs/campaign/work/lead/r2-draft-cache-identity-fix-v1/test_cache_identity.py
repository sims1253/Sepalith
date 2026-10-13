#!/usr/bin/env python3
"""CPU-only producer/DeepSpec cache identity contract test.

This creates a tiny synthetic cache with the pinned upstream writer.  It does
not load a model, read TRAIN rows, contact a provider, or run CUDA.  The fixed
target runtime helper supplies the same absolute model-directory identity that
``run_target_runtime`` now writes into its manifest.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile


PACKET = Path(__file__).resolve().parent
REFERENCE = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/"
    "docs/campaign/work/lead/r2-draft-provider-payload-v6/payload/staged/source/"
    "docs/campaign/work/r2-draft-target-runtime-v3"
)
DEEPSPEC = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/"
    "docs/campaign/work/lead/r2-draft-provider-payload-v6/payload/staged/source/"
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec"
)


def load_fixed_runtime():
    os.environ["SEPALITH_DEEPSPEC_ROOT"] = str(DEEPSPEC)
    os.environ["SEPALITH_HARDENING_ROOT"] = str(REFERENCE)
    spec = importlib.util.spec_from_file_location(
        "r2_draft_cache_identity_fix_target_runtime", PACKET / "target_runtime.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load fixed target runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def produce_synthetic_cache(module, output: Path, model_dir: Path) -> None:
    import torch

    layer_ids = (1, 10, 20, 30, 39)
    hidden_size = 4
    sequence_length = 4
    module.prepare_target_cache_output_dir(str(output))
    rank_dir = output / "_tmp" / "rank_0"
    rank_dir.mkdir(parents=True)
    writer = module.LocalTargetCacheWriter(
        rank_dir=str(rank_dir), max_shard_bytes=1 << 20
    )
    writer.write_sample(
        sample_id=0,
        input_ids=torch.tensor([0, 7, 8, 1], dtype=torch.int32),
        attention_mask=torch.ones(sequence_length, dtype=torch.uint8),
        loss_mask=torch.tensor([0, 0, 1, 1], dtype=torch.uint8),
        target_hidden_states=torch.zeros(
            sequence_length, len(layer_ids) * hidden_size, dtype=torch.bfloat16
        ),
        target_last_hidden_states=torch.zeros(
            sequence_length, hidden_size, dtype=torch.bfloat16
        ),
    )
    writer.close()
    summary = module.LocalCacheWriteSummary(
        global_rank=0,
        source_sample_start=0,
        source_sample_end=1,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    module.atomic_json_dump(summary.to_json(), rank_dir / "summary.json")
    summaries = [summary.to_json()]
    shard_map, shards = module.build_global_target_cache_shard_map(summaries)
    module.rename_local_target_cache_shards(
        output_dir=str(output),
        rank_dir=str(rank_dir),
        summary=summary.to_json(),
        shard_map=shard_map,
    )
    assert module.finalize_target_cache_index(
        output_dir=str(output), summaries=summaries, shard_map=shard_map
    ) == 1
    identity = {"model_dir": str(model_dir.resolve()), "weights_sha256": "synthetic"}
    manifest = module.build_target_cache_manifest(
        num_samples=1,
        shards=shards,
        target_layer_ids=layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            # This is the exact field used by run_target_runtime after the fix.
            "target_model_name_or_path": module._target_cache_model_name(identity),
            "target_identity": identity,
        },
    )
    module.write_target_cache_manifest(output_dir=str(output), manifest=manifest)
    module.cleanup_target_cache_tmp_dir(str(output))


def main() -> None:
    import torch

    torch.set_num_threads(2)
    module = load_fixed_runtime()
    from deepspec.data.target_cache_dataset import validate_train_cache
    with tempfile.TemporaryDirectory(prefix="r2-cache-identity-", dir="/tmp") as raw:
        root = Path(raw)
        relocated_model = root / "relocated" / "target-model"
        relocated_model.mkdir(parents=True)
        cache = root / "cache"
        produce_synthetic_cache(module, cache, relocated_model)
        dataset = module.CacheDataset(str(cache))
        try:
            expected_name = str(relocated_model.resolve())
            assert dataset.manifest["target_model_name_or_path"] == expected_name
            assert dataset.manifest["target_model_name_or_path"] != "bound-target"
            draft = SimpleNamespace(
                target_layer_ids=[1, 10, 20, 30, 39],
                config=SimpleNamespace(hidden_size=4),
            )
            validate_train_cache(
                train_dataset=dataset,
                draft_model=draft,
                target_model_name_or_path=expected_name,
            )
            try:
                validate_train_cache(
                    train_dataset=dataset,
                    draft_model=draft,
                    target_model_name_or_path=str(root / "other-target"),
                )
            except AssertionError as exc:
                mismatch_message = str(exc)
            else:
                raise AssertionError("official validator accepted a mismatched relocated path")
            try:
                module._target_cache_model_name({"name": "bound-target"})
            except ValueError as exc:
                missing_identity_message = str(exc)
            else:
                raise AssertionError("missing model_dir incorrectly fell back to a label")
            item = dataset[0]
            assert tuple(item["input_ids"].shape) == (4,)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "cpu_only": True,
                        "model_loaded": False,
                        "train_rows_read": False,
                        "provider_or_gpu_used": False,
                        "producer_target_model_name_or_path": expected_name,
                        "official_validate_train_cache": "PASS",
                        "mismatch_negative": "PASS",
                        "missing_model_dir_negative": "PASS",
                        "official_cache_dataset_roundtrip": "PASS",
                        "mismatch_message": mismatch_message,
                        "missing_identity_message": missing_identity_message,
                    },
                    sort_keys=True,
                )
            )
        finally:
            dataset.close()


if __name__ == "__main__":
    main()
