"""Actual pinned DeepSpec DSpark CPU smoke; no proxy model classes.

The model and loss below are imported from the sparse checkout at the pinned
DeepSpec commit.  A tiny dense-mask fallback is used only for eager CPU
attention because the upstream helper creates a Flex BlockMask that eager
attention cannot add.  The normal Flex path is also probed so its exact CPU
backward blocker is recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "deepspec"
sys.path.insert(0, str(VENDOR))

DEEPSPEC_REVISION = "005e03b81cec38b7da6399833d609ee89a2587f2"
SOURCE_FILES = (
    "deepspec/modeling/dspark/qwen3/modeling.py",
    "deepspec/modeling/dspark/qwen3/config.py",
    "deepspec/modeling/dspark/common.py",
    "deepspec/modeling/dspark/loss.py",
    "deepspec/modeling/dspark/markov_head.py",
    "deepspec/modeling/dspark/qwen3/__init__.py",
    "deepspec/modeling/dspark/__init__.py",
    "deepspec/modeling/dspark/gemma4/modeling.py",
    "deepspec/modeling/dspark/gemma4/config.py",
    "deepspec/modeling/dspark/gemma4/__init__.py",
    "deepspec/utils/sampling.py",
    "deepspec/utils/metrics.py",
    "deepspec/utils/config.py",
    "deepspec/utils/optim.py",
    "deepspec/utils/__init__.py",
    "deepspec/utils/constant/__init__.py",
    "deepspec/utils/constant/public.py",
    "deepspec/data/target_cache_dataset.py",
    "deepspec/data/parser.py",
    "deepspec/data/__init__.py",
)


def source_inventory() -> dict[str, str]:
    return {
        relative: hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }


def dense_dspark_attention_mask(
    *,
    anchor_positions,
    block_keep_mask,
    seq_len: int,
    block_size: int,
    device,
):
    """Dense equivalent of the pinned common.create_dspark_attention_mask.

    This is an attention-mask representation fallback, not a replacement
    model.  It is used only with the real upstream Qwen3DSparkModel and eager
    attention on CPU.
    """
    import torch

    batch_size, num_blocks = anchor_positions.shape
    query_length = num_blocks * block_size
    key_length = seq_len + query_length
    mask = torch.full(
        (batch_size, 1, query_length, key_length),
        torch.finfo(torch.float32).min,
        device=device,
    )
    for batch_index in range(batch_size):
        for block_index in range(num_blocks):
            if not bool(block_keep_mask[batch_index, block_index]):
                continue
            anchor = int(anchor_positions[batch_index, block_index])
            draft_start = seq_len + block_index * block_size
            draft_end = draft_start + block_size
            for query_offset in range(block_size):
                query_index = block_index * block_size + query_offset
                mask[batch_index, 0, query_index, :anchor] = 0
                mask[batch_index, 0, query_index, draft_start:draft_end] = 0
    return mask


def _tiny_config_and_inputs(device="cpu"):
    import torch
    from transformers import LlamaConfig

    target_config = LlamaConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=[1, 63],
        attention_dropout=0.0,
    )
    input_ids = torch.tensor(
        [[0, 4, 7, 9, 11, 13, 15, 17]], dtype=torch.long, device=device
    )
    loss_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.uint8, device=device
    )
    target_hidden_states = torch.randn(1, 8, 256 * 5, device=device)
    target_last_hidden_states = torch.randn(1, 8, 256, device=device)
    return target_config, input_ids, loss_mask, target_hidden_states, target_last_hidden_states


def _build_draft_config(target_config):
    from deepspec.modeling.dspark.qwen3.config import build_draft_config

    class ModelArgs(dict):
        __getattr__ = dict.__getitem__

    model_args = ModelArgs(
        num_draft_layers=2,
        block_size=2,
        target_layer_ids=[0, 1, 2, 3, 4],
        mask_token_id=1,
        num_anchors=1,
        confidence_head_alpha=0.0,
        markov_rank=0,
    )
    draft_config = build_draft_config(target_config, model_args)
    # This is the actual compatibility patch under test.
    draft_config.sliding_window = getattr(target_config, "sliding_window", None)
    return draft_config


def _native_flex_probe(
    target_config, input_ids, loss_mask, target_hidden, target_last, device
):
    import torch
    import deepspec.modeling.dspark.qwen3.modeling as upstream_modeling

    config = _build_draft_config(target_config)
    config._attn_implementation = "flex_attention"
    model = upstream_modeling.Qwen3DSparkModel(config).to(device)
    result: dict[str, Any] = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "backend": "flex_attention",
        "sliding_window": config.sliding_window,
    }
    try:
        # FlexAttention can execute a no-grad CPU forward.  Keep that result
        # separate from the training probe: torch's CPU flex backend rejects
        # tensors that require gradients, and the error can be raised during
        # forward before backward is reached.
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                target_hidden_states=target_hidden,
                loss_mask=loss_mask,
                target_last_hidden_states=target_last,
            )
        result["forward"] = True
        result["forward_without_grad"] = True
        result["draft_logits_shape"] = list(outputs.draft_logits.shape)
        try:
            model(
                input_ids=input_ids,
                target_hidden_states=target_hidden,
                loss_mask=loss_mask,
                target_last_hidden_states=target_last,
            ).draft_logits.sum().backward()
            result["backward"] = True
        except Exception as exc:  # exact CPU limitation is part of the receipt
            result["backward"] = False
            result["backward_error_type"] = type(exc).__name__
            result["backward_error"] = str(exc)
    except Exception as exc:
        result["forward"] = False
        result["forward_without_grad"] = False
        result["forward_error_type"] = type(exc).__name__
        result["forward_error"] = str(exc)
    return result


def _eager_dense_probe(
    target_config, input_ids, loss_mask, target_hidden, target_last, device
):
    import torch
    import torch.distributed as dist
    import deepspec.modeling.dspark.qwen3.modeling as upstream_modeling
    from deepspec.modeling.dspark.loss import compute_dspark_loss

    config = _build_draft_config(target_config)
    config._attn_implementation = "eager"
    original_mask_builder = upstream_modeling.create_dspark_attention_mask
    upstream_modeling.create_dspark_attention_mask = dense_dspark_attention_mask
    dist_file = Path(tempfile.mktemp(prefix="sepalith-upstream-dist-", dir="/tmp"))
    try:
        model = upstream_modeling.Qwen3DSparkModel(config).to(device)
        outputs = model(
            input_ids=input_ids,
            target_hidden_states=target_hidden,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last,
        )
        dist_backend = "nccl" if input_ids.is_cuda else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(
                dist_backend,
                rank=0,
                world_size=1,
                init_method="file://" + str(dist_file),
            )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=4.0,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
            confidence_head_alpha=0.0,
        )
        loss.backward()
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        return {
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "backend": "eager_attention_with_dense_cpu_mask_fallback",
            "loss_dist_backend": dist_backend,
            "forward": True,
            "draft_logits_shape": list(outputs.draft_logits.shape),
            "aligned_target_logits_shape": list(outputs.aligned_target_logits.shape),
            "eval_mask_tokens": int(outputs.eval_mask.sum().item()),
            "official_compute_dspark_loss": True,
            "loss": float(loss.detach().item()),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "gradients_finite": finite,
        }
    finally:
        upstream_modeling.create_dspark_attention_mask = original_mask_builder
        if dist.is_initialized():
            dist.destroy_process_group()
        try:
            dist_file.unlink()
        except FileNotFoundError:
            pass


def _official_cache_roundtrip() -> dict[str, Any]:
    """Exercise the pinned writer, index finalizer, manifest, and dataset.

    The temporary cache contains one tiny sample and is deleted before this
    function returns.  This proves the official byte/index protocol without
    reading campaign rows or model weights.
    """
    import torch
    from deepspec.data.target_cache_dataset import (
        CacheDataset,
        CacheCollator,
        INDEX_RECORD_SIZE,
        TARGET_CACHE_VERSION,
        LocalCacheWriteSummary,
        LocalTargetCacheWriter,
        atomic_json_dump,
        build_global_target_cache_shard_map,
        build_target_cache_manifest,
        cleanup_target_cache_tmp_dir,
        finalize_target_cache_index,
        prepare_target_cache_output_dir,
        rename_local_target_cache_shards,
        write_target_cache_manifest,
    )

    temp_root = Path(tempfile.mkdtemp(prefix="sepalith-official-cache-", dir="/tmp"))
    cache_dir = temp_root / "cache"
    rank_dir = cache_dir / "_tmp" / "rank_0"
    writer = None
    dataset = None
    try:
        prepare_target_cache_output_dir(str(cache_dir))
        rank_dir.mkdir(parents=True, exist_ok=True)
        # Token 1 is both PAD and a native EOG.  Keep it as the final real
        # token so the indexed length, rather than token identity, drives
        # attention-mask construction.
        input_ids = torch.tensor([0, 4, 7, 1], dtype=torch.int32)
        attention_mask = torch.ones(4, dtype=torch.uint8)
        loss_mask = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)
        target_hidden_states = torch.arange(4 * 5 * 2, dtype=torch.float32).reshape(4, 10)
        target_last_hidden_states = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
        writer = LocalTargetCacheWriter(
            rank_dir=str(rank_dir),
            max_shard_bytes=1 << 20,
        )
        writer.write_sample(
            sample_id=0,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
        )
        local_shard_files = list(writer.local_shard_files)
        writer.close()
        writer = None
        summary = LocalCacheWriteSummary(
            global_rank=0,
            source_sample_start=0,
            source_sample_end=1,
            num_local_samples=1,
            num_local_shards=len(local_shard_files),
            local_shard_files=local_shard_files,
        )
        atomic_json_dump(summary.to_json(), rank_dir / "summary.json")
        summaries = [summary.to_json()]
        shard_map, shards = build_global_target_cache_shard_map(summaries)
        rename_local_target_cache_shards(
            output_dir=str(cache_dir),
            rank_dir=str(rank_dir),
            summary=summary.to_json(),
            shard_map=shard_map,
        )
        num_samples = finalize_target_cache_index(
            output_dir=str(cache_dir),
            summaries=summaries,
            shard_map=shard_map,
        )
        manifest = build_target_cache_manifest(
            num_samples=num_samples,
            shards=shards,
            target_layer_ids=[0, 1, 2, 3, 4],
            hidden_size=2,
            extra_fields={
                "target_model_name_or_path": "tiny-target",
                "source_jsonl_paths": ["tiny-pretokenized"],
                "chat_template": "pretokenized",
                "max_length": 64,
                "min_loss_tokens": 1,
            },
        )
        write_target_cache_manifest(output_dir=str(cache_dir), manifest=manifest)
        cleanup_target_cache_tmp_dir(str(cache_dir))
        dataset = CacheDataset(str(cache_dir))
        item = dataset[0]
        collated = CacheCollator()([item])
        indexed_record = dataset._read_record(0)
        expected_hidden = target_hidden_states.to(torch.bfloat16)
        expected_last = target_last_hidden_states.to(torch.bfloat16)
        values_match = {
            "input_ids": bool(torch.equal(item["input_ids"], input_ids)),
            "loss_mask": bool(torch.equal(item["loss_mask"], loss_mask)),
            "target_hidden_states": bool(
                torch.equal(item["target_hidden_states"], expected_hidden)
            ),
            "target_last_hidden_states": bool(
                torch.equal(item["target_last_hidden_states"], expected_last)
            ),
        }
        shapes_match = {
            "input_ids": list(item["input_ids"].shape) == [4],
            "loss_mask": list(item["loss_mask"].shape) == [4],
            "target_hidden_states": list(item["target_hidden_states"].shape) == [4, 10],
            "target_last_hidden_states": list(item["target_last_hidden_states"].shape)
            == [4, 2],
        }
        dtypes_match = {
            "input_ids": str(item["input_ids"].dtype) == "torch.int32",
            "loss_mask": str(item["loss_mask"].dtype) == "torch.uint8",
            "target_hidden_states": str(item["target_hidden_states"].dtype)
            == "torch.bfloat16",
            "target_last_hidden_states": str(item["target_last_hidden_states"].dtype)
            == "torch.bfloat16",
        }
        return {
            "roundtrip": all(values_match.values())
            and all(shapes_match.values())
            and all(dtypes_match.values()),
            "cache_version": TARGET_CACHE_VERSION,
            "index_record_size": INDEX_RECORD_SIZE,
            "num_samples": len(dataset),
            "values_match": values_match,
            "shapes_match": shapes_match,
            "dtypes_match": dtypes_match,
            "attention_mask_writer_input": True,
            "attention_mask_dataset_output": False,
            "input_ends_native_eog": bool(input_ids[-1].item() == 1),
            "indexed_seq_len": int(indexed_record["seq_len"]),
            "indexed_seq_len_preserves_eog": int(indexed_record["seq_len"]) == 4,
            "collator_attention_mask": collated["attention_mask"].tolist(),
            "collator_attention_mask_preserves_eog": collated["attention_mask"].tolist()
            == [[1, 1, 1, 1]],
            "eos_required_by_writer": False,
        }
    finally:
        if dataset is not None:
            dataset.close()
        if writer is not None:
            writer.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def run_smoke(device="cpu") -> dict[str, Any]:
    import torch
    import transformers

    torch.set_num_threads(1)
    torch.manual_seed(0)
    target_config, input_ids, loss_mask, target_hidden, target_last = _tiny_config_and_inputs(
        device=device
    )
    native = _native_flex_probe(
        target_config, input_ids, loss_mask, target_hidden, target_last, device
    )
    eager = _eager_dense_probe(
        target_config, input_ids, loss_mask, target_hidden, target_last, device
    )
    cache = _official_cache_roundtrip()
    return {
        "deepspec_revision": DEEPSPEC_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "target_config_class": f"{type(target_config).__module__}.{type(target_config).__name__}",
        "target_config_had_sliding_window": hasattr(target_config, "sliding_window"),
        "target_config_after_patch_sliding_window": _build_draft_config(target_config).sliding_window,
        "native_flex": native,
        "eager_dense_cpu": eager,
        "official_cache_roundtrip": cache,
        "source_inventory_sha256": source_inventory(),
        "campaign_weights_read": False,
        "weights_downloaded": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        help="CPU for the local smoke; use cuda:0 only in an admitted cloud smoke.",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("requested CUDA device but torch.cuda.is_available() is false")
    print(json.dumps(run_smoke(device=args.device), sort_keys=True))
