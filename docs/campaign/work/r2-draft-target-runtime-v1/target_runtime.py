"""Bounded target generation and official DeepSpec cache runtime.

This module accepts pretokenized TRAIN rows and an explicit model hash
manifest.  It has no chat-template or authored-tail path.  The tiny self-test
uses a randomly initialized HF LlamaForCausalLM and the pinned DeepSpec hook
and cache classes; it never reads campaign weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parent
UPSTREAM_ROOT = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/"
    "work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec"
)
sys.path.insert(0, str(UPSTREAM_ROOT))
sys.path.insert(
    0,
    str(
        Path(
            "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/"
            "work/r2-draft-adapter-hardening-v1"
        )
    ),
)

from deepspec.data.target_cache_dataset import (  # noqa: E402
    CacheDataset,
    INDEX_RECORD_SIZE,
    LocalCacheWriteSummary,
    LocalTargetCacheWriter,
    TARGET_CACHE_VERSION,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    finalize_target_cache_index,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    unpack_index_record,
    write_target_cache_manifest,
)
from scripts.data.prepare_target_cache import (  # noqa: E402
    run_target_forward_with_hooks,
)
from prepare_teacher_cache import (  # noqa: E402
    TRAIN_SOURCE_PATH,
    TRAIN_SOURCE_SHA256,
    regenerate_train_rows,
)
from pretokenized_cache_bridge import (  # noqa: E402
    reconcile_written_records,
    validate_cache_sample,
)
from teacher_rows import prompt_prefix  # noqa: E402


TARGET_EOG_IDS = (1, 130073)
DEFAULT_TARGET_LAYER_IDS = (1, 10, 20, 30, 39)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every required model hash before a model loader is called."""
    required = ("model_dir", "config_sha256", "weights_file", "weights_sha256")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError("model manifest missing: " + ", ".join(missing))
    model_dir = Path(str(manifest["model_dir"])).resolve()
    config_path = model_dir / str(manifest.get("config_file", "config.json"))
    weights_path = model_dir / str(manifest["weights_file"])
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("model manifest files are not present")
    config_sha256 = sha256_file(config_path)
    weights_sha256 = sha256_file(weights_path)
    if config_sha256 != str(manifest["config_sha256"]):
        raise ValueError("model config hash does not match the explicit manifest")
    if weights_sha256 != str(manifest["weights_sha256"]):
        raise ValueError("model weights hash does not match the explicit manifest")
    result = dict(manifest)
    result.update(
        {
            "model_dir": str(model_dir),
            "config_file": str(config_path),
            "weights_file": str(weights_path),
            "config_sha256_verified": config_sha256,
            "weights_sha256_verified": weights_sha256,
        }
    )
    return result


def load_target_model(manifest: Mapping[str, Any], *, device: str = "cpu"):
    """Hash-gate then load the real HF CausalLM from a local model directory."""
    verified = validate_model_manifest(manifest)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        verified["model_dir"],
        local_files_only=True,
        dtype=torch.float32,
    )
    return model.to(device=device).eval(), verified


class GreedyTargetGenerator:
    """Generate exact pretokenized suffixes from a real HF CausalLM."""

    def __init__(
        self,
        model,
        *,
        vocab_size: int,
        stop_ids: Sequence[int] = TARGET_EOG_IDS,
    ) -> None:
        self.model = model
        self.vocab_size = int(vocab_size)
        self.stop_ids = tuple(int(token_id) for token_id in stop_ids)
        self.device = next(model.parameters()).device

    def __call__(self, prompt_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
        row = {"id": "generator-input", "input_ids": list(prompt_ids), "target_start": len(prompt_ids)}
        prefix = prompt_prefix(row, vocab_size=self.vocab_size)
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        generated: list[int] = []
        with torch.no_grad():
            for _ in range(int(max_new_tokens)):
                attention_mask = torch.ones_like(input_ids)
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                next_token = int(output.logits[0, -1].argmax(dim=-1).item())
                if not 0 <= next_token < self.vocab_size:
                    raise ValueError("target generated an out-of-vocabulary token")
                generated.append(next_token)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token]], device=self.device)], dim=1
                )
                if next_token in self.stop_ids:
                    break
        # The mapping is intentionally explicit: the hardened adapter checks
        # that this full sequence equals prefix + generated_ids.
        return {
            "input_ids": prefix + generated,
            "generated_ids": generated,
            "protocol_status": "valid",
        }


def _verify_hook_taps(
    *,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids: Sequence[int],
    captured,
) -> bool:
    """Compare official layer hooks to HF hidden_states[tap + 1]."""
    backbone = getattr(target_model, "model", target_model)
    with torch.no_grad():
        reference = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    expected = torch.cat(
        [reference.hidden_states[int(layer_id) + 1] for layer_id in target_layer_ids],
        dim=-1,
    )
    return bool(
        torch.allclose(captured.target_hidden_states, expected, atol=1e-5, rtol=1e-4)
        and torch.allclose(captured.target_last_hidden_states, reference.last_hidden_state, atol=1e-5, rtol=1e-4)
    )


def _read_index_sample_ids(index_path: Path) -> list[int]:
    raw = index_path.read_bytes()
    if len(raw) % INDEX_RECORD_SIZE:
        raise ValueError("finalized target cache index has invalid byte length")
    return [
        int(unpack_index_record(raw, offset)["sample_id"])
        for offset in range(0, len(raw), INDEX_RECORD_SIZE)
    ]


def run_target_runtime(
    *,
    rows: Sequence[Mapping[str, Any]],
    target_model,
    target_identity: Mapping[str, Any],
    output_dir: str | Path,
    source_path: str = TRAIN_SOURCE_PATH,
    source_sha256: str = TRAIN_SOURCE_SHA256,
    limit: int = 8,
    allow_bounded_run: bool = False,
    generation_cap: int = 192,
    target_layer_ids: Sequence[int] = DEFAULT_TARGET_LAYER_IDS,
    min_loss_tokens: int = 1,
) -> dict[str, Any]:
    """Generate, hook, write, finalize, and read back a bounded cache."""
    if limit <= 0 or limit > 256:
        raise ValueError("limit must be in 1..256")
    if limit > 8 and not allow_bounded_run:
        raise ValueError("limits above the 8-row profile require allow_bounded_run")
    if min_loss_tokens <= 0:
        raise ValueError("min_loss_tokens must be positive")
    if not target_layer_ids:
        raise ValueError("target_layer_ids must not be empty")
    source = {
        "path": str(source_path),
        "sha256": str(source_sha256),
    }
    if source != {"path": TRAIN_SOURCE_PATH, "sha256": TRAIN_SOURCE_SHA256}:
        raise ValueError("target runtime requires the admitted TRAIN source path/hash")
    hidden_size = int(target_model.config.hidden_size)
    vocab_size = int(target_model.config.vocab_size)
    generator = GreedyTargetGenerator(
        target_model,
        vocab_size=vocab_size,
        stop_ids=TARGET_EOG_IDS,
    )
    generation_started = time.perf_counter()
    records, generation_summary = regenerate_train_rows(
        rows,
        generator,
        limit=limit,
        generation_cap=generation_cap,
        vocab_size=vocab_size,
        target_identity=target_identity,
        source_path=source_path,
        source_sha256=source_sha256,
    )
    generation_seconds = time.perf_counter() - generation_started
    cacheable = [record for record in records if record.get("cacheable")]
    if not cacheable:
        raise RuntimeError("target regeneration produced no cacheable rows")

    output = Path(output_dir).resolve()
    prepare_target_cache_output_dir(str(output))
    rank_dir = output / "_tmp" / "rank_0"
    rank_dir.mkdir(parents=True, exist_ok=True)
    writer = LocalTargetCacheWriter(rank_dir=str(rank_dir), max_shard_bytes=1 << 30)
    written_ids: list[str] = []
    tap_checks: list[bool] = []
    cache_started = time.perf_counter()
    try:
        backbone = getattr(target_model, "model", target_model)
        for record in cacheable:
            ids = torch.tensor(record["input_ids"], dtype=torch.long, device=generator.device)
            attention = torch.ones_like(ids, dtype=torch.uint8)
            target_result = run_target_forward_with_hooks(
                target_model=backbone,
                input_ids=ids.unsqueeze(0),
                attention_mask=attention.unsqueeze(0),
                target_layer_ids=target_layer_ids,
            )
            tap_checks.append(
                _verify_hook_taps(
                    target_model=backbone,
                    input_ids=ids.unsqueeze(0),
                    attention_mask=attention.unsqueeze(0),
                    target_layer_ids=target_layer_ids,
                    captured=target_result,
                )
            )
            target_hidden = target_result.target_hidden_states[0].to(torch.bfloat16).cpu()
            target_last = target_result.target_last_hidden_states[0].to(torch.bfloat16).cpu()
            sample = validate_cache_sample(
                record,
                target_hidden,
                target_last,
                hidden_size=hidden_size,
                target_taps=len(target_layer_ids),
                vocab_size=vocab_size,
            )
            if sum(int(value) for value in sample["loss_mask"]) < min_loss_tokens:
                raise ValueError(f"row {record.get('id')!r} has too few target labels")
            row_id = str(record["id"])
            writer.write_sample(
                sample_id=len(written_ids),
                input_ids=ids.detach().to(torch.int32).cpu(),
                attention_mask=attention.cpu(),
                loss_mask=torch.tensor(record["loss_mask"], dtype=torch.uint8),
                target_hidden_states=target_hidden,
                target_last_hidden_states=target_last,
            )
            written_ids.append(row_id)
    finally:
        writer.close()
    cache_seconds = time.perf_counter() - cache_started
    reconciliation = reconcile_written_records(cacheable, written_ids)
    local_files = list(writer.local_shard_files)
    summary = LocalCacheWriteSummary(
        global_rank=0,
        source_sample_start=0,
        source_sample_end=len(rows),
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(local_files),
        local_shard_files=local_files,
    )
    atomic_json_dump(summary.to_json(), rank_dir / "summary.json")
    summaries = [summary.to_json()]
    shard_map, shards = build_global_target_cache_shard_map(summaries)
    rename_local_target_cache_shards(
        output_dir=str(output),
        rank_dir=str(rank_dir),
        summary=summary.to_json(),
        shard_map=shard_map,
    )
    num_samples = finalize_target_cache_index(
        output_dir=str(output), summaries=summaries, shard_map=shard_map
    )
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": str(target_identity.get("name", "bound-target")),
            "target_identity": dict(target_identity),
            "source_jsonl_paths": [str(source_path)],
            "source_rows_sha256": str(source_sha256),
            "selected_row_ids": [str(record.get("id")) for record in records],
            "written_row_ids": list(written_ids),
            "selected_row_count": len(records),
            "written_row_count": len(written_ids),
            "chat_template": "pretokenized",
            "max_length": max(len(record["input_ids"]) for record in cacheable),
            "min_loss_tokens": int(min_loss_tokens),
            "native_serving_stop_ids": list(TARGET_EOG_IDS),
        },
    )
    if int(manifest["num_samples"]) != len(written_ids):
        raise ValueError("manifest count does not equal actual written rows")
    write_target_cache_manifest(output_dir=str(output), manifest=manifest)
    teacher_path = output / "teacher_rows.jsonl"
    with teacher_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    cleanup_target_cache_tmp_dir(str(output))

    index_sample_ids = _read_index_sample_ids(output / "samples.idx")
    if index_sample_ids != list(range(len(written_ids))):
        raise ValueError("finalized index sample IDs are not dense")
    dataset = CacheDataset(str(output))
    try:
        if len(dataset) != len(written_ids):
            raise ValueError("CacheDataset length does not equal written row count")
        for index in range(len(dataset)):
            item = dataset[index]
            if item["input_ids"].shape[0] != len(cacheable[index]["input_ids"]):
                raise ValueError("CacheDataset sequence length differs from teacher row")
    finally:
        dataset.close()
    if not all(tap_checks):
        raise ValueError("one or more official hook outputs differ from hidden_states[tap+1]")
    return {
        "status": "target_runtime_passed",
        "profile_kind": "profile8" if limit <= 8 else "bounded_train",
        "selected_rows": len(records),
        "cacheable_rows": len(cacheable),
        "written_rows": len(written_ids),
        "written_row_ids": written_ids,
        "generation_seconds": generation_seconds,
        "cache_seconds": cache_seconds,
        "generation_summary": generation_summary,
        "tap_outputs_match_hidden_states": all(tap_checks),
        "loss_masks": [record["loss_mask"] for record in cacheable],
        "native_eog_ids": list(TARGET_EOG_IDS),
        "target_identity": dict(target_identity),
        "manifest_path": str(output / "manifest.json"),
        "cache_dir": str(output),
        "target_cache_version": TARGET_CACHE_VERSION,
        "index_record_size": INDEX_RECORD_SIZE,
        "reconciliation": reconciliation,
    }


def _tiny_model_manifest(root: Path) -> tuple[dict[str, Any], Any]:
    from transformers import LlamaConfig, LlamaForCausalLM

    model_dir = root / "tiny-target"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        # Keep the five tapped layers below the final layer so each hook
        # corresponds exactly to hidden_states[tap + 1]; the final normalized
        # output remains a separate last_hidden_state check.
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=[1, 31],
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(model_dir, safe_serialization=True)
    weights_path = model_dir / "model.safetensors"
    manifest = {
        "name": "tiny-random-target",
        "model_dir": str(model_dir),
        "config_file": "config.json",
        "config_sha256": sha256_file(model_dir / "config.json"),
        "weights_file": "model.safetensors",
        "weights_sha256": sha256_file(weights_path),
        "tokenizer_sha256": "tiny-tokenizer-not-loaded",
    }
    return manifest, model


def run_tiny_integration() -> dict[str, Any]:
    """CPU integration with a real HF CausalLM and official cache pipeline."""
    torch.set_num_threads(2)
    torch.manual_seed(0)
    temp_root = Path(tempfile.mkdtemp(prefix="sepalith-target-runtime-", dir="/tmp"))
    try:
        manifest, _ = _tiny_model_manifest(temp_root)
        model, verified = load_target_model(manifest, device="cpu")
        # Make this actual HF model's greedy first token a native EOG.  The
        # hidden state comes from the model itself; no proxy generator exists.
        prefix = torch.tensor([[0, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            hidden = model.model(input_ids=prefix, use_cache=False).last_hidden_state[0, -1]
            model.lm_head.weight.zero_()
            model.lm_head.weight[1].copy_(hidden)
        rows = [
            {
                "id": "tiny-train-0",
                "split": "cpt_train",
                "input_ids": [0, 2, 3, 29, 1],
                "target_start": 3,
            },
            {
                "id": "tiny-heldout-0",
                "split": "validation",
                "input_ids": [0, 2, 3, 28, 1],
                "target_start": 3,
            },
        ]
        output = temp_root / "cache"
        identity = dict(verified)
        identity["name"] = "tiny-random-target"
        result = run_target_runtime(
            rows=rows,
            target_model=model,
            target_identity=identity,
            output_dir=output,
            limit=2,
            target_layer_ids=(0, 1, 2, 3, 4),
        )
        assert result["written_rows"] == 1
        assert result["tap_outputs_match_hidden_states"]
        assert result["loss_masks"] == [[0, 0, 0, 1]]
        assert result["generation_summary"]["eog_terminated"] == 1
        assert result["generation_summary"]["generation_error"] == 1
        assert result["selected_rows"] == 2
        return {
            "result": result,
            "model_manifest_verified": True,
            "model_weights_loaded": True,
            "campaign_weights_read": False,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-manifest")
    parser.add_argument("--train-data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--allow-bounded-run", action="store_true")
    parser.add_argument("--generation-cap", type=int, default=192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-layer-ids", default=','.join(str(x) for x in DEFAULT_TARGET_LAYER_IDS))
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_tiny_integration(), sort_keys=True))
        return
    if not args.model_manifest or not args.train_data_path or not args.output_dir:
        parser.error("--model-manifest, --train-data-path, and --output-dir are required")
    manifest = json.loads(Path(args.model_manifest).read_text(encoding="utf-8"))
    model, verified = load_target_model(manifest, device=args.device)
    from prepare_teacher_cache import read_jsonl

    rows = read_jsonl(args.train_data_path)
    layer_ids = tuple(int(value) for value in str(args.target_layer_ids).split(',') if value)
    report = run_target_runtime(
        rows=rows,
        target_model=model,
        target_identity=verified,
        output_dir=args.output_dir,
        source_path=args.train_data_path,
        source_sha256=manifest.get("train_source_sha256", TRAIN_SOURCE_SHA256),
        limit=args.limit,
        allow_bounded_run=args.allow_bounded_run,
        generation_cap=args.generation_cap,
        target_layer_ids=layer_ids,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
