#!/usr/bin/env python3
"""Prepare or run a bounded P=2 longest-TRAIN generation memory gate.

The default data path is the admitted RL-02 train rows plus its separate
context sidecar.  Selection uses the already stored integer prompt prefix;
there is no tokenizer pass over the 8,440 rows.  The live mode is explicit and
loads the root-supplied merged-SFT parent only after all CPU checks pass.  It
generates exactly two distinct logical G=4 groups in one P=2 call and reports
actual terminal lengths, memory, and elapsed time.  It does not train, score
quality, or claim that a shorter observed completion exercised the 192-token
capacity.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


G = 4
GROUPS_PER_CALL = 2
PROMPT_MAX_TOKENS = 2048
COMPLETION_MAX_TOKENS = 192
CONTEXT_MAX_TOKENS = PROMPT_MAX_TOKENS + COMPLETION_MAX_TOKENS
MEMORY_FRACTION = 0.75
LORA_RANK = 16
LORA_ALPHA = 16
VOCAB_SIZE = 130560
BOS_ID = 0
EOS_ID = 1
NATIVE_EOG_IDS = (1, 130073)
EXPECTED_ROWS_SHA256 = "e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602"
EXPECTED_SIDECAR_SHA256 = "6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f"
EXPECTED_SELECTED_IDS_SHA256 = "24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d"
EXPECTED_PARENT_MANIFEST_SHA256 = "1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12"
EXPECTED_PARENT_MERGED_WEIGHTS_SHA256 = "499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d"
EXPECTED_TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
EXPECTED_NATIVE_BASE_WEIGHT_BYTES = 5033557128
EXPECTED_NATIVE_BASE_WEIGHT_SHA256 = "38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad"
EXPECTED_BASE_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
DEFAULT_EXECUTION_ROOT = (
    "/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/"
    "snapshots/15fb2080278022b076223c2a823413fc5cb9c014d50dab42dc9df6d57d3f2f41/source"
)
DEFAULT_ROWS = "/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/eligible-train-rows.jsonl"
DEFAULT_SIDECAR = "/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/context-sidecar.jsonl"
DEFAULT_SELECTED = "/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/selected-train-ids-lead-order-v1.json"
DEFAULT_PARENT_MANIFEST = "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/parent-manifest.json"
DEFAULT_NATIVE_BASE = "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native"


class GateError(ValueError):
    """A gate input or live generation result failed closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    except OSError as error:
        raise GateError(f"cannot hash {path}: {error}") from error
    return size, digest.hexdigest()


def _verify_file(path_value: str | Path, expected: str, name: str) -> tuple[Path, int, str]:
    path = Path(path_value).resolve()
    if not path.is_absolute() or not path.is_file():
        raise GateError(f"{name} must be an existing absolute file: {path}")
    size, actual = _sha256_file(path)
    if actual != expected:
        raise GateError(f"{name} SHA256 mismatch: expected {expected}, observed {actual}")
    return path, size, actual


def _verify_native_base(path_value: str | Path) -> dict[str, Any]:
    """Verify only small native-base metadata; never hash its weight here."""
    root = Path(path_value).resolve()
    if not root.is_dir():
        raise GateError(f"native base directory is absent: {root}")
    tokenizer = root / "tokenizer.json"
    size, digest, = _sha256_file(tokenizer) if tokenizer.is_file() else (0, "")
    if digest != EXPECTED_TOKENIZER_SHA256:
        raise GateError(
            f"native base tokenizer SHA256 mismatch: expected {EXPECTED_TOKENIZER_SHA256}, observed {digest}"
        )
    weight = root / "model.safetensors"
    if not weight.is_file() or weight.stat().st_size != EXPECTED_NATIVE_BASE_WEIGHT_BYTES:
        observed = weight.stat().st_size if weight.is_file() else None
        raise GateError(
            f"native base weight size mismatch: expected {EXPECTED_NATIVE_BASE_WEIGHT_BYTES}, observed {observed}"
        )
    return {
        "path": str(root),
        "tokenizer_json": {"bytes": size, "sha256": digest},
        "weight": {
            "bytes": EXPECTED_NATIVE_BASE_WEIGHT_BYTES,
            "expected_sha256": EXPECTED_NATIVE_BASE_WEIGHT_SHA256,
            "sha256_verified": False,
        },
        "base_revision": EXPECTED_BASE_REVISION,
        "weights_hashed": False,
    }


def _load_admitted_records(
    execution_root: Path,
    rows_path: Path,
    sidecar_path: Path,
    selected_path: Path,
) -> tuple[list[Any], Any, dict[str, Any]]:
    training = execution_root / "experiments" / "training"
    package_src = execution_root / "packages" / "sepalith" / "src"
    if not training.is_dir() or not package_src.is_dir():
        raise GateError(f"execution source tree is incomplete: {execution_root}")
    sys.path.insert(0, str(training))
    sys.path.insert(0, str(package_src))
    try:
        from campaign_rl_data import load_training_records
    except Exception as error:
        raise GateError(f"cannot import framework-free campaign_rl_data: {error}") from error
    try:
        records, manifest = load_training_records(
            rows_path,
            EXPECTED_ROWS_SHA256,
            sidecar_path,
            EXPECTED_SIDECAR_SHA256,
            selected_path,
            EXPECTED_SELECTED_IDS_SHA256,
            admission_status="admitted",
        )
    except Exception as error:
        raise GateError(f"admitted RL-02 rows/sidecar failed validation: {error}") from error
    if manifest.row_count != 8440 or len(records) != 8440:
        raise GateError(f"expected 8440 admitted TRAIN records, observed {len(records)}")
    if any(record.row.get("split") != "train" for record in records):
        raise GateError("a selected record is not split=train")
    return records, manifest, {
        "rows_path": str(rows_path),
        "rows_sha256": manifest.rows_sha256,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": manifest.context_sha256,
        "selected_ids_path": str(selected_path),
        "selected_ids_sha256": manifest.selected_ids_sha256,
        "row_count": manifest.row_count,
        "split": "train",
        "admission_status": manifest.admission_status,
    }


def select_longest_distinct(records: Sequence[Any], selected_ids: Sequence[str]) -> list[Any]:
    """Select the two longest distinct stored prompt prefixes stably by ID order."""
    order = {row_id: index for index, row_id in enumerate(selected_ids)}
    if len(order) != len(selected_ids):
        raise GateError("selected TRAIN IDs are not unique")
    ranked = sorted(records, key=lambda record: (-len(record.prompt_ids), order[record.row_id]))
    chosen: list[Any] = []
    seen_ids: set[str] = set()
    seen_prompts: set[tuple[int, ...]] = set()
    for record in ranked:
        prompt = tuple(record.prompt_ids)
        if record.row_id in seen_ids or prompt in seen_prompts:
            continue
        if len(prompt) > PROMPT_MAX_TOKENS or not prompt or prompt[0] != BOS_ID:
            continue
        if record.row.get("split") != "train":
            continue
        chosen.append(record)
        seen_ids.add(record.row_id)
        seen_prompts.add(prompt)
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise GateError("fewer than two distinct eligible TRAIN prompt prefixes fit the 2048 cap")
    return chosen


def make_generation_prompts(records: Sequence[Any]) -> list[dict[str, Any]]:
    """Repeat each selected row four times for one complete logical G group."""
    if len(records) != 2 or records[0].row_id == records[1].row_id:
        raise GateError("P2 gate requires two distinct selected row IDs")
    prompts: list[dict[str, Any]] = []
    for record in records:
        prompt = record.envelope()["prompt"]
        ids = list(prompt["ids"])
        if len(ids) > PROMPT_MAX_TOKENS or ids[0] != BOS_ID:
            raise GateError(f"selected prompt {record.row_id} violates stored prompt cap")
        prompts.extend({"text": prompt["text"], "ids": list(ids)} for _ in range(G))
    if len(prompts) != 2 * G:
        raise GateError("P2 gate did not construct two complete G=4 groups")
    return prompts


def _parent_manifest_summary(path: Path, expected_sha256: str) -> dict[str, Any]:
    size, digest = _sha256_file(path)
    if digest != expected_sha256:
        raise GateError(f"parent manifest SHA256 mismatch: expected {expected_sha256}, observed {digest}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"parent manifest is not valid JSON: {path}") from error
    if not isinstance(value, Mapping) or value.get("status") != "accepted" or value.get("kind") != "merged_sft":
        raise GateError("parent manifest is not an accepted merged_sft manifest")
    if value.get("merged_weights_sha256") != EXPECTED_PARENT_MERGED_WEIGHTS_SHA256:
        raise GateError("parent manifest merged weight identity differs from the pinned theta0 parent")
    return {
        "path": str(path),
        "bytes": size,
        "manifest_sha256": digest,
        "kind": value["kind"],
        "status": value["status"],
        "model_path": value.get("merged_model_path"),
        "merged_weights_sha256": value["merged_weights_sha256"],
        "base_model_revision": value.get("base_model_revision"),
        "tokenizer_json_sha256": value.get("tokenizer", {}).get("tokenizer_json_sha256"),
    }


def _source_summary(execution_root: Path, trainer_sha256: str) -> dict[str, Any]:
    trainer_path = execution_root / "experiments" / "training" / "campaign_rl_train.py"
    protocol_path = execution_root / "packages" / "sepalith" / "src" / "sepalith" / "campaign_protocol.py"
    trainer_bytes, trainer_actual = _sha256_file(trainer_path)
    if trainer_actual != trainer_sha256:
        raise GateError(
            f"trainer SHA256 mismatch: expected {trainer_sha256}, observed {trainer_actual}"
        )
    protocol_bytes, protocol_actual = _sha256_file(protocol_path)
    return {
        "execution_root": str(execution_root),
        "trainer_path": str(trainer_path),
        "trainer_bytes": trainer_bytes,
        "trainer_sha256": trainer_actual,
        "protocol_path": str(protocol_path),
        "protocol_bytes": protocol_bytes,
        "protocol_sha256": protocol_actual,
    }


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    execution_root = Path(args.execution_root).resolve()
    rows_path, rows_bytes, rows_sha = _verify_file(args.rows_path, EXPECTED_ROWS_SHA256, "RL-02 rows")
    sidecar_path, sidecar_bytes, sidecar_sha = _verify_file(args.sidecar_path, EXPECTED_SIDECAR_SHA256, "RL-02 sidecar")
    selected_path, selected_bytes, selected_sha = _verify_file(args.selected_ids_path, EXPECTED_SELECTED_IDS_SHA256, "RL-02 selected IDs")
    records, manifest, data = _load_admitted_records(execution_root, rows_path, sidecar_path, selected_path)
    selected = select_longest_distinct(records, manifest.selected_ids)
    prompts = make_generation_prompts(selected)
    parent = _parent_manifest_summary(Path(args.parent_manifest).resolve(), EXPECTED_PARENT_MANIFEST_SHA256)
    base = _verify_native_base(args.native_base_path)
    source = _source_summary(execution_root, args.trainer_sha256)
    selection = [
        {
            "id": record.row_id,
            "family": record.row.get("family"),
            "package_id": record.row.get("package_id"),
            "prompt_tokens": len(record.prompt_ids),
            "prompt_ids_sha256": hashlib.sha256(
                json.dumps(list(record.prompt_ids), separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "source_identity": dict(record.source_identity or {}),
            "selection_geometry": dict(record.selection_geometry or {}),
        }
        for record in selected
    ]
    return {
        "status": "prepared",
        "prepared_at_utc": _now(),
        "mode": "live_gate_available_but_not_launched",
        "geometry": {
            "candidate_count_G": G,
            "generation_groups_per_call": GROUPS_PER_CALL,
            "logical_groups": 2,
            "model_call_rows": G * GROUPS_PER_CALL,
            "completion_cap": COMPLETION_MAX_TOKENS,
            "prompt_cap": PROMPT_MAX_TOKENS,
            "context_cap": CONTEXT_MAX_TOKENS,
            "left_pad_id": EOS_ID,
            "manual_bos_id": BOS_ID,
            "native_eog_ids": list(NATIVE_EOG_IDS),
        },
        "data": data,
        "data_file_bytes": {
            "rows": rows_bytes,
            "sidecar": sidecar_bytes,
            "selected_ids": selected_bytes,
        },
        "selection": selection,
        "prompt_batch": {
            "rows": len(prompts),
            "group_row_counts": [G, G],
            "selected_ids_order": [record.row_id for record in selected],
            "prompt_ids_are_stored_prefixes": True,
            "tokenizer_audit": "two selected canonical rows only in live load; no 8440-row tokenizer audit",
        },
        "parent": parent,
        "native_base": base,
        "source": source,
        "policy": {
            "cuda_memory_fraction": MEMORY_FRACTION,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "sampling": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
                "max_new_tokens": COMPLETION_MAX_TOKENS,
                "eos_token_id": list(NATIVE_EOG_IDS),
                "pad_token_id": EOS_ID,
                "bos_token_id": BOS_ID,
                "use_cache": True,
            },
        },
        "live_argv": _live_argv(args, args.result_path),
        "quality_claim": False,
    }


def _live_argv(args: argparse.Namespace, result_path: str | Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--live",
        "--execution-root", str(Path(args.execution_root).resolve()),
        "--rows-path", str(Path(args.rows_path).resolve()),
        "--sidecar-path", str(Path(args.sidecar_path).resolve()),
        "--selected-ids-path", str(Path(args.selected_ids_path).resolve()),
        "--parent-manifest", str(Path(args.parent_manifest).resolve()),
        "--native-base-path", str(Path(args.native_base_path).resolve()),
        "--trainer-sha256", args.trainer_sha256,
        "--result-path", str(Path(result_path).resolve()),
    ]


def _device_memory(torch_module: Any, device: Any) -> dict[str, int | None]:
    if getattr(device, "type", None) != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": int(torch_module.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch_module.cuda.memory_reserved(device)),
    }


def _peak_memory(torch_module: Any, device: Any) -> dict[str, int | None]:
    if getattr(device, "type", None) != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(device)),
    }


def run_live(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Run only after explicit --live; root owns this later launch."""
    execution_root = Path(args.execution_root).resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise GateError("live P2 memory gate requires the root-owned CUDA device; CUDA_VISIBLE_DEVICES is empty")
    sys.path.insert(0, str(execution_root / "experiments" / "training"))
    sys.path.insert(0, str(execution_root / "packages" / "sepalith" / "src"))
    try:
        import torch
        from accelerate import Accelerator
        from campaign_rl_entry import (
            _attach_rl_adapter,
            _load_live_model,
            verify_merged_parent_manifest,
        )
        from campaign_rl_train import FixedIDGRPOTrainerMixin
        from campaign_sft import (
            assert_post_trainer_pinned_identity,
            training_configuration_guard,
        )
    except Exception as error:
        raise GateError(f"cannot import the pinned live generation seam: {error}") from error
    parent_wrapper = {"path": str(Path(args.parent_manifest).resolve()), "sha256": EXPECTED_PARENT_MANIFEST_SHA256}
    parent_audit = verify_merged_parent_manifest(parent_wrapper, verify_weights=True)
    if parent_audit["merged_weights_sha256"] != EXPECTED_PARENT_MERGED_WEIGHTS_SHA256:
        raise GateError("verified parent weight identity differs from the pinned theta0 parent")
    selected_records, _manifest, _data = _load_admitted_records(
        execution_root,
        Path(args.rows_path).resolve(),
        Path(args.sidecar_path).resolve(),
        Path(args.selected_ids_path).resolve(),
    )
    selected = select_longest_distinct(selected_records, json.loads(Path(args.selected_ids_path).read_text())["row_ids"])
    prompt_rows = make_generation_prompts(selected)
    model, tokenizer, reference_tokenizer, fast_model, tokenizer_audit = _load_live_model(
        parent_audit,
        [record.row for record in selected],
        model_load_max_seq_length=4096,
        cuda_memory_fraction=MEMORY_FRACTION,
    )
    model, adapter_audit = _attach_rl_adapter(model, fast_model)
    accelerator = Accelerator()
    trainer = FixedIDGRPOTrainerMixin.__new__(FixedIDGRPOTrainerMixin)
    trainer.model = model
    trainer.model_wrapped = model
    trainer.accelerator = accelerator
    trainer.args = SimpleNamespace(ds3_gather_for_generation=False)
    trainer.processing_class = tokenizer
    trainer.num_generations = G
    trainer.configure_campaign_runtime(
        generation_guard_factory=lambda actual_model: training_configuration_guard(
            actual_model, fast_model.for_training,
        ),
        post_generation_restore=lambda actual_model, actual_tokenizer: assert_post_trainer_pinned_identity(
            actual_model, actual_tokenizer, reference_tokenizer, [record.row for record in selected],
        ),
        prompt_max_tokens=PROMPT_MAX_TOKENS,
        completion_max_tokens=COMPLETION_MAX_TOKENS,
        context_max_tokens=CONTEXT_MAX_TOKENS,
        generation_kwargs={
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
        },
        generation_groups_per_call=GROUPS_PER_CALL,
    )
    device = accelerator.device
    if getattr(device, "type", None) != "cuda":
        raise GateError(f"live P2 memory gate loaded on non-CUDA device: {device}")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        prompt_ids, completions, _logprobs, _forward = trainer._generate_single_turn(prompt_rows)
        torch.cuda.synchronize(device)
    except Exception as error:
        raise GateError(f"P2 longest-TRAIN generation failed: {error}") from error
    elapsed = time.perf_counter() - started
    generation = trainer._campaign_last_generation
    accounting = generation["accounting"]
    if generation.get("generation_groups_per_call") != GROUPS_PER_CALL or generation.get("generation_call_count") != 1:
        raise GateError("patched mixin did not bind the requested P2 single-call geometry")
    if len(prompt_ids) != 8 or len(completions) != 8 or len(generation.get("groups", ())) != 2:
        raise GateError("P2 generation returned the wrong row/group count")
    completion_lengths = [len(row) for row in completions]
    return {
        "status": "complete",
        "completed_at_utc": _now(),
        "geometry": {
            "candidate_count_G": G,
            "generation_groups_per_call": GROUPS_PER_CALL,
            "generation_call_count": generation["generation_call_count"],
            "logical_groups": len(generation["groups"]),
            "returned_rows": len(prompt_ids),
        },
        "selection": plan["selection"],
        "prompt_lengths": [len(row) for row in prompt_ids[::G]],
        "completion_lengths": completion_lengths,
        "completion_cap": COMPLETION_MAX_TOKENS,
        "terminal_accounting": {
            key: accounting.get(key)
            for key in ("records", "generated_tokens", "canonical_eos", "noncanonical_eog", "cap_hits", "invalid_terminal", "padded_after_terminal")
        },
        "generation_seconds": elapsed,
        "memory_after": _device_memory(torch, device),
        "memory_peak": _peak_memory(torch, device),
        "loaded_parent_identity": {
            "manifest_sha256": parent_audit["manifest_sha256"],
            "merged_weights_sha256": parent_audit["merged_weights_sha256"],
            "model_path": parent_audit["model_path"],
        },
        "source": plan["source"],
        "tokenizer_contract": tokenizer_audit,
        "adapter": adapter_audit,
        "interpretation": "P2 longest-TRAIN generation memory/terminal gate only; observed completion lengths do not establish 192-token worst-case capacity",
        "quality_claim": False,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--execution-root", default=DEFAULT_EXECUTION_ROOT)
    parser.add_argument("--rows-path", default=DEFAULT_ROWS)
    parser.add_argument("--sidecar-path", default=DEFAULT_SIDECAR)
    parser.add_argument("--selected-ids-path", default=DEFAULT_SELECTED)
    parser.add_argument("--parent-manifest", default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--native-base-path", default=DEFAULT_NATIVE_BASE)
    parser.add_argument("--trainer-sha256", required=True)
    parser.add_argument("--result-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_path = Path(args.result_path).resolve()
    if result_path.exists():
        print(json.dumps({"status": "refused", "reason": f"result path already exists: {result_path}"}))
        return 2
    try:
        plan = prepare_plan(args)
        if args.preflight:
            _write_json(result_path, plan)
            print(json.dumps({
                "status": "prepared",
                "selected_ids": [item["id"] for item in plan["selection"]],
                "prompt_lengths": [item["prompt_tokens"] for item in plan["selection"]],
                "result_path": str(result_path),
            }, sort_keys=True))
            return 0
        live_result = run_live(args, plan)
        _write_json(result_path, live_result)
        print(json.dumps({
            "status": live_result["status"],
            "selected_ids": [item["id"] for item in live_result["selection"]],
            "prompt_lengths": live_result["prompt_lengths"],
            "completion_lengths": live_result["completion_lengths"],
            "generation_seconds": live_result["generation_seconds"],
            "memory_peak": live_result["memory_peak"],
        }, sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "status": "failed",
            "failed_at_utc": _now(),
            "error": str(error),
            "quality_claim": False,
        }
        _write_json(result_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
