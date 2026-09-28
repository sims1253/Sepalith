#!/usr/bin/env python3
"""Lead-run CUDA parity probe for the installed MiniCPM packed-attention path.

This is a measurement harness, not a production trainer.  It never steps an
optimizer or writes model state.  It compares one fixed 16-row update executed
as sixteen standalone sequences with the same update packed into <=16K
block-diagonal physical groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from varlen_contract import pack_update, verify

EXPECTED_MODEL_SHA256 = "aac456d2481869d1cec9c6e5e693c8807068ecb8a981d78c1384dbb0d4e39701"
EXPECTED_ROWS_SHA256 = "96c5e875e473e53f941318bfd8ba1b2dff8196590e87eb1bfb9c35ae0dc77c4b"
EXPECTED_PARAMETERS = 2_516_756_480


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--token-cap", type=int, default=16_384)
    p.add_argument("--seed", type=int, default=20260914)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def selected_gradient_samples(model, torch):
    families = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                "gate_proj.weight", "up_proj.weight", "down_proj.weight")
    picked = {}
    for name, parameter in model.named_parameters():
        if not any(name.endswith(family) for family in families):
            continue
        layer = next((part for part in name.split(".") if part.isdigit()), None)
        if layer not in {"0", "20", "39"} or parameter.grad is None:
            continue
        count = min(16, parameter.numel())
        idx = (torch.arange(count, device=parameter.device, dtype=torch.int64)
               * (parameter.numel() - 1) // max(1, count - 1))
        picked[name] = parameter.grad.detach().reshape(-1)[idx].float().cpu()
    if not picked:
        raise RuntimeError("no q/k/v/o/MLP gradient samples were found")
    return picked


def compare(a, b, torch) -> dict:
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("nonfinite parity tensor")
    delta = (a.float() - b.float()).abs()
    scale = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-8)
    return {"max_abs": float(delta.max()), "max_relative": float((delta / scale).max())}


def main() -> int:
    args = parse_args()
    if args.report.exists():
        raise ValueError("report path must be fresh")
    if args.token_cap != 16_384:
        raise ValueError("this bounded probe is pinned to the admitted 16K physical cap")
    weights = args.model / "model.safetensors"
    if sha256(weights) != EXPECTED_MODEL_SHA256:
        raise ValueError("model weights differ from selected representative checkpoint 66")
    if sha256(args.rows) != EXPECTED_ROWS_SHA256:
        raise ValueError("row fixture differs from the reviewed 32-row fixture")
    source_rows = [json.loads(x) for x in args.rows.read_text().splitlines() if x.strip()][:16]
    rows = []
    for i, row in enumerate(source_rows):
        if row.get("attention_mask") != [1] * len(row["input_ids"]):
            raise ValueError("fixture contains padding")
        rows.append(dict(row, _draw_position=i))
    groups = pack_update(rows, first_position=0, token_cap=args.token_cap)
    contract = verify(rows, groups, first_position=0)
    report = {
        "schema": "sepalith.sft11.actual_model_varlen_probe.v1",
        "status": "starting",
        "model_weights_sha256": EXPECTED_MODEL_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "contract": contract,
        "no_optimizer_step": True,
        "started_at": time.time(),
    }
    atomic_json(args.report, report)

    # Unsloth must precede torch so the exact installed patch is active.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    from unsloth import FastLanguageModel
    import torch
    from unsloth.utils.attention_dispatch import select_attention_backend

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one root-leased CUDA device")
    expected_allocator = "backend:native,roundup_power2_divisions:[32:256,64:128,256:64,>:32],garbage_collection_threshold:0.8"
    if os.environ.get("PYTORCH_ALLOC_CONF") != expected_allocator:
        raise RuntimeError("allocator policy changed")
    torch.cuda.memory.set_per_process_memory_fraction(0.95, device=0)
    report["allocator"] = {"effective": expected_allocator, "fraction": torch.cuda.memory.get_per_process_memory_fraction(0)}
    backend = select_attention_backend(use_varlen=True)
    if backend not in {"xformers", "flash_varlen"}:
        raise RuntimeError(f"packed throughput probe rejects backend {backend!r}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, _ = FastLanguageModel.from_pretrained(
        model_name=str(args.model), max_seq_length=args.token_cap,
        dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=True,
        float32_mixed_precision=False, fast_inference=False,
        trust_remote_code=False, use_gradient_checkpointing=False,
    )
    from saved_precision import restore_saved_fp32
    precision = restore_saved_fp32(model, weights)
    if precision["fp32_tensors_restored"] != 85 or sum(x["elements"] for x in precision["tensors"]) != 174080:
        raise RuntimeError("saved FP32 precision differs")
    report["saved_precision"] = precision
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_disable()
    except AttributeError:
        pass
    model.train()
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("model parameter count differs")
    device = next(model.parameters()).device

    def tensors(group):
        return {
            "input_ids": torch.tensor(group["input_ids"], dtype=torch.long, device=device),
            "labels": torch.tensor(group["labels"], dtype=torch.long, device=device),
            "position_ids": torch.tensor(group["position_ids"], dtype=torch.long, device=device),
            "packed_seq_lengths": torch.tensor(group["packed_seq_lengths"], dtype=torch.int32, device=device),
        }

    if len(groups) != 1:
        raise RuntimeError("reviewed fixture did not fit in one 16K physical group")
    packed = tensors(groups[0])
    denominator = contract["loss_denominator"]

    # Attention/logit isolation: compare each member to a standalone decode.
    model.eval()
    hidden_deltas, logit_deltas = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        packed_hidden = model.model(
            input_ids=packed["input_ids"], position_ids=packed["position_ids"],
            packed_seq_lengths=packed["packed_seq_lengths"], use_cache=False,
            return_dict=True,
        ).last_hidden_state
        for member, row in zip(groups[0]["members"], rows):
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            standalone = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            section = packed_hidden[:, member["start"]:member["stop"], :]
            hidden_deltas.append(compare(section, standalone, torch))
            points = sorted({0, len(row["input_ids"]) // 2, len(row["input_ids"]) - 1})
            packed_logits = model.lm_head(section[:, points, :])
            standalone_logits = model.lm_head(standalone[:, points, :])
            logit_deltas.append(compare(packed_logits, standalone_logits, torch))
        # A changed first document must not alter the second document at all.
        changed_ids = packed["input_ids"].clone()
        changed_ids[0, 1] = (changed_ids[0, 1] + 17) % model.config.vocab_size
        changed = model.model(
            input_ids=changed_ids, position_ids=packed["position_ids"],
            packed_seq_lengths=packed["packed_seq_lengths"], use_cache=False,
            return_dict=True,
        ).last_hidden_state
        second = groups[0]["members"][1]
        isolation = compare(
            packed_hidden[:, second["start"]:second["stop"], :],
            changed[:, second["start"]:second["stop"], :], torch,
        )
    del packed_hidden, changed

    # Exact objective weighting: sum 16 standalone loss numerators over the
    # common denominator, then compare one packed backward pass.
    FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    model.zero_grad(set_to_none=True)
    standalone_loss = 0.0
    for row in rows:
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            value = model(input_ids=ids, labels=labels, use_cache=False,
                          num_items_in_batch=denominator).loss
        standalone_loss += float(value.detach())
        value.backward()
    standalone_grad = selected_gradient_samples(model, torch)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        packed_loss = model(**packed, use_cache=False, num_items_in_batch=denominator).loss
    packed_loss.backward()
    packed_grad = selected_gradient_samples(model, torch)
    gradient_deltas = {name: compare(standalone_grad[name], packed_grad[name], torch)
                       for name in standalone_grad}
    result = {
        "backend": backend,
        "hidden": {"max_abs": max(x["max_abs"] for x in hidden_deltas),
                   "max_relative": max(x["max_relative"] for x in hidden_deltas)},
        "sampled_logits": {"max_abs": max(x["max_abs"] for x in logit_deltas),
                           "max_relative": max(x["max_relative"] for x in logit_deltas)},
        "cross_document_isolation": isolation,
        "standalone_loss": standalone_loss,
        "packed_loss": float(packed_loss.detach()),
        "loss_abs_delta": abs(standalone_loss - float(packed_loss.detach())),
        "gradient_samples": gradient_deltas,
        "loss_denominator": denominator,
    }
    # Isolation has an exact semantic expectation: member 1 cannot depend on a
    # token in member 0. Other BF16 deltas are measurements. This preparation
    # deliberately does not invent an acceptance tolerance before observing a
    # repeatability floor for each backend.
    if isolation["max_abs"] != 0.0:
        raise RuntimeError("cross-document attention leakage detected")
    report.update(status="measurement_complete_root_parity_decision_required", result=result,
                  completed_at=time.time())
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
