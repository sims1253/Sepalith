#!/usr/bin/env python3
"""Tiny CPU proof for native full-state save, publish, and Trainer resume."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, SequentialSampler
from transformers import Trainer, TrainerCallback, TrainingArguments

HERE = Path(__file__).resolve().parent
TRAINING = HERE / "source/experiments/training"
sys.path.insert(0, str(TRAINING))
from campaign_checkpoint import seal_checkpoint, verify_checkpoint
from full_weight_optimizer import FullWeightOptimizerTrainerMixin, OptimizerConfig
import native_checkpoint_publish
from full_weight_cpt_trainer import deterministic_sequential_dataloader, retain_native_after_durable


class Attention(nn.Module):
    def __init__(self):
        super().__init__(); self.q_proj = nn.Linear(4, 4, bias=False)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.self_attn = Attention(); self.mlp = MLP()


class Body(nn.Module):
    def __init__(self):
        super().__init__(); self.embed_tokens = nn.Embedding(11, 4); self.layers = nn.ModuleList([Block()])


class TinyAuroraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body(); self.final_norm = nn.LayerNorm(4); self.dropout = nn.Dropout(0.2); self.lm_head = nn.Linear(4, 3, bias=False)

    def forward(self, input_ids, labels, **_):
        block = self.model.layers[0]
        x = self.model.embed_tokens(input_ids).mean(1)
        hidden = block.self_attn.q_proj(x)
        hidden = hidden + block.mlp.down_proj(torch.relu(block.mlp.gate_proj(x)) + block.mlp.up_proj(x))
        logits = self.lm_head(self.dropout(self.final_norm(hidden)))
        return {"loss": nn.functional.cross_entropy(logits, labels), "logits": logits}


class Rows(Dataset):
    def __init__(self, positions): self.positions = list(positions)
    def __len__(self): return len(self.positions)
    def __getitem__(self, index):
        p = self.positions[index]
        return {"input_ids": torch.tensor([p % 11, (p + 3) % 11]), "labels": torch.tensor(p % 3), "_draw_position": torch.tensor(p)}


def collate(rows): return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def identity():
    return {"parent": {"id": "tiny"}, "tokenizer": {"sha256": "tiny"}, "renderer": {"kind": "tiny"}, "data": {"id": "rows012"}, "source": {"id": "native-v2"}, "policy": {"optimizer": "aurora_mix"}, "schedule": {"max_steps": 3, "warmup_steps": 1}}


def arguments(output):
    return TrainingArguments(output_dir=str(output), per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=3, learning_rate=3e-4, lr_scheduler_type="constant_with_warmup", warmup_steps=1, max_grad_norm=0.0, use_cpu=True, bf16=False, save_strategy="no", save_only_model=False, save_total_limit=2, logging_strategy="no", eval_strategy="no", report_to="none", disable_tqdm=True, remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=False, train_sampling_strategy="sequential", ignore_data_skip=True, seed=20260915, data_seed=20260915)


class TrackTrainer(FullWeightOptimizerTrainerMixin, Trainer):
    def __init__(self, *args, seen=None, **kwargs): self.seen = seen if seen is not None else []; super().__init__(*args, **kwargs)
    def get_train_dataloader(self): return deterministic_sequential_dataloader(self)
    def _get_train_sampler(self, train_dataset=None): return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
    def compute_loss(self, model, inputs, *args, **kwargs):
        self.seen.extend(int(x) for x in inputs.pop("_draw_position").tolist())
        return super().compute_loss(model, inputs, *args, **kwargs)


class NativeLifecycle(TrainerCallback):
    def __init__(self, durable, save_at, stop_at=None, fail_publish_at=None):
        self.durable = Path(durable); self.save_at = set(save_at); self.stop_at = stop_at; self.fail_publish_at = fail_publish_at

    def on_step_end(self, args, state, control, **_):
        if int(state.global_step) in self.save_at: control.should_save = True
        if int(state.global_step) == self.stop_at: control.should_save = True; control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **_):
        step = int(state.global_step); source = Path(args.output_dir) / f"checkpoint-{step}"
        (source / "tokenizer.json").write_text('{"tiny":true}\n')
        (source / "tokenizer_config.json").write_text('{}\n')
        (source / "config.json").write_text('{}\n')
        sampler = {"method": "sequential_frozen_draw_schedule_stage_local", "cursor": step, "stage_cursor": step, "global_step": step, "global_optimizer_step_offset": 0, "draw_schedule_sha256": "tiny", "effective_batch": 1, "ignore_data_skip": True}
        sealed = seal_checkpoint(source, identity(), step, full=True, sampler=sampler, checkpoint_kind="full_weights")
        destination = self.durable / f"checkpoint-{step}"
        original = native_checkpoint_publish.copy_hash
        if step == self.fail_publish_at:
            calls = {"n": 0}
            def interrupted(src, dst, expected):
                calls["n"] += 1
                if calls["n"] == 2: raise RuntimeError("injected_publish_interruption")
                return original(src, dst, expected)
            native_checkpoint_publish.copy_hash = interrupted
        try:
            native_checkpoint_publish.publish(source, destination, native_checkpoint_publish.digest(source / "campaign-manifest.json"), require_cross_filesystem=False)
        finally:
            native_checkpoint_publish.copy_hash = original
        retain_native_after_durable(args.output_dir, step)
        return control


def set_all_seeds(seed=20260915): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_trainer(root, name, positions, durable, save_at, stop_at=None, fail_publish_at=None, seed=20260915):
    set_all_seeds(seed); seen = []
    trainer = TrackTrainer(model=TinyAuroraModel(), args=arguments(root / name), train_dataset=Rows(positions), data_collator=collate, callbacks=[NativeLifecycle(durable, save_at, stop_at, fail_publish_at)], seen=seen)
    trainer.full_weight_optimizer_config = OptimizerConfig(arm="aurora_mix", hidden_lr=3e-4, side_lr=3e-5, weight_decay=0.0, ns_steps=2, aurora_K=1, stochastic_round_chunk_elements=128)
    return trainer, seen


def state_equal(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def run_proof(root: Path):
    root.mkdir(parents=True)
    direct, direct_seen = make_trainer(root, "direct-native", [0, 1, 2], root / "direct-E", {3})
    direct.train()
    direct_rng = torch.get_rng_state().clone()
    split, first_seen = make_trainer(root, "split-native", [0, 1, 2], root / "split-E", {2}, stop_at=2)
    split.train(); checkpoint2 = root / "split-E/checkpoint-2"
    verify_checkpoint(checkpoint2, identity(), require_full=True, expected_checkpoint_kind="full_weights")
    resumed, resumed_seen = make_trainer(root, "resumed-native", [2], root / "resumed-E", {3}, seed=7)
    resumed.train(resume_from_checkpoint=str(checkpoint2))
    resumed_rng = torch.get_rng_state().clone()
    comparisons = {
        "model": state_equal(direct.model.state_dict(), resumed.model.state_dict()),
        "optimizer": state_equal(direct.optimizer.state_dict(), resumed.optimizer.state_dict()),
        "scheduler": state_equal(direct.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict()),
        "cpu_rng": torch.equal(direct_rng, resumed_rng),
        "draws": direct_seen == [0, 1, 2] and first_seen == [0, 1] and resumed_seen == [2],
    }
    # A failed second publication must leave checkpoint-1 hot and durable.
    failing, failed_seen = make_trainer(root, "failure-native", [0, 1], root / "failure-E", {1, 2}, fail_publish_at=2)
    error = None
    try: failing.train()
    except RuntimeError as exc: error = str(exc)
    failure = {
        "error": error,
        "prior_native_survives": (root / "failure-native/checkpoint-1").is_dir(),
        "prior_durable_survives": (root / "failure-E/checkpoint-1").is_dir(),
        "failed_destination_hidden": not (root / "failure-E/checkpoint-2").exists(),
        "draws": failed_seen,
    }
    if not all(comparisons.values()) or error != "injected_publish_interruption" or not all(failure[k] for k in ("prior_native_survives", "prior_durable_survives", "failed_destination_hidden")):
        raise AssertionError({"comparisons": comparisons, "failure": failure})
    result = {"schema": "sepalith.sft11.native-cpt-resume-proof.v1", "status": "pass", "comparisons": comparisons, "failure_recovery": failure, "optimizer_dispatch": direct.full_weight_optimizer_manifest, "checkpoint2": str(checkpoint2)}
    (root / "proof-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    print(json.dumps(run_proof(args.output), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
