"""DDP contracts for an 8xA100 continuation of the exact full-weight CPT state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import torch

from full_weight_optimizer import FullWeightCompositeOptimizer, OptimizerConfig, inventory_full_weight

SCHEMA = "sepalith.sft11.a100-ddp-runtime.v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def rounding_seed(base_seed: int, global_step: int) -> int:
    require(type(base_seed) is int and 0 <= base_seed < 2**63, "rounding base seed differs")
    require(type(global_step) is int and global_step >= 0, "rounding global step differs")
    digest = hashlib.sha256(f"sepalith-ddp-rounding-v1:{base_seed}:{global_step}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class DistributedFullWeightCompositeOptimizer(FullWeightCompositeOptimizer):
    """Use one shared stochastic-rounding stream without changing rank model RNG."""
    def __init__(self, inventory, manifest, config: OptimizerConfig, *,
                 rounding_base_seed: int, completed_global_step: int,
                 world_size: int = 8):
        require(type(world_size) is int and world_size > 1, "distributed world size differs")
        require(type(completed_global_step) is int and completed_global_step >= 0,
                "completed global step differs")
        super().__init__(inventory, manifest, config)
        self.rounding_base_seed = rounding_base_seed
        self.completed_global_step = completed_global_step
        self.distributed_world_size = world_size

    def step(self, closure=None):
        seed = rounding_seed(self.rounding_base_seed, self.completed_global_step + 1)
        devices = []
        if any(p.is_cuda for g in self.param_groups for p in g["params"]):
            require(torch.cuda.is_available(), "CUDA parameter without CUDA runtime")
            devices = [torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            result = super().step(closure)
        self.completed_global_step += 1
        return result

    def state_dict(self):
        state = super().state_dict()
        state["_sepalith_distributed"] = {
            "schema": SCHEMA, "world_size": self.distributed_world_size,
            "rounding_base_seed": self.rounding_base_seed,
            "completed_global_step": self.completed_global_step,
        }
        return state

    def load_state_dict(self, state_dict: Mapping, *, admitted_source_step: int | None = None):
        meta = state_dict.get("_sepalith_distributed")
        if meta is None:
            require(admitted_source_step == self.completed_global_step,
                    "local optimizer source step differs")
        else:
            require(meta == {"schema": SCHEMA, "world_size": self.distributed_world_size,
                             "rounding_base_seed": self.rounding_base_seed,
                             "completed_global_step": self.completed_global_step},
                    "distributed optimizer metadata differs")
        base = dict(state_dict); base.pop("_sepalith_distributed", None)
        super().load_state_dict(base)


def build_distributed_optimizer(model, config: OptimizerConfig, *, base_seed: int,
                                completed_global_step: int, world_size: int = 8):
    inventory, manifest = inventory_full_weight(model, config.arm)
    return DistributedFullWeightCompositeOptimizer(
        inventory, manifest, config, rounding_base_seed=base_seed,
        completed_global_step=completed_global_step, world_size=world_size), manifest


def ddp_training_arguments(*, output_dir: str, max_steps: int, learning_rate: float,
                           warmup_steps: int, seed: int) -> dict:
    """Exact geometry: two rows/rank times eight ranks is one 16-row update."""
    return {"output_dir": output_dir, "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1, "max_steps": max_steps,
            "learning_rate": learning_rate, "lr_scheduler_type": "constant_with_warmup",
            "warmup_steps": warmup_steps, "bf16": True, "ddp_find_unused_parameters": False,
            "average_tokens_across_devices": True, "remove_unused_columns": False,
            "dataloader_num_workers": 0, "dataloader_pin_memory": False,
            "train_sampling_strategy": "sequential", "ignore_data_skip": True,
            "save_strategy": "no", "report_to": "none", "seed": seed, "data_seed": seed}


def validate_ddp_window(local_positions: Sequence[int], *, initial_cursor: int,
                        local_update: int, rank: int, world_size: int = 8) -> list[int]:
    require(world_size == 8 and 0 <= rank < world_size, "rank topology differs")
    require(len(local_positions) == 2 and all(type(x) is int for x in local_positions),
            "each rank must own exactly two draws")
    first = initial_cursor + local_update * 16 + rank * 2
    expected = [first, first + 1]
    require(list(local_positions) == expected, "rank draw positions differ")
    return expected


def validate_gathered_window(per_rank: Sequence[Sequence[int]], *, first: int) -> list[int]:
    require(len(per_rank) == 8, "gathered rank count differs")
    flat = [value for values in per_rank for value in values]
    require(flat == list(range(first, first + 16)), "global 16-draw window differs")
    return flat


def global_supervised_denominator(local_count: torch.Tensor, *, world_size: int = 8) -> torch.Tensor:
    require(local_count.numel() == 1 and local_count.dtype in (torch.int32, torch.int64),
            "local supervised count differs")
    total = local_count.clone()
    if torch.distributed.is_initialized():
        require(torch.distributed.get_world_size() == world_size, "distributed world size differs")
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
    return total


def create_initial_resume_view(source: Path, destination: Path, *, manifest_sha256: str,
                               world_size: int = 8) -> dict:
    """Create a fresh hard-link view with rank RNG aliases; never edit source."""
    source, destination = Path(source), Path(destination)
    require(world_size == 8, "resume-view world size differs")
    manifest_path = source / "campaign-manifest.json"
    require(source.is_dir() and manifest_path.is_file(), "source checkpoint missing")
    require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_sha256,
            "source checkpoint manifest differs")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("checkpoint_kind") == "full_weights" and manifest.get("full") is True,
            "source checkpoint is not full state")
    require((source / "rng_state.pth").is_file(), "canonical source RNG missing")
    require(not destination.exists(), "resume view already exists")
    destination.mkdir(parents=True); linked = []
    try:
        for name in manifest["files"]:
            src, dst = source / name, destination / name
            require(src.is_file() and not src.is_symlink(), f"source checkpoint file differs:{name}")
            os.link(src, dst); linked.append(name)
        for rank in range(world_size):
            name = f"rng_state_{rank}.pth"; os.link(source / "rng_state.pth", destination / name); linked.append(name)
        receipt = {"schema": "sepalith.sft11.a100-initial-resume-view.v1",
                   "source": str(source.resolve()), "source_manifest_sha256": manifest_sha256,
                   "source_step": manifest["step"], "world_size": world_size,
                   "rank_rng_initialization": "all ranks start from admitted canonical source RNG",
                   "linked_files": linked}
        (destination / "resume-view.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
        return receipt
    except Exception:
        shutil.rmtree(destination, ignore_errors=True); raise


def checkpoint_rng_contract(directory: Path, *, world_size: int = 8) -> dict:
    directory = Path(directory); records = {}
    for rank in range(world_size):
        path = directory / f"rng_state_{rank}.pth"
        require(path.is_file(), f"rank RNG missing:{rank}")
        records[str(rank)] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = directory / "rng_state.pth"
    if not canonical.exists(): os.link(directory / "rng_state_0.pth", canonical)
    require(hashlib.sha256(canonical.read_bytes()).hexdigest() == records["0"],
            "canonical single-device RNG does not equal rank0")
    return {"schema": SCHEMA, "world_size": world_size,
            "rank_rng_sha256": records, "single_device_resume_rng": "rng_state.pth == rank0"}


class A100DistributedTrainerMixin:
    """Mixin placed before Transformers Trainer for exact DDP window ownership."""
    ddp_initial_cursor: int
    ddp_initial_global_step: int
    ddp_rounding_base_seed: int
    full_weight_optimizer_config: OptimizerConfig

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer, self.full_weight_optimizer_manifest = build_distributed_optimizer(
                self.model, self.full_weight_optimizer_config,
                base_seed=self.ddp_rounding_base_seed,
                completed_global_step=self.ddp_initial_global_step, world_size=8)
        return self.optimizer

    def compute_loss(self, model, inputs, *args, **kwargs):
        positions = [int(x) for x in inputs.pop("_draw_position").detach().cpu().tolist()]
        rank = torch.distributed.get_rank(); world = torch.distributed.get_world_size()
        local_update = int(self.state.global_step) - self.ddp_initial_global_step
        validate_ddp_window(positions, initial_cursor=self.ddp_initial_cursor,
                            local_update=local_update, rank=rank, world_size=world)
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, positions)
        validate_gathered_window(gathered,
            first=self.ddp_initial_cursor + local_update * 16)
        # Transformers 5.5 supplies the gathered shifted-label count as
        # num_items_in_batch and scales local loss by world size when
        # average_tokens_across_devices is true. Keep that production path.
        require(kwargs.get("num_items_in_batch") is not None,
                "Trainer omitted global supervised-token denominator")
        return super().compute_loss(model, inputs, *args, **kwargs)


def publish_distributed_checkpoint(source: Path, archive_destination: Path, *,
                                   identity: Mapping, step: int, sampler: Mapping,
                                   seal_checkpoint, publish_sealed_checkpoint) -> dict | None:
    """All-rank barrier, rank0 atomic publication, then all-rank release."""
    require(torch.distributed.is_initialized() and torch.distributed.get_world_size() == 8,
            "checkpoint publication requires the admitted DDP8 group")
    torch.distributed.barrier()
    result = None
    if torch.distributed.get_rank() == 0:
        rng = checkpoint_rng_contract(source, world_size=8)
        runtime = {**rng, "step": step, "sampler": dict(sampler),
                   "optimizer_rounding": "shared deterministic per global step",
                   "portable_single_device": True}
        (Path(source) / "distributed-runtime.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        sealed = seal_checkpoint(source, identity, step, full=True, sampler=sampler,
                                 checkpoint_kind="full_weights")
        destination = publish_sealed_checkpoint(source, archive_destination, sealed)
        result = {"destination": str(destination), "manifest": sealed,
                  "distributed_runtime": runtime}
    torch.distributed.barrier()
    return result
