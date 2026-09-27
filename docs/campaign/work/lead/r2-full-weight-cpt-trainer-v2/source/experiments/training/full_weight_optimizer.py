"""Trainer-compatible full-weight AdamW/Muon/Aurora optimizer preparation.

The implementation is deliberately MiniCPM-specific and fail-closed.  It owns
one PyTorch Optimizer so Trainer schedulers and checkpoint code see ordinary
``param_groups`` and ``state_dict`` objects.  Hybrid arms contain exactly one
hidden-matrix group plus one AdamW side group.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor


HIDDEN_SUFFIXES = (
    ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
)
AURORA_SUFFIXES = (".mlp.gate_proj.weight", ".mlp.up_proj.weight")
SIDE_2D_SUFFIXES = (".embed_tokens.weight", "lm_head.weight")
VALID_ARMS = ("adamw_anchor", "muon_hybrid", "aurora_mix")


@dataclass(frozen=True)
class OptimizerConfig:
    arm: str
    hidden_lr: float
    side_lr: float
    weight_decay: float = 0.0
    momentum: float = 0.95
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    ns_steps: int = 5
    aurora_K: int = 2
    aurora_beta: float = 0.5
    rms_scale: float = 0.2
    state_dtype: str = "float32"
    bf16_update_policy: str = "stochastic_round"
    stochastic_round_chunk_elements: int = 1_048_576

    def validate(self) -> None:
        if self.arm not in VALID_ARMS:
            raise ValueError(f"unsupported optimizer arm: {self.arm}")
        if self.state_dtype != "float32":
            raise ValueError("optimizer momentum and moments must be float32")
        if self.bf16_update_policy != "stochastic_round":
            raise ValueError("BF16 full-weight updates require stochastic_round")
        if self.stochastic_round_chunk_elements <= 0:
            raise ValueError("stochastic_round_chunk_elements must be positive")
        if min(self.hidden_lr, self.side_lr) <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.weight_decay:
            raise ValueError("weight_decay must be nonnegative")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0,1)")


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _has_peft_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in ("lora_", ".adapter", "modules_to_save"))


def _classify(name: str, param: Tensor, arm: str) -> str:
    if _has_peft_name(name):
        raise ValueError(f"PEFT parameter forbidden in full-weight arm: {name}")
    is_hidden = any(name.endswith(suffix) for suffix in HIDDEN_SUFFIXES)
    if is_hidden:
        if param.ndim != 2:
            raise ValueError(f"hidden projection is not 2-D: {name} {tuple(param.shape)}")
        if arm == "adamw_anchor":
            return "adamw"
        if arm == "aurora_mix" and any(name.endswith(suffix) for suffix in AURORA_SUFFIXES):
            if param.shape[0] <= param.shape[1]:
                raise ValueError(f"Aurora projection is not tall: {name} {tuple(param.shape)}")
            return "aurora"
        return "muon"
    if param.ndim == 2 and not any(name.endswith(suffix) for suffix in SIDE_2D_SUFFIXES):
        raise ValueError(f"unknown trainable 2-D parameter; dispatch must be reviewed: {name}")
    return "adamw"


def inventory_full_weight(model: torch.nn.Module, arm: str) -> tuple[list[tuple[str, Tensor, str]], dict]:
    if arm not in VALID_ARMS:
        raise ValueError(f"unsupported optimizer arm: {arm}")
    named = list(model.named_parameters(remove_duplicate=False))
    if not named:
        raise ValueError("model has no parameters")
    frozen = [name for name, p in named if not p.requires_grad]
    if frozen:
        raise ValueError(f"full-weight arm contains frozen parameters: {frozen[:4]}")
    names = [name for name, _ in named]
    if len(names) != len(set(names)):
        raise ValueError("duplicate parameter names")
    ids = [id(param) for _, param in named]
    if len(ids) != len(set(ids)):
        raise ValueError("tied/duplicated parameter objects require an explicit reviewed policy")
    rows = []
    inventory = []
    for name, param in named:
        dispatch = _classify(name, param, arm)
        inventory.append((name, param, dispatch))
        rows.append({
            "name": name, "shape": list(param.shape), "dtype": str(param.dtype),
            "requires_grad": bool(param.requires_grad), "dispatch": dispatch,
            "parameters": int(param.numel()),
        })
    manifest = {
        "schema": "sepalith.full-weight-optimizer-dispatch.v1", "arm": arm,
        "ordered_rows": rows, "ordered_rows_sha256": _canonical_sha(rows),
        "parameter_objects": len(ids), "parameters": sum(row["parameters"] for row in rows),
        "counts": {kind: sum(row["dispatch"] == kind for row in rows)
                   for kind in ("adamw", "muon", "aurora")},
    }
    return inventory, manifest


def zeropower_fp32(momentum: Tensor, steps: int) -> Tensor:
    """Newton-Schulz polar update with FP32 input, intermediates, and output."""
    if momentum.ndim != 2:
        raise ValueError("Muon requires a 2-D tensor")
    x = momentum.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


def aurora_fp32(momentum: Tensor, steps: int, K: int, beta: float) -> Tensor:
    """Tall gate/up Aurora path; all arithmetic and temporary D are FP32."""
    if momentum.ndim != 2 or momentum.shape[0] <= momentum.shape[1]:
        raise ValueError("Aurora requires a tall 2-D tensor")
    m, n = momentum.shape
    target = math.sqrt(n / m)
    x = momentum.float()
    x = x / (x.norm() + 1e-7)
    d = torch.ones(m, device=x.device, dtype=torch.float32)
    for _ in range(K):
        row_norm = x.norm(dim=1)
        d = d.pow(beta) * row_norm.clamp_min(1e-7).pow(1.0 - beta)
        x = zeropower_fp32((target / d).unsqueeze(1) * x, steps)
    return x


@torch.no_grad()
def apply_fp32_update_(param: Tensor, update: Tensor, *, chunk_elements: int) -> None:
    """Apply an FP32 update, stochastically rounding BF16 storage in chunks.

    Deterministic round-to-nearest can permanently erase an update below half
    a BF16 ULP.  Stochastic rounding chooses one of the two adjacent BF16 values
    with probability proportional to distance, making the stored update
    unbiased without a persistent FP32 master-weight copy.  Campaign RNG state
    is therefore part of checkpoint identity.
    """
    if update.dtype != torch.float32:
        raise ValueError("assembled update must be float32")
    if param.dtype == torch.float32:
        param.sub_(update)
        return
    if param.dtype != torch.bfloat16:
        raise ValueError(f"unsupported full-weight parameter dtype: {param.dtype}")
    pflat, uflat = param.view(-1), update.view(-1)
    pos_inf = torch.tensor(float("inf"), dtype=torch.bfloat16, device=param.device)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.bfloat16, device=param.device)
    for start in range(0, param.numel(), chunk_elements):
        stop = min(start + chunk_elements, param.numel())
        target = pflat[start:stop].float().sub(uflat[start:stop])
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError("nonfinite full-weight update target")
        nearest_bf16 = target.to(torch.bfloat16)
        nearest = nearest_bf16.float()
        toward = torch.where(nearest > target, neg_inf, pos_inf)
        adjacent_bf16 = torch.nextafter(nearest_bf16, toward)
        adjacent = adjacent_bf16.float()
        lower = torch.minimum(nearest, adjacent)
        upper = torch.maximum(nearest, adjacent)
        span = upper - lower
        probability_upper = torch.where(span > 0, (target - lower) / span, torch.zeros_like(span))
        rounded = torch.where(torch.rand_like(probability_upper) < probability_upper,
                              upper.to(torch.bfloat16), lower.to(torch.bfloat16))
        pflat[start:stop].copy_(rounded)


class FullWeightCompositeOptimizer(torch.optim.Optimizer):
    """One serializable optimizer with hidden custom and AdamW side groups.

    Parameters may be BF16, while gradients, Muon/Aurora momentum, Adam first
    and second moments, orthogonalization, bias correction, and update assembly
    are FP32.  The final FP32 update is applied to the model parameter's storage
    dtype using chunked stochastic rounding.  No hidden BF16 momentum/moment
    state or deterministic round-to-nearest update is allowed.  Exact resume
    consequently requires campaign CPU/CUDA RNG restoration before the next
    optimizer step.
    """

    def __init__(self, inventory: Sequence[tuple[str, Tensor, str]], manifest: Mapping,
                 config: OptimizerConfig):
        config.validate()
        self.config = config
        self.dispatch_manifest = json.loads(json.dumps(manifest))
        hidden = [(n, p, d) for n, p, d in inventory if d in ("muon", "aurora")]
        side = [(n, p, d) for n, p, d in inventory if d == "adamw"]
        groups = []
        if hidden:
            groups.append({
                "params": [p for _, p, _ in hidden], "param_names": [n for n, _, _ in hidden],
                "dispatch": [d for _, _, d in hidden], "kind": "hidden_custom",
                "lr": config.hidden_lr, "initial_lr": config.hidden_lr,
                "weight_decay": config.weight_decay, "momentum": config.momentum,
                "ns_steps": config.ns_steps, "aurora_K": config.aurora_K,
                "aurora_beta": config.aurora_beta, "rms_scale": config.rms_scale,
            })
        if side:
            groups.append({
                "params": [p for _, p, _ in side], "param_names": [n for n, _, _ in side],
                "dispatch": [d for _, _, d in side], "kind": "side_adamw",
                "lr": config.side_lr, "initial_lr": config.side_lr,
                "weight_decay": config.weight_decay, "betas": config.adam_betas,
                "eps": config.adam_eps,
            })
        super().__init__(groups, defaults={})
        if sum(len(g["params"]) for g in self.param_groups) != len(inventory):
            raise AssertionError("parameters were lost while constructing groups")
        if config.arm != "adamw_anchor" and len(hidden) and len(self.param_groups) != 2:
            raise AssertionError("hybrid optimizer must expose exactly one hidden and one side group")

    def _structure(self, groups: Sequence[Mapping]) -> list[dict]:
        return [{"kind": g["kind"], "param_names": list(g["param_names"]),
                 "dispatch": list(g["dispatch"])} for g in groups]

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["_sepalith"] = {
            "schema": "sepalith.full-weight-optimizer-state.v1",
            "arm": self.config.arm,
            "config": asdict(self.config),
            "dispatch_manifest_sha256": self.dispatch_manifest["ordered_rows_sha256"],
            "group_structure": self._structure(state["param_groups"]),
        }
        return state

    def load_state_dict(self, state_dict: Mapping) -> None:
        meta = state_dict.get("_sepalith")
        if not isinstance(meta, Mapping) or meta.get("schema") != "sepalith.full-weight-optimizer-state.v1":
            raise ValueError("optimizer state lacks Sepalith full-weight identity")
        if meta.get("arm") != self.config.arm:
            raise ValueError("optimizer arm differs")
        if meta.get("config") != asdict(self.config):
            raise ValueError("optimizer configuration differs")
        if meta.get("dispatch_manifest_sha256") != self.dispatch_manifest["ordered_rows_sha256"]:
            raise ValueError("ordered full-weight dispatch manifest differs")
        incoming_structure = self._structure(state_dict["param_groups"])
        current_structure = self._structure(self.param_groups)
        if incoming_structure != current_structure or meta.get("group_structure") != incoming_structure:
            raise ValueError("optimizer parameter group structure differs")
        # torch.optim.Optimizer.load_state_dict casts floating state to the
        # parameter dtype.  That would silently turn our FP32 state into BF16,
        # so perform the already-validated id mapping without that cast.
        saved_groups = deepcopy(state_dict["param_groups"])
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("optimizer parameter group count differs")
        if any(len(saved["params"]) != len(current["params"])
               for saved, current in zip(saved_groups, self.param_groups)):
            raise ValueError("optimizer parameter group length differs")
        id_map = {}
        for saved, current in zip(saved_groups, self.param_groups):
            id_map.update(zip(saved["params"], current["params"]))

        def restore(value, param):
            if torch.is_tensor(value):
                dtype = torch.float32 if value.is_floating_point() else value.dtype
                return value.detach().to(device=param.device, dtype=dtype).clone()
            if isinstance(value, dict):
                return {k: restore(v, param) for k, v in value.items()}
            if isinstance(value, list):
                return [restore(v, param) for v in value]
            if isinstance(value, tuple):
                return tuple(restore(v, param) for v in value)
            return deepcopy(value)

        restored_state = defaultdict(dict)
        for saved_id, values in state_dict["state"].items():
            if saved_id not in id_map:
                raise ValueError("optimizer state contains an unbound parameter id")
            param = id_map[saved_id]
            restored_state[param] = restore(values, param)
        new_groups = []
        for saved, current in zip(saved_groups, self.param_groups):
            saved["params"] = current["params"]
            new_groups.append(saved)
        self.__setstate__({"state": restored_state, "param_groups": new_groups})
        for per_param in self.state.values():
            for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                if key in per_param and per_param[key].dtype != torch.float32:
                    raise ValueError(f"restored {key} is not float32")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["kind"] == "hidden_custom":
                self._step_hidden(group)
            elif group["kind"] == "side_adamw":
                self._step_adamw(group)
            else:
                raise RuntimeError(f"unknown optimizer group kind: {group['kind']}")
        return loss

    def _step_hidden(self, group: Mapping) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        for param, dispatch in zip(group["params"], group["dispatch"]):
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
            momentum = state["momentum_buffer"]
            momentum.lerp_(grad, 1.0 - group["momentum"])
            if dispatch == "aurora":
                direction = aurora_fp32(momentum, group["ns_steps"], group["aurora_K"], group["aurora_beta"])
            elif dispatch == "muon":
                direction = zeropower_fp32(momentum, group["ns_steps"])
            else:
                raise RuntimeError(f"non-custom dispatch in hidden group: {dispatch}")
            scale = group["rms_scale"] * math.sqrt(max(param.shape))
            update = direction.mul(lr * scale)
            if wd:
                update.add_(param.detach().float(), alpha=lr * wd)
            apply_fp32_update_(param, update,
                               chunk_elements=self.config.stochastic_round_chunk_elements)

    def _step_adamw(self, group: Mapping) -> None:
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            bias1 = 1.0 - beta1 ** state["step"]
            bias2 = 1.0 - beta2 ** state["step"]
            # Embedding/lm_head matrices are each about 1 GiB in FP32.  Work
            # in bounded flat chunks so bias correction and sqrt do not create
            # another whole-matrix temporary.
            pflat, gflat = param.view(-1), param.grad.detach().view(-1)
            mflat, vflat = exp_avg.view(-1), exp_avg_sq.view(-1)
            chunk = self.config.stochastic_round_chunk_elements
            for start in range(0, param.numel(), chunk):
                stop = min(start + chunk, param.numel())
                grad = gflat[start:stop].float()
                mean, variance = mflat[start:stop], vflat[start:stop]
                mean.lerp_(grad, 1.0 - beta1)
                variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                update = mean.div(bias1) / (variance.div(bias2).sqrt().add_(eps))
                update.mul_(lr)
                if wd:
                    update.add_(pflat[start:stop].float(), alpha=lr * wd)
                apply_fp32_update_(pflat[start:stop], update, chunk_elements=chunk)


def build_full_weight_optimizer(model: torch.nn.Module, config: OptimizerConfig):
    inventory, manifest = inventory_full_weight(model, config.arm)
    return FullWeightCompositeOptimizer(inventory, manifest, config), manifest


def install_on_trainer(trainer, config: OptimizerConfig) -> dict:
    """Install before Trainer creates its optimizer or LR scheduler."""
    if getattr(trainer, "optimizer", None) is not None:
        raise ValueError("Trainer optimizer already exists")
    if getattr(trainer, "lr_scheduler", None) is not None:
        raise ValueError("Trainer LR scheduler already exists")
    trainer.optimizer, manifest = build_full_weight_optimizer(trainer.model, config)
    trainer.full_weight_optimizer_manifest = manifest
    return manifest


class FullWeightOptimizerTrainerMixin:
    """Place before ``transformers.Trainer`` in the campaign Trainer MRO."""

    full_weight_optimizer_config: OptimizerConfig
    full_weight_optimizer_manifest: dict | None = None

    def create_optimizer(self):
        if self.optimizer is None:
            if not isinstance(self.full_weight_optimizer_config, OptimizerConfig):
                raise TypeError("full_weight_optimizer_config must be set before Trainer.create_optimizer")
            self.optimizer, self.full_weight_optimizer_manifest = build_full_weight_optimizer(
                self.model, self.full_weight_optimizer_config
            )
        return self.optimizer
