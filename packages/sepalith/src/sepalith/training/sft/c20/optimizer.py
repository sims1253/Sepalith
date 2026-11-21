"""Run the C04 optimizer one parameter at a time with FP32 state parked on CPU."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping

import torch

from sepalith.training.optim.full_weight_optimizer import (
    FullWeightCompositeOptimizer,
    OptimizerConfig,
    inventory_full_weight,
)


class OffloadedOptimizer(FullWeightCompositeOptimizer):
    """Use the existing update kernels and stage only the active parameter's moments."""

    def state_dict(self) -> dict:
        value = super().state_dict()
        value["_sepalith"]["cpu_state_offload"] = True
        return value

    def load_state_dict(self, state_dict: Mapping) -> None:
        if state_dict.get("_sepalith", {}).get("cpu_state_offload") is not True:
            raise ValueError("C20 optimizer state must use CPU offload")

        # The base loader places FP32 state on the device of each destination
        # tensor. Borrow CPU keys for the load, then remap state to the actual
        # parameters once the base loader has validated the checkpoint.
        real_groups = self.param_groups
        stand_ins = [[torch.empty(0, device="cpu") for _ in g["params"]] for g in real_groups]
        self.param_groups = [{**g, "params": keys} for g, keys in zip(real_groups, stand_ins, strict=True)]
        try:
            super().load_state_dict(state_dict)
        except Exception:
            self.param_groups = real_groups
            raise

        state = defaultdict(dict)
        for loaded, real in zip(self.param_groups, real_groups, strict=True):
            for key, param in zip(loaded["params"], real["params"], strict=True):
                if key in self.state:
                    state[param] = self.state[key]
            loaded["params"] = real["params"]
        self.state = state

    def _stage(self, group: Mapping, step: Callable[[Mapping], None]) -> None:
        for index, param in enumerate(group["params"]):
            if param.grad is None:
                continue
            state = self.state[param]
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(param.device)
            one = {**group, "params": [param], "dispatch": group["dispatch"][index:index + 1]}
            try:
                step(one)
            finally:
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.cpu()

    def _step_hidden(self, group: Mapping) -> None:
        self._stage(group, super()._step_hidden)

    def _step_adamw(self, group: Mapping) -> None:
        self._stage(group, super()._step_adamw)


def build_offloaded_optimizer(
    model: torch.nn.Module, config: OptimizerConfig,
) -> tuple[OffloadedOptimizer, dict]:
    inventory, manifest = inventory_full_weight(model, config.arm)
    return OffloadedOptimizer(inventory, manifest, config), manifest
