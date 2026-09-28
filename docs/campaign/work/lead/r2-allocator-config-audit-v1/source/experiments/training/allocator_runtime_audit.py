"""Read-only allocator configuration evidence for a production run."""
from __future__ import annotations
import importlib.metadata
import os
from typing import Any

_KEYS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")

def capture_before_unsloth() -> dict[str, Any]:
    return {
        "environment": {key: os.environ.get(key) for key in _KEYS},
        "wsl_distro_name_present": bool(os.environ.get("WSL_DISTRO_NAME")),
        "wsl_interop_present": bool(os.environ.get("WSL_INTEROP")),
        "unsloth_disable_alloc_fallback": os.environ.get("UNSLOTH_DISABLE_ALLOC_FALLBACK"),
        "unsloth_vllm_standby": os.environ.get("UNSLOTH_VLLM_STANDBY", "0"),
        "torch_distribution_version": importlib.metadata.version("torch"),
        "unsloth_distribution_version": importlib.metadata.version("unsloth"),
        "unsloth_zoo_distribution_version": importlib.metadata.version("unsloth_zoo"),
    }

def observe_after_unsloth(torch_module: Any, before: dict[str, Any]) -> dict[str, Any]:
    after = {key: os.environ.get(key) for key in _KEYS}
    backend = torch_module.cuda.memory.get_allocator_backend()
    initialized = bool(torch_module.cuda.is_initialized())
    requested = before["environment"].get("PYTORCH_ALLOC_CONF")
    effective = after.get("PYTORCH_ALLOC_CONF")
    wsl = bool(before["wsl_distro_name_present"] or before["wsl_interop_present"])
    if wsl and requested and "expandable_segments:True" in requested and (not effective or "expandable_segments:True" not in effective):
        interpretation = "expandable_segments_requested_but_removed_by_unsloth_wsl_policy"
    elif requested == effective:
        interpretation = "requested_unified_allocator_config_preserved"
    else:
        interpretation = "allocator_environment_changed_during_unsloth_import"
    return {
        "schema": "sepalith.sft11.cuda_allocator_runtime.v1",
        "before_unsloth": before,
        "after_unsloth_environment": after,
        "allocator_backend": backend,
        "cuda_initialized_at_observation": initialized,
        "interpretation": interpretation,
        "expandable_segments_effective_by_environment": bool(effective and "expandable_segments:True" in effective),
        "roundup_fallback_effective_by_environment": bool(effective and "roundup_power2_divisions:" in effective),
        "note": "backend=native does not itself distinguish native expandable-segment subconfiguration",
    }
