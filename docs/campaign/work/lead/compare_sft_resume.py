"""Compare actual continuous and resumed CUDA checkpoints using CPU only."""
import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from safetensors.torch import load_file
from campaign_checkpoint import digest, verify_checkpoint

torch.set_num_threads(1)


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(equal_state(left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(equal_state(a, b) for a, b in zip(left, right)))
    return bool(left == right)


def tensor_difference(left, right):
    assert left.keys() == right.keys()
    squared_error = squared_reference = 0.0
    max_abs = 0.0
    exact = count = 0
    for key, value in left.items():
        other = right[key]
        assert value.shape == other.shape and value.dtype == other.dtype
        assert torch.isfinite(value).all() and torch.isfinite(other).all()
        exact += int(torch.equal(value, other))
        difference = value.float() - other.float()
        squared_error += difference.double().square().sum().item()
        squared_reference += value.double().square().sum().item()
        max_abs = max(max_abs, difference.abs().max().item())
        count += value.numel()
    return {"tensor_count": len(left), "exact_tensors": exact, "elements": count,
            "max_absolute_difference": max_abs,
            "rms_difference": (squared_error / count) ** 0.5,
            "relative_l2_difference": (squared_error / squared_reference) ** 0.5
            if squared_reference else None}


def compare(left, right):
    a = verify_checkpoint(left, require_full=True)
    b = verify_checkpoint(right, a["identity"], require_full=True)
    assert a["step"] == b["step"]
    weights_a = load_file(str(left / "adapter_model.safetensors"), device="cpu")
    weights_b = load_file(str(right / "adapter_model.safetensors"), device="cpu")
    weights = tensor_difference(weights_a, weights_b)
    lora_b = tensor_difference({k: v for k, v in weights_a.items() if "lora_B" in k},
                               {k: v for k, v in weights_b.items() if "lora_B" in k})
    del weights_a, weights_b
    optimizer_a = torch.load(left / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer_b = torch.load(right / "optimizer.pt", map_location="cpu", weights_only=False)
    assert optimizer_a["state"].keys() == optimizer_b["state"].keys()
    optimizer_steps = all(equal_state(v["step"], optimizer_b["state"][k]["step"])
                          for k, v in optimizer_a["state"].items())
    moments = {}
    for key in ("exp_avg", "exp_avg_sq"):
        moments[key] = tensor_difference({k: v[key] for k, v in optimizer_a["state"].items()},
                                         {k: v[key] for k, v in optimizer_b["state"].items()})
    groups = equal_state(optimizer_a["param_groups"], optimizer_b["param_groups"])
    del optimizer_a, optimizer_b
    load = lambda folder, name: torch.load(folder / name, map_location="cpu", weights_only=False)
    rng_a, rng_b = load(left, "rng_state.pth"), load(right, "rng_state.pth")
    scheduler = equal_state(load(left, "scheduler.pt"), load(right, "scheduler.pt"))
    sampler_a = json.loads((left / "campaign-state.json").read_text())["sampler"]
    sampler_b = json.loads((right / "campaign-state.json").read_text())["sampler"]
    return {"step": a["step"], "left": str(left), "right": str(right),
            "left_manifest_sha256": digest(left / "campaign-manifest.json"),
            "right_manifest_sha256": digest(right / "campaign-manifest.json"),
            "adapter": weights, "lora_B": lora_b, "optimizer_moments": moments,
            "optimizer_steps_equal": optimizer_steps, "optimizer_groups_equal": groups,
            "scheduler_equal": scheduler, "sampler_equal": sampler_a == sampler_b,
            "rng_equal": {k: equal_state(v, rng_b[k]) for k, v in rng_a.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    control = args.root / "SFT-smoke-control/full"
    result = {
        "task": "SFT-04", "status": "measured; lead decision required",
        "independent25": compare(args.root / "SFT-smoke-a2/full/checkpoint-25", control / "checkpoint-25"),
        "resumed50": compare(args.root / "SFT-smoke-b/full/checkpoint-50", control / "checkpoint-50"),
        "interpretation": "Independent step25 repeats establish observed GPU numerical variation. State equality and measured tensor differences must be reviewed; no bit-exact trajectory claim is inferred.",
        "CUDA_started": torch.cuda.is_initialized(),
    }
    assert not result["CUDA_started"]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
