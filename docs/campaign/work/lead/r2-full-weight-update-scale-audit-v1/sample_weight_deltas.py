#!/usr/bin/env python3
"""Bounded safetensors slice comparison; never reads whole model tensors."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

LAYERS = (0, 21, 41)
MATRICES = {
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "o": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}
NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight")
BLOCK = 32
FRACTIONS = (0.0, 1 / 3, 2 / 3, 1.0)
MAX_LOGICAL_BYTES = 2 * 1024 * 1024


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def starts(size: int) -> list[int]:
    if size < BLOCK:
        raise ValueError(f"tensor dimension {size} is smaller than block {BLOCK}")
    return [round((size - BLOCK) * fraction) for fraction in FRACTIONS]


def metrics(parent: torch.Tensor, candidate: torch.Tensor) -> dict:
    p, c = parent.float(), candidate.float()
    d = c - p
    parent_rms = float(p.square().mean().sqrt())
    delta_rms = float(d.square().mean().sqrt())
    return {
        "elements": p.numel(),
        "parent_rms": parent_rms,
        "candidate_rms": float(c.square().mean().sqrt()),
        "delta_rms": delta_rms,
        "relative_delta_rms": delta_rms / parent_rms if parent_rms else None,
        "delta_mean": float(d.mean()),
        "delta_max_abs": float(d.abs().max()),
        "changed_fraction": float((d != 0).float().mean()),
    }


def sample(parent_path: Path, candidate_path: Path, parent_expected_sha: str, candidate_expected_sha: str) -> dict:
    rows, logical_bytes = [], 0
    with safe_open(parent_path, framework="pt", device="cpu") as parent, safe_open(candidate_path, framework="pt", device="cpu") as candidate:
        if set(parent.keys()) != set(candidate.keys()):
            raise ValueError("parent and candidate tensor key sets differ")
        for layer in LAYERS:
            for family, suffix in MATRICES.items():
                name = f"model.layers.{layer}.{suffix}"
                ps, cs = parent.get_slice(name), candidate.get_slice(name)
                shape = ps.get_shape()
                if shape != cs.get_shape() or len(shape) != 2:
                    raise ValueError(f"shape mismatch for {name}")
                pieces_p, pieces_c, coords = [], [], []
                for r, c in zip(starts(shape[0]), starts(shape[1])):
                    pieces_p.append(ps[r : r + BLOCK, c : c + BLOCK].reshape(-1))
                    pieces_c.append(cs[r : r + BLOCK, c : c + BLOCK].reshape(-1))
                    coords.append([r, c, BLOCK, BLOCK])
                pv, cv = torch.cat(pieces_p), torch.cat(pieces_c)
                logical_bytes += (pv.numel() * pv.element_size()) + (cv.numel() * cv.element_size())
                rows.append({"kind": "matrix", "layer": layer, "family": family, "name": name, "shape": shape, "parent_dtype": str(pv.dtype), "candidate_dtype": str(cv.dtype), "blocks": coords, **metrics(pv, cv)})
            for suffix in NORMS:
                name = f"model.layers.{layer}.{suffix}"
                pv, cv = parent.get_slice(name)[:], candidate.get_slice(name)[:]
                logical_bytes += (pv.numel() * pv.element_size()) + (cv.numel() * cv.element_size())
                rows.append({"kind": "norm", "layer": layer, "family": suffix.removesuffix(".weight"), "name": name, "shape": list(pv.shape), "parent_dtype": str(pv.dtype), "candidate_dtype": str(cv.dtype), "sampling": "complete_small_tensor", **metrics(pv, cv)})
    if logical_bytes > MAX_LOGICAL_BYTES:
        raise ValueError(f"logical payload budget exceeded: {logical_bytes}")
    family_summary = {}
    for family in list(MATRICES) + [x.removesuffix(".weight") for x in NORMS]:
        selected = [r for r in rows if r["family"] == family]
        family_summary[family] = {
            "layers": [r["layer"] for r in selected],
            "geometric_mean_relative_delta_rms": math.exp(sum(math.log(r["relative_delta_rms"]) for r in selected) / len(selected)),
            "min_relative_delta_rms": min(r["relative_delta_rms"] for r in selected),
            "max_relative_delta_rms": max(r["relative_delta_rms"] for r in selected),
        }
    return {
        "schema": "sepalith.full-weight-update-scale-slices.v1",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bindings": {
            "parent": {"path": str(parent_path), "expected_full_file_sha256": parent_expected_sha, "full_hash_recomputed_by_sampler": False, "bytes": parent_path.stat().st_size},
            "candidate": {"path": str(candidate_path), "expected_full_file_sha256": candidate_expected_sha, "full_hash_recomputed_by_sampler": False, "bytes": candidate_path.stat().st_size},
        },
        "sampling": {"layers": list(LAYERS), "matrix_blocks_per_tensor": len(FRACTIONS), "block_shape": [BLOCK, BLOCK], "logical_tensor_payload_bytes": logical_bytes, "logical_payload_limit_bytes": MAX_LOGICAL_BYTES},
        "rows": rows,
        "family_summary": family_summary,
        "rows_sha256": _sha(rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--parent-expected-sha", required=True)
    p.add_argument("--candidate-expected-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit("fresh output required")
    if any(len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in (a.parent_expected_sha, a.candidate_expected_sha)):
        raise SystemExit("expected SHA256 values must be lowercase hex")
    result = sample(a.parent, a.candidate, a.parent_expected_sha, a.candidate_expected_sha)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "logical_bytes": result["sampling"]["logical_tensor_payload_bytes"], "output": str(a.output)}))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
