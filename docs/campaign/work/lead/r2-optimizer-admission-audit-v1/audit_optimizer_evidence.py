#!/usr/bin/env python3
"""CPU/header-only audit of optimizer evidence and MiniCPM parameter geometry."""
from __future__ import annotations
import hashlib, json, math, os
from collections import Counter, defaultdict
from pathlib import Path
from safetensors import safe_open

ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
HERE = Path(__file__).resolve().parent
POC = ROOT / "experiments/training/poc_twin"
CHECKPOINT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-full15006-a/full/checkpoint-560")
MODEL = Path("/mnt/e/sepalith/campaign-20260915/models/SFT11-expanded-postsft500-e-750-merged")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    final_eval = json.loads((POC / "logs/final_eval.json").read_text())["results"]
    census = json.loads((POC / "logs/arms_census.json").read_text())
    mu = census["muon"]; au = census["aurora"]
    safepath = CHECKPOINT / "adapter_model.safetensors"
    tensors = []
    with safe_open(safepath, framework="pt", device="cpu") as f:
        for name in f.keys():
            view = f.get_slice(name); shape = list(view.get_shape()); dtype = str(view.get_dtype())
            factor = "A" if ".lora_A." in name else "B" if ".lora_B." in name else "other"
            projection = next(x for x in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if x in name)
            tensors.append({"name": name, "shape": shape, "dtype": dtype, "factor": factor,
                            "projection": projection, "parameters": math.prod(shape)})
    assert len(tensors) == 588 and sum(x["parameters"] for x in tensors) == 50_233_344
    summary = defaultdict(Counter)
    for x in tensors: summary[x["projection"]][(x["factor"], tuple(x["shape"]), x["dtype"])] += 1
    geometry = {proj: [{"factor": k[0], "shape": list(k[1]), "dtype": k[2], "count": count}
                       for k, count in sorted(items.items())] for proj, items in sorted(summary.items())}
    config = json.loads((MODEL / "config.json").read_text())
    h, i, v, layers = config["hidden_size"], config["intermediate_size"], config["vocab_size"], config["num_hidden_layers"]
    full_shapes = {"q_proj": [h, h], "k_proj": [h // (config["num_attention_heads"] // config["num_key_value_heads"]), h],
                   "v_proj": [h // (config["num_attention_heads"] // config["num_key_value_heads"]), h],
                   "o_proj": [h, h], "gate_proj": [i, h], "up_proj": [i, h], "down_proj": [h, i]}
    hidden_matrix_params = layers * sum(math.prod(shape) for shape in full_shapes.values())
    embedding_params = 2 * v * h
    norm_params = layers * 2 * h + h
    full_params = hidden_matrix_params + embedding_params + norm_params
    manifest = json.loads((CHECKPOINT / "campaign-manifest.json").read_text())
    optimizer_bytes = manifest["files"]["optimizer.pt"]["bytes"]
    expected_adam_moment_bytes = 2 * 4 * sum(x["parameters"] for x in tensors)
    evidence = {
        "schema": "sepalith.sft11.optimizer-admission-evidence.v1", "status": "cpu_header_audit_complete",
        "poc": {
            "muon_vs_adamw": {"muon_bpb": final_eval["muon"]["bpb"], "adamw_bpb": final_eval["adamw"]["bpb"],
                               "relative_muon_improvement": -final_eval["comparison"]["rel_bpb_A_vs_B"],
                               "scored_tokens": final_eval["muon"]["tokens"], "model_parameters": 206_500_000,
                               "training_tokens_per_arm": 681_574_400, "evidence_level": "raw tracked eval JSON"},
            "aurora_vs_muon": {"muon_leverage_dead_fraction": mu["leverage_census"]["mean_dead_both_1pct"],
                                "aurora_leverage_dead_fraction": au["leverage_census"]["mean_dead_both_1pct"],
                                "leverage_dead_reduction_factor": mu["leverage_census"]["mean_dead_both_1pct"] / au["leverage_census"]["mean_dead_both_1pct"],
                                "muon_near_dead_activation_fraction": mu["activation_census"]["mean_near_dead_5pct_of_median"],
                                "aurora_near_dead_activation_fraction": au["activation_census"]["mean_near_dead_5pct_of_median"],
                                "activation_near_dead_reduction_factor": mu["activation_census"]["mean_near_dead_5pct_of_median"] / au["activation_census"]["mean_near_dead_5pct_of_median"],
                                "bpb_claim": "dashboard says approximately 1% at checkpoints",
                                "bpb_auditability": "not independently reproducible from current repo: RESULTS-arms.md remains unfilled and Aurora eval/log checkpoint is absent",
                                "evidence_level": "raw tracked census plus dashboard summary"},
            "soap": {"implementation_present": False, "local_training_result_present": False,
                     "board_summary": "external standalone result 3.2561 vs 3.271 at 7000 iterations on a 124M/3.67B-token setting; reported 5-10% step overhead and high memory",
                     "evidence_level": "unverified external-intel note only"},
        },
        "current_lora": {
            "checkpoint": str(CHECKPOINT), "adapter_sha256": manifest["files"]["adapter_model.safetensors"]["sha256"],
            "adapter_tensors": len(tensors), "trainable_parameters": sum(x["parameters"] for x in tensors),
            "dtype": "F32", "layers": layers, "rank": 32, "target_projections": sorted(summary),
            "geometry": geometry, "optimizer_file_bytes": optimizer_bytes,
            "two_fp32_adam_moments_payload_bytes": expected_adam_moment_bytes,
            "optimizer_overhead_beyond_raw_moments_bytes": optimizer_bytes - expected_adam_moment_bytes,
            "aurora_original_name_dispatch_matches_lora_names": False,
            "aurora_geometry_eligible_if_explicitly_reimplemented": {"projection": ["gate_proj", "up_proj"], "factor": "B", "shape": [6144, 32], "tensors": 84},
        },
        "full_weight_geometry": {"config_sha256": sha(MODEL / "config.json"), "layers": layers,
            "projection_shapes_out_by_in": full_shapes, "hidden_matrix_parameters": hidden_matrix_params,
            "input_plus_output_embedding_parameters": embedding_params, "norm_parameters": norm_params,
            "derived_total_parameters_without_biases": full_params,
            "aurora_dispatch": {"gate_proj": [i, h], "up_proj": [i, h], "matrices": layers * 2},
            "muon_dispatch": {"projections": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"], "matrices": layers * 5},
            "side_adamw_dispatch": ["model.embed_tokens.weight", "lm_head.weight", "all RMSNorm vectors", "any non-2D trainables"]},
        "hashes": {str(p.relative_to(ROOT)): sha(p) for p in (
            POC / "RESULTS.md", POC / "RESULTS-arms.md", POC / "logs/final_eval.json",
            POC / "logs/arms_census.json", POC / "muon.py", POC / "arms/aurora.py",
            POC / "arms/train_arm.py", ROOT / "experiments/dashboard/dashboard_state.json",
            ROOT / "comms/board.md")},
        "safety": {"tensor_payload_loaded": False, "safetensors_headers_only": True,
                   "gpu_used": False, "training_or_shared_source_changed": False},
    }
    (HERE / "evidence-audit.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
