#!/usr/bin/env python3
"""C18 in-run forgetting probe: CPT 2K held-out causal NLL of one checkpoint (night only, ``cuda:0``).

The panel and evaluator are those of ``docs/training/plan-20260927/c09/cpt_forgetting.py`` (the CPT
2K validation file, 499 rows, ``campaign_cpt_eval.package_holdout_evaluator``, batch 1, at most 2,048
tokens, bf16 autocast). The model loads in plain bf16 with no FP32 norm restore, exactly as the
GRPO trainer and the DEV evaluator load it, so the probe of the starting model (``--start``) equals
the trainer's update 0 and the R6x row measures training drift only. (arm A 803's recorded 1.00341
uses the c09 harness with its saved FP32 norms restored; the result reports the difference to it too.)

    python -m sepalith.training.grpo_exec.cpt_probe --bound-recipe R --checkpoint C --output OUT.json
    python -m sepalith.training.grpo_exec.cpt_probe --bound-recipe R --start --output OUT.json
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import time
from typing import cast

from sepalith.training.grpo_exec import recipe as recipe_module

Json = dict[str, object]
RESULT_SCHEMA = "sepalith.c18.cpt-probe-result.v1"
DEVICE = "cuda:0"
EXPECTED_TENSORS, EXPECTED_PARAMETERS = 381, 2_516_756_480


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bound-recipe", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint", type=Path)
    target.add_argument("--start", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    recipe_path, output = args.bound_recipe.resolve(), args.output.resolve()
    recipe = recipe_module.load(recipe_path, check_data=False)
    recipe_sha = recipe_module.sha256(recipe_path)
    require(output.is_absolute() and not output.exists(), "fresh result required")
    output.parent.mkdir(parents=True, exist_ok=True)
    cpt = cast(Json, cast(Json, recipe["dev_gate"])["cpt"])
    panel = cast(Json, cpt["panel"])
    panel_path = Path(str(panel["path"]))
    require(recipe_module.sha256(panel_path) == panel["sha256"], "CPT 2K panel differs from its pin")
    parent = cast(Json, recipe["parent"])
    if args.start:
        checkpoint, step = Path(str(parent["path"])), 0
        for name, digest in cast(Json, parent["files"]).items():
            if name != "model.safetensors":
                require(recipe_module.sha256(checkpoint / name) == digest, f"starting model {name} differs")
    else:
        from sepalith.training.grpo.state import verify_checkpoint

        checkpoint = args.checkpoint.resolve()
        step = int(checkpoint.name.removeprefix("checkpoint-"))
        verify_checkpoint(checkpoint, recipe_sha)
    manifest_sha = recipe_module.sha256(checkpoint / "campaign-manifest.json")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""), "root CUDA lease must expose one device")
    started = time.monotonic()
    import torch
    from transformers import AutoModelForCausalLM

    from sepalith.training.eval.campaign_cpt_eval import package_holdout_evaluator

    from sepalith.training.grpo_exec.val_eval import open_cuda

    device = open_cuda(torch)
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=torch.bfloat16, device_map={"": DEVICE},
                                                 local_files_only=True, trust_remote_code=False)
    named = list(model.named_parameters())
    require(len(named) == EXPECTED_TENSORS and sum(p.numel() for _, p in named) == EXPECTED_PARAMETERS,
            "not the dense 2.52B checkpoint")
    model.config.use_cache = False
    loaded = time.monotonic()
    evaluate = package_holdout_evaluator({"validation_rows": {"path": str(panel_path)}, "validation_batch_size": 1,
                                          "parameters": {"max_sequence_tokens": int(cast(int, cpt["max_sequence_tokens"]))},
                                          "materialized_rows": True})
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        value = evaluate(model, None, checkpoint, None)
    nll = float(value["metrics"]["mean_causal_nll"])
    rows = int(value["denominators"]["validation_rows"])
    require(rows == int(cast(int, cpt["rows"])), f"evaluated {rows} rows")
    peak = int(torch.cuda.max_memory_allocated(device))
    recorded = float(cast(float, cast(Json, recipe["parent_reference"])["cpt_recorded_nll_fp32_norms"]))
    rows_path = output.with_suffix(".rows.json")
    with rows_path.open("x") as stream:
        json.dump(value["row_metrics"], stream)
    result: Json = {"schema": RESULT_SCHEMA, "status": "complete", "step": step, "start": bool(args.start),
                    "bound_recipe": {"path": str(recipe_path), "sha256": recipe_sha},
                    "checkpoint": {"path": str(checkpoint), "campaign_manifest_sha256": manifest_sha},
                    "panel": {"path": str(panel_path), "sha256": panel["sha256"], "rows": rows},
                    "mean_causal_nll": nll, "denominators": value["denominators"],
                    "delta_vs_recorded_start_fp32_norms": nll - recorded,
                    "rows": {"path": str(rows_path), "sha256": recipe_module.sha256(rows_path)},
                    "load": "plain bf16, no FP32 norm restore, bf16 autocast", "load_seconds": round(loaded - started, 1),
                    "eval_seconds": round(time.monotonic() - loaded, 1), "peak_cuda_allocated_bytes": peak,
                    "final_set_access": False}
    recipe_module.write_new(output, result)
    print(json.dumps({"status": "complete", "step": step, "mean_causal_nll": nll, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
