#!/usr/bin/env python3
"""DEV75 or DEV250 generation for one C15 checkpoint (night only, ``cuda:0``).

The same run as ``offline.dev_generation`` for GRPO checkpoints: the same
admission file, bf16 load (``campaign_eval.load_development_model``), tokenizer
contract, evaluator (``campaign_eval.development_evaluator``) with the recipe's
bound generation batching (batch 32, dynamic KV cache, greedy, cap 1,024) and
the same result schema, so R6 compares results produced exactly like the
starting model's gate-536 result. Only the recipe loader and the checkpoint
verification differ (``grpo.recipe``, ``grpo.state.verify_checkpoint``).
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import time
from typing import cast

from sepalith.training.grpo import recipe as recipe_module, rules

Json = dict[str, object]
ADMISSION_FIELDS = {"schema", "status", "purpose", "cuda_authorized", "final_set_access", "bound_recipe_sha256",
                    "checkpoint_manifest_sha256", "panel_sha256", "output", "deadline_seconds"}
RESULT_SCHEMA = "sepalith.sft11.full-weight-edit-dev-generation.v1"
DEVICE = "cuda:0"


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def allowed_steps(recipe: Json) -> set[int]:
    steps = set(rules.evaluation_steps(int(cast(int, cast(Json, recipe["pre_registered"])["max_updates"]))))
    smoke = recipe.get("smoke")
    if smoke is not None:
        steps.add(int(cast(int, cast(Json, smoke)["eval_step"])))
    return steps


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-recipe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dev-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("dev75", "dev250"), required=True)
    args = parser.parse_args(argv)
    recipe_path, checkpoint, output = args.bound_recipe.resolve(), args.checkpoint.resolve(), args.output.resolve()
    recipe = recipe_module.load(recipe_path, check_data=False)
    recipe_sha = recipe_module.sha256(recipe_path)
    gate = cast(Json, recipe["dev_gate"])
    step = int(checkpoint.name.removeprefix("checkpoint-"))
    require(step in allowed_steps(recipe), f"checkpoint {step} is not an evaluation step of this recipe")
    from sepalith.training.grpo.state import verify_checkpoint

    verify_checkpoint(checkpoint, recipe_sha)
    manifest_sha = recipe_module.sha256(checkpoint / "campaign-manifest.json")
    bound_panel = cast(Json, cast(Json, gate["panels"])[args.panel])
    panel = {"path": bound_panel["path"], "sha256": bound_panel["sha256"]}
    admission = cast(Json, json.loads(args.dev_admission.read_text()))
    require(set(admission) == ADMISSION_FIELDS, "DEV admission fields differ")
    require(admission["schema"] == "sepalith.sft11.full-weight-edit-dev-generation-admission.v1" and admission["status"] == "admitted"
            and admission["purpose"] == "development_generation", "DEV generation not admitted")
    require(admission["cuda_authorized"] is True and admission["final_set_access"] is False, "DEV CUDA/final admission differs")
    require(admission["bound_recipe_sha256"] == recipe_sha and admission["checkpoint_manifest_sha256"] == manifest_sha
            and admission["panel_sha256"] == panel["sha256"] and Path(cast(str, admission["output"])) == output,
            "DEV admission identity differs")
    require(type(admission["deadline_seconds"]) is int and 60 <= admission["deadline_seconds"] <= 3600, "DEV deadline differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""), "root CUDA lease must expose one device")
    bound = cast(Json, gate["generation_batching"])
    batch_size, kv_cache = cast(int, bound["batch_size"]), cast(str, bound["kv_cache"])
    from sepalith.training.eval.campaign_eval import development_evaluator, generation_batching, load_development_model

    batching = generation_batching(batch_size, kv_cache)
    started = time.monotonic()
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    import torch
    from transformers import AutoTokenizer

    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one root-owned CUDA device required")
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    parent = cast(Json, recipe["parent"])
    reference = load_pinned_reference_tokenizer(Path(cast(str, parent["path"])))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    require(recipe_module.sha256(checkpoint / "tokenizer.json") == cast(Json, parent["files"])["tokenizer.json"],
            "checkpoint tokenizer bytes differ")
    model = load_development_model(checkpoint, dtype="bf16", device=DEVICE)
    repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    audit = assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "dev_generation_load")
    named = list(model.named_parameters())
    require(len(named) == 381 and sum(x.numel() for _, x in named) == 2516756480 and not any("lora_" in n.lower() for n, _ in named),
            "DEV model is not the exact dense checkpoint")
    model.eval()
    case_dir = output.with_suffix(".cases")
    require(output.is_absolute() and not output.exists() and not case_dir.exists(), "fresh DEV result required")
    case_dir.mkdir(parents=True)
    rows = [json.loads(line) for line in Path(cast(str, panel["path"])).read_text().splitlines() if line]
    evaluate = development_evaluator({"development_panel": panel, "renderer_id": "zeta2-prm03-v1",
                                      "development_case_ids": sorted(row["id"] for row in rows),
                                      "development_max_new_tokens": gate["generation_max_new_tokens"],
                                      "parameters": {"max_sequence_tokens": 16384},
                                      "evaluation_output_directory": str(case_dir),
                                      "generation_batch_size": batch_size, "generation_kv_cache": kv_cache})
    summary = cast(Json, evaluate(model, tokenizer, checkpoint, step))
    require(cast(Json, summary["denominators"])["cases"] == bound_panel["cases"] and summary["panel_sha256"] == panel["sha256"],
            "DEV denominator differs")
    cases = case_dir / "cases.json"
    require(cases.is_file() and json.loads(cases.read_text())["status"] == "complete", "DEV cases incomplete")
    result: Json = {"schema": RESULT_SCHEMA, "status": "complete", "step": step,
                    "bound_recipe": {"path": str(recipe_path), "sha256": recipe_sha},
                    "checkpoint": {"path": str(checkpoint), "campaign_manifest_sha256": manifest_sha}, "panel": panel,
                    "panel_name": args.panel, "summary": summary,
                    "cases": {"path": str(cases), "sha256": recipe_module.sha256(cases)},
                    "tokenizer_contract": audit, "tokenizer_repair": repair, "elapsed_seconds": time.monotonic() - started,
                    "evaluator_source_sha256": recipe_module.evaluator_hashes(recipe_module.repository_root(recipe)),
                    "final_set_access": False}
    if batching is not None:
        result["generation_runtime"] = {"dtype": "bf16", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                                        **cast(Json, summary["generation_runtime"])}
    recipe_module.write_new(output, result)
    print(json.dumps({"status": "complete", "step": step, "output": str(output), "sha256": recipe_module.sha256(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
