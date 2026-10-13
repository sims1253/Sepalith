#!/usr/bin/env python3
"""Step-0 DEV generation on the bound parent of a v2 recipe (the C07 baseline).

``run_dev_generation`` evaluates SFT checkpoints only (it requires a gate step
and an SFT checkpoint identity). The R1 first gate compares against the parent,
so this module evaluates the parent's own files as step 0 with the same
evaluator (``campaign_eval.development_evaluator``), the same bf16 model load,
the same admission file and the same result schema. It lives outside
``training/eval`` so the evaluator modules card C05 changes stay untouched;
fold it into ``run_dev_generation`` once that interface settles.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import time

from sepalith.training.sft.recipe_v2 import _int, _object, load_bound_v2, require, sha256, write_new

ADMISSION_FIELDS = {"schema", "status", "purpose", "cuda_authorized", "final_set_access", "bound_recipe_sha256",
                    "checkpoint_manifest_sha256", "panel_sha256", "output", "deadline_seconds"}


def _panel(recipe: dict[str, object], name: str) -> tuple[dict[str, object], int]:
    bound = _object(_object(_object(recipe["dev_gate"], "dev_gate")["panels"], "panels")[name], name)
    return {"path": bound["path"], "sha256": bound["sha256"]}, _int(bound["cases"], f"{name} cases")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-recipe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="must be the recipe's bound parent directory")
    parser.add_argument("--dev-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("dev75", "dev250"), default="dev75")
    args = parser.parse_args(argv)
    recipe_path, checkpoint, output = args.bound_recipe.resolve(), args.checkpoint.resolve(), args.output.resolve()
    recipe = load_bound_v2(recipe_path)
    parent = _object(recipe["parent"], "parent")
    require(checkpoint == Path(str(parent["path"])).resolve(), "the baseline evaluates only the bound parent")
    files = _object(parent["files"], "parent files")
    manifest_sha = sha256(checkpoint / "campaign-manifest.json")
    require(manifest_sha == files["campaign-manifest.json"], "parent manifest differs")
    panel, cases = _panel(recipe, args.panel)
    admission = json.loads(args.dev_admission.read_text())
    require(set(admission) == ADMISSION_FIELDS, "DEV admission fields differ")
    require(admission["schema"] == "sepalith.sft11.full-weight-edit-dev-generation-admission.v1"
            and admission["status"] == "admitted" and admission["purpose"] == "development_generation", "DEV generation not admitted")
    require(admission["cuda_authorized"] is True and admission["final_set_access"] is False, "DEV CUDA/final admission differs")
    require(admission["bound_recipe_sha256"] == sha256(recipe_path) and admission["checkpoint_manifest_sha256"] == manifest_sha
            and admission["panel_sha256"] == panel["sha256"] and Path(admission["output"]) == output, "DEV admission identity differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""), "root CUDA lease must expose one device")
    started = time.monotonic()
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )
    from sepalith.training.eval.campaign_eval import development_evaluator

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one root-owned CUDA device required")
    reference = load_pinned_reference_tokenizer(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), torch_dtype=torch.bfloat16, device_map={"": "cuda"},
                                                 local_files_only=True, trust_remote_code=False)
    repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    audit = assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "parent_baseline_load")
    named = list(model.named_parameters())
    require(len(named) == 381 and sum(x.numel() for _, x in named) == 2516756480, "parent is not the exact dense model")
    model.eval()
    case_dir = output.with_suffix(".cases")
    require(not output.exists() and not case_dir.exists(), "fresh baseline output required")
    case_dir.mkdir(parents=True)
    rows = [json.loads(line) for line in Path(str(panel["path"])).read_text().splitlines() if line]
    gate = _object(recipe["dev_gate"], "dev_gate")
    cohort = _object(recipe["cohort"], "cohort")
    evaluate = development_evaluator({"development_panel": panel, "renderer_id": "zeta2-prm03-v1",
                                      "development_case_ids": sorted(row["id"] for row in rows),
                                      "development_max_new_tokens": gate["generation_max_new_tokens"],
                                      "parameters": {"max_sequence_tokens": cohort["max_sequence_tokens"]},
                                      "evaluation_output_directory": str(case_dir)})
    summary = evaluate(model, tokenizer, checkpoint, 0)
    require(summary["denominators"]["cases"] == cases and summary["panel_sha256"] == panel["sha256"], "DEV denominator differs")
    cases_path = case_dir / "cases.json"
    result = {"schema": "sepalith.sft11.full-weight-edit-dev-generation.v1", "status": "complete", "step": 0,
              "bound_recipe": {"path": str(recipe_path), "sha256": sha256(recipe_path)},
              "checkpoint": {"path": str(checkpoint), "campaign_manifest_sha256": manifest_sha, "role": "parent_baseline"},
              "panel": panel, "panel_name": args.panel, "summary": summary,
              "cases": {"path": str(cases_path), "sha256": sha256(cases_path)}, "tokenizer_contract": audit,
              "tokenizer_repair": repair, "elapsed_seconds": time.monotonic() - started, "final_set_access": False}
    write_new(output, result)
    print(json.dumps({"status": "complete", "step": 0, "output": str(output), "sha256": sha256(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
