#!/usr/bin/env python3
"""Re-score old LoRA-SFT models on the current plan evaluator (measurement only).

Each model in ``models.json`` (merged bf16 weights) goes through DEV75 and DEV250
with exactly the evaluator and settings of arm A's gates (C07/C08/C09):
``campaign_eval.load_development_model`` in bf16, the pinned tokenizer of the bound
recipe's parent (byte-identical to arm A checkpoint 536's tokenizer files), the
same tokenizer contract (``restore_trainer_eog_alignment``, ``assert_runtime_tokenizer``),
``campaign_eval.development_evaluator`` with renderer zeta2-prm03-v1, the bound
recipe's panels, cap (1,024), ``max_sequence_tokens`` and generation batching
(batch 32, dynamic KV cache, left padding, longest prompt first). The R1 counts
come from ``gate_rule.metrics_from_summaries``. Arm A 536's gate results are
read back from their files (sha256-checked) and put alongside. No rule is
applied and nothing is adopted.

Modes:

- ``--dry-run`` (CPU only, no CUDA): checks every model directory (LlamaForCausalLM
  config, one complete safetensors file with 381 bf16 tensors and no LoRA keys,
  tokenizer files), hashes the weights against the recorded sha256, checks that
  each model's own tokenizer has the pinned vocabulary and renders all 325 DEV
  prompts to the same ids as the pinned tokenizer, recomputes arm A's row, and
  writes a receipt (``--dry-run-output``). ``--cpu-smoke LABEL`` additionally
  runs the real evaluator on the CPU on the two shortest DEV75 prompts with a
  16-token cap (a subset panel in a temporary directory).
- default (GPU, night window only): creates a fresh attempt directory under
  ``--output-dir``, takes the CUDA lock through ``cuda_host_guard`` and runs
  itself with ``--inner``. Weights are re-verified against the dry-run receipt
  (size and mtime; a full sha256 if either differs). Before each model it checks
  that enough time is left before ``SEPALITH_DEADLINE_EPOCH``; models that do
  not fit are recorded as skipped. One model's failure does not stop the others.
  Per-panel results (schema ``sepalith.sft11.full-weight-edit-dev-generation.v1``)
  and case files go to the attempt directory; the summary receipt (JSON) and
  table (markdown) go to ``--summary-dir`` and are rewritten after every model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
RECEIPT_SCHEMA = "sepalith.lora-baseline.rescore-receipt.v1"
DRY_RUN_SCHEMA = "sepalith.lora-baseline.rescore-dry-run.v1"
RESULT_SCHEMA = "sepalith.sft11.full-weight-edit-dev-generation.v1"
DEFAULT_RECIPE = HERE.parent / "c04/bound/sft-v2-lr1e-5.json"
DEFAULT_MODELS = HERE / "models.json"
DEFAULT_DRY_RUN = HERE.parent / "receipts/LORA-rescore-dev250-dry-run.json"
DEFAULT_SUMMARY_DIR = HERE / "results"
CUDA_LOCK = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock")
GATES = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates/SFT-v2-r4-lr1e-5")
ARM_A = {
    "label": "arm-a-536",
    "name": "arm A 536",
    "checkpoint": "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536",
    "training": "full-weight SFT on CPT 11,586, lr 1e-5, target-only loss (exact_target_body_terminal_eos_only), bound recipe sft-v2-lr1e-5",
    "dev75": {"path": str(GATES / "results/dev-dev75-step-536.20261014T021133-673743425.json"),
              "sha256": "c6e77e49f8fc6323244b576fe2d0ee730c7c524c844490872ca0b0f0c3796e67"},
    "dev250": {"path": str(GATES / "results/dev-dev250-step-536.20261014T021133-673743425.json"),
               "sha256": "0a1dbbd903dedc4ec392b2848f8b2ea89641e5228a466ca3a79fe1c4936ab9d3"},
    # C09.json gate_table, decision-step-536.
    "recorded": {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 117},
}
WEIGHTS_BYTES = 5033557128
TENSORS = 381
PARAMETERS = 2516756480
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
REQUIRED_FILES = ("config.json", "generation_config.json", "model.safetensors", *TOKENIZER_FILES)
PANELS = ("dev75", "dev250")
# Arm A took 171 s (DEV75) + 371 s (DEV250) plus the load; the parent, which runs to the cap on
# nearly every case, took 354 s + 983 s at the same settings (C07 validation). A model that runs to the
# cap more often takes longer. Start a model only with this much time left.
MODEL_BUDGET_SECONDS = 25 * 60
SMOKE_CASES = 2
SMOKE_CAP = 16


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE, help="arm A's bound recipe (panels, cap, batching, parent tokenizer)")
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--only", default="", help="comma-separated model labels (default: all, in models.json order)")
    p.add_argument("--dry-run", action="store_true", help="CPU checks only; writes --dry-run-output")
    p.add_argument("--dry-run-output", type=Path, default=DEFAULT_DRY_RUN)
    p.add_argument("--cpu-smoke", default="", help="with --dry-run: run the evaluator on the CPU for this model label")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for --dry-run (1 to 8)")
    p.add_argument("--output-dir", type=Path, help="GPU run: a fresh attempt directory is created inside")
    p.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR, help="receipt JSON and markdown table")
    p.add_argument("--guard-seconds", type=int, default=9000, help="cuda_host_guard hard limit (the hard stop caps it)")
    p.add_argument("--no-guard", action="store_true", help="run the inner process directly (tests only)")
    p.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--attempt-dir", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        p.error("--output-dir is required without --dry-run")
    if args.cpu_smoke and not args.dry_run:
        p.error("--cpu-smoke goes with --dry-run")
    if not 1 <= args.threads <= 8:
        p.error("--threads must be 1 to 8")
    return args


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deadline_left():
    value = os.environ.get("SEPALITH_DEADLINE_EPOCH")
    return math.inf if value in (None, "") else float(value) - time.time()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# Inputs -------------------------------------------------------------------------------------------

def load_models(path, only=""):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "sepalith.lora-baseline.models.v1":
        raise SystemExit("models.json schema differs")
    models = data["models"]
    labels = [m["label"] for m in models]
    if len(set(labels)) != len(labels) or ARM_A["label"] in labels:
        raise SystemExit("model labels must be unique and must not be arm A's")
    for m in models:
        if not {"label", "name", "path", "source", "weights_sha256", "step", "training"} <= set(m):
            raise SystemExit(f"model {m.get('label')!r} lacks fields")
        if not Path(m["path"]).is_absolute() or len(m["weights_sha256"]) != 64 or type(m["step"]) is not int:
            raise SystemExit(f"model {m['label']!r}: absolute path, sha256 and integer step required")
    wanted = [x for x in only.split(",") if x]
    if unknown := set(wanted) - set(labels):
        raise SystemExit(f"unknown model labels: {sorted(unknown)}")
    return [m for m in models if not wanted or m["label"] in wanted]


def load_recipe(path):
    """The bound recipe's evaluator settings, without the trainer's full binding checks."""
    recipe = json.loads(Path(path).read_text())
    gate = recipe["dev_gate"]
    parent = recipe["parent"]
    return {
        "path": str(Path(path).resolve()), "sha256": sha256(path),
        "parent_path": parent["path"], "parent_files": parent["files"],
        "panels": {name: {"path": gate["panels"][name]["path"], "sha256": gate["panels"][name]["sha256"],
                          "cases": gate["panels"][name]["cases"]} for name in PANELS},
        "cap": gate["generation_max_new_tokens"],
        "batch_size": gate["generation_batching"]["batch_size"],
        "kv_cache": gate["generation_batching"]["kv_cache"],
        "max_sequence_tokens": recipe["cohort"]["max_sequence_tokens"],
    }


def settings_record(recipe):
    return {"dtype": "bf16", "renderer_id": "zeta2-prm03-v1", "cap": recipe["cap"], "batch_size": recipe["batch_size"],
            "kv_cache": recipe["kv_cache"], "max_sequence_tokens": recipe["max_sequence_tokens"],
            "tokenizer": f"pinned tokenizer of the recipe parent ({recipe['parent_path']}), byte-identical to arm A 536's",
            "evaluator": "sepalith.training.eval.campaign_eval.development_evaluator",
            "metrics": "sepalith.training.sft.gate_rule.metrics_from_summaries"}


def safetensors_header(path):
    with Path(path).open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        if not 0 < size < 64 * 1024 * 1024:
            raise ValueError("implausible safetensors header size")
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    return size, header


def check_model_dir(model):
    """Static completeness checks (no weights loaded); returns a record, raises on failure."""
    path = Path(model["path"])
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"{model['label']}: missing {missing}")
    shards = sorted(p.name for p in path.glob("*.safetensors"))
    if shards != ["model.safetensors"] or (path / "model.safetensors.index.json").exists():
        raise ValueError(f"{model['label']}: expected one model.safetensors, found {shards}")
    if any(path.glob("adapter_*")):
        raise ValueError(f"{model['label']}: adapter files present; a merged checkpoint is required")
    config = json.loads((path / "config.json").read_text())
    expected = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": 130560, "num_hidden_layers": 42,
                "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2, "intermediate_size": 6144,
                "tie_word_embeddings": False, "bos_token_id": 0, "pad_token_id": 1}
    if differs := {k: config.get(k) for k, v in expected.items() if config.get(k) != v}:
        raise ValueError(f"{model['label']}: config differs {differs}")
    if list(config.get("eos_token_id") or []) != [1, 130073]:
        raise ValueError(f"{model['label']}: config eos_token_id differs")
    weights = path / "model.safetensors"
    stat = weights.stat()
    if stat.st_size != WEIGHTS_BYTES:
        raise ValueError(f"{model['label']}: model.safetensors has {stat.st_size} bytes, expected {WEIGHTS_BYTES}")
    header_bytes, header = safetensors_header(weights)
    dtypes = sorted({v["dtype"] for v in header.values()})
    numel = sum(math.prod(v["shape"]) for v in header.values())
    lora = [k for k in header if "lora_" in k.lower()]
    if len(header) != TENSORS or dtypes != ["BF16"] or numel != PARAMETERS or lora:
        raise ValueError(f"{model['label']}: tensors {len(header)} dtypes {dtypes} numel {numel} lora {len(lora)}")
    end = max(v["data_offsets"][1] for v in header.values())
    if 8 + header_bytes + end != stat.st_size:
        raise ValueError(f"{model['label']}: safetensors data does not end at the file end (truncated?)")
    return {"files": {name: {"bytes": (path / name).stat().st_size,
                             "sha256": None if name == "model.safetensors" else sha256(path / name)} for name in REQUIRED_FILES},
            "architecture": "LlamaForCausalLM", "tensors": len(header), "parameters": numel, "dtype": "BF16",
            "weights_bytes": stat.st_size, "weights_mtime_ns": stat.st_mtime_ns}


def arm_a_reference():
    from sepalith.training.sft import gate_rule

    summaries, files = {}, {}
    for name in PANELS:
        record = ARM_A[name]
        if sha256(record["path"]) != record["sha256"]:
            raise SystemExit(f"arm A {name} gate result sha256 differs: {record['path']}")
        result = json.loads(Path(record["path"]).read_text())
        summaries[name] = result["summary"]
        files[name] = {**record, "generation_runtime": result.get("generation_runtime"),
                       "elapsed_seconds": result.get("elapsed_seconds"), "panel": result["panel"]}
    metrics = gate_rule.metrics_from_summaries(summaries["dev75"], summaries["dev250"])
    if metrics != ARM_A["recorded"]:
        raise SystemExit(f"arm A 536 recomputed metrics {metrics} differ from C09 {ARM_A['recorded']}")
    return {"label": ARM_A["label"], "name": ARM_A["name"], "checkpoint": ARM_A["checkpoint"], "training": ARM_A["training"],
            "status": "complete", "results": files, "metrics": metrics, "detail": detail(summaries["dev75"], summaries["dev250"])}


def detail(dev75, dev250):
    c75, c250 = dev75["counts"], dev250["counts"]
    return {
        "dev75": {"valid": c75["protocol_valid"], "cap": c75["cap_hit"], "edit_exact_of_43": c75["edit_exact"],
                  "noop_correct_of_32": c75["strict_noop_correct"], "nofp": c75["strict_noop_false_suggestions"],
                  "target_nll": dev75.get("target_nll"), "prompt_nll": dev75.get("prompt_nll")},
        "dev250": {"valid": c250["protocol_valid"], "cap": c250["cap_hit"], "edit_exact_of_200": c250["edit_exact"],
                   "noop_correct_of_50": c250["strict_noop_correct"], "nofp": c250["strict_noop_false_suggestions"],
                   "score250": c250["edit_exact"] + c250["strict_noop_correct"],
                   "target_nll": dev250.get("target_nll"), "prompt_nll": dev250.get("prompt_nll")},
    }


def panel_rows(panel):
    rows = []
    for line in Path(panel["path"]).read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            if case["split"] != "dev":
                raise SystemExit("only DEV rows may enter this evaluation")
            rows.append(case)
    if len(rows) != panel["cases"]:
        raise SystemExit(f"panel has {len(rows)} rows, expected {panel['cases']}")
    return rows


def render(rows, tokenizer, max_sequence_tokens, cap):
    from sepalith.campaign_protocol import PromptContext, build_training_row
    from sepalith.training.eval import campaign_eval

    out = {}
    for case in rows:
        row = build_training_row(PromptContext.from_mapping(case["context"]), operation=case["operation"],
                                 region_new=case["region_new"], tokenizer=tokenizer, row_id=case["id"],
                                 family=case["family"], package_id=case["package_id"], split="dev")
        campaign_eval.validate_development_capacity(row, max_sequence_tokens, cap, case["id"])
        out[case["id"]] = (row["input_ids"], row["target_start"])
    return out


# Dry run (CPU) ------------------------------------------------------------------------------------

def dry_run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer

    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer

    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        raise SystemExit("the dry run must not see a CUDA device")
    started = time.monotonic()
    recipe = load_recipe(args.recipe)
    models = load_models(args.models, args.only)
    parent = Path(recipe["parent_path"])
    for name in TOKENIZER_FILES:
        if sha256(parent / name) != recipe["parent_files"][name]:
            raise SystemExit(f"recipe parent {name} bytes differ")
    reference = load_pinned_reference_tokenizer(parent)
    rows = {}
    for name in PANELS:
        panel = recipe["panels"][name]
        if sha256(panel["path"]) != panel["sha256"]:
            raise SystemExit(f"{name} panel sha256 differs")
        rows[name] = panel_rows(panel)
    rendered_reference = {name: render(rows[name], reference, recipe["max_sequence_tokens"], recipe["cap"]) for name in PANELS}
    receipt = {"schema": DRY_RUN_SCHEMA, "status": "running", "started": now(), "cuda_started": False,
               "final_set_access": False, "recipe": {"path": recipe["path"], "sha256": recipe["sha256"]},
               "models_file": {"path": str(args.models.resolve()), "sha256": sha256(args.models)},
               "panels": recipe["panels"], "settings": settings_record(recipe), "models": {}}
    receipt["arm_a"] = arm_a_reference()
    problems = []
    for model in models:
        record = {"path": model["path"], "expected_weights_sha256": model["weights_sha256"]}
        receipt["models"][model["label"]] = record
        try:
            record.update(check_model_dir(model))
            t0 = time.monotonic()
            actual = sha256(Path(model["path"]) / "model.safetensors")
            record["weights_sha256"] = actual
            record["files"]["model.safetensors"]["sha256"] = actual
            record["hash_seconds"] = round(time.monotonic() - t0, 1)
            if actual != model["weights_sha256"]:
                raise ValueError(f"weights sha256 {actual} differs from the recorded {model['weights_sha256']}")
            own = AutoTokenizer.from_pretrained(model["path"], local_files_only=True, trust_remote_code=False)
            if own.get_vocab() != reference.get_vocab() or len(own) != len(reference):
                raise ValueError("the model's own tokenizer vocabulary differs from the pinned tokenizer")
            if (own.bos_token_id, own.eos_token_id, own.pad_token_id) != (0, 1, 1):
                raise ValueError("the model's own tokenizer BOS/EOS/PAD differ")
            parity = {}
            for name in PANELS:
                mine = render(rows[name], own, recipe["max_sequence_tokens"], recipe["cap"])
                differ = sorted(k for k, v in mine.items() if v != rendered_reference[name][k])
                parity[name] = {"cases": len(mine), "identical_prompt_and_target_ids": len(mine) - len(differ), "differ": differ}
                if differ:
                    raise ValueError(f"{name}: {len(differ)} cases render differently with the model's own tokenizer")
            record["own_tokenizer"] = {
                "tokenizer_json_sha256": record["files"]["tokenizer.json"]["sha256"],
                "same_bytes_as_pinned": all(record["files"][n]["sha256"] == recipe["parent_files"][n] for n in TOKENIZER_FILES),
                "vocab_equal_pinned": True, "render_parity": parity}
            record["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - record and continue with the next model
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            problems.append(f"{model['label']}: {error}")
        write_json(args.dry_run_output, receipt)
    lengths = {name: sorted(len(ids[:start]) for ids, start in rendered_reference[name].values()) for name in PANELS}
    receipt["prompt_tokens"] = {name: {"min": v[0], "max": v[-1], "median": v[len(v) // 2]} for name, v in lengths.items()}
    if args.cpu_smoke:
        receipt["cpu_smoke"] = cpu_smoke(args, recipe, models, rows["dev75"], rendered_reference["dev75"], reference)
        if receipt["cpu_smoke"]["status"] != "complete":
            problems.append(f"cpu smoke: {receipt['cpu_smoke'].get('error')}")
    receipt.update(status="pass" if not problems else "fail", problems=problems, finished=now(),
                   elapsed_seconds=round(time.monotonic() - started, 1))
    write_json(args.dry_run_output, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(args.dry_run_output), "problems": problems}), flush=True)
    return 0 if not problems else 1


def cpu_smoke(args, recipe, models, dev75_rows, rendered, reference):
    """The real load, tokenizer contract and evaluator on the CPU: two shortest DEV75 prompts, cap 16."""
    import torch
    from transformers import AutoTokenizer

    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )
    from sepalith.training.eval import campaign_eval
    from sepalith.training.sft import gate_rule

    model_record = next((m for m in models if m["label"] == args.cpu_smoke), None)
    if model_record is None:
        return {"status": "failed", "error": f"unknown --cpu-smoke label {args.cpu_smoke!r}"}
    chosen = sorted(rendered, key=lambda k: (rendered[k][1], k))[:SMOKE_CASES]
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="lora-rescore-smoke-") as tmp:
            subset = Path(tmp) / "dev75-smoke.jsonl"
            subset.write_text("".join(json.dumps(r) + "\n" for r in dev75_rows if r["id"] in chosen))
            tokenizer = AutoTokenizer.from_pretrained(recipe["parent_path"], local_files_only=True, trust_remote_code=False)
            model = campaign_eval.load_development_model(Path(model_record["path"]), dtype="bf16", device="cpu")
            repair = restore_trainer_eog_alignment(model, tokenizer, reference)
            audit = assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "lora_rescore_cpu_smoke")
            named = list(model.named_parameters())
            if len(named) != TENSORS or sum(x.numel() for _, x in named) != PARAMETERS:
                raise ValueError("not the exact dense model")
            model.eval()
            cases_dir = Path(tmp) / "cases"
            cases_dir.mkdir()
            evaluate = campaign_eval.development_evaluator({
                "development_panel": {"path": str(subset), "sha256": sha256(subset)}, "renderer_id": "zeta2-prm03-v1",
                "development_case_ids": sorted(chosen), "development_max_new_tokens": SMOKE_CAP,
                "parameters": {"max_sequence_tokens": recipe["max_sequence_tokens"]},
                "evaluation_output_directory": str(cases_dir), "generation_batch_size": recipe["batch_size"],
                "generation_kv_cache": recipe["kv_cache"]})
            with torch.no_grad():
                summary = evaluate(model, tokenizer, Path(model_record["path"]), model_record["step"])
            # metrics_from_summaries needs two summaries; reuse the subset for both (shape check only).
            gate_rule.metrics_from_summaries(summary, summary)
            cases = json.loads((cases_dir / "cases.json").read_text())["results"]
        return {"status": "complete", "model": model_record["label"], "case_ids": chosen, "cap": SMOKE_CAP,
                "batch_size": recipe["batch_size"], "counts": summary["counts"], "target_nll": summary.get("target_nll"),
                "generated_tokens": [c["generated_tokens"] for c in cases],
                "prompt_tokens": [c["prompt_tokens"] for c in cases], "tokenizer_contract": audit,
                "tokenizer_repair_changed": repair["backend_repair"]["changed"],
                "seconds": round(time.monotonic() - started, 1),
                "note": "pipeline check only (CPU bf16, 16-token cap); the counts are not a score"}
    except Exception as error:  # noqa: BLE001 - the smoke result records the failure
        return {"status": "failed", "model": model_record["label"], "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.monotonic() - started, 1)}


# GPU run (inner) ----------------------------------------------------------------------------------

def verify_weights(model, dry):
    """Size and mtime against the dry-run receipt; a full sha256 when either differs."""
    record = (dry.get("models") or {}).get(model["label"]) or {}
    weights = Path(model["path"]) / "model.safetensors"
    stat = weights.stat()
    if (record.get("status") == "ok" and record.get("weights_sha256") == model["weights_sha256"]
            and record.get("weights_bytes") == stat.st_size and record.get("weights_mtime_ns") == stat.st_mtime_ns):
        return {"method": "dry-run receipt (sha256, size, mtime_ns unchanged)", "sha256": model["weights_sha256"]}
    actual = sha256(weights)
    if actual != model["weights_sha256"]:
        raise ValueError(f"weights sha256 {actual} differs from {model['weights_sha256']}")
    return {"method": "full sha256 in the job", "sha256": actual}


def inner(args):
    import torch
    from transformers import AutoTokenizer

    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )
    from sepalith.training.eval import campaign_eval
    from sepalith.training.sft import gate_rule

    attempt = args.attempt_dir
    recipe = load_recipe(args.recipe)
    models = load_models(args.models, args.only)
    dry = json.loads(args.dry_run_output.read_text()) if args.dry_run_output.is_file() else {}
    if dry.get("schema") != DRY_RUN_SCHEMA or dry.get("status") != "pass":
        raise SystemExit(f"a passing dry-run receipt is required: {args.dry_run_output}")
    if not (torch.cuda.is_available() and torch.cuda.device_count() == 1):
        raise SystemExit("exactly one visible CUDA device required")
    parent = Path(recipe["parent_path"])
    for name in TOKENIZER_FILES:
        if sha256(parent / name) != recipe["parent_files"][name]:
            raise SystemExit(f"recipe parent {name} bytes differ")
    for name in PANELS:
        if sha256(recipe["panels"][name]["path"]) != recipe["panels"][name]["sha256"]:
            raise SystemExit(f"{name} panel sha256 differs")
    stamp = attempt.name
    summary_json = args.summary_dir / f"rescore-{stamp}.json"
    summary_md = args.summary_dir / f"rescore-{stamp}.md"
    receipt = {"schema": RECEIPT_SCHEMA, "status": "running", "started": now(), "attempt_dir": str(attempt),
               "purpose": "measurement only: no adoption decision, no rule change", "final_set_access": False,
               "recipe": {"path": recipe["path"], "sha256": recipe["sha256"]},
               "models_file": {"path": str(args.models.resolve()), "sha256": sha256(args.models)},
               "dry_run": {"path": str(args.dry_run_output.resolve()), "sha256": sha256(args.dry_run_output)},
               "panels": recipe["panels"], "settings": settings_record(recipe),
               "device": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
               "arm_a": arm_a_reference(), "models": {}}

    def save():
        write_json(attempt / "receipt.json", receipt)
        write_json(summary_json, receipt)
        summary_md.write_text(markdown(receipt))

    save()
    reference = load_pinned_reference_tokenizer(parent)
    rows = {name: panel_rows(recipe["panels"][name]) for name in PANELS}
    for model_record in models:
        label = model_record["label"]
        record = {"name": model_record["name"], "path": model_record["path"], "step": model_record["step"],
                  "training": model_record["training"], "status": "running"}
        receipt["models"][label] = record
        if deadline_left() < MODEL_BUDGET_SECONDS:
            record.update(status="skipped", reason=f"less than {MODEL_BUDGET_SECONDS} s before the window deadline")
            save()
            continue
        save()
        started = time.monotonic()
        model = None
        try:
            record["weights"] = verify_weights(model_record, dry)
            tokenizer = AutoTokenizer.from_pretrained(str(parent), local_files_only=True, trust_remote_code=False)
            model = campaign_eval.load_development_model(Path(model_record["path"]), dtype="bf16", device="cuda")
            repair = restore_trainer_eog_alignment(model, tokenizer, reference)
            audit = assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "lora_rescore_load")
            named = list(model.named_parameters())
            if len(named) != TENSORS or sum(x.numel() for _, x in named) != PARAMETERS or any("lora_" in n.lower() for n, _ in named):
                raise ValueError("not the exact dense checkpoint")
            model.eval()
            record["load_seconds"] = round(time.monotonic() - started, 1)
            summaries = {}
            for name in PANELS:
                panel = {"path": recipe["panels"][name]["path"], "sha256": recipe["panels"][name]["sha256"]}
                output = attempt / label / f"dev-{name}.json"
                case_dir = output.with_suffix(".cases")
                case_dir.mkdir(parents=True)
                evaluate = campaign_eval.development_evaluator({
                    "development_panel": panel, "renderer_id": "zeta2-prm03-v1",
                    "development_case_ids": sorted(r["id"] for r in rows[name]),
                    "development_max_new_tokens": recipe["cap"],
                    "parameters": {"max_sequence_tokens": recipe["max_sequence_tokens"]},
                    "evaluation_output_directory": str(case_dir),
                    "generation_batch_size": recipe["batch_size"], "generation_kv_cache": recipe["kv_cache"]})
                t0 = time.monotonic()
                summary = evaluate(model, tokenizer, Path(model_record["path"]), model_record["step"])
                seconds = time.monotonic() - t0
                if summary["denominators"]["cases"] != recipe["panels"][name]["cases"] or summary["panel_sha256"] != panel["sha256"]:
                    raise ValueError(f"{name} denominator differs")
                cases = case_dir / "cases.json"
                if json.loads(cases.read_text())["status"] != "complete":
                    raise ValueError(f"{name} cases incomplete")
                result = {"schema": RESULT_SCHEMA, "status": "complete", "step": model_record["step"],
                          "bound_recipe": {"path": recipe["path"], "sha256": recipe["sha256"]},
                          "checkpoint": {"path": model_record["path"], "label": label,
                                         "weights_sha256": record["weights"]["sha256"]},
                          "panel": panel, "panel_name": name, "summary": summary,
                          "cases": {"path": str(cases), "sha256": sha256(cases)}, "tokenizer_contract": audit,
                          "tokenizer_repair": repair, "elapsed_seconds": seconds, "final_set_access": False,
                          "generation_runtime": {"dtype": "bf16", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                                 **(summary.get("generation_runtime") or {})}}
                write_json(output, result)
                summaries[name] = summary
                record.setdefault("results", {})[name] = {"path": str(output), "sha256": sha256(output),
                                                           "seconds": round(seconds, 1),
                                                           "generation_runtime": result["generation_runtime"]}
                save()
            record["metrics"] = gate_rule.metrics_from_summaries(summaries["dev75"], summaries["dev250"])
            record["detail"] = detail(summaries["dev75"], summaries["dev250"])
            record["status"] = "complete"
        except Exception as error:  # noqa: BLE001 - record and continue with the next model
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            if isinstance(error, torch.cuda.OutOfMemoryError):
                record["error"] = record["error"].splitlines()[0]
        finally:
            record["seconds"] = round(time.monotonic() - started, 1)
            del model
            torch.cuda.empty_cache()
        save()
    states = [r["status"] for r in receipt["models"].values()]
    receipt["status"] = "complete" if all(s == "complete" for s in states) else "partial"
    receipt["finished"] = now()
    save()
    print(json.dumps({"status": receipt["status"], "receipt": str(summary_json)}), flush=True)
    return 0


def markdown(receipt):
    def fmt(value, digits=4):
        return "" if value is None else f"{value:.{digits}f}"

    lines = [
        "# LoRA-SFT baseline re-scored on the plan evaluator",
        "",
        f"Status: {receipt['status']}. Started {receipt['started']}"
        + (f", finished {receipt['finished']}" if receipt.get("finished") else "") + ".",
        "Measurement only: no adoption decision, no rule change.",
        "",
        f"Evaluator: `campaign_eval.development_evaluator`, bf16, cap {receipt['settings']['cap']}, batch "
        f"{receipt['settings']['batch_size']} ({receipt['settings']['kv_cache']} KV cache), renderer zeta2-prm03-v1, "
        f"max_sequence_tokens {receipt['settings']['max_sequence_tokens']}, the pinned parent tokenizer; "
        f"recipe `{Path(receipt['recipe']['path']).name}` (sha256 {receipt['recipe']['sha256'][:12]}). "
        f"DEV75 sha256 {receipt['panels']['dev75']['sha256'][:12]}, DEV250 sha256 {receipt['panels']['dev250']['sha256'][:12]}.",
        "",
        "| Model | Step | DEV75 valid | cap | edit /43 | nofp /32 | score250 | DEV250 edit /200 | DEV250 no-op /50 | DEV250 valid | DEV250 cap | DEV75 target NLL | DEV250 target NLL | Minutes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [(receipt["arm_a"]["name"], 536, receipt["arm_a"], None)]
    rows += [(r["name"], r["step"], r, r.get("seconds")) for r in receipt["models"].values()]
    for name, step, r, seconds in rows:
        if r.get("status") != "complete":
            reason = r.get("error") or r.get("reason") or ""
            lines.append(f"| {name} | {step} | {r.get('status')}: {reason} |" + " |" * 12)
            continue
        m, d = r["metrics"], r["detail"]
        lines.append(f"| {name} | {step} | {m['valid']} | {m['cap']} | {m['edit']} | {m['nofp']} | {m['score250']} | "
                     f"{d['dev250']['edit_exact_of_200']} | {d['dev250']['noop_correct_of_50']} | {d['dev250']['valid']} | "
                     f"{d['dev250']['cap']} | {fmt(d['dev75']['target_nll'])} | {fmt(d['dev250']['target_nll'])} | "
                     f"{'' if seconds is None else f'{seconds / 60:.1f}'} |")
    lines += ["", "Arm A 536 is read back from its gate results (sha256-checked); its row is not re-run.", "", "## Training differences", ""]
    lines.append(f"- {receipt['arm_a']['name']}: {receipt['arm_a']['training']}")
    for r in receipt["models"].values():
        t = r["training"]
        lines.append(f"- {r['name']}: {t['run']}; base {t['base']}; {t['adapter']}; loss {t['loss']}.")
    lines += ["", "Target NLL is teacher-forced on the reference target; it is not comparable across models trained "
              "with different loss masks (θ0 trained on prompt and target).", ""]
    return "\n".join(lines)


# Outer process: fresh attempt directory, CUDA lock and host-memory guard -------------------------

def outer(args):
    attempt = args.output_dir / time.strftime("%Y%m%dT%H%M%S")
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--inner", "--attempt-dir", str(attempt),
               "--recipe", str(args.recipe), "--models", str(args.models), "--only", args.only,
               "--dry-run-output", str(args.dry_run_output), "--output-dir", str(args.output_dir),
               "--summary-dir", str(args.summary_dir)]
    if args.no_guard:
        launch = command
    else:
        CUDA_LOCK.parent.mkdir(parents=True, exist_ok=True)
        CUDA_LOCK.touch(exist_ok=True)
        write_json(attempt / "command.json", command)
        hard = os.environ.get("SEPALITH_HARD_STOP_EPOCH")
        seconds = args.guard_seconds if hard in (None, "") else min(args.guard_seconds, float(hard) - time.time() - 60)
        launch = [sys.executable, "-B", "-m", "sepalith.training.guards.cuda_host_guard", "--command-json",
                  str(attempt / "command.json"), "--output", str(attempt / "guard"), "--seconds", str(int(max(60, seconds)))]
    with (attempt / "outer.log").open("wb") as log:
        process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
        signal.signal(signal.SIGTERM, lambda *_: process.terminate())
        code = process.wait()
    receipt_path = attempt / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    models = (receipt or {}).get("models") or {}
    table = {label: (r.get("metrics") or r.get("status")) for label, r in models.items()}
    summary = {"exit": code, "status": (receipt or {}).get("status"), "receipt": str(receipt_path),
               "summary_dir": str(args.summary_dir), "metrics": table,
               "arm_a_536": ((receipt or {}).get("arm_a") or {}).get("metrics")}
    needs_user = None
    if code != 0 or receipt is None or receipt.get("status") != "complete":
        needs_user = f"LoRA re-score did not complete for every model (measurement only); see {attempt}"
    result = os.environ.get("SEPALITH_JOB_RESULT")
    if result:
        evaluations = [str(receipt_path)] + [x["path"] for r in models.values() for x in (r.get("results") or {}).values()]
        write_json(Path(result), {"summary": json.dumps(summary, sort_keys=True, ensure_ascii=False),
                                  "evaluations": evaluations, "needs_user": needs_user})
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
    return 0 if code == 0 and receipt is not None and receipt.get("status") in ("complete", "partial") else 1


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    if args.inner:
        return inner(args)
    return outer(args)


if __name__ == "__main__":
    raise SystemExit(main())
