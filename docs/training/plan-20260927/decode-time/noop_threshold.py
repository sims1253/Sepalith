#!/usr/bin/env python3
"""No-op threshold for arm A 536: decode-time, no training, measurement only.

PRM03 spells a no-op as ``[NO_EDIT]`` + LF + ``>>>>>>> UPDATED`` + EOS. Its first token is
``[`` (id 80). In the C12 census every TRAIN gold target starts with 80 exactly when it is
a no-op, and 8,759 of the 8,767 arm A samples that start with 80 are ``[NO_EDIT]``. So the
first generated token is the no-op decision, and its probability ``p_noop`` can be exposed
as a user threshold: stay silent only when ``p_noop >= tau``, otherwise propose an edit.

For every case this job records, with arm A checkpoint 536 in bf16:

- the first-token distribution after the prompt (one unpadded forward over the prompt
  plus the no-op target): ``log p(80)``, the log-mass of every other token, the best
  other token, the top 5, ``p_noop``, the logit margin ``log p(80) - log(1 - p(80))``
  and the max margin ``log p(80) - max_other log p`` (greedy picks no-op iff it is > 0),
  plus the log-probability of each no-op target token and whether greedy would emit the
  whole no-op string;
- the forced-edit output: greedy generation with token 80 suppressed at the first step
  only (``begin_suppress_tokens``), through the C07 batched generator
  (``campaign_eval.batched_greedy_generate``: batch 32, dynamic KV cache, left padding,
  longest prompt first, cap 1,024, stop ids 1 and 130073), classified by
  ``campaign_eval.classify`` exactly like the gate. For cases whose greedy first token is
  not 80 this is the ordinary greedy output.

A forced no-op is the canonical no-op output, so every threshold's outcome follows from
the two records: ``p_noop >= tau`` -> no-op, else the forced-edit output. The greedy
reference (no-op iff the first-token argmax is 80) is recomputed the same way and
compared with the gate's per-case results.

Panels: DEV75 and DEV250 (the bound recipe's panels, sha256-checked) and TRAIN1000, a
seeded stratified sample of the C12 census prompts (TRAIN only): 200 no-op and 800 edit
prompts, edits stratified by family, i.e. DEV250's 4:1 edit/no-op composition. The
recommended default threshold is calibrated on TRAIN1000 only (highest exact edits +
correct no-ops on the predeclared grid; ties go to the higher threshold, i.e. more
proposals, because no-op false positives are the lowest priority). DEV numbers are
reported at that threshold and along the whole grid; nothing is selected on DEV.

Modes:

- ``--dry-run`` (CPU only, no CUDA): checks the checkpoint (config, size, full sha256),
  the recipe, the pinned parent tokenizer, the DEV panels and arm A's gate results
  (official counts recomputed); selects TRAIN1000, checks that ``build_training_row``
  reproduces the census prompt and target ids for every selected row and writes the
  subset file; checks the no-op target ids. ``--cpu-smoke`` then runs the real model on
  the CPU through the same measurement code as the GPU run (2 DEV75 + 2 TRAIN cases,
  cap 16), summarises it and re-runs it to prove the resume path skips finished batches.
- default (GPU, night window only): resumes the latest unfinished attempt under
  ``--output-dir`` or creates one, takes the CUDA lock through ``cuda_host_guard`` and
  runs itself with ``--inner``. Records are appended per batch (fsync) to
  ``records/<panel>.jsonl``; a restart skips finished batches. Before each batch it stops
  if the stop-request file exists, SIGTERM arrived, fewer than 4 minutes remain before
  ``SEPALITH_DEADLINE_EPOCH`` or the run budget (140 min) is used; it then exits 75 and
  the next window resumes. The curves are rewritten after every panel and at the end
  (also after a crash) into the attempt and ``--summary-dir``. Exit 0 when every panel is
  complete, 75 when stopped with work left, 1 when nothing was measured or the run failed.

CUDA: the model goes to ``cuda:0`` (``CUDA_VISIBLE_DEVICES=0`` leaves one device); every
CUDA call names device 0 explicitly (``gpu_device``, ``cuda_report``); there is no
``set_device``, ``synchronize`` or ``mem_get_info``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
DRY_RUN_SCHEMA = "sepalith.decode-time.noop-threshold-dry-run.v1"
RECEIPT_SCHEMA = "sepalith.decode-time.noop-threshold-receipt.v1"
SUMMARY_SCHEMA = "sepalith.decode-time.noop-threshold-summary.v1"
RECORD_SCHEMA = "sepalith.decode-time.noop-threshold-record.v1"
SUBSET_SCHEMA = "sepalith.decode-time.noop-train-subset.v1"
DEFAULT_RECIPE = PLAN / "c04/bound/sft-v2-lr1e-5.json"
DEFAULT_DRY_RUN = PLAN / "receipts/DT-noop-threshold-dry-run.json"
DEFAULT_SUMMARY_DIR = HERE / "results"
STATE = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/decode-time")
DEFAULT_SUBSET = STATE / "noop-train-subset.jsonl"
CUDA_LOCK = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock")

CHECKPOINT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536")
STEP = 536
# HF LFS hash == local manifest (C03 status, C12 census receipt).
WEIGHTS_SHA256 = "befe587a0d4969e33b0daaa8fe66247371ec32f331616db74724616878ad6030"
WEIGHTS_BYTES = 5033904496
TENSORS = 381
PARAMETERS = 2516756480
SAVED_DTYPES = {"BF16": 296, "F32": 85}  # the trainer saved the RMSNorm weights in F32
GATES = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates/SFT-v2-r4-lr1e-5/results")
GATE_RESULTS = {
    "dev75": (GATES / "dev-dev75-step-536.20261014T021133-673743425.json",
              "c6e77e49f8fc6323244b576fe2d0ee730c7c524c844490872ca0b0f0c3796e67"),
    "dev250": (GATES / "dev-dev250-step-536.20261014T021133-673743425.json",
               "0a1dbbd903dedc4ec392b2848f8b2ea89641e5228a466ca3a79fe1c4936ab9d3"),
}
# C09 gate table, decision-step-536.
GATE_RECORDED = {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 117}
CENSUS = Path("/home/m0hawk/.local/state/sepalith/c12/census-arm-a")
CENSUS_PROMPTS_SHA256 = "55194312168f7d636d78c8c1728b7f7457b06b07bd7bdad7fb4d65b93f445025"
SIDECAR = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/context-sidecar.jsonl")
SIDECAR_SHA256 = "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a"
FORBIDDEN = ("DAT-07-final", "sealed-final", "DAT-08", "spec_traces")

# "[NO_EDIT]\n>>>>>>> UPDATED" + EOS under the pinned tokenizer; the dry run checks it
# against build_training_row for every no-op case.
NOOP_TARGET = (80, 12991, 84, 2747, 2091, 1618, 97666, 24779, 651, 8089, 23877, 1)
NOOP_FIRST = NOOP_TARGET[0]
PANELS = ("dev75", "dev250", "train1000")
TRAIN_SEED = 20261022
TRAIN_NOOP = 200
TRAIN_EDIT = 800
# Predeclared thresholds on p_noop: stay silent iff p_noop >= tau. None = always propose.
TAU_GRID = (0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99,
            0.995, 0.999, 0.9999, None)
BATCH_RESERVE_SECONDS = 240
DEFAULT_BUDGET_MINUTES = 140
SMOKE_CAP = 16


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE, help="arm A's bound recipe (panels, cap, batching, tokenizer)")
    p.add_argument("--dry-run", action="store_true", help="CPU checks only; writes --dry-run-output and the TRAIN subset")
    p.add_argument("--dry-run-output", type=Path, default=DEFAULT_DRY_RUN)
    p.add_argument("--train-subset", type=Path, default=DEFAULT_SUBSET, help="TRAIN1000 rows (written by --dry-run)")
    p.add_argument("--cpu-smoke", action="store_true", help="with --dry-run: run the real model on the CPU on 4 cases")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for --dry-run (1 to 8)")
    p.add_argument("--output-dir", type=Path, help="GPU run: attempts live here; the latest unfinished one is resumed")
    p.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR, help="summary JSON and markdown")
    p.add_argument("--panels", default=",".join(PANELS), help="comma-separated subset of dev75,dev250,train1000 (in this order)")
    p.add_argument("--budget-minutes", type=float, default=DEFAULT_BUDGET_MINUTES, help="stop starting batches after this")
    p.add_argument("--guard-seconds", type=int, default=9000, help="cuda_host_guard hard limit (the hard stop caps it)")
    p.add_argument("--fresh", action="store_true", help="start a new attempt instead of resuming")
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
    panels = [x for x in args.panels.split(",") if x]
    if not panels or any(x not in PANELS for x in panels) or len(set(panels)) != len(panels):
        p.error(f"--panels must be a subset of {','.join(PANELS)}")
    args.panels = [x for x in PANELS if x in panels]
    if not 1 <= args.budget_minutes <= 150:
        p.error("--budget-minutes must be 1 to 150 (the job stays under 2.5 h)")
    if args.inner and args.attempt_dir is None:
        p.error("--inner needs --attempt-dir")
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


def guard_path(path):
    if any(token in str(path) for token in FORBIDDEN):
        raise SystemExit(f"refusing a sealed/final or pre-split input: {path}")
    return Path(path)


def deadline_left():
    value = os.environ.get("SEPALITH_DEADLINE_EPOCH")
    return math.inf if value in (None, "") else float(value) - time.time()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def finite(value, bound=1e4):
    value = float(value)
    if math.isnan(value):
        raise ValueError("NaN in a log-probability")
    return max(-bound, min(bound, value))


# CUDA (the only CUDA-touching code; device index 0 everywhere) --------------------------------------

def gpu_device():
    """The one visible GPU, with an explicit index (torch rejects a bare "cuda" in several calls)."""
    import torch

    return torch.device("cuda", 0)


def require_one_gpu():
    import torch

    if not (torch.cuda.is_available() and torch.cuda.device_count() == 1):
        raise SystemExit("exactly one visible CUDA device required")


def cuda_report(reset=False):
    import torch

    if reset:
        # reset_peak_memory_stats raises "Invalid device argument" before the CUDA context exists (2026-10-22 run).
        torch.empty(0, device=torch.device("cuda", 0))
        torch.cuda.reset_peak_memory_stats(0)
        return None
    return {"name": torch.cuda.get_device_name(0), "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 2**20),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 2**20)}


def release_cuda_cache():
    import torch

    torch.cuda.empty_cache()


# Inputs -------------------------------------------------------------------------------------------

def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    gate = recipe["dev_gate"]
    parent = recipe["parent"]
    return {
        "path": str(Path(path).resolve()), "sha256": sha256(path),
        "parent_path": parent["path"], "parent_files": parent["files"],
        "panels": {name: {"path": gate["panels"][name]["path"], "sha256": gate["panels"][name]["sha256"],
                          "cases": gate["panels"][name]["cases"]} for name in ("dev75", "dev250")},
        "cap": gate["generation_max_new_tokens"],
        "batch_size": gate["generation_batching"]["batch_size"],
        "kv_cache": gate["generation_batching"]["kv_cache"],
        "max_sequence_tokens": recipe["cohort"]["max_sequence_tokens"],
    }


def check_parent_tokenizer(recipe):
    parent = Path(recipe["parent_path"])
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha256(parent / name) != recipe["parent_files"][name]:
            raise SystemExit(f"recipe parent {name} bytes differ")
    return parent


def check_checkpoint(full_hash=True):
    path = CHECKPOINT
    for name in ("config.json", "generation_config.json", "model.safetensors"):
        if not (path / name).is_file():
            raise SystemExit(f"checkpoint lacks {name}")
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != ["LlamaForCausalLM"] or list(config.get("eos_token_id") or []) != [1, 130073] \
            or config.get("vocab_size") != 130560:
        raise SystemExit("checkpoint config differs")
    weights = path / "model.safetensors"
    stat = weights.stat()
    if stat.st_size != WEIGHTS_BYTES:
        raise SystemExit(f"model.safetensors has {stat.st_size} bytes, expected {WEIGHTS_BYTES}")
    with weights.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    numel = sum(math.prod(v["shape"]) for v in header.values())
    dtypes = dict(Counter(v["dtype"] for v in header.values()))
    if len(header) != TENSORS or numel != PARAMETERS or dtypes != SAVED_DTYPES:
        raise SystemExit(f"checkpoint tensors differ: {len(header)} tensors, {numel} parameters, dtypes {dtypes}")
    record = {"path": str(path), "weights_bytes": stat.st_size, "weights_mtime_ns": stat.st_mtime_ns,
              "tensors": len(header), "parameters": numel, "saved_dtypes": dtypes,
              "load_dtype": "bf16 (campaign_eval.load_development_model casts the 85 F32 norm weights to bf16)"}
    if full_hash:
        actual = sha256(weights)
        if actual != WEIGHTS_SHA256:
            raise SystemExit(f"weights sha256 {actual} differs from {WEIGHTS_SHA256}")
        record["weights_sha256"] = actual
    return record


def verify_weights(dry):
    """Size and mtime against the dry-run receipt; a full sha256 when either differs."""
    record = dry.get("checkpoint") or {}
    stat = (CHECKPOINT / "model.safetensors").stat()
    if (record.get("weights_sha256") == WEIGHTS_SHA256 and record.get("weights_bytes") == stat.st_size
            and record.get("weights_mtime_ns") == stat.st_mtime_ns):
        return {"method": "dry-run receipt (sha256, size, mtime_ns unchanged)", "sha256": WEIGHTS_SHA256}
    actual = sha256(CHECKPOINT / "model.safetensors")
    if actual != WEIGHTS_SHA256:
        raise SystemExit(f"weights sha256 {actual} differs from {WEIGHTS_SHA256}")
    return {"method": "full sha256 in the job", "sha256": actual}


def dev_cases(recipe, name, tokenizer):
    """DEV rows rendered exactly as the evaluator renders them."""
    from sepalith.campaign_protocol import PromptContext, build_training_row
    from sepalith.training.eval import campaign_eval

    panel = recipe["panels"][name]
    guard_path(panel["path"])
    if sha256(panel["path"]) != panel["sha256"]:
        raise SystemExit(f"{name} panel sha256 differs")
    cases = []
    for line in Path(panel["path"]).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] != "dev":
            raise SystemExit("only DEV rows may enter a DEV panel")
        context = PromptContext.from_mapping(row["context"])
        built = build_training_row(context, operation=row["operation"], region_new=row["region_new"], tokenizer=tokenizer,
                                   row_id=row["id"], family=row["family"], package_id=row["package_id"], split="dev")
        campaign_eval.validate_development_capacity(built, recipe["max_sequence_tokens"], recipe["cap"], row["id"])
        cases.append(make_case(row["id"], row["family"], "dev", built, context, row["region_new"]))
    if len(cases) != panel["cases"]:
        raise SystemExit(f"{name} has {len(cases)} rows, expected {panel['cases']}")
    return cases


def make_case(case_id, family, split, built, context, region_new):
    start = built["target_start"]
    return {"id": case_id, "family": family, "split": split, "prompt_ids": list(built["input_ids"][:start]),
            "target_ids": list(built["input_ids"][start:]), "context": context, "region_new": list(region_new),
            "expected_noop": list(context.region_old) == list(region_new)}


def train_cases(path, recipe, expected_sha256=None):
    from sepalith.campaign_protocol import PromptContext

    guard_path(path)
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise SystemExit("TRAIN subset sha256 differs from the dry-run receipt")
    cases = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("schema") != SUBSET_SCHEMA or row["split"] != "train":
            raise SystemExit("TRAIN subset rows must be TRAIN rows of this schema")
        context = PromptContext.from_mapping(row["context"])
        if len(row["prompt_ids"]) + recipe["cap"] > recipe["max_sequence_tokens"]:
            raise SystemExit(f"TRAIN row exceeds the context capacity: {row['id']}")
        cases.append({"id": row["id"], "family": row["family"], "split": "train", "prompt_ids": row["prompt_ids"],
                      "target_ids": row["target_ids"], "context": context, "region_new": row["region_new"],
                      "expected_noop": list(context.region_old) == list(row["region_new"])})
    if len(cases) != TRAIN_NOOP + TRAIN_EDIT or len({c["id"] for c in cases}) != len(cases):
        raise SystemExit("TRAIN subset size or ids differ")
    return cases


def select_train_subset(recipe, tokenizer, output):
    """Seeded stratified TRAIN1000 from the C12 census prompts; prompt/target parity is checked per row."""
    from sepalith.campaign_protocol import PromptContext, build_training_row, parse_output

    guard_path(SIDECAR)
    summary = json.loads((CENSUS / "census-summary.json").read_text())
    if summary["data"]["context"]["sha256"] != SIDECAR_SHA256 or summary["files"]["census-prompts.jsonl"]["sha256"] != CENSUS_PROMPTS_SHA256:
        raise SystemExit("census identities differ")
    if sha256(CENSUS / "census-prompts.jsonl") != CENSUS_PROMPTS_SHA256:
        raise SystemExit("census-prompts.jsonl sha256 differs")
    prompts = {}
    with (CENSUS / "census-prompts.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "train":
                raise SystemExit("census prompt is not TRAIN")
            prompts[row["id"]] = row
    by_family = {}
    for rid in sorted(prompts):
        by_family.setdefault(prompts[rid]["family"], []).append(rid)
    rng = random.Random(TRAIN_SEED)
    for family in sorted(by_family):
        rng.shuffle(by_family[family])
    edit_families = sorted(f for f in by_family if f != "no_op")
    total_edit = sum(len(by_family[f]) for f in edit_families)
    exact = {f: TRAIN_EDIT * len(by_family[f]) / total_edit for f in edit_families}
    quota = {f: int(exact[f]) for f in edit_families}
    for f in sorted(edit_families, key=lambda f: (-(exact[f] - quota[f]), f))[:TRAIN_EDIT - sum(quota.values())]:
        quota[f] += 1
    quota["no_op"] = TRAIN_NOOP
    # Candidates in shuffled order, three times the quota, so parity failures can be replaced.
    wanted = {rid for f, q in quota.items() for rid in by_family[f][:3 * q + 10]}
    contexts = {}
    with SIDECAR.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["row_id"] in wanted:
                if row["split"] != "train":
                    raise SystemExit("sidecar row is not TRAIN")
                contexts[row["row_id"]] = (row["context"], row["package_id"])
    rows, rejected = [], Counter()
    for family in sorted(quota):
        taken = 0
        for rid in by_family[family]:
            if taken == quota[family]:
                break
            if rid not in contexts:
                rejected["no_context"] += 1
                continue
            census = prompts[rid]
            mapping, package_id = contexts[rid]
            context = PromptContext.from_mapping(mapping)
            target_ids = json.loads(census["target_ids"]) if isinstance(census["target_ids"], str) else census["target_ids"]
            prompt_ids = json.loads(census["prompt_token_ids"]) if isinstance(census["prompt_token_ids"], str) else census["prompt_token_ids"]
            if target_ids[-1] != 1:
                rejected["target_without_eos"] += 1
                continue
            raw = tokenizer.decode(target_ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            parsed = parse_output(raw, context)
            if parsed.status != "accepted":
                rejected["target_unparsed"] += 1
                continue
            operation = parsed.operation
            region_new = list(context.region_old) if operation == "no_op" else list(parsed.body)
            try:
                built = build_training_row(context, operation=operation, region_new=region_new, tokenizer=tokenizer,
                                           row_id=rid, family=family, package_id=package_id, split="train")
            except ValueError:
                rejected["build_failed"] += 1
                continue
            start = built["target_start"]
            if list(built["input_ids"][:start]) != list(prompt_ids) or list(built["input_ids"][start:]) != list(target_ids):
                rejected["census_parity"] += 1
                continue
            if start + recipe["cap"] > recipe["max_sequence_tokens"]:
                rejected["capacity"] += 1
                continue
            if (family == "no_op") != (operation == "no_op"):
                rejected["family_operation_mismatch"] += 1
                continue
            rows.append({"schema": SUBSET_SCHEMA, "id": rid, "family": family, "split": "train", "package_id": package_id,
                         "operation": operation, "region_new": region_new, "context": mapping,
                         "prompt_ids": list(prompt_ids), "target_ids": list(target_ids)})
            taken += 1
        if taken != quota[family]:
            raise SystemExit(f"TRAIN subset: only {taken} of {quota[family]} {family} rows pass the checks")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, output)
    ids = sorted(r["id"] for r in rows)
    return {"path": str(output), "sha256": sha256(output), "rows": len(rows), "seed": TRAIN_SEED, "quota": quota,
            "families": dict(Counter(r["family"] for r in rows)), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "rejected_candidates": dict(rejected),
            "sources": {"census_prompts": {"path": str(CENSUS / "census-prompts.jsonl"), "sha256": CENSUS_PROMPTS_SHA256},
                        "context_sidecar": {"path": str(SIDECAR), "sha256": SIDECAR_SHA256,
                                            "note": "identity from census-summary.json"}},
            "note": "seeded stratified TRAIN sample: 200 no-op + 800 edit prompts (edits by family share), DEV250's 4:1 mix; "
                    "build_training_row reproduces the census prompt and target ids for every row"}


def gate_reference():
    """Arm A 536's gate results (sha256-checked) and their per-case outcomes."""
    from sepalith.training.sft import gate_rule

    summaries, outcomes, files = {}, {}, {}
    for name, (path, digest) in GATE_RESULTS.items():
        if sha256(path) != digest:
            raise SystemExit(f"arm A {name} gate result sha256 differs")
        result = json.loads(path.read_text())
        cases = Path(result["cases"]["path"])
        if sha256(cases) != result["cases"]["sha256"]:
            raise SystemExit(f"arm A {name} cases sha256 differs")
        summaries[name] = result["summary"]
        outcomes[name] = {r["id"]: {k: r[k] for k in ("protocol_valid", "exact_region", "predicted_noop", "suggestion", "cap_hit")}
                          | {"first_token": r["generated_ids"][0] if r["generated_ids"] else None}
                          for r in json.loads(cases.read_text())["results"]}
        files[name] = {"path": str(path), "sha256": digest, "cases": result["cases"]}
    metrics = gate_rule.metrics_from_summaries(summaries["dev75"], summaries["dev250"])
    if metrics != GATE_RECORDED:
        raise SystemExit(f"arm A 536 recomputed metrics {metrics} differ from C09 {GATE_RECORDED}")
    return {"metrics": metrics, "files": files}, outcomes


# Measurement (shared by the GPU run and the CPU smoke) ------------------------------------------------

class SuppressNoopFirst:
    """Hand ``batched_greedy_generate`` a model whose generate bans token 80 at the first step only."""

    def __init__(self, model):
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def generate(self, **kwargs):
        return self.model.generate(**kwargs, begin_suppress_tokens=[NOOP_FIRST])


def first_token_record(model, prompt_ids, device):
    """One unpadded forward over prompt + no-op target; log-probs of the first token and of the no-op string."""
    import torch

    ids = torch.tensor([list(prompt_ids) + list(NOOP_TARGET[:-1])], dtype=torch.long, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=len(NOOP_TARGET))
    logits = output.logits[0, -len(NOOP_TARGET):].float()
    del output
    logprobs = torch.log_softmax(logits, dim=-1)
    first = logprobs[0]
    other = first.clone()
    other[NOOP_FIRST] = -math.inf
    best_other_lp, best_other_id = other.max(dim=-1)
    top_lp, top_id = first.topk(5)
    target = torch.tensor(NOOP_TARGET, device=logprobs.device)
    chain = logprobs.gather(1, target[:, None])[:, 0]
    lp_noop = finite(first[NOOP_FIRST].item())
    lse_other = finite(torch.logsumexp(other, dim=-1).item())
    return {
        "lp_noop": lp_noop, "lse_other": lse_other, "p_noop": math.exp(lp_noop),
        "margin_logit": finite(lp_noop - lse_other), "margin_max": finite(lp_noop - best_other_lp.item()),
        "argmax_id": int(first.argmax().item()), "best_other_id": int(best_other_id.item()),
        "best_other_lp": finite(best_other_lp.item()),
        "top5": [[int(i), finite(v)] for i, v in zip(top_id.tolist(), top_lp.tolist(), strict=True)],
        "noop_chain_lps": [finite(v) for v in chain.tolist()], "noop_chain_lp": finite(chain.sum().item()),
        "noop_chain_greedy": bool((logprobs.argmax(dim=-1) == target).all().item()),
    }


def classify_generation(tokenizer, case, generated, cap):
    from sepalith.campaign_protocol import NATIVE_EOG_IDS
    from sepalith.training.eval import campaign_eval

    termination = campaign_eval.generation_termination(generated, cap, NATIVE_EOG_IDS)
    raw_ids = generated[:-1] if termination["response_complete"] else generated
    raw = tokenizer.decode(raw_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    outcome = campaign_eval.classify(raw, case["context"], case["region_new"], generated)
    return {**termination, **outcome, "raw_output": raw, "generated_ids": list(generated),
            "generated_tokens": len(generated), "first_token": generated[0] if generated else None}


def read_records(path):
    """Finished records of a panel; a torn trailing line (crash mid-write) is dropped and truncated away."""
    path = Path(path)
    if not path.is_file():
        return []
    data = path.read_bytes()
    records, good = [], 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
        good += len(line)
    if good != len(data):
        with path.open("r+b") as stream:
            stream.truncate(good)
    return records


def append_records(path, records):
    with Path(path).open("a") as stream:
        stream.write("".join(json.dumps(r, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n" for r in records))
        stream.flush()
        os.fsync(stream.fileno())


def measure_panel(model, tokenizer, name, cases, *, device, cap, batch_size, kv_cache, records_path, should_stop,
                  log=print):
    """Margins + forced-edit generation per batch; returns the panel status. Finished batches are skipped."""
    import torch

    from sepalith.campaign_protocol import NATIVE_EOG_IDS
    from sepalith.training.eval import campaign_eval

    done = {r["id"] for r in read_records(records_path)}
    if not done <= {c["id"] for c in cases}:
        raise SystemExit(f"{name}: records contain foreign case ids")
    batches = campaign_eval.generation_batches([(c["id"], len(c["prompt_ids"])) for c in cases], batch_size)
    started, new, stopped = time.monotonic(), 0, None
    wrapped = SuppressNoopFirst(model)
    for number, indexes in enumerate(batches):
        ids = [cases[i]["id"] for i in indexes]
        if all(i in done for i in ids):
            continue
        if any(i in done for i in ids):
            raise SystemExit(f"{name}: batch {number} is partly recorded")
        if (stopped := should_stop()) is not None:
            break
        t0 = time.monotonic()
        with campaign_eval.case_evaluation_guard(model):
            with torch.no_grad():
                firsts = [first_token_record(model, cases[i]["prompt_ids"], device) for i in indexes]
                forward_seconds = time.monotonic() - t0
                t1 = time.monotonic()
                generated = campaign_eval.batched_greedy_generate(
                    wrapped, [cases[i]["prompt_ids"] for i in indexes], cap=cap, stop_ids=NATIVE_EOG_IDS,
                    kv_cache=kv_cache, device=device)
                generation_seconds = time.monotonic() - t1
        records = []
        for i, first, out in zip(indexes, firsts, generated, strict=True):
            case = cases[i]
            forced = classify_generation(tokenizer, case, out, cap)
            if forced["first_token"] == NOOP_FIRST:
                raise RuntimeError(f"{case['id']}: the suppressed no-op token was generated first")
            records.append({"schema": RECORD_SCHEMA, "panel": name, "id": case["id"], "family": case["family"],
                            "split": case["split"], "expected_noop": case["expected_noop"],
                            "prompt_tokens": len(case["prompt_ids"]), "batch": number, "first": first,
                            "forced_edit": forced, "batch_forward_seconds": round(forward_seconds, 3),
                            "batch_generation_seconds": round(generation_seconds, 3)})
        append_records(records_path, records)
        done.update(ids)
        new += len(ids)
        log(json.dumps({"panel": name, "batch": number, "batches": len(batches), "cases": len(ids), "done": len(done),
                        "seconds": round(time.monotonic() - t0, 1)}), flush=True)
    status = "complete" if len(done) == len(cases) else ("stopped" if stopped else "incomplete")
    return {"status": status, "done": len(done), "cases": len(cases), "new": new, "stop_reason": stopped,
            "seconds": round(time.monotonic() - started, 1), "batches": len(batches)}


# Curves -------------------------------------------------------------------------------------------

NOOP_OUTCOME = {"protocol_valid": True, "predicted_noop": True, "suggestion": False, "cap_hit": False}


def decide_tau(tau):
    return (lambda r: "edit") if tau is None else (lambda r: "noop" if r["first"]["p_noop"] >= tau else "edit")


def decide_greedy(r):
    return "noop" if r["first"]["argmax_id"] == NOOP_FIRST else "edit"


def outcome(record, decision):
    if decision == "noop":
        return {**NOOP_OUTCOME, "exact_region": record["expected_noop"]}
    return record["forced_edit"]


def metrics(records, decide):
    m = Counter()
    for r in records:
        o = outcome(r, decide(r))
        m["cases"] += 1
        m["valid"] += o["protocol_valid"]
        m["cap"] += o["cap_hit"]
        m["suggestions"] += o["suggestion"]
        m["predicted_noop"] += o["predicted_noop"]
        if r["expected_noop"]:
            m["noop_cases"] += 1
            m["noop_correct"] += o["predicted_noop"]
            m["nofp"] += o["suggestion"]
        else:
            m["edit_cases"] += 1
            m["edit_exact"] += o["exact_region"]
            m["missed_edit"] += o["predicted_noop"]
    out = {k: m[k] for k in ("cases", "valid", "cap", "suggestions", "predicted_noop", "noop_cases", "noop_correct",
                             "nofp", "edit_cases", "edit_exact", "missed_edit")}
    out["score"] = out["edit_exact"] + out["noop_correct"]
    out["noop_recall"] = round(out["noop_correct"] / out["noop_cases"], 4) if out["noop_cases"] else None
    out["edit_exact_rate"] = round(out["edit_exact"] / out["edit_cases"], 4) if out["edit_cases"] else None
    return out


def curve(records):
    return [{"tau": tau, "rule": "always propose" if tau is None else f"no-op iff p_noop >= {tau}",
             **metrics(records, decide_tau(tau))} for tau in TAU_GRID]


def calibrate(train_curve):
    """Highest score on the grid; ties go to the higher threshold (more proposals). None (always propose) ranks highest."""
    order = {tau: i for i, tau in enumerate(TAU_GRID)}
    best = max(train_curve, key=lambda row: (row["score"], order[row["tau"]]))
    return best["tau"]


def roc_auc(records):
    """P(margin of a no-op case > margin of an edit case); ties count half."""
    pos = sorted(r["first"]["margin_logit"] for r in records if r["expected_noop"])
    neg = sorted(r["first"]["margin_logit"] for r in records if not r["expected_noop"])
    if not pos or not neg:
        return None
    import bisect

    wins = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos)
    return round(wins / (len(pos) * len(neg)), 4)


def greedy_check(records, gate_outcomes=None):
    """The recomputed greedy reference, its consistency and (DEV) its agreement with the gate's per-case results."""
    unverified = sum(1 for r in records if decide_greedy(r) == "noop" and not r["first"]["noop_chain_greedy"])
    first_differs = sum(1 for r in records if decide_greedy(r) == "edit" and r["forced_edit"]["first_token"] != r["first"]["argmax_id"])
    check = {"metrics": metrics(records, decide_greedy), "noop_first_but_chain_not_greedy": unverified,
             "edit_first_token_differs_from_forward_argmax": first_differs}
    if gate_outcomes is not None:
        agree, differ = 0, []
        for r in records:
            g = gate_outcomes.get(r["id"])
            o = outcome(r, decide_greedy(r))
            same = g is not None and all(bool(o[k]) == bool(g[k]) for k in ("protocol_valid", "exact_region", "predicted_noop"))
            agree += same
            if not same:
                differ.append(r["id"])
        check["gate_case_agreement"] = {"agree": agree, "of": len(records), "differ": differ}
    return check


def per_family(records, decide):
    families = sorted({r["family"] for r in records})
    return {f: metrics([r for r in records if r["family"] == f], decide) for f in families}


def summarize(attempt, summary_dir, panels, gate=None, gate_outcomes=None):
    """Curves from whatever records exist; rewritten after every panel and at the end."""
    records = {name: read_records(Path(attempt) / "records" / f"{name}.jsonl") for name in panels}
    receipt_path = Path(attempt) / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    expected = {name: (receipt.get("panels") or {}).get(name, {}).get("cases") for name in panels}
    summary = {"schema": SUMMARY_SCHEMA, "written": now(), "attempt_dir": str(attempt),
               "purpose": "measurement only: decode-time no-op threshold for arm A 536; calibrated on TRAIN1000; DEV is report-only",
               "final_set_access": False, "status": receipt.get("status"), "tau_grid": list(TAU_GRID),
               "rule": "stay silent (no-op) iff p_noop >= tau, else greedy with the no-op first token suppressed",
               "gate_reference": gate, "panels": {}}
    for name in panels:
        rows = records[name]
        complete = expected[name] is not None and len(rows) == expected[name]
        entry = {"records": len(rows), "expected": expected[name], "complete": complete}
        if rows:
            entry.update(curve=curve(rows), greedy=greedy_check(rows, (gate_outcomes or {}).get(name)),
                         auc_margin_logit=roc_auc(rows),
                         p_noop_quantiles={kind: quantiles([r["first"]["p_noop"] for r in rows if r["expected_noop"] == flag])
                                           for kind, flag in (("noop_cases", True), ("edit_cases", False))},
                         forced_edit_on_noop_cases=dict(Counter(
                             "echo_region_parsed_noop" if r["forced_edit"]["predicted_noop"] else
                             "suggestion" if r["forced_edit"]["suggestion"] else "invalid"
                             for r in rows if r["expected_noop"])),
                         seconds=batch_seconds(rows))
        summary["panels"][name] = entry
    train = summary["panels"].get("train1000") or {}
    if train.get("curve"):
        tau = calibrate(train["curve"])
        summary["calibration"] = {
            "on": "train1000", "complete": train["complete"], "recommended_tau": tau,
            "objective": "exact edits + correct no-ops on TRAIN1000 (800 edit / 200 no-op); ties -> higher tau (more proposals)",
            "train_at_tau": metrics(records["train1000"], decide_tau(tau)),
            "train_greedy": train["greedy"]["metrics"],
            "edit_max_tau": max(train["curve"], key=lambda row: (row["edit_exact"], -TAU_GRID.index(row["tau"])))["tau"],
        }
        for name in ("dev75", "dev250"):
            if records.get(name):
                summary["calibration"][f"{name}_at_tau"] = metrics(records[name], decide_tau(tau))
                summary["calibration"][f"{name}_families_at_tau"] = per_family(records[name], decide_tau(tau))
        if not train["complete"]:
            summary["calibration"]["warning"] = "TRAIN1000 is incomplete; the recommendation is provisional"
    write_json(Path(attempt) / "summary.json", summary)
    if summary_dir is not None:
        stamp = Path(attempt).name
        write_json(Path(summary_dir) / f"noop-threshold-{stamp}.json", summary)
        (Path(summary_dir) / f"noop-threshold-{stamp}.md").write_text(markdown(summary))
    return summary


def batch_seconds(rows):
    batches = {r["batch"]: (r["batch_forward_seconds"], r["batch_generation_seconds"]) for r in rows}
    return {"forward": round(sum(f for f, _ in batches.values()), 1), "generation": round(sum(g for _, g in batches.values()), 1),
            "batches": len(batches)}


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]  # noqa: E731
    return {"n": len(values), "p10": round(pick(0.1), 5), "p25": round(pick(0.25), 5), "median": round(pick(0.5), 5),
            "p75": round(pick(0.75), 5), "p90": round(pick(0.9), 5)}


def markdown(summary):
    lines = ["# No-op threshold for arm A 536 (decode-time, measurement only)", "",
             f"Status: {summary.get('status')}. Written {summary['written']}.", "",
             "Rule: stay silent (no-op) iff the first-token probability of `[NO_EDIT]` (token 80) is at least tau; "
             "otherwise generate greedily with token 80 suppressed at the first step. tau is calibrated on TRAIN1000 "
             "only (200 no-op + 800 edit census prompts); DEV is reported, not used for selection.", ""]
    cal = summary.get("calibration")
    if cal:
        lines += [f"Recommended default (TRAIN1000{'' if cal['complete'] else ', provisional'}): tau = {cal['recommended_tau']}.",
                  f"Edit-maximising tau on TRAIN1000: {cal['edit_max_tau']}.", ""]
    for name, entry in summary["panels"].items():
        lines += [f"## {name} ({entry['records']} of {entry['expected']} cases)", ""]
        if not entry.get("curve"):
            lines += ["No records.", ""]
            continue
        g = entry["greedy"]["metrics"]
        lines += [f"Greedy reference: score {g['score']}, edit exact {g['edit_exact']}/{g['edit_cases']}, no-op correct "
                  f"{g['noop_correct']}/{g['noop_cases']}, nofp {g['nofp']}, valid {g['valid']}, cap {g['cap']}."]
        if "gate_case_agreement" in entry["greedy"]:
            a = entry["greedy"]["gate_case_agreement"]
            lines.append(f"Per-case agreement with the gate's outcomes: {a['agree']} of {a['of']}.")
        lines += [f"ROC AUC of the logit margin (no-op vs edit cases): {entry['auc_margin_logit']}.", "",
                  "| tau | predicted no-op | nofp | no-op recall | edit exact | missed edits | score | valid | cap |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in entry["curve"]:
            tau = "always propose" if row["tau"] is None else row["tau"]
            lines.append(f"| {tau} | {row['predicted_noop']} | {row['nofp']}/{row['noop_cases']} | {row['noop_recall']} | "
                         f"{row['edit_exact']}/{row['edit_cases']} | {row['missed_edit']} | {row['score']} | {row['valid']} | {row['cap']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# Dry run (CPU) ------------------------------------------------------------------------------------

def dry_run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer

    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        raise SystemExit("the dry run must not see a CUDA device")
    started = time.monotonic()
    recipe = load_recipe(args.recipe)
    parent = check_parent_tokenizer(recipe)
    reference = load_pinned_reference_tokenizer(parent)
    receipt = {"schema": DRY_RUN_SCHEMA, "status": "running", "started": now(), "cuda_started": False,
               "final_set_access": False, "recipe": {"path": recipe["path"], "sha256": recipe["sha256"]},
               "settings": settings_record(recipe), "noop_target_ids": list(NOOP_TARGET), "tau_grid": list(TAU_GRID)}
    receipt["checkpoint"] = check_checkpoint(full_hash=True)
    gate, _ = gate_reference()
    receipt["gate_reference"] = gate
    problems = []
    panels = {}
    for name in ("dev75", "dev250"):
        cases = dev_cases(recipe, name, reference)
        panels[name] = cases
        receipt.setdefault("panels", {})[name] = {**recipe["panels"][name], **panel_stats(cases)}
    receipt["train_subset"] = select_train_subset(recipe, reference, args.train_subset)
    panels["train1000"] = train_cases(args.train_subset, recipe, receipt["train_subset"]["sha256"])
    receipt["panels"]["train1000"] = {"path": str(args.train_subset), "sha256": receipt["train_subset"]["sha256"],
                                      "cases": len(panels["train1000"]), **panel_stats(panels["train1000"])}
    # The no-op target and the first-token signal, on every case's reference target.
    signal_check = Counter()
    for name, cases in panels.items():
        for c in cases:
            starts = c["target_ids"][:1] == [NOOP_FIRST]
            if c["expected_noop"]:
                signal_check[f"{name}:noop_target_is_NOOP_TARGET"] += tuple(c["target_ids"]) == NOOP_TARGET
                signal_check[f"{name}:noop_cases"] += 1
            else:
                signal_check[f"{name}:edit_target_starts_with_80"] += starts
    receipt["noop_signal_check"] = dict(signal_check)
    for name in panels:
        if signal_check[f"{name}:noop_target_is_NOOP_TARGET"] != signal_check[f"{name}:noop_cases"]:
            problems.append(f"{name}: a no-op target differs from NOOP_TARGET")
        if signal_check[f"{name}:edit_target_starts_with_80"]:
            problems.append(f"{name}: an edit target starts with token 80")
    write_json(args.dry_run_output, receipt)
    if args.cpu_smoke:
        receipt["cpu_smoke"] = cpu_smoke(args, recipe, reference, parent, panels)
        if receipt["cpu_smoke"]["status"] != "complete":
            problems.append(f"cpu smoke: {receipt['cpu_smoke'].get('error')}")
    receipt.update(status="pass" if not problems else "fail", problems=problems, finished=now(),
                   elapsed_seconds=round(time.monotonic() - started, 1))
    write_json(args.dry_run_output, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(args.dry_run_output), "problems": problems}), flush=True)
    return 0 if not problems else 1


def panel_stats(cases):
    lengths = sorted(len(c["prompt_ids"]) for c in cases)
    return {"noop_cases": sum(c["expected_noop"] for c in cases), "families": dict(Counter(c["family"] for c in cases)),
            "prompt_tokens": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1],
                              "sum": sum(lengths)}}


def settings_record(recipe):
    return {"dtype": "bf16", "renderer_id": "zeta2-prm03-v1", "cap": recipe["cap"], "batch_size": recipe["batch_size"],
            "kv_cache": recipe["kv_cache"], "max_sequence_tokens": recipe["max_sequence_tokens"],
            "generation": "sepalith.training.eval.campaign_eval.batched_greedy_generate + begin_suppress_tokens=[80]",
            "classification": "sepalith.training.eval.campaign_eval.classify",
            "tokenizer": f"pinned tokenizer of the recipe parent ({recipe['parent_path']}), byte-identical to arm A 536's",
            "device": "cuda:0 (CUDA_VISIBLE_DEVICES=0)"}


def load_model_and_tokenizer(parent, reference, device_name):
    from transformers import AutoTokenizer

    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )
    from sepalith.training.eval import campaign_eval

    tokenizer = AutoTokenizer.from_pretrained(str(parent), local_files_only=True, trust_remote_code=False)
    model = campaign_eval.load_development_model(CHECKPOINT, dtype="bf16", device=device_name)
    repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    audit = assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "noop_threshold_load")
    named = list(model.named_parameters())
    if len(named) != TENSORS or sum(x.numel() for _, x in named) != PARAMETERS:
        raise ValueError("not the exact dense checkpoint")
    model.eval()
    return model, tokenizer, {"tokenizer_contract": audit, "tokenizer_repair_changed": repair["backend_repair"]["changed"]}


def cpu_smoke(args, recipe, reference, parent, panels):
    """The real model on the CPU through measure_panel + summarize, then a resume.

    Per panel (DEV75, TRAIN1000) the shortest no-op case and the shortest edit case, cap 16.
    """
    import torch

    started = time.monotonic()
    try:
        model, tokenizer, contract = load_model_and_tokenizer(parent, reference, "cpu")
        device = torch.device("cpu")
        def shortest(cases):  # the shortest no-op case and the shortest edit case
            order = sorted(cases, key=lambda c: (len(c["prompt_ids"]), c["id"]))
            return [next(c for c in order if c["expected_noop"]), next(c for c in order if not c["expected_noop"])]

        smoke = {"dev75": shortest(panels["dev75"]), "train1000": shortest(panels["train1000"])}
        with tempfile.TemporaryDirectory(prefix="noop-threshold-smoke-") as tmp:
            attempt = Path(tmp) / "attempt"
            (attempt / "records").mkdir(parents=True)
            write_json(attempt / "receipt.json", {"status": "running",
                                                  "panels": {n: {"cases": len(c)} for n, c in smoke.items()}})
            results, rerun = {}, {}
            for name, cases in smoke.items():
                results[name] = measure_panel(model, tokenizer, name, cases, device=device, cap=SMOKE_CAP,
                                              batch_size=recipe["batch_size"], kv_cache=recipe["kv_cache"],
                                              records_path=attempt / "records" / f"{name}.jsonl", should_stop=lambda: None)
            for name, cases in smoke.items():
                rerun[name] = measure_panel(model, tokenizer, name, cases, device=device, cap=SMOKE_CAP,
                                            batch_size=recipe["batch_size"], kv_cache=recipe["kv_cache"],
                                            records_path=attempt / "records" / f"{name}.jsonl", should_stop=lambda: None)
            summary = summarize(attempt, None, list(smoke))
            records = {n: read_records(attempt / "records" / f"{n}.jsonl") for n in smoke}
        if any(r["new"] for r in rerun.values()):
            raise RuntimeError("the resume re-measured finished batches")
        return {"status": "complete", "cap": SMOKE_CAP, "panels": results, "resume_new_cases": {n: r["new"] for n, r in rerun.items()},
                "cases": {n: [{"id": r["id"], "expected_noop": r["expected_noop"], "p_noop": r["first"]["p_noop"],
                               "argmax_id": r["first"]["argmax_id"], "noop_chain_greedy": r["first"]["noop_chain_greedy"],
                               "forced_first_token": r["forced_edit"]["first_token"],
                               "forced_generated_tokens": r["forced_edit"]["generated_tokens"]} for r in rs]
                          for n, rs in records.items()},
                "summary_calibration": {k: v for k, v in (summary.get("calibration") or {}).items() if not k.endswith("families_at_tau")},
                **contract,
                "seconds": round(time.monotonic() - started, 1),
                "note": "pipeline check only (CPU bf16, 16-token cap); not a measurement"}
    except Exception as error:  # noqa: BLE001 - the smoke result records the failure
        return {"status": "failed", "error": f"{type(error).__name__}: {error}", "seconds": round(time.monotonic() - started, 1)}


# GPU run (inner) ----------------------------------------------------------------------------------

def inner(args):
    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer

    attempt = args.attempt_dir
    (attempt / "records").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stop = {"signal": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("signal", True))
    receipt_path = attempt / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {
        "schema": RECEIPT_SCHEMA, "started": now(), "attempt_dir": str(attempt), "runs": [],
        "purpose": "measurement only: no adoption decision, no rule change", "final_set_access": False}
    run = {"started": now(), "pid": os.getpid(), "budget_minutes": args.budget_minutes}
    receipt["runs"].append(run)
    receipt["status"] = "running"
    gate = gate_outcomes = None

    def save():
        write_json(receipt_path, receipt)

    def should_stop():
        stop_file = os.environ.get("SEPALITH_STOP_REQUEST_FILE")
        if stop["signal"]:
            return "sigterm"
        if stop_file and Path(stop_file).exists():
            return "stop_request_file"
        if deadline_left() < BATCH_RESERVE_SECONDS:
            return "window_deadline"
        if time.monotonic() - started > args.budget_minutes * 60:
            return "run_budget"
        return None

    try:
        recipe = load_recipe(args.recipe)
        dry = json.loads(args.dry_run_output.read_text()) if args.dry_run_output.is_file() else {}
        if dry.get("schema") != DRY_RUN_SCHEMA or dry.get("status") != "pass":
            raise SystemExit(f"a passing dry-run receipt is required: {args.dry_run_output}")
        require_one_gpu()
        parent = check_parent_tokenizer(recipe)
        for name in ("dev75", "dev250"):
            if recipe["panels"][name]["sha256"] != dry["panels"][name]["sha256"]:
                raise SystemExit(f"{name} panel differs from the dry run")
        receipt.update(recipe={"path": recipe["path"], "sha256": recipe["sha256"]}, settings=settings_record(recipe),
                       dry_run={"path": str(args.dry_run_output.resolve()), "sha256": sha256(args.dry_run_output)})
        gate, gate_outcomes = gate_reference()
        receipt["gate_reference"] = gate
        reference = load_pinned_reference_tokenizer(parent)
        cases = {}
        for name in args.panels:
            cases[name] = (train_cases(args.train_subset, recipe, dry["train_subset"]["sha256"]) if name == "train1000"
                           else dev_cases(recipe, name, reference))
            receipt.setdefault("panels", {}).setdefault(name, {})["cases"] = len(cases[name])
        save()
        remaining = [n for n in args.panels if len(read_records(attempt / "records" / f"{n}.jsonl")) < len(cases[n])]
        if remaining and should_stop() is None:
            run["weights"] = verify_weights(dry)
            device = gpu_device()
            cuda_report(reset=True)
            t0 = time.monotonic()
            model, tokenizer, contract = load_model_and_tokenizer(parent, reference, str(device))
            run.update(load_seconds=round(time.monotonic() - t0, 1), **contract)
            if next(model.parameters()).device != device:
                raise RuntimeError(f"model is on {next(model.parameters()).device}, expected {device}")
            save()
            try:
                for name in remaining:
                    result = measure_panel(model, tokenizer, name, cases[name], device=device, cap=recipe["cap"],
                                           batch_size=recipe["batch_size"], kv_cache=recipe["kv_cache"],
                                           records_path=attempt / "records" / f"{name}.jsonl", should_stop=should_stop)
                    receipt["panels"][name].update(status=result["status"], done=result["done"])
                    run.setdefault("panels", {})[name] = result
                    run["cuda"] = cuda_report()
                    save()
                    summarize(attempt, args.summary_dir, args.panels, gate, gate_outcomes)
                    if result["status"] != "complete":
                        break
            finally:
                del model
                release_cuda_cache()
        done = {n: len(read_records(attempt / "records" / f"{n}.jsonl")) for n in args.panels}
        complete = all(done[n] == len(cases[n]) for n in args.panels)
        receipt["status"] = "complete" if complete else "stopped"
        run.update(finished=now(), stop_reason=None if complete else should_stop() or "incomplete", done=done)
        save()
        summarize(attempt, args.summary_dir, args.panels, gate, gate_outcomes)
        print(json.dumps({"status": receipt["status"], "done": done, "attempt": str(attempt)}), flush=True)
        return 0 if complete else 75
    except BaseException as error:
        receipt["status"] = "failed"
        run.update(finished=now(), error=f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}")
        save()
        try:
            summarize(attempt, args.summary_dir, args.panels, gate, gate_outcomes)
        except Exception as summary_error:  # noqa: BLE001 - keep the original error
            run["summary_error"] = f"{type(summary_error).__name__}: {summary_error}"
            save()
        raise


# Outer process: attempt directory (resume), CUDA lock and host-memory guard --------------------------

def find_attempt(output_dir, fresh=False):
    """The latest attempt whose receipt is not complete (resume), else a new one."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not fresh:
        for attempt in sorted((p for p in output_dir.iterdir() if p.is_dir()), reverse=True):
            receipt = attempt / "receipt.json"
            status = json.loads(receipt.read_text()).get("status") if receipt.is_file() else None
            if status != "complete":
                return attempt, True
            break
    attempt = output_dir / time.strftime("%Y%m%dT%H%M%S")
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt, False


def measured(attempt, panels):
    return {n: len(read_records(Path(attempt) / "records" / f"{n}.jsonl")) for n in panels}


def outer(args):
    attempt, resumed = find_attempt(args.output_dir, args.fresh)
    before = measured(attempt, args.panels)
    run = len(list(attempt.glob("guard-*"))) + 1
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--inner", "--attempt-dir", str(attempt),
               "--recipe", str(args.recipe), "--dry-run-output", str(args.dry_run_output), "--train-subset", str(args.train_subset),
               "--output-dir", str(args.output_dir), "--summary-dir", str(args.summary_dir), "--panels", ",".join(args.panels),
               "--budget-minutes", str(args.budget_minutes)]
    if args.no_guard:
        launch = command
    else:
        CUDA_LOCK.parent.mkdir(parents=True, exist_ok=True)
        CUDA_LOCK.touch(exist_ok=True)
        write_json(attempt / f"command-{run}.json", command)
        hard = os.environ.get("SEPALITH_HARD_STOP_EPOCH")
        seconds = args.guard_seconds if hard in (None, "") else min(args.guard_seconds, float(hard) - time.time() - 60)
        launch = [sys.executable, "-B", "-m", "sepalith.training.guards.cuda_host_guard", "--command-json",
                  str(attempt / f"command-{run}.json"), "--output", str(attempt / f"guard-{run}"), "--seconds", str(int(max(60, seconds)))]
    with (attempt / f"outer-{run}.log").open("wb") as log:
        process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
        signal.signal(signal.SIGTERM, lambda *_: process.terminate())
        code = process.wait()
    receipt_path = attempt / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    after = measured(attempt, args.panels)
    total = sum(after.values())
    expected = {n: (receipt.get("panels") or {}).get(n, {}).get("cases") for n in args.panels}
    complete = receipt.get("status") == "complete" and all(after[n] == expected[n] for n in args.panels)
    summary_path = attempt / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    calibration = summary.get("calibration") or {}
    digest = {"exit": code, "status": receipt.get("status"), "resumed": resumed, "attempt": str(attempt),
              "measured_before": before, "measured_after": after, "expected": expected,
              "recommended_tau": calibration.get("recommended_tau"),
              "train_at_tau": calibration.get("train_at_tau"), "dev250_at_tau": calibration.get("dev250_at_tau"),
              "dev75_at_tau": calibration.get("dev75_at_tau"), "summary_dir": str(args.summary_dir)}
    if complete:
        result_code = 0
    elif total == 0:
        result_code = 1  # nothing measured at all
    elif receipt.get("status") == "stopped" and total > sum(before.values()):
        result_code = 75  # saved at a batch boundary with work left; the next window resumes this attempt
    else:
        result_code = 1  # crash, guard stop, or no progress: flagged for the user; a retry resumes
    needs_user = None
    if result_code == 1:
        needs_user = f"no-op threshold job did not finish (measurement only, {total} cases recorded); see {attempt}"
    result = os.environ.get("SEPALITH_JOB_RESULT")
    if result:
        evaluations = [str(p) for p in (receipt_path, summary_path) if p.is_file()]
        write_json(Path(result), {"summary": json.dumps(digest, sort_keys=True, ensure_ascii=False),
                                  "evaluations": evaluations, "needs_user": needs_user})
    print(json.dumps(digest, sort_keys=True, ensure_ascii=False), flush=True)
    return result_code


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    if args.inner:
        return inner(args)
    return outer(args)


if __name__ == "__main__":
    raise SystemExit(main())
