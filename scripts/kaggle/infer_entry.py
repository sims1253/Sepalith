#!/usr/bin/env python3
"""Batched inference for Sepalith checkpoints on Kaggle T4s (card C03).

Runs inside the kernel that ``scripts/kaggle/infer_push.py`` generates, or on
any CUDA host. Generation goes through the C05/C07 evaluator helpers in
``sepalith.training.eval.campaign_eval``, so the batching policy is the one
the PC gates are bound to: prompts sorted by prompt length descending, then
case id; left padding with pad id 1 and a mask; dynamic KV cache; per-row stop
on the end-of-generation ids 1 and 130073; cap 1,024 by default.

Two input modes:

- ``--panel dev75|dev250``: a DEV panel through
  ``campaign_eval.development_evaluator`` (teacher-forced NLL, protocol
  classification, R1 counts). The panel comes from the public HF dataset with
  its sha256 pinned, or from ``--panel-source``.
- ``--prompts SOURCE``: a JSONL file of pre-tokenized prompts, one object per
  line with ``id``, ``prompt_token_ids`` and optional ``split``, ``family``,
  ``package_id`` and ``metadata``. Rows whose ``split`` is not ``train`` or
  ``dev`` are refused. Greedy by default; ``--temperature T --samples K``
  samples K continuations per prompt.

Sources (model, panel, prompts): ``hf:<owner>/<repo>/<folder>[@rev]`` (model
repository folder), ``hf-dataset:<owner>/<repo>/<path>[@rev]`` (dataset
file), ``kaggle:<owner>/<slug>[/<subpath>]`` (an attached Kaggle dataset) or
``path:<absolute path>``. No secret is needed for public sources; none is read.

Work splits at batch granularity, so every batch has the same cases and the
same padding as on the PC: global batch ``b`` belongs to session shard
``b % shard_count``, and within a shard its ``j``-th batch goes to worker
``j % workers``. ``--workers 2`` runs both T4s as two independent processes,
each with one GPU and its own model copy.

Outputs in ``--output-dir``: ``generations.jsonl`` (one record per prompt and
sample: generated ids, decoded text, finish reason, stop id, cap-hit flag),
``summary.json`` (runtime, inputs and their hashes, per-worker timing,
throughput, non-finite logit count, and for panels the evaluator counts and
the R1 counts), and for panels ``cases.json`` in the evaluator's case schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

SCHEMA = "sepalith.c03.inference.v1"
HF_DATASET = "scholzmx/sepalith"
PANELS = {
    "dev75": {"source": f"hf-dataset:{HF_DATASET}/campaign-20260915/dev/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl",
              "sha256": "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035", "cases": 75},
    "dev250": {"source": f"hf-dataset:{HF_DATASET}/campaign-20260915/dev/dev250-v1/dev250-v1.jsonl",
               "sha256": "75f3a5ee216400fdfe1e2fb1fb63224374f418d1c248a1c278ae0d09be487f22", "cases": 250},
}
MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "campaign-manifest.json")
REQUIRED_MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
MAX_SEQUENCE_TOKENS = 16384  # bound recipe cohort.max_sequence_tokens
DENSE_PARAMETERS = (381, 2_516_756_480)
ALLOWED_SPLITS = ("train", "dev")
KAGGLE_INPUT = Path("/kaggle/input")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="names the output directory and the HF upload folder")
    p.add_argument("--model", required=True, help="model source (hf:, kaggle: or path:)")
    p.add_argument("--model-sha256", help="expected model.safetensors sha256 (verified before loading)")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--panel", choices=tuple(PANELS), help="DEV panel through the evaluator")
    inputs.add_argument("--prompts", help="pre-tokenized prompts JSONL source")
    p.add_argument("--panel-source", help="panel file source (default: the public HF copy); its sha256 stays pinned")
    p.add_argument("--prompts-sha256", help="expected sha256 of the prompts file")
    p.add_argument("--case-limit", type=int, help="seeded subset of this many cases or prompts (smoke runs)")
    p.add_argument("--case-seed", type=int, default=3407)
    p.add_argument("--dtype", choices=("fp16", "fp32", "bf16"), default="fp16")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32, help="prompts per generate call (PC gates: 32)")
    p.add_argument("--kv-cache", choices=("dynamic", "static"), default="dynamic")
    p.add_argument("--workers", type=int, default=1, help="independent GPU workers in this session (0 = all visible GPUs)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default); > 0 samples (prompts mode)")
    p.add_argument("--samples", type=int, default=1, help="continuations per prompt when sampling")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407, help="sampling seed; batch b uses seed + b")
    p.add_argument("--output-dir", type=Path, help="default: /kaggle/working/<run-id>")
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/sepalith-inference"), help="downloads go here")
    p.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    p.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.batch_size < 1 or a.max_new_tokens < 1 or a.workers < 0 or a.samples < 1:
        p.error("batch size, max new tokens and samples must be positive; workers must be >= 0")
    if not 0 <= a.shard_index < a.shard_count:
        p.error("--shard-index must be in [0, --shard-count)")
    if a.temperature < 0 or (a.temperature == 0 and a.samples != 1):
        p.error("greedy decoding takes exactly one sample; use --temperature > 0 for --samples")
    if a.panel and a.temperature > 0:
        p.error("DEV panels are greedy (the evaluator's metrics); sample with --prompts")
    if a.output_dir is None:
        a.output_dir = Path("/kaggle/working") / a.run_id
    return a


# Small helpers ------------------------------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def log(message):
    print(f"[infer {time.strftime('%H:%M:%S')}] {message}", flush=True)


# Sources ------------------------------------------------------------------------------------------

def split_revision(rest):
    if "@" in rest:
        rest, revision = rest.rsplit("@", 1)
        return rest, revision
    return rest, None


def kaggle_mounts(owner, slug):
    """The dataset mount layouts seen on Kaggle (docs/research/2026-09-06-kaggle-compute-integration.md)."""
    return [KAGGLE_INPUT / slug, KAGGLE_INPUT / slug / slug, KAGGLE_INPUT / "datasets" / owner / slug,
            KAGGLE_INPUT / "datasets" / owner / slug / slug]


def resolve_source(spec, cache_dir, *, want="file"):
    """Return a local path for ``spec``: a file, or with ``want="model"`` a checkpoint directory."""
    scheme, sep, rest = spec.partition(":")
    if not sep or not rest:
        raise ValueError(f"source needs a scheme (hf:, hf-dataset:, kaggle:, path:): {spec!r}")
    if scheme == "path":
        path = Path(rest)
        if not path.is_absolute() or not path.exists():
            raise ValueError(f"path source must be an existing absolute path: {rest}")
        return path
    if scheme in ("hf", "hf-dataset"):
        rest, revision = split_revision(rest)
        parts = rest.strip("/").split("/")
        if len(parts) < 3:
            raise ValueError(f"{scheme} source needs <owner>/<repo>/<path>: {spec!r}")
        repo_id, inner = "/".join(parts[:2]), "/".join(parts[2:])
        local = Path(cache_dir) / scheme / repo_id.replace("/", "--")
        if scheme == "hf-dataset":
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id, inner, repo_type="dataset", revision=revision, local_dir=str(local)))
        if want != "model":
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id, inner, revision=revision, local_dir=str(local)))
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id, revision=revision, local_dir=str(local),
                          allow_patterns=[f"{inner}/{name}" for name in MODEL_FILES])
        return local / inner
    if scheme == "kaggle":
        parts = rest.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"kaggle source needs <owner>/<slug>: {spec!r}")
        owner, slug, inner = parts[0], parts[1], "/".join(parts[2:])
        for mount in kaggle_mounts(owner, slug):
            candidate = mount / inner if inner else mount
            if want == "model" and (candidate / "model.safetensors").is_file():
                return candidate
            if want == "file" and candidate.is_file():
                return candidate
        if want == "model":
            for mount in kaggle_mounts(owner, slug):
                if mount.is_dir():
                    for dirpath, _, filenames in os.walk(mount):
                        if "model.safetensors" in filenames and "config.json" in filenames:
                            return Path(dirpath)
        tops = sorted(p.name for p in KAGGLE_INPUT.iterdir()) if KAGGLE_INPUT.is_dir() else []
        raise ValueError(f"Kaggle dataset {owner}/{slug} ({inner or 'root'}) not found under /kaggle/input (top: {tops[:8]})")
    raise ValueError(f"unknown source scheme {scheme!r}")


# Work items and the batch plan --------------------------------------------------------------------

def panel_items(panel_path, tokenizer, cap):
    """DEV panel rows -> work items with their evaluator prompt lengths (target_start)."""
    from sepalith.campaign_protocol import PromptContext, build_training_row
    from sepalith.training.eval.campaign_eval import validate_development_capacity

    items = []
    for line in Path(panel_path).read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if case.get("split") != "dev":
            raise ValueError("only DEV rows may enter a panel run")
        row = build_training_row(PromptContext.from_mapping(case["context"]), operation=case["operation"],
                                 region_new=case["region_new"], tokenizer=tokenizer, row_id=case["id"],
                                 family=case["family"], package_id=case["package_id"], split="dev")
        validate_development_capacity(row, MAX_SEQUENCE_TOKENS, cap, case["id"])
        prompt = row["input_ids"][:row["target_start"]]
        items.append({"id": case["id"], "prompt_tokens": len(prompt), "prompt_sha256": sha256_ids(prompt),
                      "family": case["family"], "package_id": case["package_id"], "line": line})
    return items


def prompt_items(prompts_path, cap):
    items, seen = [], set()
    for row in read_jsonl(prompts_path):
        case_id, ids = row.get("id"), row.get("prompt_token_ids")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("prompts need unique non-empty string ids")
        if row.get("split", "train") not in ALLOWED_SPLITS:
            raise ValueError(f"prompt {case_id}: split {row.get('split')!r} is not TRAIN or DEV; refused")
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError(f"prompt {case_id}: prompt_token_ids must be a non-empty list of token ids")
        if len(ids) + cap > MAX_SEQUENCE_TOKENS:
            raise ValueError(f"prompt {case_id} plus the cap exceeds {MAX_SEQUENCE_TOKENS} tokens")
        seen.add(case_id)
        items.append({"id": case_id, "prompt_tokens": len(ids), "prompt_sha256": sha256_ids(ids),
                      "family": row.get("family"), "package_id": row.get("package_id"), "split": row.get("split"),
                      "metadata": row.get("metadata")})
    return items


def subset(items, limit, seed):
    if limit is None or limit >= len(items):
        return items
    if limit < 1:
        raise ValueError("--case-limit must be positive")
    chosen = set(random.Random(seed).sample(sorted(item["id"] for item in items), limit))
    return [item for item in items if item["id"] in chosen]


def batch_plan(items, batch_size, *, shard_index=0, shard_count=1, workers=1):
    """Global batches (the evaluator's order) assigned to this shard's workers."""
    from sepalith.training.eval.campaign_eval import generation_batches

    batches = generation_batches([(item["id"], item["prompt_tokens"]) for item in items], batch_size)
    plan = []
    for global_index, indexes in enumerate(batches):
        if global_index % shard_count != shard_index:
            continue
        local = global_index // shard_count
        plan.append({"global_index": global_index, "worker": local % workers,
                     "ids": [items[index]["id"] for index in indexes]})
    return {"batch_size": batch_size, "global_batches": len(batches), "shard_index": shard_index,
            "shard_count": shard_count, "workers": workers, "batches": plan}


def check_rebatching(worker_items, worker_batches, batch_size):
    """A worker's subset must re-batch into exactly its assigned global batches (same cases, same padding)."""
    from sepalith.training.eval.campaign_eval import generation_batches

    again = generation_batches([(item["id"], item["prompt_tokens"]) for item in worker_items], batch_size)
    observed = [[worker_items[index]["id"] for index in indexes] for indexes in again]
    expected = [batch["ids"] for batch in worker_batches]
    if observed != expected:
        raise ValueError("worker subset does not re-batch into its assigned global batches")


# Model --------------------------------------------------------------------------------------------

def load_checkpoint(path, dtype, device):
    """The PC baseline's load sequence (parent_baseline.py) in the requested dtype."""
    from transformers import AutoTokenizer

    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )
    from sepalith.training.eval.campaign_eval import load_development_model

    reference = load_pinned_reference_tokenizer(Path(path))
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model = load_development_model(path, dtype=dtype, device=device)
    repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    audit = assert_runtime_tokenizer(model, tokenizer, reference, embedding_identity(model), "c03_inference_load")
    named = list(model.named_parameters())
    if (len(named), sum(x.numel() for _, x in named)) != DENSE_PARAMETERS:
        raise ValueError("checkpoint is not the exact dense Sepalith model")
    model.eval()
    return model, tokenizer, {"tokenizer_repair": repair, "tokenizer_contract": audit}


class NonFiniteLogits:
    """Counts output-head calls whose logits contain NaN or inf; the outputs are left unchanged."""

    def __init__(self, model):
        import torch

        head = model.get_output_embeddings()
        self.count = torch.zeros((), dtype=torch.long, device=next(model.parameters()).device)
        self.calls = 0
        self.handle = head.register_forward_hook(self.hook)

    def hook(self, _module, _inputs, output):
        import torch

        self.calls += 1
        self.count += (~torch.isfinite(output)).any().to(torch.long)

    def close(self):
        self.handle.remove()
        return {"output_head_calls": self.calls, "calls_with_nonfinite_logits": int(self.count.item())}


def sampled_generate(model, prompts, *, cap, stop_ids, samples, temperature, top_p, seed, device):
    """Sampled counterpart of ``campaign_eval.batched_greedy_generate`` (same padding and stop cut)."""
    import torch

    from sepalith.training.eval.campaign_eval import GENERATION_BOS_TOKEN_ID, GENERATION_PAD_TOKEN_ID, truncate_at_stop

    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full((len(prompts), width), GENERATION_PAD_TOKEN_ID, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        input_ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
        attention_mask[row, width - len(prompt):] = 1
    torch.manual_seed(seed)
    sequence = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=cap, do_sample=True,
                              temperature=temperature, top_p=top_p, top_k=0, num_beams=1,
                              num_return_sequences=samples, eos_token_id=list(stop_ids),
                              pad_token_id=GENERATION_PAD_TOKEN_ID, bos_token_id=GENERATION_BOS_TOKEN_ID,
                              repetition_penalty=1.0, use_cache=True)
    rows = [truncate_at_stop(row, stop_ids, cap) for row in sequence[:, width:].tolist()]
    return [rows[index * samples:(index + 1) * samples] for index in range(len(prompts))]


def generation_record(case_id, sample, generated, tokenizer, cap, stop_ids, **extra):
    stop = generated[-1] if generated and generated[-1] in stop_ids else None
    text_ids = generated[:-1] if stop is not None else generated
    return {"id": case_id, "sample": sample, "generated_ids": generated, "generated_tokens": len(generated),
            "text": tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "finish_reason": "stop" if stop is not None else "length", "stop_token_id": stop,
            "cap_hit": stop is None and len(generated) == cap, **extra}


# Worker -------------------------------------------------------------------------------------------

def run_worker(args, plan, model, tokenizer, worker, out_dir):
    """Run this worker's batches on a loaded model; return its summary. Testable on CPU."""
    import torch

    from sepalith.campaign_protocol import NATIVE_EOG_IDS
    from sepalith.training.eval import campaign_eval

    items = {item["id"]: item for item in plan["items"]}
    batches = [batch for batch in plan["batches"] if batch["worker"] == worker]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap, stop_ids = args.max_new_tokens, tuple(NATIVE_EOG_IDS)
    device = next(model.parameters()).device
    guard = NonFiniteLogits(model)
    started = time.monotonic()
    summary = {"worker": worker, "batches": [batch["global_index"] for batch in batches],
               "cases": sum(len(batch["ids"]) for batch in batches)}
    records = []
    try:
        if not batches:
            pass
        elif plan["mode"] == "panel":
            ordered_ids = {case_id for batch in batches for case_id in batch["ids"]}
            mine = [item for item in plan["items"] if item["id"] in ordered_ids]
            check_rebatching(mine, batches, args.batch_size)
            sub_panel = out_dir / "panel-subset.jsonl"
            sub_panel.write_text("".join(item["line"] + "\n" for item in mine))
            case_dir = out_dir / "evaluator"
            case_dir.mkdir(exist_ok=True)
            evaluate = campaign_eval.development_evaluator({
                "development_panel": {"path": str(sub_panel.resolve()), "sha256": sha256_file(sub_panel)},
                "renderer_id": "zeta2-prm03-v1", "development_case_ids": sorted(item["id"] for item in mine),
                "development_max_new_tokens": cap, "parameters": {"max_sequence_tokens": MAX_SEQUENCE_TOKENS},
                "evaluation_output_directory": str(case_dir.resolve()), "generation_batch_size": args.batch_size,
                "generation_kv_cache": args.kv_cache})
            summary["evaluator_summary"] = evaluate(model, tokenizer, Path(plan["model_path"]), 0)
            cases = json.loads((case_dir / "cases.json").read_text())["results"]
            batch_of = {case_id: batch["global_index"] for batch in batches for case_id in batch["ids"]}
            for case in cases:
                if case["prompt_tokens"] != items[case["id"]]["prompt_tokens"]:
                    raise ValueError(f"prompt length differs from the plan: {case['id']}")
                records.append(generation_record(case["id"], 0, case["generated_ids"], tokenizer, cap, stop_ids,
                                                 prompt_tokens=case["prompt_tokens"],
                                                 prompt_sha256=items[case["id"]]["prompt_sha256"],
                                                 batch_index=batch_of[case["id"]], worker=worker,
                                                 generation_seconds=case["generation_seconds"]))
        else:
            prompts = {row["id"]: row["prompt_token_ids"] for row in read_jsonl(plan["prompts_path"])}
            for batch in batches:
                batch_started = time.monotonic()
                with torch.no_grad():
                    if args.temperature > 0:
                        generated = sampled_generate(model, [prompts[i] for i in batch["ids"]], cap=cap,
                                                     stop_ids=stop_ids, samples=args.samples,
                                                     temperature=args.temperature, top_p=args.top_p,
                                                     seed=args.seed + batch["global_index"], device=device)
                    else:
                        generated = [[ids] for ids in campaign_eval.batched_greedy_generate(
                            model, [prompts[i] for i in batch["ids"]], cap=cap, stop_ids=stop_ids,
                            kv_cache=args.kv_cache, device=device)]
                seconds = (time.monotonic() - batch_started) / len(batch["ids"])
                for case_id, rows in zip(batch["ids"], generated, strict=True):
                    item = items[case_id]
                    for sample, ids in enumerate(rows):
                        records.append(generation_record(
                            case_id, sample, ids, tokenizer, cap, stop_ids, prompt_tokens=item["prompt_tokens"],
                            prompt_sha256=item["prompt_sha256"], family=item.get("family"),
                            batch_index=batch["global_index"], worker=worker, generation_seconds=seconds))
                write_jsonl(out_dir / "generations.partial.jsonl", records)
    finally:
        summary["nonfinite"] = guard.close()
    summary["wall_seconds"] = time.monotonic() - started
    summary["generated_tokens"] = sum(record["generated_tokens"] for record in records)
    summary["tokens_per_second"] = summary["generated_tokens"] / summary["wall_seconds"] if summary["wall_seconds"] else None
    if device.type == "cuda":
        summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        summary["device_name"] = torch.cuda.get_device_name(0)
    write_jsonl(out_dir / "generations.jsonl", records)
    write_json(out_dir / "worker.json", summary)
    return summary


def worker_main(args):
    plan = json.loads(args.plan.read_text())
    out_dir = Path(plan["output_dir"]) / f"worker-{args.worker}"
    load_started = time.monotonic()
    import torch

    if args.device == "cuda" and not (torch.cuda.is_available() and torch.cuda.device_count() == 1):
        raise SystemExit("each worker needs exactly one visible CUDA device")
    model, tokenizer, contract = load_checkpoint(Path(plan["model_path"]), args.dtype, args.device)
    load_seconds = time.monotonic() - load_started
    log(f"worker {args.worker}: model loaded in {load_seconds:.1f} s on {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    summary = run_worker(args, plan, model, tokenizer, args.worker, out_dir)
    summary.update(load_seconds=load_seconds, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), **contract)
    write_json(out_dir / "worker.json", summary)
    log(f"worker {args.worker}: {summary['cases']} cases, {summary['generated_tokens']} tokens in "
        f"{summary['wall_seconds']:.1f} s")
    return 0


# Merge --------------------------------------------------------------------------------------------

def evaluator_counts(results):
    """``development_evaluator``'s summary counts, recomputed over the merged cases."""
    counts = {name: sum(bool(r[name]) for r in results)
              for name in ("protocol_valid", "exact_region", "predicted_noop", "suggestion", "cap_hit")}
    counts.update(strict_noop_correct=sum(r["expected_noop"] and r["predicted_noop"] for r in results),
                  strict_noop_false_suggestions=sum(r["expected_noop"] and r["suggestion"] for r in results),
                  edit_exact=sum(not r["expected_noop"] and r["exact_region"] for r in results))
    return counts


def r1_counts(counts):
    """The DEV75 R1 counts (gate_rule.metrics_from_summaries without DEV250)."""
    return {"valid": counts["protocol_valid"], "cap": counts["cap_hit"], "edit": counts["edit_exact"],
            "nofp": counts["strict_noop_false_suggestions"]}


def merge(args, plan, workers):
    out = Path(plan["output_dir"])
    order = {item["id"]: index for index, item in enumerate(plan["items"])}
    records = []
    for worker in range(plan["workers"]):
        path = out / f"worker-{worker}" / "generations.jsonl"
        if path.is_file():
            records.extend(read_jsonl(path))
    records.sort(key=lambda r: (order[r["id"]], r["sample"]))
    planned = {case_id for batch in plan["batches"] for case_id in batch["ids"]}
    if {r["id"] for r in records} != planned or len(records) != len(planned) * args.samples:
        raise ValueError("merged generations do not cover the planned prompts exactly once per sample")
    write_jsonl(out / "generations.jsonl", records)
    summary = {"records": len(records), "prompts": len(planned)}
    if plan["mode"] == "panel":
        cases = []
        for worker in range(plan["workers"]):
            path = out / f"worker-{worker}" / "evaluator" / "cases.json"
            if path.is_file():
                cases.extend(json.loads(path.read_text())["results"])
        cases.sort(key=lambda r: order[r["id"]])
        counts = evaluator_counts(cases)
        summary.update(counts=counts, cases=len(cases), cases_path=str(out / "cases.json"))
        if plan["panel"] == "dev75" and len(cases) == PANELS["dev75"]["cases"]:
            summary["r1_dev75"] = r1_counts(counts)
        losses = [c["loss"] for c in cases]
        if losses:
            summary["prompt_nll"] = sum(x["prompt_nll_sum"] for x in losses) / max(1, sum(x["prompt_tokens"] for x in losses))
            summary["target_nll"] = sum(x["target_nll_sum"] for x in losses) / max(1, sum(x["target_tokens"] for x in losses))
            summary["nll_finite"] = all(math.isfinite(x["prompt_nll_sum"]) and math.isfinite(x["target_nll_sum"]) for x in losses)
        write_json(out / "cases.json", {"status": "complete", "results": cases})
    nonfinite = sum(w.get("nonfinite", {}).get("calls_with_nonfinite_logits", 0) for w in workers)
    summary["calls_with_nonfinite_logits"] = nonfinite
    summary["generated_tokens"] = sum(r["generated_tokens"] for r in records)
    summary["cap_hits"] = sum(r["cap_hit"] for r in records)
    walls = [w.get("wall_seconds", 0) for w in workers if w.get("cases")]
    summary["generation_wall_seconds"] = max(walls) if walls else 0
    summary["session_tokens_per_second"] = (summary["generated_tokens"] / summary["generation_wall_seconds"]
                                            if summary["generation_wall_seconds"] else None)
    return summary


# Session driver -----------------------------------------------------------------------------------

def runtime_versions():
    import importlib.metadata as metadata

    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "tokenizers", "accelerate", "huggingface_hub", "safetensors"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def visible_gpus():
    try:
        names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                               text=True, check=True, timeout=60).stdout.split("\n")
    except (OSError, subprocess.SubprocessError):
        return []
    return [name.strip() for name in names if name.strip()]


def main(argv=None):
    args = parse_args(argv)
    if args.worker is not None:
        return worker_main(args)
    started = time.monotonic()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {"schema": SCHEMA, "run_id": args.run_id, "status": "running",
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": {k: str(v) for k, v in vars(args).items()},
               "final_set_access": False}
    write_json(out / "summary.json", summary)
    try:
        gpus = visible_gpus() if args.device == "cuda" else []
        workers = args.workers or max(1, len(gpus))
        if args.device == "cuda" and len(gpus) < workers:
            raise ValueError(f"{workers} workers need {workers} GPUs; visible: {gpus}")
        summary.update(gpus=gpus, workers=workers, runtime=runtime_versions())
        log(f"GPUs {gpus}; {workers} worker(s); resolving the model {args.model}")
        model_path = resolve_source(args.model, args.cache_dir, want="model").resolve()
        missing = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]
        if missing:
            raise ValueError(f"model directory lacks {missing}")
        model_sha = sha256_file(model_path / "model.safetensors")
        if args.model_sha256 and model_sha != args.model_sha256:
            raise ValueError("model.safetensors sha256 differs from --model-sha256")
        summary["model"] = {"source": args.model, "path": str(model_path), "model_safetensors_sha256": model_sha,
                            "tokenizer_json_sha256": sha256_file(model_path / "tokenizer.json")}
        write_json(out / "summary.json", summary)
        plan = {"output_dir": str(out.resolve()), "model_path": str(model_path), "workers": workers}
        if args.panel:
            from transformers import AutoTokenizer

            pinned = PANELS[args.panel]
            panel_path = resolve_source(args.panel_source or pinned["source"], args.cache_dir)
            if sha256_file(panel_path) != pinned["sha256"]:
                raise ValueError(f"{args.panel} panel sha256 differs from the pinned panel")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
            items = panel_items(panel_path, tokenizer, args.max_new_tokens)
            if len(items) != pinned["cases"]:
                raise ValueError(f"{args.panel} has {len(items)} cases, expected {pinned['cases']}")
            plan.update(mode="panel", panel=args.panel)
            summary["inputs"] = {"panel": args.panel, "source": args.panel_source or pinned["source"],
                                 "sha256": pinned["sha256"], "cases": len(items)}
        else:
            prompts_path = resolve_source(args.prompts, args.cache_dir)
            prompts_sha = sha256_file(prompts_path)
            if args.prompts_sha256 and prompts_sha != args.prompts_sha256:
                raise ValueError("prompts sha256 differs from --prompts-sha256")
            items = prompt_items(prompts_path, args.max_new_tokens)
            plan.update(mode="prompts", prompts_path=str(prompts_path))
            summary["inputs"] = {"prompts": args.prompts, "sha256": prompts_sha, "prompts_total": len(items)}
        items = subset(items, args.case_limit, args.case_seed)
        plan.update(items=items, **batch_plan(items, args.batch_size, shard_index=args.shard_index,
                                              shard_count=args.shard_count, workers=workers))
        summary["plan"] = {"cases_selected": len(items), "case_limit": args.case_limit, "case_seed": args.case_seed,
                           "global_batches": plan["global_batches"], "batches_this_shard": len(plan["batches"]),
                           "shard": [args.shard_index, args.shard_count]}
        summary["generation"] = {
            "dtype": args.dtype, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
            "kv_cache": args.kv_cache, "ordering": "prompt_tokens_descending_then_case_id_ascending",
            "padding_side": "left", "pad_token_id": 1, "stop_ids": [1, 130073],
            "decoding": "greedy" if args.temperature == 0 else {"temperature": args.temperature, "top_p": args.top_p,
                                                                "samples": args.samples, "seed": args.seed},
            "attention": "sdpa (transformers default)", "work_split": "whole global batches per shard and worker"}
        plan_path = out / "plan.json"
        write_json(plan_path, plan)
        write_json(out / "summary.json", summary)
        log(f"{len(items)} cases in {plan['global_batches']} global batches; {len(plan['batches'])} in this shard")
        base = [sys.executable, "-u", str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]
        children = []
        for worker in range(workers):
            env = dict(os.environ)
            if args.device == "cuda":
                from sepalith.training.eval.campaign_eval import pin_cuda_device

                pin_cuda_device(worker, environ=env)
            log_path = out / f"worker-{worker}.log"
            stream = log_path.open("wb")
            children.append((worker, subprocess.Popen([*base, "--worker", str(worker), "--plan", str(plan_path)],
                                                      env=env, stdout=stream, stderr=subprocess.STDOUT), stream, log_path))
        codes = {}
        for worker, process, stream, _ in children:
            codes[worker] = process.wait()
            stream.close()
        for worker, _, _, log_path in children:
            sys.stdout.write(log_path.read_text(errors="replace")[-4000:])
        sys.stdout.flush()
        if any(codes.values()):
            raise RuntimeError(f"worker exit codes {codes}")
        worker_summaries = [json.loads((out / f"worker-{w}" / "worker.json").read_text()) for w in range(workers)]
        summary["workers_detail"] = [{k: v for k, v in w.items() if k not in ("evaluator_summary", "tokenizer_contract")}
                                     for w in worker_summaries]
        summary["result"] = merge(args, plan, worker_summaries)
        summary["status"] = "complete"
    except BaseException as error:  # record the failure, then re-raise
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(out / "summary.json", summary)
        log(json.dumps({k: summary.get(k) for k in ("status", "elapsed_seconds", "result", "error")}, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
