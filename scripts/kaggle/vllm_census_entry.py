#!/usr/bin/env python3
"""Bounded vLLM census-throughput trial on Kaggle T4s (card C12, user decision 2026-10-16).

Runs inside the kernel that ``scripts/kaggle/vllm_census_push.py`` generates.
The bound-batching transformers path (``infer_entry.py``) stays the parity and
evaluation reference; this trial measures whether vLLM (CUDA graphs,
continuous batching, paged attention) is fast enough for the C12 census and
checks its fp16 numerics.

Phases, in one session:

1. Inputs. The model (default: public ``scholzmx/sepalith-2b-edit-sft/arm-a``,
   checkpoint 536, sha256 pinned), the public TRAIN rows (DAT10 v4-braceindent,
   sha256 pinned; every row must have ``split == "train"``) and the pinned DEV75
   and DEV250 panels. A seeded subset of ``--train-prompts`` TRAIN prompts
   (``input_ids[:target_start]``) is sampled; DEV is used for greedy parity
   only, never for sampling. Nothing from the sealed final set is read.
2. Reference (``--hf-dev75``): DEV75 greedy through ``infer_entry.py`` at the
   bound batching (fp16, batch 32, both T4s), the same harness C03 uses.
   vLLM is installed into a separate venv meanwhile, so Kaggle's torch and the
   harness pins stay untouched.
3. vLLM, one engine per T4 (data parallel, two independent processes), fp16,
   ``max_model_len`` 4,096, token-id prompts (``skip_tokenizer_init``), stop
   ids 1 and 130073, the model's generation_config ignored. Engine
   configurations are tried in order until one starts (the default attention
   backend, then TRITON_ATTN, FLEX_ATTENTION, and finally eager mode without
   CUDA graphs, which is recorded as such). Per worker:
   a. greedy on its share of DEV75 + DEV250 (one call), then DEV75 again alone
      (a different batch composition: vLLM's own noise floor);
   b. census sampling on its share of the TRAIN subset: k = ``--samples``
      samples per prompt (``n``), temperature 1.0, top_p 1.0, top_k off, cap
      ``--max-new-tokens``, seed ``--seed + prompt index`` per prompt, in
      chunks of ``--chunk-prompts`` prompts, stopping before the session
      deadline so the trial stays inside its GPU-hour budget;
   c. teacher-forced target NLL on its DEV share (prompt logprobs), last,
      because it is the step most likely to run out of memory.
4. DEV greedy outputs are classified with the evaluator's ``classify`` (PRM03
   contract) so they carry ``protocol_valid``, ``exact_region``, ``suggestion``
   and ``cap_hit`` like the PC references.

Outputs in ``--output-dir``: ``summary.json`` (inputs, versions, engine,
timings, throughput), ``vllm-dev-greedy.jsonl`` and ``vllm-dev75-rerun.jsonl``
(per case: generated ids incl. the stop id, classification),
``vllm-dev-nll.jsonl``, ``vllm-train-samples.jsonl`` (per prompt and sample:
generated ids incl. the stop id, finish reason, cap hit), ``train-prompts.jsonl``
(the sampled TRAIN subset: ids, family, prompt and target token ids),
``hf-dev75/`` (the harness run), and every log.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_entry import PANELS, resolve_source, sha256_file, sha256_ids, write_json, write_jsonl  # noqa: E402

SCHEMA = "sepalith.c12.vllm-trial.v1"
MODEL = ("hf:scholzmx/sepalith-2b-edit-sft/arm-a", "befe587a0d4969e33b0daaa8fe66247371ec32f331616db74724616878ad6030")
TRAIN_ROWS = ("hf-dataset:scholzmx/sepalith/campaign-20260915/sft/DAT10-finish-source-repair-v4-braceindent/"
              "train-token-rows.jsonl", "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef")
STOP_IDS = (1, 130073)
MAX_MODEL_LEN = 4096  # C11 length policy: 3,072 prompt tokens incl. BOS + 1,024 completion tokens incl. EOS
VENV = Path("/tmp/vllm-venv")
# Engine configurations, tried in order until one starts. "backend" None = vLLM's own choice.
LADDER = (
    {"name": "auto", "backend": None, "eager": False},
    {"name": "triton-attn", "backend": "TRITON_ATTN", "eager": False},
    {"name": "flex-attention", "backend": "FLEX_ATTENTION", "eager": False},
    {"name": "triton-attn-eager", "backend": "TRITON_ATTN", "eager": True},
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/sepalith-vllm"))
    p.add_argument("--model", default=MODEL[0])
    p.add_argument("--model-sha256", default=MODEL[1])
    p.add_argument("--train-prompts", type=int, default=384, help="seeded TRAIN subset size")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--chunk-prompts", type=int, default=96, help="TRAIN prompts per generate call and worker")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--hf-dev75", action=argparse.BooleanOptionalAction, default=True,
                   help="run the bound-batching transformers DEV75 reference in this session")
    p.add_argument("--vllm-specs", default="vllm==0.31.0,vllm==0.25.0",
                   help="pip specs tried in order (each into a fresh venv) until an engine starts")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--session-budget-minutes", type=float, default=38.0,
                   help="no sampling chunk may run past this many minutes after the entry started")
    p.add_argument("--first-chunk-prompts", type=int, default=32, help="calibration chunk (sizes the deadline check)")
    p.add_argument("--train-rows", default=TRAIN_ROWS[0], help="TRAIN rows source (its sha256 stays pinned)")
    p.add_argument("--cpu-dry-run", action="store_true",
                   help="inputs, vLLM install and import only (no GPU work); for a CPU kernel")
    p.add_argument("--dev-nll", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-worker", type=int, help=argparse.SUPPRESS)
    p.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--attempt", type=int, default=0, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.output_dir is None:
        a.output_dir = Path("/kaggle/working") / a.run_id
    return a


def log(message):
    print(f"[vllm-trial {time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# Inputs (system Python, harness pins) ----------------------------------------------------------------

def build_inputs(args, model_path, out):
    from transformers import AutoTokenizer

    from sepalith.campaign_protocol import PromptContext, build_training_row

    rows_path = resolve_source(args.train_rows, args.cache_dir)
    if sha256_file(rows_path) != TRAIN_ROWS[1]:
        raise ValueError("TRAIN rows sha256 differs from the pinned file")
    rows = read_jsonl(rows_path)
    if len(rows) != 15006 or any(row.get("split") != "train" for row in rows):
        raise ValueError("expected 15,006 rows, all split == 'train'")
    by_id = {row["id"]: row for row in rows}
    chosen = random.Random(args.seed).sample(sorted(by_id), args.train_prompts)  # random order, not length-sorted
    train = []
    for case_id in chosen:
        row = by_id[case_id]
        prompt = row["input_ids"][:row["target_start"]]
        if len(prompt) + args.max_new_tokens > MAX_MODEL_LEN:
            raise ValueError(f"TRAIN prompt {case_id} plus the cap exceeds {MAX_MODEL_LEN}")
        train.append({"id": case_id, "family": row["family"], "split": "train", "prompt_token_ids": prompt,
                      "prompt_sha256": sha256_ids(prompt), "target_ids": row["input_ids"][row["target_start"]:]})
    write_jsonl(out / "train-prompts.jsonl", train)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    dev, panels = [], {}
    for name in ("dev75", "dev250"):
        pinned = PANELS[name]
        path = resolve_source(pinned["source"], args.cache_dir)
        if sha256_file(path) != pinned["sha256"]:
            raise ValueError(f"{name} sha256 differs from the pinned panel")
        count = 0
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            if case.get("split") != "dev":
                raise ValueError("only DEV rows may enter a panel")
            row = build_training_row(PromptContext.from_mapping(case["context"]), operation=case["operation"],
                                     region_new=case["region_new"], tokenizer=tokenizer, row_id=case["id"],
                                     family=case["family"], package_id=case["package_id"], split="dev")
            prompt = row["input_ids"][:row["target_start"]]
            if len(row["input_ids"]) > MAX_MODEL_LEN or len(prompt) + args.max_new_tokens > MAX_MODEL_LEN:
                raise ValueError(f"DEV case {case['id']} exceeds {MAX_MODEL_LEN} tokens")
            dev.append({"key": f"{name}:{case['id']}", "panel": name, "id": case["id"], "family": case["family"],
                        "prompt_token_ids": prompt, "prompt_sha256": sha256_ids(prompt),
                        "input_ids": row["input_ids"], "target_start": row["target_start"]})
            count += 1
        if count != pinned["cases"]:
            raise ValueError(f"{name}: {count} cases, expected {pinned['cases']}")
        panels[name] = {"source": pinned["source"], "sha256": pinned["sha256"], "cases": count}
    write_jsonl(out / "dev-prompts.jsonl", dev)
    families = {}
    for item in train:
        families[item["family"]] = families.get(item["family"], 0) + 1
    return {"train_rows": {"source": args.train_rows, "sha256": TRAIN_ROWS[1], "rows": len(rows)},
            "train_subset": {"prompts": len(train), "seed": args.seed, "families": families,
                             "prompt_tokens_mean": sum(len(t["prompt_token_ids"]) for t in train) / len(train)},
            "dev_panels": panels}


def classify_dev(model_path, cache_dir, records):
    """Evaluator classification (campaign_eval.classify) for vLLM DEV greedy records, in place."""
    from transformers import AutoTokenizer

    from sepalith.campaign_protocol import PromptContext
    from sepalith.training.eval.campaign_eval import classify, generation_termination

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    panels = {}
    for name in ("dev75", "dev250"):
        path = resolve_source(PANELS[name]["source"], cache_dir)
        for line in Path(path).read_text().splitlines():
            if line.strip():
                case = json.loads(line)
                panels[f"{name}:{case['id']}"] = case
    for record in records:
        case = panels[record["key"]]
        context = PromptContext.from_mapping(case["context"])
        generated = record["generated_ids"]
        termination = generation_termination(generated, record["max_new_tokens"], STOP_IDS)
        raw_ids = generated[:-1] if termination["response_complete"] else generated
        raw = tokenizer.decode(raw_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        record.update(termination, expected_noop=list(context.region_old) == case["region_new"], raw_output=raw,
                      **classify(raw, context, case["region_new"], generated))
    return records


def dev_counts(records):
    counts = {name: sum(bool(r[name]) for r in records)
              for name in ("protocol_valid", "exact_region", "predicted_noop", "suggestion", "cap_hit")}
    counts.update(strict_noop_false_suggestions=sum(r["expected_noop"] and r["suggestion"] for r in records),
                  edit_exact=sum(not r["expected_noop"] and r["exact_region"] for r in records),
                  cases=len(records))
    return counts


# vLLM install (background, separate venv) -----------------------------------------------------------

VERSION_PROBE = """
import importlib.metadata as m, json
versions = {}
for name in ("vllm", "torch", "triton", "transformers", "flashinfer-python", "xformers", "numpy"):
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
print(json.dumps(versions))
"""

def install_vllm(spec, venv, log_path):
    """uv venv + uv pip install; returns a record with timing and versions."""
    started = time.monotonic()
    record = {"spec": spec, "venv": str(venv)}
    with Path(log_path).open("a") as stream:
        def run(cmd, timeout):
            stream.write(f"$ {' '.join(map(str, cmd))}\n")
            stream.flush()
            subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        try:
            run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], 300)
            uv = [sys.executable, "-m", "uv"]
            run([*uv, "venv", "--clear", "--python", sys.executable, str(venv)], 300)
            run([*uv, "pip", "install", "--python", str(venv / "bin/python"), spec], 1500)
            probe = subprocess.run([str(venv / "bin/python"), "-c", VERSION_PROBE], capture_output=True, text=True,
                                   timeout=600, env=dict(os.environ, PYTHONPATH=""))
            record["versions"] = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 else None
            record["probe_stderr"] = probe.stderr[-1500:] if probe.returncode else None
            record["status"] = "installed" if probe.returncode == 0 else "import_failed"
        except Exception as error:  # recorded; the caller decides
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


# vLLM worker (venv Python) ---------------------------------------------------------------------------

def make_engine(plan, config):
    from vllm import LLM

    kwargs = dict(model=plan["model_path"], dtype="float16", max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=plan["gpu_memory_utilization"], max_num_seqs=plan["max_num_seqs"],
                  enable_prefix_caching=True, skip_tokenizer_init=True, generation_config="vllm",
                  seed=plan["seed"], tensor_parallel_size=1, enforce_eager=config["eager"], trust_remote_code=False)
    if config["backend"]:
        os.environ["VLLM_ATTENTION_BACKEND"] = config["backend"]
        kwargs["attention_config"] = {"backend": config["backend"]}
    try:
        return LLM(**kwargs), kwargs
    except TypeError as error:
        if "attention_config" not in str(error):
            raise
        kwargs.pop("attention_config")
        return LLM(**kwargs), kwargs


def sampling_params(**kwargs):
    from vllm import SamplingParams

    base = dict(stop_token_ids=list(STOP_IDS), detokenize=False, skip_special_tokens=False)
    base.update(kwargs)
    return SamplingParams(**base)


def tokens_prompt(ids):
    try:
        from vllm.inputs import TokensPrompt
    except ImportError:  # pragma: no cover - older layouts
        return {"prompt_token_ids": list(ids)}
    return TokensPrompt(prompt_token_ids=list(ids))


def with_stop(completion):
    """Generated ids with the stop id kept as the last token (the harness's convention)."""
    ids = list(completion.token_ids)
    reason = getattr(completion, "stop_reason", None)
    if completion.finish_reason == "stop" and (not ids or ids[-1] not in STOP_IDS) and reason in STOP_IDS:
        ids.append(reason)
    return ids


def completion_record(completion, cap):
    ids = with_stop(completion)
    stop = ids[-1] if ids and ids[-1] in STOP_IDS else None
    return {"generated_ids": ids, "generated_tokens": len(ids), "finish_reason": completion.finish_reason,
            "stop_token_id": stop, "cap_hit": stop is None and len(ids) >= cap, "max_new_tokens": cap}


def engine_info(llm):
    info = {}
    try:
        config = llm.llm_engine.vllm_config
        info["compilation_config"] = str(config.compilation_config)[:1500]
        info["cache_config"] = {k: getattr(config.cache_config, k, None)
                                for k in ("block_size", "num_gpu_blocks", "enable_prefix_caching", "cache_dtype")}
        info["scheduler_config"] = {k: getattr(config.scheduler_config, k, None)
                                    for k in ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill")}
        attention = getattr(config, "attention_config", None)
        info["attention_config"] = str(attention)[:300] if attention is not None else None
    except Exception as error:  # informative only
        info["error"] = f"{type(error).__name__}: {error}"
    return info


def vllm_worker(args):
    plan = json.loads(args.plan.read_text())
    worker, config = args.vllm_worker, LADDER[args.attempt]
    out = Path(plan["output_dir"]) / f"vllm-worker-{worker}"
    out.mkdir(parents=True, exist_ok=True)
    result = {"worker": worker, "attempt": args.attempt, "config": config,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    started = time.monotonic()
    import torch
    import vllm

    llm, kwargs = make_engine(plan, config)
    result.update(vllm=vllm.__version__, torch=torch.__version__, engine_seconds=round(time.monotonic() - started, 1),
                  engine_kwargs={k: v for k, v in kwargs.items() if k != "model"}, engine=engine_info(llm),
                  device_name=torch.cuda.get_device_name(0))
    write_json(out / "engine.json", result)  # marks a started engine: later failures are not retried
    log(f"worker {worker}: engine '{config['name']}' up in {result['engine_seconds']} s")
    cap = plan["max_new_tokens"]

    # Warm-up (not timed): a few short greedy requests.
    dev = [item for index, item in enumerate(read_jsonl(plan["dev_prompts"])) if index % plan["workers"] == worker]
    llm.generate([tokens_prompt(d["prompt_token_ids"]) for d in dev[:4]], sampling_params(temperature=0.0, max_tokens=8),
                 use_tqdm=False)

    def greedy(items, label):
        t0 = time.monotonic()
        outputs = llm.generate([tokens_prompt(d["prompt_token_ids"]) for d in items],
                               sampling_params(temperature=0.0, max_tokens=cap), use_tqdm=False)
        seconds = time.monotonic() - t0
        records = []
        for item, output in zip(items, outputs, strict=True):
            records.append({"key": item["key"], "panel": item["panel"], "id": item["id"], "family": item["family"],
                            "prompt_sha256": item["prompt_sha256"], "prompt_tokens": len(item["prompt_token_ids"]),
                            "worker": worker, **completion_record(output.outputs[0], cap)})
        tokens = sum(r["generated_tokens"] for r in records)
        result[label] = {"cases": len(records), "seconds": round(seconds, 2), "generated_tokens": tokens,
                         "tokens_per_second": round(tokens / seconds, 1) if seconds else None}
        write_jsonl(out / f"{label}.jsonl", records)
        write_json(out / "worker.json", result)
        log(f"worker {worker}: {label} {len(records)} cases, {tokens} tokens in {seconds:.1f} s")

    greedy(dev, "dev_greedy")
    greedy([d for d in dev if d["panel"] == "dev75"], "dev75_rerun")

    # Census sampling on this worker's TRAIN share, chunked, bounded by the session deadline.
    train = [item for index, item in enumerate(read_jsonl(plan["train_prompts"])) if index % plan["workers"] == worker]
    index_of = {item["id"]: index for index, item in enumerate(read_jsonl(plan["train_prompts"]))}
    samples, chunks = [], []
    sampling_started = time.monotonic()
    bounds, first = [], 0
    while first < len(train):
        size = plan["first_chunk_prompts"] if not bounds else plan["chunk_prompts"]
        bounds.append((first, min(len(train), first + size)))
        first += size
    for first, last in bounds:
        remaining = plan["deadline_epoch"] - time.time()
        per_prompt = (sum(c["seconds"] for c in chunks) / sum(c["prompts"] for c in chunks)) if chunks else 0
        if remaining < 1.2 * per_prompt * (last - first) or remaining < 60:
            result["sampling_stopped_early"] = {"before_prompt": first, "remaining_seconds": round(remaining, 1)}
            log(f"worker {worker}: deadline - stop sampling before prompt {first}")
            break
        chunk = train[first:last]
        # top_k left at vLLM's default (disabled); repetition penalty 1; the model's generation_config is ignored.
        params = [sampling_params(n=plan["samples"], temperature=plan["temperature"], top_p=plan["top_p"],
                                  max_tokens=cap, seed=plan["seed"] + index_of[item["id"]]) for item in chunk]
        t0 = time.monotonic()
        outputs = llm.generate([tokens_prompt(item["prompt_token_ids"]) for item in chunk], params, use_tqdm=False)
        seconds = time.monotonic() - t0
        tokens = 0
        for item, output in zip(chunk, outputs, strict=True):
            if len(output.outputs) != plan["samples"]:
                raise ValueError(f"{item['id']}: {len(output.outputs)} samples, expected {plan['samples']}")
            for completion in sorted(output.outputs, key=lambda c: c.index):
                record = {"id": item["id"], "family": item["family"], "sample": completion.index,
                          "prompt_tokens": len(item["prompt_token_ids"]), "prompt_sha256": item["prompt_sha256"],
                          "seed": plan["seed"] + index_of[item["id"]], "worker": worker,
                          **completion_record(completion, cap)}
                tokens += record["generated_tokens"]
                samples.append(record)
        chunks.append({"prompts": len(chunk), "samples": len(chunk) * plan["samples"], "seconds": round(seconds, 2),
                       "generated_tokens": tokens, "tokens_per_second": round(tokens / seconds, 1)})
        result["sampling"] = {"chunks": chunks, "prompts": sum(c["prompts"] for c in chunks),
                              "samples": len(samples), "generated_tokens": sum(c["generated_tokens"] for c in chunks),
                              "seconds": round(sum(c["seconds"] for c in chunks), 2),
                              "wall_seconds": round(time.monotonic() - sampling_started, 2)}
        result["sampling"]["tokens_per_second"] = round(result["sampling"]["generated_tokens"]
                                                        / result["sampling"]["seconds"], 1)
        write_jsonl(out / "train_samples.jsonl", samples)
        write_json(out / "worker.json", result)
        log(f"worker {worker}: chunk {len(chunks)}: {len(chunk)} prompts, {tokens} tokens in {seconds:.1f} s "
            f"({tokens / seconds:.0f} tok/s)")

    # Teacher-forced target NLL on the DEV share (prompt logprobs of the full rows), last.
    if plan["dev_nll"] and time.time() < plan["deadline_epoch"] + 300:
        try:
            t0 = time.monotonic()
            nll = []
            for first in range(0, len(dev), 8):
                batch = dev[first:first + 8]
                outputs = llm.generate([tokens_prompt(d["input_ids"]) for d in batch],
                                       sampling_params(temperature=0.0, max_tokens=1, prompt_logprobs=0),
                                       use_tqdm=False)
                for item, output in zip(batch, outputs, strict=True):
                    logprobs = output.prompt_logprobs
                    ids = item["input_ids"]
                    target = [logprobs[i][ids[i]].logprob for i in range(item["target_start"], len(ids))]
                    prompt = [logprobs[i][ids[i]].logprob for i in range(1, item["target_start"])]
                    nll.append({"key": item["key"], "panel": item["panel"], "target_tokens": len(target),
                                "target_nll_sum": -sum(target), "prompt_tokens": len(prompt),
                                "prompt_nll_sum": -sum(prompt)})
            write_jsonl(out / "dev_nll.jsonl", nll)
            result["dev_nll"] = {"cases": len(nll), "seconds": round(time.monotonic() - t0, 1)}
        except Exception as error:  # informative; the throughput result stands
            result["dev_nll"] = {"error": f"{type(error).__name__}: {error}"[:1500]}
        write_json(out / "worker.json", result)
    result["status"] = "complete"
    result["wall_seconds"] = round(time.monotonic() - started, 1)
    write_json(out / "worker.json", result)
    llm = None
    gc.collect()
    return 0


# Orchestrator -----------------------------------------------------------------------------------------

def run_hf_dev75(args, model_path, out):
    """The bound-batching transformers reference (infer_entry.py, unchanged), DEV75 greedy fp16 batch 32."""
    entry = Path(__file__).resolve().parent / "infer_entry.py"
    target = out / "hf-dev75"
    cmd = [sys.executable, "-u", str(entry), "--run-id", "hf-dev75", "--output-dir", str(target),
           "--cache-dir", str(args.cache_dir), "--model", f"path:{model_path}", "--model-sha256", args.model_sha256,
           "--panel", "dev75", "--dtype", "fp16", "--batch-size", "32", "--workers", str(args.workers)]
    started = time.monotonic()
    with (out / "hf-dev75.log").open("wb") as stream:
        code = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT).returncode
    record = {"exit": code, "seconds": round(time.monotonic() - started, 1)}
    summary = target / "summary.json"
    if summary.is_file():
        value = json.loads(summary.read_text())
        record["result"] = value.get("result")
        record["status"] = value.get("status")
    return record


def run_vllm_workers(args, plan_path, python, out):
    """One process per GPU; each walks the engine ladder until an engine starts."""
    results = {}

    def one(worker):
        attempts = []
        for attempt in range(len(LADDER)):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker), PYTHONPATH="", VLLM_LOGGING_LEVEL="INFO",
                       VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false")
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # vLLM's own allocator settings
            engine_marker = out / f"vllm-worker-{worker}" / "engine.json"
            if engine_marker.exists():
                engine_marker.unlink()
            log_path = out / f"vllm-worker-{worker}-attempt-{attempt}.log"
            t0 = time.monotonic()
            with log_path.open("wb") as stream:
                code = subprocess.run([python, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
                                       "--output-dir", str(out), "--vllm-worker", str(worker), "--plan", str(plan_path),
                                       "--attempt", str(attempt)], env=env, stdout=stream,
                                      stderr=subprocess.STDOUT).returncode
            tail = log_path.read_text(errors="replace")[-3000:]
            attempts.append({"attempt": attempt, "config": LADDER[attempt]["name"], "exit": code,
                             "seconds": round(time.monotonic() - t0, 1), "engine_started": engine_marker.exists(),
                             "log_tail": tail if code else None})
            if code == 0 or engine_marker.exists():
                break
            if time.time() > json.loads(plan_path.read_text())["deadline_epoch"]:
                break
        results[worker] = attempts

    threads = [threading.Thread(target=one, args=(w,)) for w in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def merge_vllm(out, workers, label):
    records = []
    for worker in range(workers):
        path = out / f"vllm-worker-{worker}" / f"{label}.jsonl"
        if path.is_file():
            records.extend(read_jsonl(path))
    return records


def throughput(worker_results):
    """Session tokens/s for sampling: all generated tokens over the slowest worker's sampling time."""
    done = [w.get("sampling") for w in worker_results if w.get("sampling")]
    if not done:
        return None
    tokens = sum(s["generated_tokens"] for s in done)
    seconds = max(s["seconds"] for s in done)
    return {"workers": len(done), "generated_tokens": tokens, "slowest_worker_seconds": seconds,
            "session_tokens_per_second": round(tokens / seconds, 1),
            "per_worker_tokens_per_second": [s["tokens_per_second"] for s in done],
            "prompts": sum(s["prompts"] for s in done), "samples": sum(s["samples"] for s in done)}


def main(argv=None):
    args = parse_args(argv)
    if args.vllm_worker is not None:
        return vllm_worker(args)
    t0 = time.time()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    deadline = t0 + 60 * args.session_budget_minutes
    summary = {"schema": SCHEMA, "run_id": args.run_id, "status": "running", "final_set_access": False,
               "args": {k: str(v) for k, v in vars(args).items()}, "session_t0_epoch": t0,
               "sampling_deadline_epoch": deadline, "timeline": []}

    def mark(step):
        summary["timeline"].append([step, round(time.time() - t0, 1)])
        write_json(out / "summary.json", summary)
        log(step)

    write_json(out / "summary.json", summary)
    try:
        mark("start")
        model_path = resolve_source(args.model, args.cache_dir, want="model").resolve()
        model_sha = sha256_file(model_path / "model.safetensors")
        if model_sha != args.model_sha256:
            raise ValueError(f"model.safetensors sha256 {model_sha} differs from {args.model_sha256}")
        summary["model"] = {"source": args.model, "path": str(model_path), "model_safetensors_sha256": model_sha}
        mark("model")
        summary["inputs"] = build_inputs(args, model_path, out)
        mark("inputs")

        specs = [s.strip() for s in args.vllm_specs.split(",") if s.strip()]
        installs = {}
        install_thread = threading.Thread(
            target=lambda: installs.update({0: install_vllm(specs[0], VENV, out / "vllm-install.log")}))
        install_thread.start()
        if args.hf_dev75 and not args.cpu_dry_run:
            summary["hf_dev75"] = run_hf_dev75(args, model_path, out)
            mark("hf-dev75")
        install_thread.join()
        summary["vllm_install"] = [installs[0]]
        mark("vllm-install")
        if args.cpu_dry_run:
            summary["status"] = "complete"
            return 0

        plan = {"output_dir": str(out.resolve()), "model_path": str(model_path), "workers": args.workers,
                "dev_prompts": str(out / "dev-prompts.jsonl"), "train_prompts": str(out / "train-prompts.jsonl"),
                "max_new_tokens": args.max_new_tokens, "samples": args.samples, "temperature": args.temperature,
                "top_p": args.top_p, "seed": args.seed, "chunk_prompts": args.chunk_prompts,
                "first_chunk_prompts": args.first_chunk_prompts,
                "gpu_memory_utilization": args.gpu_memory_utilization, "max_num_seqs": args.max_num_seqs,
                "deadline_epoch": deadline, "dev_nll": args.dev_nll}
        plan_path = out / "vllm-plan.json"
        write_json(plan_path, plan)
        summary["vllm_attempts"] = []
        for index, spec in enumerate(specs):
            venv = VENV if index == 0 else Path(f"/tmp/vllm-venv-{index}")
            if index > 0:
                if time.time() > deadline - 600:
                    summary["vllm_attempts"].append({"spec": spec, "skipped": "too close to the session deadline"})
                    break
                summary["vllm_install"].append(install_vllm(spec, venv, out / "vllm-install.log"))
                mark(f"vllm-install-{index}")
            if summary["vllm_install"][-1]["status"] != "installed":
                summary["vllm_attempts"].append({"spec": spec, "skipped": "install failed"})
                continue
            attempts = run_vllm_workers(args, plan_path, str(venv / "bin/python"), out)
            summary["vllm_attempts"].append({"spec": spec, "workers": attempts})
            mark(f"vllm-run-{index}")
            if any(a["engine_started"] for runs in attempts.values() for a in runs):
                break
        worker_results = []
        for worker in range(args.workers):
            path = out / f"vllm-worker-{worker}" / "worker.json"
            if path.is_file():
                worker_results.append(json.loads(path.read_text()))
        summary["vllm_workers"] = worker_results
        summary["throughput"] = throughput(worker_results)
        dev = classify_dev(model_path, args.cache_dir, merge_vllm(out, args.workers, "dev_greedy"))
        rerun = classify_dev(model_path, args.cache_dir, merge_vllm(out, args.workers, "dev75_rerun"))
        write_jsonl(out / "vllm-dev-greedy.jsonl", dev)
        write_jsonl(out / "vllm-dev75-rerun.jsonl", rerun)
        write_jsonl(out / "vllm-dev-nll.jsonl", merge_vllm(out, args.workers, "dev_nll"))
        samples = merge_vllm(out, args.workers, "train_samples")
        write_jsonl(out / "vllm-train-samples.jsonl", samples)
        summary["vllm_dev_counts"] = {panel: dev_counts([r for r in dev if r["panel"] == panel])
                                      for panel in ("dev75", "dev250")}
        summary["vllm_dev75_rerun_counts"] = dev_counts(rerun) if rerun else None
        mark("merged")
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        summary["elapsed_seconds"] = round(time.time() - t0, 1)
        write_json(out / "summary.json", summary)
        log(json.dumps({k: summary.get(k) for k in ("status", "elapsed_seconds", "throughput", "error")},
                       default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
