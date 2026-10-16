#!/usr/bin/env python3
"""LoRA-on-T4 feasibility probe for the editing-SFT parent (research, 2026-10-16).

Runs inside the kernel that ``scripts/kaggle/lora_probe_push.py`` generates.
It measures whether LoRA SFT of checkpoint 11,586 fits and how fast it runs on
Kaggle's T4s (fp16 only, no bf16), using the real SFT v2 TRAIN rows and the
real SFT v2 draw schedule, both public on HF with their sha256 pinned:

- rows: ``campaign-20260915/sft/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl``
- schedule: ``.../schedule-v2/draw-schedule-v2.json`` (1,071 updates of 16 rows,
  13 edit + 3 no-op, length-bucketed batches)

Every row is checked to be ``split == "train"``. Nothing from DEV or the sealed
final set is read.

Training step (one optimizer update = 16 schedule rows, the recipe's effective
batch): micro-batches of ``--micro-batch`` rows (right padded with pad id 1),
prompt-masked loss (labels only from ``target_start``, exactly as
``sepalith.training.sft.campaign_edit_data``), loss summed over target tokens
and divided by the update's target-token count. The output head runs only on
the target positions (the loss is identical; it avoids 130,560-wide logits for
prompt tokens). Base weights fp16 (or 4-bit NF4 for QLoRA), LoRA adapters fp32,
forward under fp16 autocast, ``torch.amp.GradScaler``. Gradient checkpointing
(non-reentrant) is on for rows longer than ``--gc-threshold`` tokens (0 = all).

Phases (``--phases``): each phase is a ``+``-separated set of configs run
concurrently, one process per GPU; ``ddp:<config>`` runs one config over both
GPUs (torchrun, each rank takes half of every update's rows, the scaled LoRA
gradients are all-reduced before unscaling so both ranks skip together).

Outputs in ``--output-dir``: ``summary.json`` (environment, per-config memory,
timing, loss and scaler records, extrapolation to the 268-update pilot and the
1,071-update arm) and ``<config>/updates.jsonl`` + ``<config>/result.json``.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_entry import resolve_source, sha256_file, write_json  # noqa: E402

SCHEMA = "sepalith.kaggle-lora-probe.v1"
HF = "hf-dataset:scholzmx/sepalith/campaign-20260915/sft/DAT10-finish-source-repair-v4-braceindent"
ROWS = (f"{HF}/train-token-rows.jsonl", "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef")
SCHEDULE = (f"{HF}/schedule-v2/draw-schedule-v2.json", "6820a84738ba2131911146f1541aaa525b41b08300b924261a287f6a404cc7bc")
MODEL = ("hf:scholzmx/sepalith-2b-cpt/checkpoint-11586", "4bf617f9ea66a31e547397b93f0998372686e076d03ba22961780abd285a3c51")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PILOT_UPDATES, ARM_UPDATES = 268, 1071

# name -> settings. "quant": none|nf4. gc_threshold: rows longer than this use gradient checkpointing (0 = all rows).
CONFIGS = {
    "r16": {"rank": 16, "quant": "none", "gc_threshold": 0, "micro_batch": 1},
    "r64": {"rank": 64, "quant": "none", "gc_threshold": 0, "micro_batch": 1},
    "r16-gc2048": {"rank": 16, "quant": "none", "gc_threshold": 2048, "micro_batch": 1},
    "r16-nogc": {"rank": 16, "quant": "none", "gc_threshold": 1 << 30, "micro_batch": 1},
    "r16-mb4": {"rank": 16, "quant": "none", "gc_threshold": 0, "micro_batch": 4},
    "q4-r16": {"rank": 16, "quant": "nf4", "gc_threshold": 0, "micro_batch": 1},
    "q4-r64": {"rank": 64, "quant": "nf4", "gc_threshold": 0, "micro_batch": 1},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/sepalith-lora"))
    p.add_argument("--phases", default="r16+r64,r16-gc2048+q4-r16,ddp:r16",
                   help="comma-separated phases; '+' joins configs that run concurrently on separate GPUs")
    p.add_argument("--updates", type=int, default=40, help="optimizer updates per single-GPU config")
    p.add_argument("--ddp-updates", type=int, default=30)
    p.add_argument("--config-seconds", type=float, default=1200, help="stop a config after this many training seconds")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--selftest-tiny", action="store_true", help=argparse.SUPPRESS)  # CPU logic test, tiny random model
    # worker mode (internal)
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--ddp", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def log(msg):
    print(f"[lora {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Plan: model, rows and the sampled updates --------------------------------------------------------

def build_plan(args):
    if args.selftest_tiny:
        return selftest_plan(args)
    model_dir = resolve_source(MODEL[0], args.cache_dir, want="model")
    model_sha = sha256_file(model_dir / "model.safetensors")
    if model_sha != MODEL[1]:
        raise SystemExit(f"model sha256 differs: {model_sha}")
    rows_path = resolve_source(ROWS[0], args.cache_dir)
    sched_path = resolve_source(SCHEDULE[0], args.cache_dir)
    for path, want in ((rows_path, ROWS[1]), (sched_path, SCHEDULE[1])):
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path.name} sha256 differs: {got} != {want}")
    schedule = json.loads(sched_path.read_text())
    updates = collections.defaultdict(list)
    for draw in schedule["draws"]:
        updates[draw["update"]].append(draw)
    if len(updates) != ARM_UPDATES or any(len(v) != 16 for v in updates.values()):
        raise SystemExit("schedule shape differs from 1,071 x 16")
    features = {u: update_features([d["total_tokens"] for d in v], [d["supervised_target_tokens"] for d in v],
                                   v[0]["batch_sequence_bucket"]) for u, v in updates.items()}
    # Sample: evenly spaced updates over the whole arm (bucket mix close to the arm's), with the update that holds
    # the longest row first so an out-of-memory shows up in the first minute.
    count = max(args.updates, args.ddp_updates)
    step = ARM_UPDATES / count
    chosen = sorted({1 + int(i * step) for i in range(count)})
    longest = max(updates, key=lambda u: max(d["total_tokens"] for d in updates[u]))
    if longest in chosen:
        chosen.remove(longest)
    else:
        chosen.pop()
    chosen = [longest] + chosen
    need = {d["row_id"] for u in chosen for d in updates[u]}
    rows = {}
    with rows_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in need:
                if row.get("split") != "train":
                    raise SystemExit(f"row {row['id']} has split {row.get('split')!r}; only TRAIN rows may be used")
                rows[row["id"]] = {"input_ids": row["input_ids"], "target_start": row["target_start"],
                                   "family": row["family"]}
    missing = need - set(rows)
    if missing:
        raise SystemExit(f"{len(missing)} scheduled rows missing from the rows file")
    plan = {"model_dir": str(model_dir), "model_sha256": model_sha, "rows_sha256": ROWS[1], "schedule_sha256": SCHEDULE[1],
            "updates": [{"update": u, "bucket": updates[u][0]["batch_sequence_bucket"],
                         "rows": [rows[d["row_id"]] for d in updates[u]]} for u in chosen],
            "schedule_features": {str(u): f for u, f in features.items()}}
    return plan


def update_features(lengths, targets, bucket):
    return {"bucket": bucket, "tokens": sum(lengths), "sq": sum(n * n for n in lengths), "max": max(lengths),
            "targets": sum(targets)}


def selftest_plan(args):
    import random

    rng = random.Random(args.seed)
    ups, feats = [], {}
    for u in range(1, ARM_UPDATES + 1):
        lengths = [rng.randint(20, 90) for _ in range(16)]
        feats[str(u)] = update_features(lengths, [5] * 16, 512)
    for u in range(1, max(args.updates, args.ddp_updates) + 1):
        rows = []
        for _ in range(16):
            n = rng.randint(20, 90)
            ids = [0] + [rng.randint(2, 255) for _ in range(n - 2)] + [1]
            rows.append({"input_ids": ids, "target_start": n - rng.randint(3, 10), "family": "synthetic"})
        ups.append({"update": u, "bucket": 512, "rows": rows})
    return {"model_dir": "tiny", "model_sha256": None, "updates": ups, "schedule_features": feats, "selftest": True}


# Worker -------------------------------------------------------------------------------------------

def load_model(plan, cfg, device, selftest):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    t0 = time.monotonic()
    if selftest:
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                             num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                             pad_token_id=1, bos_token_id=0, eos_token_id=1))
    elif cfg["quant"] == "nf4":
        from transformers import BitsAndBytesConfig

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(plan["model_dir"], dtype=torch.float16, quantization_config=q,
                                                     device_map={"": device}, attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(plan["model_dir"], dtype=torch.float16, device_map={"": device},
                                                     attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=cfg["rank"], lora_alpha=cfg["rank"], lora_dropout=0.0, target_modules=TARGETS, bias="none",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()  # fp32 master adapter weights
    model.train()
    base = model.get_base_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    info = {"load_seconds": round(time.monotonic() - t0, 1), "trainable_parameters": sum(p.numel() for p in trainable),
            "trainable_dtypes": sorted({str(p.dtype) for p in trainable}),
            "frozen_dtypes": sorted({str(p.dtype) for p in model.parameters() if not p.requires_grad})}
    return model, base, trainable, info


def set_gc(base, on):
    for module in base.modules():
        if hasattr(module, "gradient_checkpointing") and module is not base:
            module.gradient_checkpointing = on


def micro_batches(rows, size):
    order = sorted(rows, key=lambda r: len(r["input_ids"]), reverse=True)
    return [order[i:i + size] for i in range(0, len(order), size)]


def micro_loss(base, batch, device, amp):
    """Sum of target-token cross-entropy (prompt-masked) and the target-token count."""
    import torch
    import torch.nn.functional as F

    width = max(len(r["input_ids"]) for r in batch)
    ids = torch.full((len(batch), width), 1, dtype=torch.long)
    mask = torch.zeros_like(ids)
    pos_b, pos_t, targets = [], [], []
    for i, r in enumerate(batch):
        n, s = len(r["input_ids"]), r["target_start"]
        ids[i, :n] = torch.tensor(r["input_ids"])
        mask[i, :n] = 1
        pos_b += [i] * (n - s)
        pos_t += list(range(s - 1, n - 1))  # position t predicts token t + 1
        targets += r["input_ids"][s:n]
    ids, mask = ids.to(device), mask.to(device)
    pos_b, pos_t = torch.tensor(pos_b, device=device), torch.tensor(pos_t, device=device)
    targets = torch.tensor(targets, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        hidden = base.model(input_ids=ids, attention_mask=mask).last_hidden_state
        logits = base.lm_head(hidden[pos_b, pos_t])
    loss = F.cross_entropy(logits.float(), targets, reduction="sum")
    return loss, int(targets.numel())


def run_worker(args):
    import torch

    plan = json.loads(args.plan.read_text())
    selftest = bool(plan.get("selftest"))
    name = args.worker
    cfg = CONFIGS[name]
    rank, world = 0, 1
    if args.ddp:
        import torch.distributed as dist

        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        dist.init_process_group("gloo" if selftest else "nccl")
    tag = f"ddp-{name}" if args.ddp else name
    out = args.output_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available() and not selftest
    device = torch.device(f"cuda:{rank}" if args.ddp else "cuda:0") if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    amp = cuda  # fp16 autocast + GradScaler on GPU; fp32 for the CPU selftest
    torch.manual_seed(args.seed)
    result = {"config": name, "settings": cfg, "ddp": args.ddp, "world": world, "rank": rank, "status": "starting",
              "device": torch.cuda.get_device_name(device) if cuda else "cpu"}
    updates_log = (out / f"updates-rank{rank}.jsonl").open("w")
    try:
        model, base, trainable, info = load_model(plan, cfg, device, selftest)
        result.update(info)
        if cuda:
            torch.cuda.synchronize()
            result["static_allocated_gib"] = round(torch.cuda.memory_allocated(device) / 2**30, 3)
            torch.cuda.reset_peak_memory_stats(device)
        opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                                fused=cuda)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        n_updates = args.ddp_updates if args.ddp else args.updates
        records, t_train = [], 0.0
        for k, up in enumerate(plan["updates"][:n_updates]):
            lr = args.lr * min(1.0, (k + 1) / max(1, args.warmup))
            for group in opt.param_groups:
                group["lr"] = lr
            rows = up["rows"][rank::world]
            total_targets = sum(len(r["input_ids"]) - r["target_start"] for r in up["rows"])
            if cuda:
                torch.cuda.synchronize()
            t0 = time.monotonic()
            loss_sum, nonfinite_micro = 0.0, 0
            for mb in micro_batches(rows, cfg["micro_batch"]):
                set_gc(base, max(len(r["input_ids"]) for r in mb) > cfg["gc_threshold"])
                loss, _ = micro_loss(base, mb, device, amp)
                if not torch.isfinite(loss):
                    nonfinite_micro += 1
                loss_sum += float(loss.detach())
                scaler.scale(loss / total_targets).backward()
            if args.ddp:
                import torch.distributed as dist

                flat = torch.cat([p.grad.reshape(-1) for p in trainable])
                dist.all_reduce(flat)
                offset = 0
                for p in trainable:
                    p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
                    offset += p.numel()
                ls = torch.tensor([loss_sum, nonfinite_micro], dtype=torch.float64, device=device)
                dist.all_reduce(ls)
                loss_sum, nonfinite_micro = float(ls[0]), int(ls[1])
            scale_before = scaler.get_scale() if amp else 1.0
            scaler.unscale_(opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.synchronize()
            dt = time.monotonic() - t0
            t_train += dt
            scale_after = scaler.get_scale() if amp else 1.0
            rec = {"k": k, "update": up["update"], "bucket": up["bucket"], "seconds": round(dt, 3),
                   "tokens": sum(len(r["input_ids"]) for r in up["rows"]),
                   "max_len": max(len(r["input_ids"]) for r in up["rows"]), "targets": total_targets,
                   "loss": loss_sum / total_targets, "grad_norm": grad_norm if math.isfinite(grad_norm) else str(grad_norm),
                   "scale_before": scale_before, "scale_after": scale_after,
                   "skipped": bool(amp and scale_after < scale_before), "nonfinite_micro": nonfinite_micro, "lr": lr}
            if cuda:
                rec["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
            records.append(rec)
            updates_log.write(json.dumps({k2: (v if not isinstance(v, float) or math.isfinite(v) else str(v))
                                          for k2, v in rec.items()}) + "\n")
            updates_log.flush()
            if rank == 0:
                log(f"{tag} update {k + 1}/{n_updates} (sched {up['update']}, bucket {up['bucket']}, max {rec['max_len']}) "
                    f"{dt:.2f}s loss {rec['loss']:.4f} gnorm {rec['grad_norm']} scale {scale_after:g}"
                    + (f" peak {rec.get('peak_allocated_gib')} GiB" if cuda else ""))
            if t_train > args.config_seconds:
                result["stopped_early"] = f"training time {t_train:.0f}s > {args.config_seconds}s"
                break
        result["records"] = records
        result["status"] = "complete"
        if cuda:
            result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            result["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
            free, total = torch.cuda.mem_get_info(device)
            result["device_total_gib"] = round(total / 2**30, 3)
    except BaseException as error:  # record OOM and dtype errors instead of losing the phase
        result.update(status="failed", error=f"{type(error).__name__}: {str(error)[:2000]}",
                      traceback=traceback.format_exc()[-6000:])
        if cuda:
            result["peak_allocated_gib_at_failure"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        log(f"{tag} FAILED: {result['error'][:500]}")
    finally:
        updates_log.close()
        write_json(out / f"result-rank{rank}.json", sanitize(result))
        if args.ddp:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
    return 0 if result["status"] == "complete" else 1


def sanitize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


# Orchestrator -------------------------------------------------------------------------------------

def environment():
    env = {"python": sys.version.split()[0]}
    try:
        import torch

        env.update(torch=torch.__version__, cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(i) for i in
                                                                           range(torch.cuda.device_count())])
        if torch.cuda.device_count():
            env["capability"] = list(torch.cuda.get_device_capability(0))
            env["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as error:  # noqa: BLE001
        env["torch_error"] = repr(error)
    for mod in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as error:  # noqa: BLE001
            env[mod] = f"unavailable: {type(error).__name__}"
    try:
        env["host_mem_gib"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        env["cpus"] = os.cpu_count()
    except (ValueError, OSError):
        pass
    return env


def extrapolate(records, features, world):
    """Seconds per update for the pilot (updates 1..268) and the arm (1..1,071) from the measured updates.

    Two estimates: per-bucket mean (excluding the first, warm-up update) times the schedule's bucket counts, and a
    least-squares fit seconds = a + b * tokens + c * sum(len^2) over the measured updates applied per update."""
    recs = [r for r in records[1:] if isinstance(r.get("seconds"), (int, float))]
    if len(recs) < 4:
        return {"error": "too few measured updates"}
    by_bucket = collections.defaultdict(list)
    for r in recs:
        by_bucket[r["bucket"]].append(r["seconds"])
    overall = sum(r["seconds"] for r in recs) / len(recs)
    bucket_mean = {b: sum(v) / len(v) for b, v in by_bucket.items()}
    # least squares on [1, tokens, sq]
    xs = [[1.0, r["tokens"] / 1e4, r.get("sq", 0) / 1e7] for r in recs]
    ys = [r["seconds"] for r in recs]
    coef = lstsq(xs, ys)

    def predict(upto):
        fs = [features[str(u)] for u in range(1, upto + 1)]
        bucket = sum(bucket_mean.get(f["bucket"], overall) for f in fs)
        fit = sum(coef[0] + coef[1] * f["tokens"] / 1e4 + coef[2] * f["sq"] / 1e7 for f in fs) if coef else None
        return {"bucket_mean_hours": round(bucket / 3600, 3), "fit_hours": round(fit / 3600, 3) if fit else None}

    return {"measured_updates": len(recs), "mean_seconds_per_update": round(overall, 3),
            "bucket_mean_seconds": {str(b): round(v, 3) for b, v in sorted(bucket_mean.items())},
            "bucket_counts": {str(b): len(v) for b, v in sorted(by_bucket.items())},
            "fit_coefficients": coef, "measured_tokens_per_second": round(
                sum(r["tokens"] for r in recs) / sum(r["seconds"] for r in recs), 1),
            "pilot_268": predict(PILOT_UPDATES), "arm_1071": predict(ARM_UPDATES), "gpus": world}


def lstsq(xs, ys):
    """Tiny normal-equation solver (3 unknowns), no numpy dependency in the orchestrator."""
    n = len(xs[0])
    a = [[sum(x[i] * x[j] for x in xs) for j in range(n)] + [sum(x[i] * y for x, y in zip(xs, ys))] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(n):
            if r != col:
                f = a[r][col] / a[col][col]
                a[r] = [v - f * w for v, w in zip(a[r], a[col])]
    return [round(a[i][n] / a[i][i], 6) for i in range(n)]


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        return run_worker(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    summary = {"schema": SCHEMA, "run_id": args.run_id, "phases": args.phases, "lr": args.lr, "warmup": args.warmup,
               "updates": args.updates, "ddp_updates": args.ddp_updates, "status": "starting", "steps": []}

    def mark(step):
        summary["steps"].append([step, round(time.monotonic() - t0, 1)])
        summary["elapsed_seconds"] = round(time.monotonic() - t0, 1)
        write_json(args.output_dir / "summary.json", sanitize(summary))
        log(step)

    phases = [p.split("+") for p in args.phases.split(",") if p]
    unknown = [c for ph in phases for c in ph if c.removeprefix("ddp:") not in CONFIGS]
    if unknown or any(len(ph) > 2 or (len(ph) > 1 and any(c.startswith("ddp:") for c in ph)) for ph in phases):
        raise SystemExit(f"bad --phases {args.phases!r}: unknown {unknown}, at most 2 configs per phase, ddp alone")
    if not args.selftest_tiny:
        # Kaggle's image ships torchao 0.10.0; peft 0.20.0's LoRA dispatcher raises ImportError for torchao < 0.16
        # on every plain nn.Linear (probe attempt 1). The probe does not use torchao, so remove it.
        r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True, text=True,
                           timeout=300)
        summary["torchao_uninstall"] = {"exit": r.returncode, "tail": (r.stdout or r.stderr)[-300:]}
    if any(c.removeprefix("ddp:").startswith("q4") for ph in phases for c in ph) and not args.selftest_tiny:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "bitsandbytes==0.50.2"],
                           capture_output=True, text=True, timeout=600)
        summary["bitsandbytes_install"] = {"exit": r.returncode, "tail": (r.stderr or r.stdout)[-800:]}
    summary["environment"] = environment()
    mark("environment")
    plan = build_plan(args)
    plan_path = args.cache_dir / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan))
    sq = {}
    for up in plan["updates"]:
        sq[up["update"]] = sum(len(r["input_ids"]) ** 2 for r in up["rows"])
    summary["inputs"] = {"model": MODEL[0], "model_sha256": plan["model_sha256"], "rows": ROWS[0],
                         "rows_sha256": plan.get("rows_sha256"), "schedule": SCHEDULE[0],
                         "schedule_sha256": plan.get("schedule_sha256"),
                         "sampled_updates": [u["update"] for u in plan["updates"]],
                         "sampled_bucket_counts": dict(collections.Counter(str(u["bucket"]) for u in plan["updates"]))}
    mark("plan")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2")
    common = ["--run-id", args.run_id, "--output-dir", str(args.output_dir), "--cache-dir", str(args.cache_dir),
              "--plan", str(plan_path), "--updates", str(args.updates), "--ddp-updates", str(args.ddp_updates),
              "--config-seconds", str(args.config_seconds), "--lr", str(args.lr), "--warmup", str(args.warmup),
              "--seed", str(args.seed)]
    results = {}
    for ph in phases:
        procs = []
        for gpu, name in enumerate(ph):
            if name.startswith("ddp:"):
                cfg = name[4:]
                cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                       str(Path(__file__).resolve()), "--worker", cfg, "--ddp", *common]
                penv = dict(env)
            else:
                cfg = name
                cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", cfg, *common]
                penv = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
            if args.selftest_tiny:
                penv["CUDA_VISIBLE_DEVICES"] = ""
            procs.append((name, subprocess.Popen(cmd, env=penv)))
        for name, proc in procs:
            code = proc.wait()
            tag = f"ddp-{name[4:]}" if name.startswith("ddp:") else name
            path = args.output_dir / tag / "result-rank0.json"
            res = json.loads(path.read_text()) if path.is_file() else {"status": "no result", "exit": code}
            res["exit"] = code
            if args.ddp or name.startswith("ddp:"):
                r1 = args.output_dir / tag / "result-rank1.json"
                if r1.is_file():
                    other = json.loads(r1.read_text())
                    res["rank1_peak_allocated_gib"] = other.get("peak_allocated_gib")
                    res["rank1_status"] = other.get("status")
            for rec in res.get("records", []):
                rec["sq"] = sq.get(rec["update"], 0)
            res["extrapolation"] = extrapolate(res.get("records", []), plan["schedule_features"],
                                               2 if name.startswith("ddp:") else 1)
            losses = [r["loss"] for r in res.get("records", []) if isinstance(r.get("loss"), (int, float))]
            res["loss_summary"] = {"first": losses[0] if losses else None, "last": losses[-1] if losses else None,
                                   "all_finite": all(math.isfinite(x) for x in losses) and len(losses) == len(
                                       res.get("records", [])),
                                   "skipped_updates": sum(1 for r in res.get("records", []) if r.get("skipped")),
                                   "final_scale": (res.get("records") or [{}])[-1].get("scale_after")}
            results[tag] = {k: v for k, v in res.items() if k != "traceback"} | (
                {"traceback_tail": res["traceback"][-1500:]} if "traceback" in res else {})
            summary["results"] = results
            mark(f"done {tag}: {res.get('status')}")
    summary["status"] = "complete"
    mark("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
