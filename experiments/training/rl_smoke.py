#!/usr/bin/env python3
"""RL pipeline trial: GRPO with validator rewards on zeta2 edit scenarios.

First real RL run on our own model (night 2026-08-20). Thesis under test:
environments-as-data — single-model GRPO where the reward is the SAME
exact/validator score the eval harness uses, on TRAIN-split rows only.

Method (kept deliberately simple; SAO lesson: single-rollout-style RL is
fine for agentic/edit tasks, GRPO with small groups is the same spirit):

  1. MERGE the v6 SFT LoRA to a plain base (/tmp) via
     save_pretrained_merged — RL trains a FRESH LoRA on top, never the
     original adapter in place (artifacts stay separable).
  2. GRPO (trl GRPOTrainer, unsloth-patched) on TRAIN-split rows from
     /mnt/h/sepalith/datasets/sft_v6/train.jsonl for the chosen scenario
     families. Prompts/targets are the EXACT rows the assembler rendered
     with edit_row()/render_zeta2() — zero re-rendering, zero skew. Any
     row whose prompt appears in the materialized holdout split
     (sft_v3/eval.jsonl, the authoritative 3% package holdout that
     eval_scenarios.py scores against) is EXCLUDED (verified: 5 rows).
  3. Reward per completion = exact + 0.2 * line_f1 where
       exact    = parsed prediction == normalized target lines
                  (same norm as eval_scenarios' exact: rstrip per line,
                  trailing blanks popped — run_eval.parse_pred + norm),
       line_f1  = scenarios.exact_reward (verbatim copy below; the
                  scenarios module cannot load in .venv-sft because
                  tree_sitter_r is not installed there — the function is
                  pure difflib and identical to the eval path).
     Families whose target is an UNCHANGED region (no_op) reward
     emit-nothing correctness through the same code path: their target
     minus the UPDATED marker is empty, so the reward is 1.2 iff the
     model emits just the marker.
  4. BOS parity: SFT tokenized text with add_special_tokens=True (BOS id
     0 prepended); trl's GRPO tokenizes prompts with
     add_special_tokens=False, so every prompt string here starts with
     the tokenizer's literal BOS token — assert-checked at startup.

Usage (.venv-sft, GPU free per day-queue):
  python rl_smoke.py --merge                 # step 1 only
  python rl_smoke.py --smoke                 # 10 steps, 32 prompts, /tmp out
  python rl_smoke.py --steps 300             # full run
Output: --out (default /mnt/h/sepalith/runs/rl_grpo_v1): checkpoints,
final_lora/, rl_metrics.jsonl (one line per optimizer step).
"""
import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))           # run_eval (light deps)
from run_eval import norm, parse_pred                    # noqa: E402  exact eval path

SFT_LORA = Path("/mnt/h/sepalith/runs/sft_v6_minicpm5/final_lora")
MERGED_BASE = Path("/tmp/merged_rl_v6_base")            # RL base (merged v6)
TRAIN_JSONL = Path("/mnt/h/sepalith/datasets/sft_v6/train.jsonl")
HOLDOUT_REF = Path("/mnt/h/sepalith/datasets/sft_v3/eval.jsonl")
UPDATED_MARK = ">>>>>>> UPDATED"
STOP = ">>>>>>> UPDATED"

# verbatim copy of scenarios.exact_reward (experiments/synthetic-data/
# scenarios.py) — scenarios.py imports tree_sitter_r at module load, which
# is absent in .venv-sft; the function is pure difflib and byte-identical.
import difflib                                           # noqa: E402


def exact_reward(pred_lines, region_new_lines) -> float:
    """1.0 on exact match after rstrip-normalisation, else line-F1 (difflib)."""
    p = [l.rstrip() for l in (pred_lines or [])]
    g = [l.rstrip() for l in (region_new_lines or [])]
    while p and p[-1] == "":
        p.pop()
    while g and g[-1] == "":
        g.pop()
    if p == g:
        return 1.0
    if not p or not g:
        return 0.0
    sm = difflib.SequenceMatcher(a=p, b=g, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    prec = matched / len(p)
    rec = matched / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


FAMILY_QUOTA = {          # train-split draws; held-out packages excluded
    "rename_propagation": 1400,   # v6 exact 0.820 — headroom, not cold
    "format_propagation": 1400,   # v6 exact 0.522 — most headroom
    "no_op": 350,                 # emit-nothing discipline (intent guard)
    "pipe_rewrite": 150,          # tiny anchor for the near-ceiling 0.944
}
SHAPING = 0.2             # reward = exact + SHAPING * line_f1 (max 1.2)

# ---------------------------------------------------------------------------
# dataset: train.jsonl -> {prompt (BOS-prefixed), target, family}
# ---------------------------------------------------------------------------


def gt_lines(target: str):
    """Normalized target region lines: strip the UPDATED marker, eval norm."""
    body = target
    for suf in (f"\n{UPDATED_MARK}", UPDATED_MARK):
        if body.endswith(suf):
            body = body[: -len(suf)]
            break
    return norm(body.splitlines())


def build_dataset(tok, quotas=FAMILY_QUOTA, seed=3407):
    import random
    holdout = set()
    for line in open(HOLDOUT_REF):
        r = json.loads(line)
        if r.get("family") in ("rename_propagation", "pipe_rewrite",
                               "format_propagation"):
            holdout.add(r["prompt"])
    pools, excluded = {f: [] for f in quotas}, {"holdout": 0, "dupe": 0}
    seen = set()
    for line in open(TRAIN_JSONL):
        r = json.loads(line)
        fam = r.get("family")
        if fam not in quotas:
            continue
        if r["prompt"] in holdout:
            excluded["holdout"] += 1
            continue
        if r["prompt"] in seen:
            excluded["dupe"] += 1
            continue
        seen.add(r["prompt"])
        pools[fam].append(r)
    rng = random.Random(seed)
    rows = []
    for fam, pool in pools.items():
        rng.shuffle(pool)                       # random draw, not file order
        rows.extend(pool[: quotas[fam]])
    rng.shuffle(rows)
    # BOS parity + length guard: prompts must fit max_prompt_length tokens
    # (trl truncates from the RIGHT, which would cut the <[fim-middle]> cue),
    # and targets must fit the completion cap (completions stop at the cap —
    # the SFT model emits no EOS — so rows with longer targets would be
    # un-winnable and are dropped, count logged). Prompt cap 480: HF generate
    # prefill computes full-vocab logits at EVERY prompt position of the
    # padded generation batch (32 x L x 99.6k x 2B); with L~800 that spiked
    # 6.25GB and OOM'd against the twin-coexistence fraction cap (step 41
    # of the first full attempt); 480 keeps the spike ~3GB.
    bos = tok.bos_token
    max_prompt_tok, max_target_tok = 0, 0
    dropped_len = 0
    out = []
    for r in rows:
        p = bos + r["prompt"]
        pt = len(tok(p, add_special_tokens=False)["input_ids"])
        tt = len(tok(r["target"], add_special_tokens=False)["input_ids"])
        if pt > 480 or tt > 170:
            dropped_len += 1
            continue
        max_prompt_tok, max_target_tok = max(max_prompt_tok, pt), max(max_target_tok, tt)
        out.append({"prompt": p, "target": r["target"], "family": r["family"]})
    stats = {f: sum(1 for x in out if x["family"] == f) for f in quotas}
    excluded["len"] = dropped_len
    return out, dict(excluded=excluded, counts=stats,
                     max_prompt_tok=max_prompt_tok, max_target_tok=max_target_tok)


# reward-fn metric stash: (family, exact, reward) per completion, flushed
# per optimizer step by the callback below (on-policy: one reward call per
# generation round == one optimizer step)
STASH = deque()


def scenario_reward(prompts, completions, completion_ids=None, target=None,
                    family=None, trainer_state=None, **kw):
    out = []
    for fam, comp, tgt in zip(family, completions, target):
        pred = parse_pred("zeta2", comp)      # exact eval parsing path
        gt = gt_lines(tgt)
        ex = int(pred == gt)
        rew = ex + SHAPING * exact_reward(pred, gt)
        STASH.append((fam, ex, rew))
        out.append(rew)
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def gpu_guard(limit_mib=20000):
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    util, mem = [x.strip().split()[0] for x in gpu.split(",")]
    print(f"GPU check: util={util}% mem={mem}MiB", flush=True)
    # shared-machine policy is abort >8GB; tonight's 03:40 user override lets
    # the junior twin POC (~13.7GB cap) share the 5090 with RL (senior), so
    # the launch gate is 20GB instead — RL needs ~15GB and 15+13.7 < 32.
    if float(mem) > limit_mib:
        raise SystemExit(f"GPU busy (>{limit_mib}MiB used) — aborting")


def do_merge():
    from unsloth import FastLanguageModel
    if (MERGED_BASE / "config.json").exists():
        print(f"merge: {MERGED_BASE} already exists, skipping")
        return
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(SFT_LORA), max_seq_length=2048, dtype=None,
        load_in_4bit=False)
    model.save_pretrained_merged(str(MERGED_BASE), tokenizer,
                                 save_method="merged_16bit")
    print(f"merged -> {MERGED_BASE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="only merge v6 LoRA")
    ap.add_argument("--smoke", action="store_true", help="10 steps, 32 prompts")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8, help="per-device batch (completions)")
    ap.add_argument("--ga", type=int, default=4, help="grad accumulation")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--memfrac", type=float, default=0.62,
                    help="cuda memory fraction cap (twin POC co-existence)")
    ap.add_argument("--families", default=",".join(FAMILY_QUOTA))
    ap.add_argument("--out", default="/mnt/h/sepalith/runs/rl_grpo_v1")
    args = ap.parse_args()

    if args.merge:
        do_merge()
        return

    gpu_guard()
    from unsloth import FastLanguageModel          # patch BEFORE trl import
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    quotas = {f: FAMILY_QUOTA[f] for f in
              [x.strip() for x in args.families.split(",")] if f in FAMILY_QUOTA}
    if args.smoke:
        quotas = {f: min(n, 8) for f, n in quotas.items()}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MERGED_BASE), max_seq_length=2048, dtype=None,
        load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=3407)
    model.config.use_cache = True      # KV cache for GRPO's generate calls
    if args.memfrac:
        torch.cuda.set_per_process_memory_fraction(args.memfrac)
        print(f"cuda memory fraction capped at {args.memfrac}", flush=True)

    # BOS parity self-check: our prompt strings must tokenize to BOS-first
    ids = tokenizer(tokenizer.bos_token + "x", add_special_tokens=False)["input_ids"]
    assert ids[0] == tokenizer.bos_token_id, \
        f"BOS parity broken: {ids[:3]} vs bos_id={tokenizer.bos_token_id}"

    rows, dstat = build_dataset(tokenizer, quotas)
    print(json.dumps(dict(dataset=dstat, n_rows=len(rows),
                          bos=tokenizer.bos_token)), flush=True)
    ds = Dataset.from_list(rows)
    max_prompt_len = 512
    assert dstat["max_prompt_tok"] + 1 <= max_prompt_len, \
        f"prompt overflow: {dstat['max_prompt_tok']} > {max_prompt_len - 1}"
    # 192-token cap: the SFT model emits no EOS, so every completion runs to
    # the cap; on WSL2 each decode step is launch-overhead-bound (~0.12s
    # regardless of batch), so cap length directly sets step time (~23s/round)
    max_completion_len = 192
    assert dstat["max_target_tok"] + 8 <= max_completion_len, \
        f"target overflow: {dstat['max_target_tok']}"

    steps = 10 if args.smoke else args.steps
    out = Path("/tmp/rl_smoke_out" if args.smoke else args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "rl_metrics.jsonl"

    class MetricsCb(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()
            self.last_flush = 0

        def on_step_end(self, a, state, control, **kw):
            recs = list(STASH)
            STASH.clear()
            if not recs:
                return
            n = len(recs)
            line = dict(step=state.global_step, n=n,
                        reward=round(sum(r for _, _, r in recs) / n, 4),
                        exact=round(sum(e for _, e, _ in recs) / n, 4))
            for fam in sorted(set(f for f, _, _ in recs)):
                fr = [x for x in recs if x[0] == fam]
                line[f"exact_{fam}"] = round(sum(e for _, e, _ in fr) / len(fr), 4)
                line[f"n_{fam}"] = len(fr)
            line["elapsed_s"] = round(time.time() - self.t0, 1)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            print("RLMETRIC " + json.dumps(line), flush=True)

        def on_train_end(self, a, state, control, **kw):
            # greedy probe: does the model still emit zeta2 completions?
            import torch as _t
            model = kw.get("model")
            was_training = model.training
            model.eval()
            probes = rows[:2]
            with _t.no_grad():
                for p in probes:
                    enc = tokenizer(p["prompt"], return_tensors="pt",
                                    add_special_tokens=False).to(model.device)
                    o = model.generate(**enc, max_new_tokens=120,
                                       do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
                    txt = tokenizer.decode(o[0][enc.input_ids.shape[1]:],
                                           skip_special_tokens=True)
                    print("PROBE[" + p["family"] + "] " + repr(txt[:220]), flush=True)
            if was_training:
                model.train()

    cfg = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        steps_per_generation=args.ga,      # gen batch == 1 optim step, on-policy
        num_generations=args.num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        max_steps=steps,
        learning_rate=args.lr, lr_scheduler_type="constant_with_warmup",
        warmup_steps=10 if not args.smoke else 2,
        temperature=args.temp, beta=args.beta,
        loss_type="bnpo",                  # trl 0.24 default
        scale_rewards="group",
        bf16=True, seed=3407, report_to="none",
        logging_steps=10, log_completions=True, num_completions_to_print=2,
        save_strategy="steps" if not args.smoke else "no",
        save_steps=50, save_total_limit=2,
    )
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, reward_funcs=scenario_reward,
        args=cfg, train_dataset=ds,
        callbacks=[MetricsCb()])
    trainer.train()

    final = out / "final_lora"
    model.save_pretrained(str(final))
    print(f"SAVED_LORA {final}")
    if not args.smoke:
        merged = Path("/tmp/merged_rl_grpo_v1")
        model.save_pretrained_merged(str(merged), tokenizer,
                                     save_method="merged_16bit")
        print(f"SAVED_MERGED {merged}")
    print("DONE")


if __name__ == "__main__":
    main()
