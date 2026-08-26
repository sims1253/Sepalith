#!/usr/bin/env python3
"""RL-run-4: GRPO + frozen privileged critic, TETHER advantage blend.

The live test of the POC verdict: identical config to rl_grpo_v1 (v6
merged base, fresh LoRA r16, run-1 family quotas, seed 3407, bs 8 x ga 4,
num_generations 4, temp 1.0, beta 0, 220 steps) with ONE variable — the
advantage. Instead of trl's mean-group baseline (std-normalized), each
completion's advantage is the TETHER blend (Le Critique):

    A_i = R_i - [(1-rho) * LOO_i + rho * V_i]

with V_i from the FROZEN privileged critic trained in 01_train_value.py
(pvf_value_priv: v7 backbone + LoRA + bounded head, conditioned on the
reference block the policy never sees), LOO the leave-one-out group mean,
rho fixed at the offline-fitted value from 02_tether.py. Unnormalized
advantages by design (paper recipe; also our dr_grpo stance).

Mechanics: trl 0.24 computes advantages inline after the reward call, so
(a) the reward fn computes R, LOO and batched-critic V per completion and
stashes adv keyed by (prompt, completion); (b) a GRPOTrainer subclass
swaps inputs["advantages"] in compute_loss (decode row ids -> text ->
lookup; rows trl can't match keep trl's own advantage, counted+logged).
The critic runs inside the reward fn — it receives the full grouped batch
in one call (contiguous, prompt-major), so group structure is exact.

Prompt -> reference-block map: scenarios_v1 rows re-rendered with the
assembler path and matched against the sft_v6 train prompts (same
machinery as 00_replay.py's fidelity gate, v6 edition). Coverage is
reported; unmatched prompts run with an empty reference block (counted).

Output: /mnt/h/sepalith/runs/rl_grpo_v4_tether — rl_metrics.jsonl (same
schema as v1: step, n, reward, exact, per-family exact) + extra fields
rho, v_mean, lookup_miss. Compare against /mnt/h/sepalith/runs/
rl_grpo_v1/rl_metrics.jsonl step-by-step.

Usage (.venv-sft, GPU free):
  python 03_grpo_tether.py --rho 0.7 --steps 220
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # rl_smoke
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "eval"))
sys.path.insert(0, str(HERE.parent.parent / "post-processing"))
import rl_smoke                                     # noqa: E402
from rl_smoke import (build_dataset, do_merge, gpu_guard,          # noqa: E402
                      exact_reward, gt_lines, FAMILY_QUOTA)
from run_eval import parse_pred                     # noqa: E402
from assemble_sft_v5 import edit_row                # noqa: E402

OUT_DIR = Path("/mnt/h/sepalith/runs/rl_grpo_v4_tether")
CRITIC_BASE = Path("/tmp/merged_pvf_v7_base")
CRITIC_LORA = Path("/mnt/h/sepalith/runs/pvf_value_priv/final_lora")
CRITIC_HEAD = Path("/mnt/h/sepalith/runs/pvf_value_priv/head.pt")
SCEN_DIR = Path("/mnt/h/sepalith/datasets/scenarios_v1")
TRAIN_V6 = Path("/mnt/h/sepalith/datasets/sft_v6/train.jsonl")
REWARD_MAX = 1.2
PRIV_MARK = "#~ critic reference (never shown to the policy):"


# ---------------------------------------------------------------------------
# prompt -> privileged reference block (v6 edition of 00_replay's gate)
# ---------------------------------------------------------------------------

def ref_block(family, ref):
    lines = [PRIV_MARK, f"#~ task note: {ref.get('note', '')}"]
    if family == "no_op":
        lines.append("#~ correct action: emit nothing (region unchanged):")
        lines += [f"#~ | {l}" for l in (ref.get("region_old") or [])[:8]]
    else:
        lines.append("#~ correct region after the edit:")
        lines += [f"#~ | {l}" for l in (ref.get("region_new") or [])[:12]]
    return "\n".join(lines)


def build_ref_map():
    """Render the 4 scenario families; map sft_v6 train prompt -> ref block."""
    fams = set(FAMILY_QUOTA)
    v6_prompts = set()
    for line in open(TRAIN_V6):
        r = json.loads(line)
        if r.get("family") in fams:
            v6_prompts.add(r["prompt"])
    ref_map, matched = {}, 0
    for fam in fams:
        for line in open(SCEN_DIR / f"{fam}.jsonl"):
            row = json.loads(line)
            if fam == "no_op":
                rr = edit_row(row, fam, row["package"], fd=0,
                              cursor_after=row.get("cursor_idx", 0))
            else:
                rr = edit_row(row, fam, row["package"])
            if rr is not None and rr["prompt"] in v6_prompts:
                ref_map[rr["prompt"]] = ref_block(fam, row)
                matched += 1
    print(json.dumps(dict(ref_map=len(ref_map), v6_scenario_prompts=len(
        v6_prompts))), flush=True)
    return ref_map


# ---------------------------------------------------------------------------
# frozen privileged critic
# ---------------------------------------------------------------------------

class Critic:
    def __init__(self, device="cuda"):
        import torch
        import torch.nn as nn
        from unsloth import FastLanguageModel
        self.torch = torch
        # unsloth loader: the merged base was saved by save_pretrained_merged
        # and its attention forward needs unsloth's patches (apply_qkv/o) —
        # plain AutoModelForCausalLM crashes with "'LlamaAttention' object
        # has no attribute 'apply_qkv'" (found in smoke).
        model, self.tok = FastLanguageModel.from_pretrained(
            model_name=str(CRITIC_LORA), max_seq_length=1024, dtype=None,
            load_in_4bit=False)
        model = model.merge_and_unload() if hasattr(model, "merge_and_unload") \
            else model
        self.model = model.to(device).eval()
        self.head = nn.Linear(self.model.config.hidden_size, 1).to(
            device=device, dtype=torch.bfloat16)
        self.head.load_state_dict(
            {k: v.to(torch.bfloat16) for k, v in
             torch.load(CRITIC_HEAD, map_location=device).items()})
        self.head.eval()
        self.device = device
        self.bos = self.tok.bos_token

    def values(self, prompt_completions):
        """[(prompt, completion, ref_block)] -> [V,...] batched, sigmoid-bounded."""
        torch = self.torch
        ids_list = []
        for p, c, ref in prompt_completions:
            text = self.bos + p + c + ("\n" + ref if ref else "")
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            ids_list.append(ids[:1024])
        n = max(len(x) for x in ids_list)
        x = torch.full((len(ids_list), n), self.tok.pad_token_id,
                       dtype=torch.long)
        m = torch.zeros((len(ids_list), n), dtype=torch.long)
        for i, ids in enumerate(ids_list):
            x[i, : len(ids)] = torch.tensor(ids)
            m[i, : len(ids)] = 1
        x, m = x.to(self.device), m.to(self.device)
        with torch.no_grad():
            h = self.model(input_ids=x, attention_mask=m,
                           output_hidden_states=True).hidden_states[-1]
            idx = m.sum(1) - 1
            v = self.head(h[torch.arange(len(ids_list), device=self.device),
                            idx]).squeeze(-1)
            return (torch.sigmoid(v) * REWARD_MAX).float().tolist()


# ---------------------------------------------------------------------------
# reward fn: R + LOO + critic V, TETHER advantages stashed
# ---------------------------------------------------------------------------

ADV = {}            # (prompt, completion) -> tether advantage
VLOG = []           # per-step critic stats for the metrics callback


def make_reward_fn(critic, rho, ref_map, num_generations):
    SHAPING = rl_smoke.SHAPING

    def tether_reward(prompts, completions, completion_ids=None, target=None,
                      family=None, trainer_state=None, **kw):
        import statistics
        out_rewards = []
        rows = []
        for fam, comp, tgt, pr in zip(family, completions, target, prompts):
            pred = parse_pred("zeta2", comp)
            gt = gt_lines(tgt)
            ex = int(pred == gt)
            rew = ex + SHAPING * exact_reward(pred, gt)
            rl_smoke.STASH.append((fam, ex, rew))     # v1-compatible metrics
            out_rewards.append(rew)
            rows.append(dict(prompt=pr, completion=comp, reward=rew,
                             family=fam))
        # critic values for the whole generation round, one batch
        pc = [(r["prompt"].removeprefix(critic.bos), r["completion"],
               ref_map.get(r["prompt"].removeprefix(critic.bos), ""))
              for r in rows]
        try:
            vals = critic.values(pc)
        except Exception as e:                        # critic down -> pure LOO
            print(f"CRITIC_FAIL {e}", flush=True)
            vals = [None] * len(rows)
        K = num_generations
        advs = []
        for gi in range(0, len(rows), K):
            grp = rows[gi: gi + K]
            s = sum(r["reward"] for r in grp)
            for j, r in enumerate(grp):
                loo = (s - r["reward"]) / (len(grp) - 1)
                v = vals[gi + j]
                adv = r["reward"] - ((1 - rho) * loo +
                                     (rho * v if v is not None else 0.0))
                advs.append(adv)
                ADV[(norm_bos(r["prompt"], critic.bos), r["completion"])] = adv
        if vals and vals[0] is not None:
            vv = [v for v in vals if v is not None]
            VLOG.append((round(sum(vv) / len(vv), 4),
                         round(statistics.pstdev(vv), 4)))
        return out_rewards

    return tether_reward


# ---------------------------------------------------------------------------
# trainer subclass: swap advantages at compute_loss
# ---------------------------------------------------------------------------

def norm_bos(text, bos):
    """Collapse repeated leading BOS literals to exactly one.

    trl tokenizes prompts with add_special_tokens=True, adding a BOS id on
    top of our literal-BOS prompt strings (run-1 trained on double-BOS
    prompts); decoded ids therefore carry <s><s> while the reward fn saw
    one. Normalizing both sides to exactly one BOS makes the keys match."""
    while text.startswith(bos):
        text = text[len(bos):]
    return bos + text


def build_trainer_class():
    from trl import GRPOTrainer

    class TetherGRPOTrainer(GRPOTrainer):
        misses = 0
        debug_shown = 0

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            tok = self.processing_class
            advs = []
            for i in range(inputs["prompt_ids"].shape[0]):
                pm = inputs["prompt_mask"][i].bool()
                cm = inputs["completion_mask"][i].bool()
                p = norm_bos(tok.decode(inputs["prompt_ids"][i][pm],
                                        skip_special_tokens=False),
                             tok.bos_token)
                c = tok.decode(inputs["completion_ids"][i][cm],
                               skip_special_tokens=False)
                adv = ADV.get((p, c))
                if adv is None:
                    self.misses += 1
                    if self.debug_shown < 3:
                        self.debug_shown += 1
                        key = next(iter(ADV)) if ADV else None
                        print(f"ADV_MISS decoded p={p[:60]!r} c={c[:60]!r} "
                              f"mask_sums=({int(pm.sum())},{int(cm.sum())}) "
                              f"stash_key={str(key)[:120]!r}", flush=True)
                    adv = inputs["advantages"][i].item()
                advs.append(adv)
            if self.debug_shown == 0:
                import torch as _t
                print(f"ADV_SWAP mine(mean|{round(float(_t.tensor(advs).abs().mean()), 3)}) "
                      f"trl(mean|{round(float(inputs['advantages'].abs().mean()), 3)})",
                      flush=True)
                self.debug_shown = 1
            inputs = dict(inputs)
            import torch
            inputs["advantages"] = torch.tensor(
                advs, dtype=inputs["advantages"].dtype,
                device=inputs["advantages"].device)
            return super().compute_loss(model, inputs, return_outputs,
                                        num_items_in_batch)

    return TetherGRPOTrainer


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, required=True,
                    help="TETHER blend weight (offline-fitted from 02)")
    ap.add_argument("--steps", type=int, default=220)   # v1's actual length
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    gpu_guard()
    if not (rl_smoke.MERGED_BASE / "config.json").exists():
        do_merge()                                     # v6 base for the policy
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig

    torch.cuda.set_per_process_memory_fraction(0.92)    # card is ours tonight
    print("building prompt->reference map (v6)...", flush=True)
    ref_map = build_ref_map()
    critic = Critic()
    num_generations = 4

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(rl_smoke.MERGED_BASE), max_seq_length=2048, dtype=None,
        load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=3407)
    model.config.use_cache = True

    quotas = {f: min(n, 8) for f, n in FAMILY_QUOTA.items()} \
        if args.smoke else FAMILY_QUOTA
    rows, dstat = build_dataset(tokenizer, quotas)
    print(json.dumps(dict(dataset=dstat, n_rows=len(rows))), flush=True)
    ref_hits = sum(1 for r in rows if r["prompt"].removeprefix(
        tokenizer.bos_token) in ref_map)
    print(json.dumps(dict(ref_block_coverage=f"{ref_hits}/{len(rows)}")),
          flush=True)
    ds = Dataset.from_list(rows)

    out = Path("/tmp/rl_tether_smoke" if args.smoke else OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "rl_metrics.jsonl"

    class MetricsCb(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()

        def on_step_end(self, a, state, control, **kw):
            recs = list(rl_smoke.STASH)
            rl_smoke.STASH.clear()
            if not recs:
                return
            n = len(recs)
            line = dict(step=state.global_step, n=n, rho=args.rho,
                        reward=round(sum(r for _, _, r in recs) / n, 4),
                        exact=round(sum(e for _, e, _ in recs) / n, 4))
            for fam in sorted(set(f for f, _, _ in recs)):
                fr = [x for x in recs if x[0] == fam]
                line[f"exact_{fam}"] = round(
                    sum(e for _, e, _ in fr) / len(fr), 4)
                line[f"n_{fam}"] = len(fr)
            if VLOG:
                v = VLOG.pop()
                line["v_mean"], line["v_std"] = v
            line["adv_lookup_miss"] = getattr(self.trainer_ref, "misses", -1) \
                if getattr(self, "trainer_ref", None) else -1
            line["elapsed_s"] = round(time.time() - self.t0, 1)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            print("RLMETRIC " + json.dumps(line), flush=True)

    cfg = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        steps_per_generation=4,
        num_generations=num_generations,
        max_prompt_length=512, max_completion_length=192,
        max_steps=10 if args.smoke else args.steps,
        learning_rate=2e-5, lr_scheduler_type="constant_with_warmup",
        warmup_steps=10, temperature=1.0, beta=0.0,
        loss_type="bnpo", scale_rewards="group",   # trl's own advantages are
        # computed then REPLACED per-row in compute_loss; rows that fail
        # lookup keep trl's (counted in adv_lookup_miss)
        bf16=True, seed=3407, report_to="none",
        logging_steps=10, save_strategy="steps", save_steps=50,
        save_total_limit=2,
    )
    cb = MetricsCb()
    trainer = build_trainer_class()(
        model=model, processing_class=tokenizer,
        reward_funcs=make_reward_fn(critic, args.rho, ref_map,
                                    num_generations),
        args=cfg, train_dataset=ds, callbacks=[cb])
    cb.trainer_ref = trainer
    trainer.train()
    model.save_pretrained(str(out / "final_lora"))
    print(f"DONE adv_lookup_miss={trainer.misses} "
          f"rho={args.rho} steps={args.steps}")


if __name__ == "__main__":
    main()
