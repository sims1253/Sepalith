#!/usr/bin/env python3
"""PVF POC step 1: plain vs privileged value-function bake-off (EVAFUL).

Tests the Le Critique / BPCO privileged-critic claim on our distribution:
does a critic that sees the REFERENCE (region_new/note — what the policy
never sees) explain materially more within-group reward variance than a
critic restricted to the policy's own view (prompt + completion)?

Data: 00_replay.py groups (K=8 sampled completions/prompt, verbatim
rl_smoke reward). Model: MiniCPM5-1B + merged v7 SFT LoRA backbone, fresh
LoRA (r16, same modules as rl_smoke) + linear value head on the last real
token's top hidden state. Bounded head per BPCO: value = sigmoid(logit) *
1.2 (the reward range); loss = MSE(value, reward) + 0.3 * BCE(logit,
exact) (Le Critique's binary-reward observation; our exact flag is the
binary part). Terminal MC targets are the data by construction (offline,
sequence-level rewards).

Arms (--arm):
  plain  input = BOS + prompt + completion            (policy's view)
  priv   input = BOS + prompt + completion + ref block (critic-only suffix:
         "#~ critic reference" comment lines, same marker family as the
         simulator's proposal-history channel — a channel the product
         already renders, so nothing new tokenizes weirdly)

EVAFUL (Le Critique): 1 - sum_g sum_i (r_i - V_i)^2 / sum_g sum_i (r_i -
mean_g(r))^2, computed on VAL groups (packages unseen in training — the
replay split is by package). Reported overall + per family, plus the
zero-variance-group fraction (our frac_reward_zero_std analogue) and both
with/without zero-var groups. Val predictions dump per arm for 02_tether.

Usage (.venv-sft, GPU free; replay must have finished — llama-server down):
  python 01_train_value.py --merge                 # backbone only
  python 01_train_value.py --arm plain --epochs 1  # pipeline check
  python 01_train_value.py --arm both              # the bake-off (default)
"""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLAY = Path("/mnt/h/sepalith/datasets/pvf_poc_v1/replay_scenarios.jsonl")
SFT_LORA = Path("/mnt/h/sepalith/runs/sft_v7_minicpm5/final_lora")
MERGED_BASE = Path("/tmp/merged_pvf_v7_base")
RUNS = Path("/mnt/h/sepalith/runs")
SEED = 3407
MAX_LEN = 1024          # prompt<=512 + completion<=200 + ref<=~280 budget
PRIV_MARK = "#~ critic reference (never shown to the policy):"
LR = 1e-4
BS = 8
GA = 2
BCE_W = 0.3
REWARD_MAX = 1.2


def gpu_guard(limit_mib=20000):
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,"
                          "memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    util, mem = [x.strip().split()[0] for x in gpu.split(",")]
    print(f"GPU check: util={util}% mem={mem}MiB", flush=True)
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


def ref_block(family, ref):
    """Privileged text: the reference answer in product comment syntax."""
    lines = [PRIV_MARK, f"#~ task note: {ref.get('note', '')}"]
    if family == "no_op":
        lines.append("#~ correct action: emit nothing (region unchanged):")
        lines += [f"#~ | {l}" for l in (ref.get("region_old") or [])[:8]]
    else:
        lines.append("#~ correct region after the edit:")
        lines += [f"#~ | {l}" for l in (ref.get("region_new") or [])[:12]]
    return "\n".join(lines)


def load_examples(tok, arm):
    """(input_ids without special tokens' padding, reward, exact, group id,
    family, split) per completion. BOS prepended as a literal token (rl_smoke
    parity); groups whose prompt overflows 512 tokens are dropped (counted).
    Truncation: the reference block is cut first, then the completion tail —
    the prompt is never cut (its <[fim-middle]> cue is right-most)."""
    bos = tok.bos_token
    ids = tok(bos + "x", add_special_tokens=False)["input_ids"]
    assert ids[0] == tok.bos_token_id, "BOS parity broken"

    ex, dropped = [], {"prompt_len": 0}
    for line in open(REPLAY):
        g = json.loads(line)
        p_ids = tok(g["prompt"], add_special_tokens=False)["input_ids"]
        if len(p_ids) > 512:
            dropped["prompt_len"] += 1
            continue
        block = ref_block(g["family"], g["ref"]) if arm == "priv" else ""
        b_ids = tok("\n" + block, add_special_tokens=False)["input_ids"] \
            if block else []
        for comp, rew, ex_flag in zip(g["completions"], g["rewards"],
                                      g["exact"]):
            c_ids = tok(comp, add_special_tokens=False)["input_ids"]
            room = MAX_LEN - len(p_ids) - len(b_ids) - 1
            if len(c_ids) > room:                       # cut completion tail
                c_ids = c_ids[:room]
                dropped["comp_trunc"] = dropped.get("comp_trunc", 0) + 1
            ids = p_ids + c_ids + b_ids + [tok.eos_token_id]
            ex.append(dict(ids=ids, reward=rew, exact=ex_flag,
                           pid=g["pid"], family=g["family"],
                           split=g["split"]))
    return ex, dropped


def batches(rows, bs, seed):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    for i in range(0, len(rows) - bs + 1, bs):
        yield rows[i: i + bs]


def collate(batch, pad_id, device):
    import torch
    n = max(len(b["ids"]) for b in batch)
    x = torch.full((len(batch), n), pad_id, dtype=torch.long)
    m = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b["ids"])] = torch.tensor(b["ids"])
        m[i, : len(b["ids"])] = 1
    r = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
    e = torch.tensor([b["exact"] for b in batch], dtype=torch.float32)
    return x.to(device), m.to(device), r.to(device), e.to(device)


def evaful(groups):
    """1 - residual-variance / around-group-mean-variance, pooled.

    groups: {pid: dict(rewards=[...], values=[...], family=...)}.
    Returns (evaful_nonzero, evaful_all, n_zero_var_groups, per_family)."""
    def pooled(sel):
        num = den = 0.0
        for g in sel.values():
            rs, vs = g["rewards"], g["values"]
            mean = sum(rs) / len(rs)
            den += sum((r - mean) ** 2 for r in rs)
            num += sum((r - v) ** 2 for r, v in zip(rs, vs))
        return (1.0 - num / den) if den > 0 else None

    nz = {p: g for p, g in groups.items()
          if len(set(g["rewards"])) > 1}
    per_fam = {}
    for fam in sorted({g["family"] for g in groups.values()}):
        sub = {p: g for p, g in nz.items() if g["family"] == fam}
        per_fam[fam] = dict(evaful=round(pooled(sub), 4) if sub else None,
                            n_groups=len(sub))
    return pooled(nz), pooled(groups), len(groups) - len(nz), per_fam


def predict_groups(model, tok, head, rows, device, bs=16):
    import torch
    model.eval()
    groups = {}
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            batch = rows[i: i + bs]
            x, m, _, _ = collate(batch, tok.pad_token_id, device)
            h = model(input_ids=x, attention_mask=m,
                      output_hidden_states=True).hidden_states[-1]
            idx = m.sum(1) - 1
            v = head(h[torch.arange(len(batch), device=device), idx]
                     ).squeeze(-1)
            vals = (torch.sigmoid(v) * REWARD_MAX).float().tolist()
            for b, val in zip(batch, vals):
                g = groups.setdefault(b["pid"], dict(rewards=[], values=[],
                                                     family=b["family"]))
                g["rewards"].append(b["reward"])
                g["values"].append(round(val, 5))
    model.train()
    return groups


def train_arm(arm, args):
    import torch
    from unsloth import FastLanguageModel
    torch.manual_seed(SEED)
    out_dir = RUNS / f"pvf_value_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(MERGED_BASE), max_seq_length=MAX_LEN, dtype=None,
        load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=SEED)
    if args.memfrac:
        torch.cuda.set_per_process_memory_fraction(args.memfrac)
    device = model.device
    head = torch.nn.Linear(model.config.hidden_size, 1).to(
        device=device, dtype=torch.bfloat16)
    torch.nn.init.zeros_(head.weight)     # start at value 0.6 (sigmoid .5)
    torch.nn.init.zeros_(head.bias)

    ex, dropped = load_examples(tok, arm)
    train = [e for e in ex if e["split"] == "train"]
    val = [e for e in ex if e["split"] == "val"]
    print(json.dumps(dict(arm=arm, examples=len(ex), train=len(train),
                          val=len(val), dropped=dropped)), flush=True)

    params = [p for p in model.parameters() if p.requires_grad] + \
        list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    steps_per_ep = len(train) // (args.bs * args.ga)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps_per_ep * args.epochs,
        pct_start=0.05)
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        run, seen = 0.0, 0
        for bi, batch in enumerate(batches(train, args.bs, SEED + ep)):
            x, m, r, e = collate(batch, tok.pad_token_id, device)
            h = model(input_ids=x, attention_mask=m,
                      output_hidden_states=True).hidden_states[-1]
            idx = m.sum(1) - 1
            logit = head(h[torch.arange(len(batch), device=device), idx]
                         ).squeeze(-1).float()
            value = torch.sigmoid(logit) * REWARD_MAX
            loss = ((value - r) ** 2).mean() + \
                BCE_W * torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, e)
            (loss / args.ga).backward()
            run += loss.item()
            seen += 1
            if (bi + 1) % args.ga == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(json.dumps(dict(arm=arm, epoch=ep + 1, step=step,
                                          of=steps_per_ep * args.epochs,
                                          loss=round(run / seen, 5),
                                          elapsed_s=round(
                                              time.time() - t0, 1))),
                          flush=True)
                    run, seen = 0.0, 0
        groups = predict_groups(model, tok, head, val, device)
        ev_nz, ev_all, n_zero, per_fam = evaful(groups)
        print(json.dumps(dict(arm=arm, epoch=ep + 1,
                              evaful_nonzero=None if ev_nz is None
                              else round(ev_nz, 4),
                              evaful_all=None if ev_all is None
                              else round(ev_all, 4),
                              n_val_groups=len(groups),
                              n_zero_var=n_zero, per_family=per_fam)),
              flush=True)

    model.save_pretrained(str(out_dir / "final_lora"))
    import torch as _t
    _t.save(head.state_dict(), out_dir / "head.pt")
    groups = predict_groups(model, tok, head, val, device)
    with open(out_dir / "predictions_val.jsonl", "w") as f:
        for pid, g in groups.items():
            f.write(json.dumps(dict(pid=pid, family=g["family"],
                                    rewards=g["rewards"],
                                    values=g["values"])) + "\n")
    ev_nz, ev_all, n_zero, per_fam = evaful(groups)
    print(json.dumps(dict(arm=arm, FINAL=True,
                          evaful_nonzero=None if ev_nz is None
                          else round(ev_nz, 4),
                          evaful_all=None if ev_all is None
                          else round(ev_all, 4),
                          n_val_groups=len(groups), n_zero_var=n_zero,
                          per_family=per_fam,
                          elapsed_s=round(time.time() - t0, 1))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true",
                    help="only merge the v7 SFT LoRA backbone")
    ap.add_argument("--arm", default="both", choices=["plain", "priv", "both"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--memfrac", type=float, default=0.8)
    ap.add_argument("--bs", type=int, default=BS,
                    help="per-device batch (priv arm OOMs at 8: its "
                         "sequences hit the 1024 cap and hidden_states "
                         "retention defeats grad checkpointing)")
    ap.add_argument("--ga", type=int, default=GA)
    args = ap.parse_args()
    if args.merge:
        do_merge()
        return
    if not REPLAY.exists():
        sys.exit(f"missing replay data: {REPLAY} (run 00_replay.py first)")
    gpu_guard()
    arms = ["plain", "priv"] if args.arm == "both" else [args.arm]
    for arm in arms:
        train_arm(arm, args)


if __name__ == "__main__":
    main()
