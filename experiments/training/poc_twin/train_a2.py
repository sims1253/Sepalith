"""A2 trainer: the twin discipline + matryoshka exits + MTP ramp.

Same data order/schedule/optimizer/QK-Clip as the verified twin trainer
(pieces imported from train.py), with the A2 structure:
  loss = ce_top + sum(w_e * ce_e) + mtp_lambda(step) * ce_mtp
  mtp_lambda: 0 until 10% of steps, then 0.5 (the spec ramp)
Per-exit telemetry every log step (the 16L-vs-24L instrument).

CLUSTER LAUNCH (single big GPU — the zero-friction path):
  POC_MEM_FRACTION=0.95 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_a2.py --arm muon --lr 0.01 --d-model 2048 --n-layers 24 \
    --n-q 16 --n-kv 2 --head-dim 128 --ffn-hidden 8192 --vocab 32768 \
    --exits 8,16 --steps <N> --tokens-per-step 524288 --micro-bs <auto> \
    --data <blocks> ...
Multi-GPU: not yet DDP-wrapped (documented next step); one 80GB card
runs the 1.5B at micro-bs 8-12 single-GPU (~3-4x the 5090 rate).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import model_config  # noqa: E402
from model_a2 import A2Model  # noqa: E402
import train as base  # noqa: E402  PackedData, lr_at, build_optim, logging


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["muon", "adamw"], default="muon")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-embed", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tokens-per-step", type=int, default=262144)
    ap.add_argument("--micro-bs", type=int, default=0,
                    help="0 = auto-shape from free VRAM (cluster-ready)")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1273)
    ap.add_argument("--data", default="/tmp/poc_twin/train_blocks.npy")
    ap.add_argument("--eval-data", default="/tmp/poc_twin/eval_blocks.npy")
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-q", type=int, default=None)
    ap.add_argument("--n-kv", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=None)
    ap.add_argument("--ffn-hidden", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--exits", default=None,
                    help="comma exit layers (default: 8,16 at 24L; 4,8 at 12L)")
    ap.add_argument("--exit-weights", default="0.25,0.125")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--mtp-lambda", type=float, default=0.5)
    ap.add_argument("--mtp-from", type=float, default=0.10,
                    help="fraction of steps before MTP loss ramps in")
    ap.add_argument("--tau", type=float, default=100.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tag", default="a2")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--gate-min-free-mib", type=int, default=16 * 1024)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    if args.no_gate is False and args.gate_min_free_mib > 0:
        base.LaunchGate(args.gate_min_free_mib).wait()

    over = {"max_seq": args.seq}
    for k, a in (("d_model", args.d_model), ("n_layers", args.n_layers),
                 ("n_q", args.n_q), ("n_kv", args.n_kv),
                 ("head_dim", args.head_dim), ("ffn_hidden", args.ffn_hidden),
                 ("vocab", args.vocab)):
        if a is not None:
            over[k] = a
    n_layers = over.get("n_layers") or model_config()["n_layers"]
    exits = ([int(x) for x in args.exits.split(",")] if args.exits
             else ([8, 16] if n_layers >= 24 else [4, 8]))
    over["exit_layers"] = exits
    over["exit_weights"] = [float(x) for x in args.exit_weights.split(",")]
    over["use_mtp"] = not args.no_mtp

    cfg = model_config(**over)
    model = A2Model(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps(dict(event="cfg", params=n_params,
                          exits=exits, mtp=over["use_mtp"], cfg=cfg)),
          flush=True)

    # auto-shape micro-bs (cluster-ready): target ~60% of free VRAM for
    # activations; the 206M twin ~0.5GB/row, 1.5B ~2.5GB/row at seq 1024
    # (measured edges: 5090-32GB mb2@1.5B; scale down conservatively)
    if args.micro_bs <= 0:
        free = torch.cuda.mem_get_info()[0] / 2**20
        per_row = {12: 500, 24: 2600}.get(n_layers, 1000)
        args.micro_bs = max(1, min(16, int(free * 0.25 / per_row)))
    accum = args.tokens_per_step // (args.seq * args.micro_bs)
    print(json.dumps(dict(event="shape", micro_bs=args.micro_bs,
                          accum=accum)), flush=True)

    torch.manual_seed(args.seed)
    data = base.PackedData(args.data, args.seed, args.micro_bs * max(accum, 1))
    lr_embed = args.lr_embed if args.lr_embed is not None else args.lr
    muon, extra_opts, desc = base.build_optim(
        args.arm, model, args.lr, lr_embed, args.wd)
    opts = [muon] + extra_opts

    log_path = os.path.join(HERE, "logs", f"{args.tag}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "a")

    def log(d):
        d["tag"] = args.tag
        d["ts"] = time.time()
        log_f.write(json.dumps(d) + "\n")
        log_f.flush()
        print(json.dumps(d), flush=True)

    mtp_from_step = int(args.steps * args.mtp_from)
    t0 = time.time()
    tokens_seen = 0
    for step in range(1, args.steps + 1):
        lr_now = base.lr_at(step, args.steps, args.lr)
        lr_emb_now = base.lr_at(step, args.steps, lr_embed)
        for g in opts[0].param_groups:
            g["lr"] = lr_now
        for o in opts[1:]:
            for g in o.param_groups:
                g["lr"] = lr_emb_now
        for o in opts:
            o.zero_grad(set_to_none=True)
        xb = data.batch(step)
        mtp_w = args.mtp_lambda if step >= mtp_from_step else 0.0
        sums = {}
        for mi in range(accum):
            rows = xb[mi * args.micro_bs:(mi + 1) * args.micro_bs]
            x = torch.from_numpy(rows.astype(np.int64)).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, ld = model(x, targets=x, probe=(mi == 0),
                              mtp_weight=mtp_w)
                loss = ld["ce_top"] + mtp_w * ld.get("ce_mtp",
                                                     torch.zeros((), device="cuda"))
                for e, w in zip(model.exit_layers, model.exit_weights):
                    loss = loss + w * ld[f"ce_{e}"]
            (loss / accum).backward()
            for k, v in ld.items():
                sums[k] = sums.get(k, 0.0) + float(v) / accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        for o in opts:
            o.step()
        n_clip, qk_max = model.qk_clip_all(args.tau, args.alpha)
        tokens_seen += args.tokens_per_step
        if step % args.log_every == 0 or step == 1:
            out = dict(event="step", step=step, tokens=tokens_seen,
                       grad_norm=round(gn, 4), qk_max=round(qk_max, 1),
                       qk_clipped_heads=n_clip,
                       tok_per_s=round(tokens_seen / (time.time() - t0), 1),
                       elapsed_s=round(time.time() - t0, 1))
            for k, v in sums.items():
                out[k] = round(v / accum, 4)
            log(out)

    ckpt_dir = os.path.join("/tmp/poc_twin", f"ckpt_{args.tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg,
                "args": vars(args)},
               os.path.join(ckpt_dir, "final.pt"))
    log(dict(event="done", steps=args.steps, tokens=tokens_seen,
             ckpt=ckpt_dir))


if __name__ == "__main__":
    main()
