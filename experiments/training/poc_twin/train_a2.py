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
    --mixture <a2_mixture_manifest.json>
  latest.pt every --ckpt-every steps into --ckpt-dir (default
  <repo>/checkpoints/<tag> — instance disk, never /tmp); --resume
  latest.pt restarts exactly (step-pure data order both loaders).
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
import train as base  # noqa: E402  PackedData, lr_at, build_optim, save_ckpt


class MixtureData:
    """Manifest-driven multi-stratum loader (the runbook §3.2 sampler).

    Every block slot draws its stratum by draw_share, then takes the next
    block from that stratum's seeded per-epoch permutation — repetition-
    by-sampling, no merged file, memmap per stratum. batch(step) is a PURE
    function of (manifest, seed, step): stratum cursors are reconstructed
    by replaying slot draws on first use, so a resumed run sees the exact
    data order a continuous run would have (the resume-discipline backstop
    for interruptible rented hosts). Missing strata re-cut their share
    pro-rata across the packed ones (the manifest's adaptive rule).
    """

    def __init__(self, manifest_path, seed):
        man = json.load(open(manifest_path))
        self.seed = seed
        self.names, self.paths, self.shares, self.ns = [], [], [], []
        for name, s in sorted(man["strata"].items()):
            if not s.get("blocks"):
                print(f"[mixture] {name}: not packed — share re-cut "
                      f"pro-rata", flush=True)
                continue
            self.names.append(name)
            self.paths.append(s["path"])
            self.shares.append(s["draw_share"])
            self.ns.append(s["blocks"])
        tot = sum(self.shares)
        self.p = [x / tot for x in self.shares]
        self.blocks = [np.load(p, mmap_mode="r") for p in self.paths]
        self._cursors = None
        self._perm = {}
        print(f"[mixture] {len(self.names)} strata, "
              f"{sum(n*1024 for n in self.ns)/1e9:.2f}B tok avail", flush=True)

    def _slot_draws(self, upto_step):
        """Per-stratum slot counts over steps 1..upto_step (replay)."""
        counts = np.zeros(len(self.names), dtype=np.int64)
        for s in range(1, upto_step + 1):
            slots = np.random.RandomState(self.seed + s).choice(
                len(self.names), size=self.seq_per_step, p=self.p)
            counts += np.bincount(slots, minlength=len(self.names))
        return counts

    def _perm_for(self, si, epoch):
        key = (si, epoch)
        if key not in self._perm:
            keep = {k: v for k, v in self._perm.items() if k[1] >= epoch}
            keep[key] = np.random.RandomState(
                self.seed + 977 * (si + 1) + epoch).permutation(self.ns[si])
            self._perm = keep
        return self._perm[key]

    def batch(self, step):
        if self._cursors is None:
            self._cursors = self._slot_draws(step - 1).tolist()
        slots = np.random.RandomState(self.seed + step).choice(
            len(self.names), size=self.seq_per_step, p=self.p)
        xb = np.empty((self.seq_per_step, self.blocks[0].shape[1]),
                      dtype=np.int32)
        for i, si in enumerate(slots):
            n = self.ns[si]
            ep, within = divmod(self._cursors[si], n)
            xb[i] = self.blocks[si][self._perm_for(si, ep)[within]]
            self._cursors[si] += 1
        return xb

    def epoch_float(self, step):
        if self._cursors is None:
            self._cursors = self._slot_draws(step - 1).tolist()
        eps = [c / n for c, n in zip(self._cursors, self.ns)]
        return float(min(eps))


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
    ap.add_argument("--mixture", default=None,
                    help="a2_mixture_manifest.json (overrides --data; the "
                         "cluster path — per-slot stratum sampling)")
    ap.add_argument("--order-file", default=None,
                    help="decay/CMA POC draw dir (draw_manifest.json + "
                         "uniform_order.idx.npy); overrides --mixture/--data")
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--floor-ratio", type=float, default=0.1)
    ap.add_argument("--const-tail-frac", type=float, default=0.0)
    ap.add_argument("--wd-muon", type=float, default=None,
                    help="weight decay for the Muon group only (arm H: 0)")
    ap.add_argument("--elr", action="store_true",
                    help="log per-window mean ||dW||_F/||W||_F (Muon group)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="default: <repo>/checkpoints/<tag> (instance disk, "
                         "NOT /tmp — survives restarts)")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--resume", default=None,
                    help="latest.pt from a killed run (step-pure data order "
                         "makes resume exact)")
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
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        step0 = ck["step"]
        print(f"[resume] model from step {step0}", flush=True)
    else:
        step0 = 0

    # auto-shape micro-bs (cluster-ready): target ~60% of free VRAM for
    # activations; the 206M twin ~0.5GB/row, 1.5B ~2.5GB/row at seq 1024
    # (measured edges: 5090-32GB mb2@1.5B; scale down conservatively)
    if args.micro_bs <= 0:
        free = torch.cuda.mem_get_info()[0] / 2**20
        per_row = {12: 500, 24: 2600}.get(n_layers, 1000)
        args.micro_bs = max(1, min(16, int(free * 0.25 / per_row)))
    accum = args.tokens_per_step // (args.seq * args.micro_bs)
    seq_per_step = args.micro_bs * accum
    print(json.dumps(dict(event="shape", micro_bs=args.micro_bs,
                          accum=accum)), flush=True)

    torch.manual_seed(args.seed)
    if args.order_file:
        data = base.OrderFileData(args.order_file, seq_per_step)
    elif args.mixture:
        data = MixtureData(args.mixture, args.seed)
        data.seq_per_step = seq_per_step
    else:
        data = base.PackedData(args.data, args.seed, seq_per_step)
    lr_embed = args.lr_embed if args.lr_embed is not None else args.lr
    muon, extra_opts, desc = base.build_optim(
        args.arm, model, args.lr, lr_embed, args.wd,
        track_updates=args.elr, wd_muon=args.wd_muon)
    opts = [muon] + extra_opts
    if args.resume:
        for o, sd in zip(opts, ck["opt"]):
            o.load_state_dict(sd)
        torch.set_rng_state(ck["torch_rng"])
        if ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        print(f"[resume] optim + rng restored", flush=True)

    ckpt_dir = args.ckpt_dir or os.path.join(
        HERE, "checkpoints", args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)

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
    tokens_seen = step0 * args.tokens_per_step
    for step in range(step0 + 1, args.steps + 1):
        lr_now = base.lr_at(step, args.steps, args.lr,
                            decay_frac=args.decay_frac,
                            floor_ratio=args.floor_ratio,
                            const_tail_frac=args.const_tail_frac)
        lr_emb_now = base.lr_at(step, args.steps, lr_embed,
                                decay_frac=args.decay_frac,
                                floor_ratio=args.floor_ratio,
                                const_tail_frac=args.const_tail_frac)
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
        if step % args.ckpt_every == 0:
            base.save_ckpt(os.path.join(ckpt_dir, "latest.pt"),
                           model, opts, step, cfg, args)
        if step % args.log_every == 0 or step == 1:
            out = dict(event="step", step=step, tokens=tokens_seen,
                       grad_norm=round(gn, 4), qk_max=round(qk_max, 1),
                       qk_clipped_heads=n_clip,
                       tok_per_s=round(
                           (tokens_seen - step0 * args.tokens_per_step)
                           / (time.time() - t0), 1),
                       elapsed_s=round(time.time() - t0, 1))
            for k, v in sums.items():
                out[k] = round(v / accum, 4)
            log(out)

    base.save_ckpt(os.path.join(ckpt_dir, "final.pt"),
                   model, opts, args.steps, cfg, args)
    log(dict(event="done", steps=args.steps, tokens=tokens_seen,
             ckpt=ckpt_dir))


if __name__ == "__main__":
    main()
