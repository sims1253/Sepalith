#!/usr/bin/env python3
"""Task 4: trainer for the OT-coupled twin — POC-DDOT.

Mirrors poc_diff's train_md.py discipline EXACTLY (identical data order,
optimizer recipe, schedule, seed — the paired-arm rule; only the model
adds a position head and the loss is the OT-coupled objective):
  - model: MDGQA + pos_head (nn.Linear d->1, predicts the clean slot
    coordinate at span positions; side AdamW at the embed lr — a 1-row
    matrix is a vector, not a Muon target);
  - data: poc_diff's flat ids bin + THIS package's train_slots.bin
    (float16 slot stream aligned 1:1 with the flat bin's span regions;
    SlotsData reads the same row_lens);
  - loss: ot_step -> objective_ot.ot_mdlm_loss (per-example Sinkhorn on
    the span window, value CE soft-routed through the plan, positions
    noised p~ = p + t*eps and denoised by the head);
  - telemetry beyond train_md's: ot_plan_entropy (the collapse-to-
    identity watch — if the plan is always ~identity, the coupling is
    inert and that IS the finding), sinkhorn iters, row-mass min/max,
    pos_share = position/(value+1e-8). The plan's pre-registered single
    lambda adjustment (1.0 -> 0.3 if the position term dominates the
    value term) is a LOGGED trigger, not an automatic one — apply it as
    a conscious act with a board note if the telemetry says so.
  - checkpoints: {cfg, model, pos_head, opt, step, args} staged to
    /tmp/poc_ddot/ckpt then rsync to /mnt/h/sepalith/runs/poc_ddot/
    (eval_ot.load_ot reads this format).

Usage:
  smoke: uv run python -m experiments.training.poc_ddot.train_ot --steps 40 --smoke
  full:  uv run python -m experiments.training.poc_ddot.train_ot --steps 3815 --compile
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from experiments.training.poc_twin import train as base
from experiments.training.poc_diff import TMP as DIFF_TMP
from experiments.training.poc_diff import objective, span_region
from experiments.training.poc_diff.model_md import MDGQA, model_config_md
from experiments.training.poc_diff.train_md import (
    TripleData, micro_batches, rsync_out)
try:
    from .objective_ot import noise_positions, ot_mdlm_loss
    from .eval_ot import PosModel  # noqa: F401  (re-exported for tests/loading)
except ImportError:                          # script/test path (dir on sys.path)
    from objective_ot import noise_positions, ot_mdlm_loss
    from eval_ot import PosModel  # noqa: F401

POC_DDOT_TMP = "/tmp/poc_ddot"
POC_DDOT_DIR = os.path.dirname(os.path.abspath(__file__))


class SlotsData:
    """float16 slot stream aligned with poc_diff's flat-ids span regions."""

    def __init__(self, tmp=POC_DDOT_TMP, lens_path=None):
        self.slots = np.memmap(os.path.join(tmp, "train_slots.bin"),
                               dtype=np.float16, mode="r")
        lens = np.frombuffer(
            open(lens_path or os.path.join(DIFF_TMP, "train_row_lens.bin"),
                 "rb").read(), dtype=np.int32).reshape(-1, 2)
        self.region_len = lens[:, 1]
        self.offsets = np.concatenate([[0], np.cumsum(self.region_len)])
        self.n_rows = len(lens)

    def slots_for(self, row):
        return self.slots[self.offsets[row]:
                          self.offsets[row] + self.region_len[row]]

    def extend_micro(self, micro_rows, pad_len, span_starts):
        """(B, pad_len) float32 slot tensor, garbage-0 outside spans."""
        out = torch.zeros(len(micro_rows), pad_len, dtype=torch.float32)
        for i, (r, p) in enumerate(zip(micro_rows, span_starts)):
            s = torch.from_numpy(np.asarray(self.slots_for(r),
                                            dtype=np.float32))
            out[i, p:p + s.numel()] = s
        return out


def ot_step(model, pos_head, x, span_pos, valid, slots, generator=None,
            eps=0.05, kappa=None, lam=1.0, t_override=None, n_iters=50):
    """One forward + joint OT loss on a micro-batch (testable on CPU).

    Returns dict(total, value, position, telemetry, t, mask). Autocast is
    the caller's context (the trainer wraps this in bf16 autocast)."""
    B = x.size(0)
    device = x.device
    t = t_override if t_override is not None else objective.sample_rates(
        B, generator=generator, device=device)
    m = objective.bernoulli_mask(span_pos, t, generator=generator)
    x_in = x.masked_fill(m, model.mask_id)
    h = model.trunk(x_in, probe=False, attn_mask=valid[:, None, None, :])
    pos_pred = pos_head(h).squeeze(-1)
    noised, sigma = noise_positions(slots, m, t, generator=generator)
    out = ot_mdlm_loss(h, x, m, t, span_pos.sum(1), model.embed.weight,
                       slots_noised=noised, slots_true=slots,
                       span_pos=span_pos, eps=eps, kappa=kappa,
                       lam=lam, pos_pred=pos_pred, sigma=sigma,
                       n_iters=n_iters)
    return dict(total=out.total, value=out.value, position=out.position,
                telemetry=out.telemetry, t=t, mask=m)


def build_micro_tensors_ot(micro, data, slots_data, device):
    """train_md.build_micro_tensors + the slots field."""
    rows, pad_len = micro
    n = len(rows)
    x = torch.full((n, pad_len), 1, dtype=torch.long)
    span_pos = torch.zeros(n, pad_len, dtype=torch.bool)
    valid = torch.zeros(n, pad_len, dtype=torch.bool)
    span_starts = []
    for i, r in enumerate(rows):
        p, s = int(data.prompt_len[r]), int(data.region_len[r])
        ids = torch.from_numpy(
            np.asarray(data.ids[data.offsets[r]:data.offsets[r] + p + s],
                       dtype=np.int64))
        x[i, :p + s] = ids
        span_pos[i, p:p + s] = True
        valid[i, :p + s] = True
        span_starts.append(p)
    slots = slots_data.extend_micro(rows, pad_len, span_starts)
    return (x.to(device, non_blocking=True), span_pos.to(device),
            valid.to(device), slots.to(device, non_blocking=True))


def save_ot(path, model, pos_head, opts, step, cfg, args):
    torch.save(dict(cfg=cfg, model=model.state_dict(),
                    pos_head=pos_head.state_dict(),
                    opt=[o.state_dict() for o in opts],
                    step=step, args=vars(args)), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-embed", type=float, default=4e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3815)
    ap.add_argument("--tokens-per-step", type=int, default=524288)
    ap.add_argument("--seed", type=int, default=1273)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--tau", type=float, default=100.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--gate-min-free-mib", type=int, default=16 * 1024)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--out", default="/mnt/h/sepalith/runs/poc_ddot")
    args = ap.parse_args()
    if args.smoke:
        args.steps = min(args.steps, 40)

    if not args.no_gate:
        base.LaunchGate(min_free_mib=args.gate_min_free_mib).wait()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    frac = float(os.environ.get("POC_MEM_FRACTION", "0.42"))
    torch.cuda.set_per_process_memory_fraction(frac)
    device = torch.device("cuda")

    cfg = model_config_md()
    model = MDGQA(cfg).to(device)
    pos_head = torch.nn.Linear(cfg["d_model"], 1).to(device)
    opt, extra_opts, _ = base.build_optim(
        "muon", model, args.lr, args.lr_embed, args.wd)
    opt_pos = torch.optim.AdamW(pos_head.parameters(), lr=args.lr_embed,
                                betas=(0.9, 0.95), weight_decay=args.wd)
    opts = [opt] + extra_opts + [opt_pos]
    cuda_gen = torch.Generator(device=device).manual_seed(args.seed)

    data = TripleData(seed=args.seed)
    slots_data = SlotsData()

    step0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        pos_head.load_state_dict(
            {k: v.float() for k, v in ck["pos_head"].items()})
        for o, sd in zip(opts, ck["opt"]):
            o.load_state_dict(sd)
        step0 = ck["step"]
        print(f"[resume] from step {step0}", flush=True)

    fwd_trunk = model.trunk
    if args.compile:
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = 64
        try:
            fwd_trunk = torch.compile(model.trunk)
            x0 = torch.zeros(8, 512, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fwd_trunk(x0, probe=False)
            print("[compile] ok (bucketed shapes compile lazily)", flush=True)
        except Exception as e:
            print(f"[compile] FAILED ({type(e).__name__}: {e}); eager",
                  flush=True)
            fwd_trunk = model.trunk

    ckpt_dir = os.path.join(POC_DDOT_TMP, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    def log(rec):
        rec.update(tag="ot", ts=time.time())
        print(json.dumps(rec), flush=True)
        with open(os.path.join(POC_DDOT_DIR, "logs_ot.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    n_ckpt = {"n": 0}

    def save(path, step):
        save_ot(path, model, pos_head, opts, step, cfg, args)
        n_ckpt["n"] += 1
        if n_ckpt["n"] % 2 == 0 or "final" in os.path.basename(path):
            rsync_out(ckpt_dir, args.out)

    @torch.no_grad()
    def quick_eval(n_rows=64):
        """Held-out OT loss on eval triples with inline slots (seeded t)."""
        rows = [json.loads(l) for l in
                open(os.path.join(POC_DDOT_TMP, "eval_triples_pos.jsonl"))]
        rows = rows[:n_rows]
        losses = []
        for i in range(0, len(rows), 8):
            batch = rows[i:i + 8]
            pad = max(len(r["prompt_ids"]) + len(r["slots"]) for r in batch)
            x = torch.full((len(batch), pad), 1, dtype=torch.long,
                           device=device)
            sp = torch.zeros(len(batch), pad, dtype=torch.bool, device=device)
            vl = torch.ones(len(batch), pad, dtype=torch.bool, device=device)
            sl = torch.zeros(len(batch), pad, device=device)
            for j, r in enumerate(batch):
                ids = r["prompt_ids"] + span_region(r["span_ids"])
                p = len(r["prompt_ids"])
                x[j, :len(ids)] = torch.tensor(ids, device=device)
                sp[j, p:len(ids)] = True
                sl[j, p:len(ids)] = torch.tensor(r["slots"], device=device)
            t = torch.rand(len(batch), generator=cuda_gen, device=device)
            m = objective.bernoulli_mask(sp, t, generator=cuda_gen)
            x_in = x.masked_fill(m, model.mask_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = fwd_trunk(x_in, probe=False,
                              attn_mask=vl[:, None, None, :])
                pp = pos_head(h).squeeze(-1)
                noised, sigma = noise_positions(sl, m, t)
                out = ot_mdlm_loss(h, x, m, t, sp.sum(1), model.embed.weight,
                                   slots_noised=noised, slots_true=sl,
                                   span_pos=sp, eps=args.eps,
                                   kappa=args.kappa, pos_pred=pp,
                                   sigma=sigma)
            losses.append(float(out.total))
        return float(np.mean(losses))

    print(f"[cfg] MDGQA+pos_head; steps={args.steps} x "
          f"{args.tokens_per_step} tok = "
          f"{args.steps*args.tokens_per_step/1e9:.2f}B; eps={args.eps} "
          f"kappa={args.kappa} lam={args.lam}", flush=True)

    watchdog = None
    if not args.smoke:
        watchdog = base.GPUWatchdog()
        watchdog.start()

    t_start = time.time()
    tokens_seen = step0 * args.tokens_per_step
    win = {"loss": [], "value": [], "pos": [], "ent": [], "iters": [],
           "pos_share": [], "gn": [], "qk": -1.0, "clip": 0, "t0": time.time()}

    step = step0
    while step < args.steps:
        if watchdog is not None and watchdog.event.is_set():
            save(os.path.join(ckpt_dir, "yield.pt"), step)
            torch.cuda.empty_cache()
            log(dict(event="yield", step=step, gpu_mib=watchdog.last_reading))
            while watchdog.event.is_set():
                time.sleep(120)
            log(dict(event="resume_after_yield", step=step))

        lr_now = base.lr_at(step, args.steps, args.lr)
        lr_emb_now = base.lr_at(step, args.steps, args.lr_embed)
        for g in opt.param_groups:
            g["lr"] = lr_now if g is opt.param_groups[0] else lr_emb_now
        for o in extra_opts + [opt_pos]:
            for g in o.param_groups:
                g["lr"] = lr_emb_now

        rows, acc = data.rows_for_step(step, args.tokens_per_step)
        micros = micro_batches(rows, data)
        for o in opts:
            o.zero_grad(set_to_none=True)
        step_loss, step_val, step_pos = 0.0, 0.0, 0.0
        ent, iters, pos_share = [], [], []
        for mi, micro in enumerate(micros):
            x, span_pos, valid, slots = build_micro_tensors_ot(
                micro, data, slots_data, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = ot_step(model, pos_head, x, span_pos, valid, slots,
                              generator=cuda_gen, eps=args.eps,
                              kappa=args.kappa, lam=args.lam)
            (out["total"] / len(micros)).backward()
            step_loss += float(out["total"])
            step_val += float(out["value"])
            step_pos += float(out["position"])
            tel = out["telemetry"]
            ent.append(tel["plan_entropy"])
            iters.append(tel["iters_mean"])
            pos_share.append(float(out["position"]) /
                             (float(out["value"]) + 1e-8))
        gn = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(pos_head.parameters()), 1.0).item()
        opt.step()
        for o in extra_opts + [opt_pos]:
            o.step()
        n_clip, qk_now = model.qk_clip_all(tau=args.tau, alpha=args.alpha)

        step += 1
        tokens_seen += acc
        win["loss"].append(step_loss / len(micros))
        win["value"].append(step_val / len(micros))
        win["pos"].append(step_pos / len(micros))
        win["ent"] += ent
        win["iters"] += iters
        win["pos_share"] += pos_share
        win["gn"].append(gn)
        win["qk"] = max(win["qk"], qk_now)
        win["clip"] += n_clip

        if step % args.log_every == 0:
            dt = time.time() - win["t0"]
            rec = dict(step=step, tokens=tokens_seen,
                       loss=float(np.mean(win["loss"])),
                       value=float(np.mean(win["value"])),
                       position=float(np.mean(win["pos"])),
                       ot_plan_entropy=float(np.mean(win["ent"])),
                       sinkhorn_iters=float(np.mean(win["iters"])),
                       pos_share=float(np.mean(win["pos_share"])),
                       grad_norm=float(np.mean(win["gn"])),
                       qk_max=win["qk"], qk_clipped_heads=win["clip"],
                       tok_per_s=round(args.log_every *
                                       args.tokens_per_step / dt, 1),
                       epoch=round(data.epoch_float(step,
                                                    args.tokens_per_step), 2),
                       gpu_mib=watchdog.last_reading if watchdog else None)
            if not np.isfinite(rec["loss"]):
                rec["event"] = "LOSS-NAN — aborting"
                log(rec)
                save(os.path.join(ckpt_dir, "nan_abort.pt"), step)
                raise SystemExit(2)
            log(rec)
            win = {"loss": [], "value": [], "pos": [], "ent": [],
                   "iters": [], "pos_share": [], "gn": [], "qk": -1.0,
                   "clip": 0, "t0": time.time()}

        if step % args.eval_every == 0:
            log(dict(event="eval", step=step,
                     eval_loss=round(quick_eval(), 5)))

        if step % args.ckpt_every == 0:
            save(os.path.join(ckpt_dir, "latest.pt"), step)

    save(os.path.join(ckpt_dir, "ot_final.pt"), step)
    log(dict(event="done", step=step, tokens=tokens_seen,
             total_s=round(time.time() - t_start, 1)))
    if watchdog:
        watchdog.stop.set()


if __name__ == "__main__":
    main()
