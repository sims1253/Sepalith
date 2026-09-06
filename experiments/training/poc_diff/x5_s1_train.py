"""X5-S1 trainer: FRM self-conditioning (Stage A) + Fixed-Point Forcing
(Stage B) continuation on the MD span head.

Queue docs/EXPERIMENT-QUEUE.md §3 X5 S1 row (pre-registration verbatim):
continuation from banked md_final.pt with a zero-init carry channel
ingesting prev-pass RAW span logits; Stage A = two-pass self-conditioning;
Stage B = FPF — carry = stopgrad rollout prediction from t_start with
t_start ~ Beta(2,2) capped at supervision t, objective otherwise
unchanged. FRM v2 (arXiv 2606.29150) App-A details fetched 2026-09-06:
null carry = zeros, pass 1 under stopgrad, two-pass prob 0.5, FPF rollout
prob 0.5, supervised input stays TEACHER-FORCED (the rollout only
produces the carry). NOTE (flagged): the paper's code draws t_start ~
U(0,t); the queue row pre-registers Beta(2,2)-capped — the row wins here
(support is [t,1] in our mask-rate mirror either way).

PRE-REGISTERED HEALTH CHECKS (B8 gates pattern; unit-tested CPU-side in
test_x5_s1.py before any GPU run):
  G1 carry-channel zero-init verified at load: the only missing key vs a
     banked ckpt is carry_proj.weight and it stays all-zero; the trunk's
     carry=None path never touches it (banked behavior byte-identical).
  G2 null-carry safety: a backward on a zero/null carry leaves
     carry_proj.weight.grad exactly zero — null-mode training can never
     move the channel.
  G3 two-pass shapes: logits_to_carry returns (B,T,d) fp32, zero rows
     outside span_pos; stage-A carry rows are zeroed for null examples.
  G4 nested masks: m subset-of m_start (same noise draw u); m_start_rate
     = max(Beta(2,2), t) per example.
  G5 rollout: the FPF rollout commits only m_start\\m positions (greedy),
     never touches context or supervision-masked positions; its carry
     carries no grad.
  G6 loss estimator unchanged: stage-A/B losses equal objective.mdlm_loss
     on the pass-2 hidden states (objective otherwise unchanged).

Stage/step budget (pre-registered, calibrated to the ~4-6h GPU window):
  Stage A ~400 steps x 524,288 tok, lr 0.003 (0.3x anchor peak) WSD,
  embed 1.2e-3; Stage B ~260 steps, lr 0.0015, embed 6e-4. Fresh Muon
  optimizer per stage (their protocol: fresh optimizer at stage bound-
ary). Cost model: A ~1.33x anchor step (2 fwd + bwd), B ~1.67x (2 rollout
fwd + supervised fwd + bwd).

Usage:
  A: uv run python -m experiments.training.poc_diff.x5_s1_train \
        --stage A --resume /mnt/h/sepalith/runs/poc_diff/md_final.pt \
        --steps 400 --compile
  B: uv run python -m experiments.training.poc_diff.x5_s1_train \
        --stage B --resume /mnt/h/sepalith/runs/x5_s1/x5_s1_a_final.pt \
        --steps 260 --compile
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from experiments.training.poc_twin import train as base
from experiments.training.poc_diff import TMP
from experiments.training.poc_diff.model_md import MDGQA, model_config_md, \
    logits_to_carry
from experiments.training.poc_diff import objective
from experiments.training.poc_diff.sample import _chunked_probs
from experiments.training.poc_diff.train_md import (
    POC_DIFF_DIR, TripleData, build_micro_tensors, micro_batches, rsync_out,
)

P_SC = 0.5     # two-pass probability (their Table 3, all datasets)
P_FPF = 0.5    # FPF rollout probability (their Table 3)


def beta22(n, generator=None, device="cpu"):
    """Beta(2,2) draws via Gamma(2,1) = -log u1 - log u2 (deterministic
    under the seeded generator; torch.distributions has no generator arg)."""
    u1 = torch.rand(n, generator=generator, device=device).clamp_min(1e-12)
    u2 = torch.rand(n, generator=generator, device=device).clamp_min(1e-12)
    g1 = -torch.log(u1) - torch.log(u2)
    u1 = torch.rand(n, generator=generator, device=device).clamp_min(1e-12)
    u2 = torch.rand(n, generator=generator, device=device).clamp_min(1e-12)
    g2 = -torch.log(u1) - torch.log(u2)
    return g1 / (g1 + g2)


def nested_masks(span_pos, t, m_start_rate, u):
    """Supervision mask m and the noisier rollout start m_start from ONE
    shared noise draw u (their 'same noise eps' construction): m = span &
    u < t (identical in law to objective.bernoulli_mask), m_start = span &
    u < m_start_rate with m_start_rate >= t, hence m is a subset of
    m_start."""
    m = span_pos & (u < t[:, None])
    m_start = span_pos & (u < m_start_rate[:, None])
    return m, m_start


@torch.no_grad()
def stage_a_carry(model, fwd_trunk, x_in, span_pos, valid, sc_mask,
                  autocast_dtype=torch.bfloat16, chunk=1024):
    """Two-pass self-conditioning carry (their App-A): pass 1 on x_in
    under stopgrad, carry = its raw span logits as the soft clean-
    prediction embedding; rows zeroed for null-carry examples (sc_mask
    (B,) bool)."""
    attn = valid[:, None, None, :]
    with torch.autocast("cuda", dtype=autocast_dtype,
                        enabled=x_in.is_cuda):
        h1 = fwd_trunk(x_in, probe=False, attn_mask=attn)
    c = logits_to_carry(h1, span_pos, model.embed.weight, chunk=chunk)
    return c * sc_mask[:, None, None].float()


@torch.no_grad()
def fpf_rollout(model, fwd_trunk, x, m, m_start, span_pos, valid, depth=2,
                autocast_dtype=torch.bfloat16, chunk=1024):
    """Fixed-Point Forcing carry (their §3.2 re-derived for the remasking
    sampler): roll the model's OWN inference dynamics from the noisier
    m_start state down to the supervision mask m (committing greedy picks
    at m_start\\m positions, split evenly across the depth-1 intermediate
    forwards — the discrete integration toward t), feeding the carry back
    recurrently; return the FINAL forward's span carry (stopgrad)."""
    attn = valid[:, None, None, :]
    x_roll = x.masked_fill(m_start, model.mask_id)
    to_commit = m_start & ~m
    carry = None
    for j in range(depth):
        with torch.autocast("cuda", dtype=autocast_dtype,
                            enabled=x.is_cuda):
            h = fwd_trunk(x_roll, probe=False, attn_mask=attn, carry=carry)
        carry = logits_to_carry(h, span_pos, model.embed.weight, chunk=chunk)
        remaining = depth - 1 - j
        if j < depth - 1 and to_commit.any():
            bidx, pidx = to_commit.nonzero(as_tuple=True)
            probs = _chunked_probs(h[bidx, pidx], model.embed.weight,
                                   chunk=chunk, temperature=0.0)
            pconf, picks = probs.max(dim=-1)
            quota = -(-int(to_commit.sum().item()) // max(1, remaining))
            order = pconf.argsort(descending=True)[:quota]
            x_roll[bidx[order], pidx[order]] = picks[order]
            to_commit[bidx[order], pidx[order]] = False
    return carry


def stage_loss(model, fwd_trunk, x, span_pos, valid, t, u, active_mask,
               stage, fpf_depth=2, chunk=1024, generator=None, probe=False):
    """One training micro's loss (G6: objective otherwise unchanged).

    stage 'A': carry = two-pass self-conditioning (active_mask = sc draw).
    stage 'B': carry = FPF rollout prediction (active_mask = fpf draw);
               the supervised input stays the canonical teacher-forced
               x_in exactly as in their formulation. `probe` rides the
               supervised (grad) pass only — the anchor's QK-Clip
               telemetry discipline."""
    attn = valid[:, None, None, :]
    m = span_pos & (u < t[:, None])
    x_in = x.masked_fill(m, model.mask_id)
    if stage == "A":
        carry = stage_a_carry(model, fwd_trunk, x_in, span_pos, valid,
                              active_mask, chunk=chunk)
    else:
        m_start_rate = torch.maximum(
            beta22(x.size(0), generator=generator, device=x.device), t)
        m_start = span_pos & (u < m_start_rate[:, None])
        carry = fpf_rollout(model, fwd_trunk, x, m, m_start, span_pos,
                            valid, depth=fpf_depth, chunk=chunk)
        carry = carry * active_mask[:, None, None].float()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        h2 = fwd_trunk(x_in, probe=probe, attn_mask=attn, carry=carry)
        loss = objective.mdlm_loss(h2, x, m, t, span_pos.sum(dim=1),
                                   model.embed.weight)
    stats = dict(t_mean=float(t.mean()),
                 mask_rate=float(m.sum() / span_pos.sum().clamp(min=1)),
                 active_frac=float(active_mask.float().mean()))
    return loss, stats


def load_resume(model, path):
    """G1: resume with the carry-channel load gate. A PRE-X5 ckpt (no
    carry key) must leave the zero-init channel untouched; an S1 ckpt
    carries its trained channel and loads it."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert set(missing) <= {"carry_proj.weight"}, f"missing: {missing}"
    if "carry_proj.weight" in missing:
        assert torch.count_nonzero(model.carry_proj.weight) == 0, \
            "G1 violated: carry_proj nonzero after a pre-X5 resume"
    return ck.get("step", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--resume", required=True,
                    help="input ckpt: md_final.pt (A) or stage-A final (B)")
    ap.add_argument("--steps", type=int, default=None,
                    help="default 400 (A) / 260 (B)")
    ap.add_argument("--lr", type=float, default=None,
                    help="default 3e-3 (A) / 1.5e-3 (B)")
    ap.add_argument("--lr-embed", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--tokens-per-step", type=int, default=524288)
    ap.add_argument("--seed", type=int, default=1273)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--tau", type=float, default=100.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--fpf-depth", type=int, default=2)
    ap.add_argument("--p-active", type=float, default=None,
                    help="carry-active probability (default 0.5, their "
                         "Table 3 for both stages)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--gate-min-free-mib", type=int, default=16 * 1024)
    ap.add_argument("--out", default="/mnt/h/sepalith/runs/x5_s1")
    args = ap.parse_args()
    if args.steps is None:
        args.steps = 400 if args.stage == "A" else 260
    if args.lr is None:
        args.lr = 3e-3 if args.stage == "A" else 1.5e-3
    if args.lr_embed is None:
        args.lr_embed = 1.2e-3 if args.stage == "A" else 6e-4
    if args.p_active is None:
        args.p_active = P_SC if args.stage == "A" else P_FPF
    if args.smoke:
        args.steps = min(args.steps, 12)
        args.log_every = min(args.log_every, 2)
        args.eval_every = min(args.eval_every, 6)
        args.ckpt_every = min(args.ckpt_every, 6)

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
    step0 = load_resume(model, args.resume)
    print(f"[resume] {args.resume} (banked step {step0}); stage {args.stage}",
          flush=True)

    opt, extra_opts, _ = base.build_optim(
        "muon", model, args.lr, args.lr_embed, args.wd)
    opts = [opt] + extra_opts
    cuda_gen = torch.Generator(device=device).manual_seed(args.seed)

    data = TripleData(tmp=TMP, seed=args.seed)

    fwd_trunk = model.trunk
    if args.compile:
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = 64
        try:
            fwd_trunk = torch.compile(model.trunk)
            x0 = torch.zeros(4, 512, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fwd_trunk(x0, probe=False)
                c0 = torch.zeros(4, 512, cfg["d_model"], device=device)
                fwd_trunk(x0, probe=False, carry=c0)
            print("[compile] ok (null + carry paths)", flush=True)
        except Exception as e:
            print(f"[compile] FAILED ({type(e).__name__}: {e}); eager",
                  flush=True)
            fwd_trunk = model.trunk

    ckpt_dir = os.path.join(TMP, "ckpt_x5_s1")  # own staging (not ckpt/)
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"x5_s1_{args.stage.lower()}"

    def log(rec):
        rec.update(tag=tag, ts=time.time())
        print(json.dumps(rec), flush=True)
        with open(os.path.join(POC_DIFF_DIR, f"logs_{tag}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    n_ckpt = {"n": 0}

    def save(name, step):
        path = os.path.join(ckpt_dir, name)
        base.save_ckpt(path, model, opts, step, cfg, args)
        n_ckpt["n"] += 1
        if n_ckpt["n"] % 2 == 0 or "final" in name:
            rsync_out(ckpt_dir, args.out)

    @torch.no_grad()
    def quick_eval(n_rows=128):
        """Held-out null-carry MDLM loss (anchor-comparable telemetry)."""
        from experiments.training.poc_diff import span_region
        rows = [json.loads(l) for l in
                open(os.path.join(TMP, "eval_triples.jsonl"))][:n_rows]
        losses = []
        for i in range(0, len(rows), 16):
            batch = rows[i:i + 16]
            prompts = [r["prompt_ids"] for r in batch]
            regions = [span_region(r["span_ids"]) for r in batch]
            pad = max(len(p) + len(s) for p, s in zip(prompts, regions))
            x = torch.full((len(batch), pad), 1, dtype=torch.long,
                           device=device)
            sp = torch.zeros(len(batch), pad, dtype=torch.bool, device=device)
            vl = torch.zeros(len(batch), pad, dtype=torch.bool, device=device)
            for j, (p, s) in enumerate(zip(prompts, regions)):
                ids = p + s
                x[j, :len(ids)] = torch.tensor(ids, device=device)
                sp[j, len(p):len(ids)] = True
                vl[j, :len(ids)] = True
            t = torch.rand(len(batch), generator=cuda_gen, device=device)
            m = objective.bernoulli_mask(sp, t, generator=cuda_gen)
            x_in = x.masked_fill(m, model.mask_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = fwd_trunk(x_in, probe=False, attn_mask=vl[:, None, None, :])
                loss = objective.mdlm_loss(h, x, m, t, sp.sum(1),
                                           model.embed.weight)
            losses.append(float(loss))
        return float(np.mean(losses))

    n_all = sum(p.numel() for p in model.parameters())
    print(f"[cfg] MDGQA+carry {n_all/1e6:.1f}M; stage {args.stage} "
          f"steps={args.steps} x {args.tokens_per_step} tok = "
          f"{args.steps*args.tokens_per_step/1e9:.2f}B; lr={args.lr} "
          f"p_active={args.p_active} fpf_depth={args.fpf_depth}", flush=True)

    watchdog = None
    if not args.smoke:
        watchdog = base.GPUWatchdog()
        watchdog.start()

    t_start = time.time()
    tokens_seen = 0
    win = {"loss": [], "gn": [], "t": [], "mask": [], "act": [], "cn": 0.0,
           "qk": -1.0, "clip": 0, "t0": time.time()}

    step = 0  # stage-local steps (fresh WSD per stage)
    while step < args.steps:
        if watchdog is not None and watchdog.event.is_set():
            save(f"{tag}_yield.pt", step)
            torch.cuda.empty_cache()
            log(dict(event="yield", step=step,
                     gpu_mib=watchdog.last_reading))
            while watchdog.event.is_set():
                time.sleep(120)
            log(dict(event="resume_after_yield", step=step))

        lr_now = base.lr_at(step, args.steps, args.lr)
        lr_emb_now = base.lr_at(step, args.steps, args.lr_embed)
        for g in opt.param_groups:
            g["lr"] = lr_now if g is opt.param_groups[0] else lr_emb_now
        for o in extra_opts:
            for g in o.param_groups:
                g["lr"] = lr_emb_now

        rows, acc = data.rows_for_step(step, args.tokens_per_step)
        micros = micro_batches(rows, data)
        for o in opts:
            o.zero_grad(set_to_none=True)
        step_nats, step_stats = 0.0, None
        for mi, micro in enumerate(micros):
            x, span_pos, valid = build_micro_tensors(micro, data, device)
            B = x.size(0)
            t = objective.sample_rates(B, generator=cuda_gen, device=device)
            u = torch.rand(span_pos.shape, generator=cuda_gen, device=device)
            active = torch.rand(B, generator=cuda_gen, device=device) \
                < args.p_active
            loss, st = stage_loss(model, fwd_trunk, x, span_pos, valid, t,
                                  u, active, args.stage,
                                  fpf_depth=args.fpf_depth,
                                  generator=cuda_gen, probe=(mi == 0))
            (loss / len(micros)).backward()
            step_nats += float(loss)
            step_stats = st
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        for o in extra_opts:
            o.step()
        n_clip, qk_now = model.qk_clip_all(tau=args.tau, alpha=args.alpha)

        step += 1
        tokens_seen += acc
        win["loss"].append(step_nats / len(micros))
        win["gn"].append(gn)
        if step_stats:
            win["t"].append(step_stats["t_mean"])
            win["mask"].append(step_stats["mask_rate"])
            win["act"].append(step_stats["active_frac"])
        win["qk"] = max(win["qk"], qk_now)
        win["clip"] += n_clip
        win["cn"] = max(win["cn"], float(model.carry_proj.weight.abs().max()))

        if step % args.log_every == 0:
            dt = time.time() - win["t0"]
            rec = dict(step=step, tokens=tokens_seen,
                       loss=float(np.mean(win["loss"])), lr=lr_now,
                       grad_norm=float(np.mean(win["gn"])),
                       t_mean=float(np.mean(win["t"])),
                       mask_rate=float(np.mean(win["mask"])),
                       active_frac=float(np.mean(win["act"])),
                       carry_w_absmax=win["cn"],
                       qk_max=win["qk"], qk_clipped_heads=win["clip"],
                       tok_per_s=round(args.log_every * args.tokens_per_step
                                       / dt, 1),
                       gpu_mib=watchdog.last_reading if watchdog else None)
            if not np.isfinite(rec["loss"]):
                rec["event"] = "LOSS-NAN — aborting"
                log(rec)
                save(f"{tag}_nan_abort.pt", step)
                sys.exit(2)
            log(rec)
            win = {"loss": [], "gn": [], "t": [], "mask": [], "act": [],
                   "cn": 0.0, "qk": -1.0, "clip": 0, "t0": time.time()}

        if step % args.eval_every == 0:
            log(dict(event="eval", step=step,
                     eval_loss=round(quick_eval(), 5)))

        if step % args.ckpt_every == 0:
            save(f"{tag}_latest.pt", step)

    save(f"{tag}_final.pt", step)
    log(dict(event="done", stage=args.stage, step=step, tokens=tokens_seen,
             carry_w_absmax=win.get("cn", 0.0),
             total_s=round(time.time() - t_start, 1)))
    if watchdog:
        watchdog.stop.set()


if __name__ == "__main__":
    main()
