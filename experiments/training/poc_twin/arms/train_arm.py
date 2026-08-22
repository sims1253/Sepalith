"""
Trainer for the twin-POC follow-up arms C+D (2026-08-21, agent session).

Reuses last night's trainer (../train.py) as-is: model, PackedData, WSD
lr_at, LaunchGate, GPUWatchdog, save_ckpt, quick-eval discipline, QK-Clip,
grad clip, seeds, telemetry — all imported, none modified. Two deltas:

  1. --arm {muon, aurora}: the aurora arm swaps ONLY the orthogonalization
     on the tall MLP up/gate projections (see aurora.py); --arm muon is
     bit-identical in behavior to ../train.py (same build_optim call).
  2. checkpoints go to /tmp/poc_twin/ckpt_{TAG}/ (not ckpt_{arm}) so this
     session's runs never clobber last night's /tmp scratch checkpoints.

Arm C (aurora):  --arm aurora --lr 0.01 --lr-embed 0.004 --steps 1300
                 --tokens-per-step 524288 --compile --tag aurora
Arm D (muon_half): same but --arm muon --data .../train_blocks_half.npy
                 --tag muon_half
Paired baselines for both arms = last night's checkpoints/muon_final.pt
(2.78 epochs over the full 245.4M-token corpus, identical lr/schedule/
seed/order discipline).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))            # poc_twin/
sys.path.insert(0, HERE)                              # poc_twin/arms/
from model import TinyGQA, model_config, chunked_ce  # noqa: E402
from muon import Muon                                  # noqa: E402
from aurora import Aurora                              # noqa: E402
import train as base                                   # noqa: E402

POC_DIR = base.POC_DIR
TMP = base.TMP


def build_optim_arms(arm, model, lr, lr_embed, wd):
    hidden_named, other = [], []
    for n_, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in n_:
            hidden_named.append((n_, p))
        else:
            other.append(p)
    adam = torch.optim.AdamW(other, lr=lr_embed, betas=(0.9, 0.95),
                             eps=1e-8, weight_decay=wd, fused=True)
    if arm == "muon":
        # identical to the parent trainer's Muon arm
        opt = Muon([p for _, p in hidden_named], lr=lr, momentum=0.95,
                   ns_steps=5, weight_decay=wd)
        desc = f"Muon(hidden, lr={lr}, wd={wd}) + AdamW(embed/norms, lr={lr_embed}, wd={wd})"
    elif arm == "aurora":
        opt = Aurora(hidden_named, lr=lr, momentum=0.95, ns_steps=5,
                     weight_decay=wd, aurora_beta=0.5, aurora_K=2,
                     aurora_names=("Wg.weight", "Wu.weight"))
        desc = (f"Aurora(Wg/Wu tall, K=2, beta=0.5, lr={lr}, wd={wd}) + "
                f"Muon(other hidden, lr={lr}, wd={wd}) + AdamW(embed/norms, lr={lr_embed}, wd={wd})")
    else:
        raise ValueError(arm)
    return opt, [adam], desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["muon", "aurora"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--lr-embed", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=1300)
    ap.add_argument("--tokens-per-step", type=int, default=524288)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1273)
    ap.add_argument("--data", default=f"{TMP}/train_blocks.npy")
    ap.add_argument("--eval-data", default=f"{TMP}/eval_blocks.npy")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--tau", type=float, default=100.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--gate-min-free-mib", type=int, default=16 * 1024)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()
    if args.lr_embed is None:
        args.lr_embed = args.lr

    if not args.no_gate:
        base.LaunchGate(min_free_mib=args.gate_min_free_mib).wait()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    torch.cuda.set_per_process_memory_fraction(0.80)  # sole GPU owner now
    dev = torch.cuda.current_device()

    cfg = model_config(max_seq=args.seq)
    model = TinyGQA(cfg).cuda()
    n_all, n_emb, n_hid = (sum(p.numel() for p in model.parameters()),
                           model.embed.weight.numel(),
                           sum(p.numel() for n_, p in model.named_parameters()
                               if p.ndim == 2 and "embed" not in n_))
    opt, extra_opts, opt_desc = build_optim_arms(args.arm, model,
                                                 args.lr, args.lr_embed, args.wd)
    opts = [opt] + extra_opts

    step0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        for o, sd in zip(opts, ck["opt"]):
            o.load_state_dict(sd)
        step0 = ck["step"]
        print(f"[resume] from step {step0}", flush=True)

    fwd_trunk = model.trunk
    if args.compile:
        try:
            fwd_trunk = torch.compile(model.trunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = fwd_trunk(torch.zeros(args.micro_bs, args.seq, dtype=torch.long,
                                          device="cuda"), probe=False)
                chunked_ce(h, model.embed.weight,
                           torch.zeros(args.micro_bs, args.seq, dtype=torch.long,
                                       device="cuda")).backward()
            print("[compile] ok (trunk compiled; CE stays eager)", flush=True)
        except Exception as e:
            print(f"[compile] FAILED ({type(e).__name__}: {e}); running eager", flush=True)
            fwd_trunk = model.trunk

    data = base.PackedData(args.data, args.tokens_per_step // args.seq, args.seed)
    accum = args.tokens_per_step // (args.seq * args.micro_bs)
    tag = args.tag or (f"probe_{args.arm}_{args.lr:g}" if args.probe else args.arm)
    log_path = os.path.join(POC_DIR, "logs", f"{tag}.jsonl")
    logf = open(log_path, "a")
    ckpt_dir = os.path.join(TMP, f"ckpt_{tag}")   # DELTA 2: tag-keyed

    def log(rec):
        rec.update(arm=args.arm, tag=tag, ts=time.time())
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    print(f"[cfg] arm={args.arm} {opt_desc} steps={args.steps} "
          f"tokens/step={args.tokens_per_step} micro_bs={args.micro_bs} accum={accum} "
          f"params: total={n_all/1e6:.1f}M embed={n_emb/1e6:.1f}M hidden={n_hid/1e6:.1f}M "
          f"blocks={data.n_blocks} ({data.n_blocks*args.seq/1e6:.0f}M tokens/epoch) "
          f"data={args.data}", flush=True)

    watchdog = None
    if not args.probe:
        watchdog = base.GPUWatchdog()
        watchdog.start()

    t_start = time.time()
    tokens_seen = step0 * args.tokens_per_step
    win_loss, win_gn, win_qk, win_clip, win_t0 = [], [], -1.0, 0, time.time()
    yields = 0

    @torch.no_grad()
    def quick_eval(n_blocks=64, bs=8):
        eb = np.load(args.eval_data, mmap_mode="r")
        nats, toks = 0.0, 0
        for i in range(0, min(n_blocks, len(eb)), bs):
            x = torch.from_numpy(eb[i:i + bs].astype(np.int64)).cuda()
            h = fwd_trunk(x[:, :-1], probe=False)
            logits = F.linear(h, model.embed.weight)
            lg = logits.view(-1, logits.size(-1))
            tg = x[:, 1:].reshape(-1)
            for c in range(0, lg.size(0), 4096):
                nats += F.cross_entropy(lg[c:c + 4096].float(), tg[c:c + 4096],
                                        reduction="sum").item()
            toks += tg.numel()
        return nats / toks

    step = step0
    while step < args.steps:
        if watchdog is not None and watchdog.event.is_set():
            ck = os.path.join(ckpt_dir, "yield.pt")
            base.save_ckpt(ck, model, opts, step, cfg, args)
            torch.cuda.empty_cache()
            yields += 1
            log(dict(event="yield", step=step, gpu_mib=watchdog.last_reading,
                     note="total>29GB; yielding"))
            waited = 0
            while watchdog.event.is_set():
                time.sleep(120)
                waited += 120
                if waited % 1800 == 0:
                    log(dict(event="still_yielding", step=step, waited_s=waited,
                             gpu_mib=watchdog.last_reading))
            log(dict(event="resume_after_yield", step=step, waited_s=waited,
                     gpu_mib=watchdog.last_reading))

        if args.probe:
            lr_now = base.lr_at(step, 40, args.lr, warmup_frac=0.125, decay_frac=0.0)
            lr_emb_now = base.lr_at(step, 40, args.lr_embed, warmup_frac=0.125, decay_frac=0.0)
        else:
            lr_now = base.lr_at(step, args.steps, args.lr)
            lr_emb_now = base.lr_at(step, args.steps, args.lr_embed)
        for g in opt.param_groups:
            g["lr"] = lr_now if g is opt.param_groups[0] else lr_emb_now
        for o in extra_opts:
            for g in o.param_groups:
                g["lr"] = lr_emb_now

        xb = data.batch(step)
        micro_nats_sum = 0.0
        for o in opts:
            o.zero_grad(set_to_none=True)
        t_micro = time.time()
        for mi in range(accum):
            rows = xb[mi * args.micro_bs:(mi + 1) * args.micro_bs]
            x = torch.from_numpy(rows.astype(np.int64)).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = fwd_trunk(x[:, :-1], probe=(mi == 0))
                loss = chunked_ce(h, model.embed.weight, x[:, 1:])
            (loss / accum).backward()
            micro_nats_sum += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        for o in extra_opts:
            o.step()
        n_clip, qk_now = model.qk_clip_all(tau=args.tau, alpha=args.alpha)
        win_qk = max(win_qk, qk_now)
        win_clip += n_clip

        step += 1
        tokens_seen += args.tokens_per_step
        win_loss.append(micro_nats_sum / accum)
        win_gn.append(gn)

        if step % args.log_every == 0:
            dt = time.time() - win_t0
            rec = dict(step=step, tokens=tokens_seen,
                       loss=float(np.mean(win_loss)), lr=lr_now,
                       lr_embed=lr_emb_now, grad_norm=float(np.mean(win_gn)),
                       qk_max=win_qk, qk_clipped_heads=win_clip,
                       tok_per_s=round(args.log_every * args.tokens_per_step / dt, 1),
                       epoch=round(data.epoch_float(step), 3),
                       elapsed_s=round(time.time() - t_start, 1),
                       gpu_mib=watchdog.last_reading if watchdog else None)
            if not math.isfinite(rec["loss"]):
                rec["event"] = "LOSS-NAN — aborting at checkpoint"
                log(rec)
                base.save_ckpt(os.path.join(ckpt_dir, "nan_abort.pt"),
                               model, opts, step, cfg, args)
                sys.exit(2)
            log(rec)
            win_loss, win_gn, win_qk, win_clip, win_t0 = [], [], -1.0, 0, time.time()

        if not args.probe:
            if step % args.eval_every == 0:
                ev = quick_eval()
                log(dict(event="eval", step=step, tokens=tokens_seen,
                         eval_loss=round(ev, 5)))
            if step % args.ckpt_every == 0:
                base.save_ckpt(os.path.join(ckpt_dir, "latest.pt"),
                               model, opts, step, cfg, args)
                if abs(step - args.steps // 2) < args.ckpt_every // 2:
                    base.save_ckpt(os.path.join(ckpt_dir, "mid.pt"),
                                   model, opts, step, cfg, args)

    if args.probe:
        ev = quick_eval(n_blocks=32)
        train_s = time.time() - t_start
        log(dict(event="probe_final", lr=args.lr, arm=args.arm,
                 last_loss=float(np.mean(win_loss[-5:])) if win_loss else None,
                 eval_loss=round(ev, 5),
                 tok_per_s=round(tokens_seen / max(1e-9, train_s), 1),
                 train_s=round(train_s, 1)))
        return

    base.save_ckpt(os.path.join(ckpt_dir, "final.pt"), model, opts, step, cfg, args)
    log(dict(event="done", step=step, tokens=tokens_seen,
             total_s=round(time.time() - t_start, 1), yields=yields))
    if watchdog:
        watchdog.stop.set()


if __name__ == "__main__":
    main()
