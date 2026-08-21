"""
Twin POC trainer: Muon(+QK-Clip) vs AdamW on identical data order/schedule.

GPU discipline (junior job, shares the 5090 with the senior RL trial):
  - torch.cuda.set_per_process_memory_fraction(0.42)  (~13.7GB hard cap)
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set by launcher)
  - background watchdog polls nvidia-smi every 10 min; if TOTAL used memory
    > ~29GB (RL job stressed), we checkpoint, empty_cache, and yield for
    30+ min before resuming. We never touch the RL job's processes.
Artifacts: code/configs/logs in the repo dir; run checkpoints in /tmp/poc_twin
(NAS/drvfs avoided during training).
"""
import argparse, json, math, os, subprocess, sys, threading, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyGQA, model_config  # noqa: E402
from muon import Muon  # noqa: E402

POC_DIR = "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin"
TMP = "/tmp/poc_twin"


class LaunchGate:
    """Coordinator mandate (RL smoke holds ~31GB): BEFORE creating any CUDA
    context, require >=16GB free. Poll every 10 min, indefinitely; checks
    are logged. A failed pre-allocation could destabilize the RL run, so we
    simply do not touch the driver until the gate opens."""

    def __init__(self, min_free_mib=16 * 1024, interval_s=600):
        self.min_free_mib = min_free_mib
        self.interval_s = interval_s

    @staticmethod
    def free_mib():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
            return int(out[0])
        except Exception:
            return 0

    def wait(self):
        n = 0
        while True:
            free = self.free_mib()
            if free >= self.min_free_mib:
                if n:
                    print(f"[gate] OPEN: {free}MiB free after {n} checks", flush=True)
                else:
                    print(f"[gate] OPEN: {free}MiB free", flush=True)
                return
            if n % 6 == 0:
                print(f"[gate] CLOSED: {free}MiB free < {self.min_free_mib}MiB; "
                      f"sleeping {self.interval_s}s (check {n})", flush=True)
            time.sleep(self.interval_s)
            n += 1


class GPUWatchdog(threading.Thread):
    """Junior-job yield discipline: poll total GPU memory; flag if the
    machine is over the shared budget (RL job = senior)."""

    def __init__(self, threshold_mib=29 * 1024, clear_mib=28 * 1024, interval_s=600):
        super().__init__(daemon=True)
        self.threshold_mib = threshold_mib
        self.clear_mib = clear_mib
        self.interval_s = interval_s
        self.event = threading.Event()
        self.stop = threading.Event()
        self.last_reading = None

    def _used_mib(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
            return int(out[0])
        except Exception:
            return None

    def run(self):
        while not self.stop.is_set():
            used = self._used_mib()
            self.last_reading = used
            if used is not None:
                if used > self.threshold_mib and not self.event.is_set():
                    self.event.set()
                elif used < self.clear_mib and self.event.is_set():
                    self.event.clear()
            self.stop.wait(self.interval_s)


class PackedData:
    """Deterministic block order from (seed, epoch): both arms see the
    identical token sequence for a given step."""

    def __init__(self, path, seq_per_step, seed):
        self.blocks = np.load(path, mmap_mode="r")
        self.n_blocks = len(self.blocks)
        self.seq_per_step = seq_per_step
        self.seed = seed
        self._order_cache = {}

    def _order(self, epoch):
        if epoch not in self._order_cache:
            self._order_cache = {epoch: np.random.RandomState(self.seed + epoch).permutation(self.n_blocks)}
        return self._order_cache[epoch]

    def batch(self, step):
        pos = step * self.seq_per_step
        epoch = pos // self.n_blocks
        within = pos % self.n_blocks
        order = self._order(epoch)
        need = self.seq_per_step
        if within + need <= self.n_blocks:
            idxs = order[within:within + need]
        else:  # wrap into next epoch's permutation
            nxt = self._order(epoch + 1)
            idxs = np.concatenate([order[within:], nxt[:need - (self.n_blocks - within)]])
        x = np.ascontiguousarray(self.blocks[idxs])  # (bs, 1025) int32
        return x

    def epoch_float(self, step):
        return (step * self.seq_per_step) / self.n_blocks


def lr_at(step, total_steps, peak, warmup_frac=0.015, decay_frac=0.2, floor_ratio=0.1):
    """WSD: linear warmup -> constant -> linear decay to floor_ratio*peak."""
    w = max(1, int(total_steps * warmup_frac))
    if step < w:
        return peak * (step + 1) / w
    d_start = int(total_steps * (1 - decay_frac))
    if step < d_start:
        return peak
    frac = (step - d_start) / max(1, total_steps - d_start)
    return peak * (floor_ratio + (1 - floor_ratio) * (1 - frac))


def build_optim(arm, model, lr, lr_embed, wd):
    hidden, other = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in n:
            hidden.append(p)
        else:
            other.append(p)
    if arm == "adamw":
        opt = torch.optim.AdamW(hidden + other, lr=lr, betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=wd, fused=True)
        return opt, [], f"AdamW(all) lr={lr}"
    muon = Muon(hidden, lr=lr, momentum=0.95, ns_steps=5, weight_decay=wd)
    adam = torch.optim.AdamW(other, lr=lr_embed, betas=(0.9, 0.95),
                             eps=1e-8, weight_decay=wd, fused=True)
    return muon, [adam], f"Muon(hidden, lr={lr}, wd={wd}) + AdamW(embed/norms, lr={lr_embed}, wd={wd})"


def save_ckpt(path, model, opts, step, cfg, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step, "cfg": cfg,
        "model": {k: v.bfloat16() for k, v in model.state_dict().items()},
        "opt": [o.state_dict() for o in opts],
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    }, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["muon", "adamw"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--lr-embed", type=float, default=None,
                    help="AdamW lr for embed/norms in the Muon arm (default: same as AdamW arm winner)")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=1600)
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
    ap.add_argument("--probe", action="store_true", help="LR probe mode: short, no ckpt/eval/watchdog")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--gate-min-free-mib", type=int, default=16 * 1024)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()
    if args.lr_embed is None:
        args.lr_embed = args.lr

    # LAUNCH GATE: must run before ANY CUDA context creation
    if not args.no_gate:
        LaunchGate(min_free_mib=args.gate_min_free_mib).wait()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    torch.cuda.set_per_process_memory_fraction(0.42)  # junior-job hard cap ~13.7GB
    dev = torch.cuda.current_device()

    cfg = model_config(max_seq=args.seq)
    model = TinyGQA(cfg).cuda()
    n_all, n_emb, n_hid = (sum(p.numel() for p in model.parameters()),
                           model.embed.weight.numel(),
                           sum(p.numel() for n_, p in model.named_parameters()
                               if p.ndim == 2 and "embed" not in n_))
    opt, extra_opts, opt_desc = build_optim(args.arm, model, args.lr, args.lr_embed, args.wd)
    opts = [opt] + extra_opts

    step0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        for o, sd in zip(opts, ck["opt"]):
            o.load_state_dict(sd)
        step0 = ck["step"]
        print(f"[resume] from step {step0}", flush=True)

    from model import chunked_ce  # eager, checkpointed, chunked CE

    fwd_trunk = model.trunk
    if args.compile:
        try:
            fwd_trunk = torch.compile(model.trunk)
            # warm up with REAL training shapes so autotune memory lands now
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

    data = PackedData(args.data, args.tokens_per_step // args.seq, args.seed)
    accum = args.tokens_per_step // (args.seq * args.micro_bs)
    tag = args.tag or (f"probe_{args.arm}_{args.lr:g}" if args.probe else args.arm)
    log_path = os.path.join(POC_DIR, "logs", f"{tag}.jsonl")
    logf = open(log_path, "a")

    def log(rec):
        rec.update(arm=args.arm, tag=tag, ts=time.time())
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    print(f"[cfg] arm={args.arm} {opt_desc} steps={args.steps} "
          f"tokens/step={args.tokens_per_step} micro_bs={args.micro_bs} accum={accum} "
          f"params: total={n_all/1e6:.1f}M embed={n_emb/1e6:.1f}M hidden={n_hid/1e6:.1f}M "
          f"blocks={data.n_blocks} ({data.n_blocks*args.seq/1e6:.0f}M tokens/epoch)", flush=True)

    watchdog = None
    if not args.probe:
        watchdog = GPUWatchdog()
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
        # --- junior-job yield discipline ---
        if watchdog is not None and watchdog.event.is_set():
            ck = os.path.join(TMP, f"ckpt_{args.arm}", "yield.pt")
            save_ckpt(ck, model, opts, step, cfg, args)
            torch.cuda.empty_cache()
            yields += 1
            log(dict(event="yield", step=step, gpu_mib=watchdog.last_reading,
                     note="total>29GB; yielding to senior RL job (30min cycles)"))
            waited = 0
            while watchdog.event.is_set():  # no cap: we resume only when clear
                time.sleep(120)
                waited += 120
                if waited % 1800 == 0:
                    log(dict(event="still_yielding", step=step, waited_s=waited,
                             gpu_mib=watchdog.last_reading))
            log(dict(event="resume_after_yield", step=step, waited_s=waited,
                     gpu_mib=watchdog.last_reading))

        if args.probe:
            # short warmup, constant LR, no decay — ranking only
            lr_now = lr_at(step, 40, args.lr, warmup_frac=0.125, decay_frac=0.0)
            lr_emb_now = lr_at(step, 40, args.lr_embed, warmup_frac=0.125, decay_frac=0.0)
        else:
            lr_now = lr_at(step, args.steps, args.lr)
            lr_emb_now = lr_at(step, args.steps, args.lr_embed)
        for g in opt.param_groups:
            g["lr"] = lr_now if g is opt.param_groups[0] else lr_emb_now
        for o in extra_opts:
            for g in o.param_groups:
                g["lr"] = lr_emb_now

        xb = data.batch(step)  # (bs, 1025)
        micro_nats_sum = 0.0
        for o in opts:
            o.zero_grad(set_to_none=True)
        t_micro = time.time()
        for mi in range(accum):
            rows = xb[mi * args.micro_bs:(mi + 1) * args.micro_bs]
            x = torch.from_numpy(rows.astype(np.int64)).cuda(non_blocking=True)
            # QK-Clip probe on the first micro of each step only (K2 probes
            # per step; per-micro is overkill and the logit max is stable
            # across micros of the same step)
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
                save_ckpt(os.path.join(TMP, f"ckpt_{args.arm}", "nan_abort.pt"),
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
                save_ckpt(os.path.join(TMP, f"ckpt_{args.arm}", "latest.pt"),
                          model, opts, step, cfg, args)
                if abs(step - args.steps // 2) < args.ckpt_every // 2:
                    save_ckpt(os.path.join(TMP, f"ckpt_{args.arm}", "mid.pt"),
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

    save_ckpt(os.path.join(TMP, f"ckpt_{args.arm}", "final.pt"), model, opts, step, cfg, args)
    log(dict(event="done", step=step, tokens=tokens_seen,
             total_s=round(time.time() - t_start, 1), yields=yields))
    if watchdog:
        watchdog.stop.set()


if __name__ == "__main__":
    main()
