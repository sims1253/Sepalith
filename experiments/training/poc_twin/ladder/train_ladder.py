"""
FIM dose-ladder trainer (2026-08-22, ladder agent).

Four arms, IDENTICAL everything except the PSM-FIM token share (dose):
0% / 10% / 20% / 35%, each 668 steps x 524,288 tok = 350.2M tokens on the
206.5M TinyGQA (Muon lr .01 / embed 4e-3 / wd .1 / WSD / QK-Clip tau=100
alpha=.5 / clip 1.0 / seed 1273 — the adopted twin-POC recipe).

Paired data design: every arm draws from the SAME two packed streams over
the SAME astfim fixed-corpus rows —
    fim stream     /tmp/poc_twin/train_blocks.npy          (PSM text as-is)
    causal stream  /tmp/poc_twin/ladder/train_blocks_causal.npy (plain docs)
Slot assignment is a NESTED Bernoulli mask: u_s ~ U[0,1) from ONE fixed RNG
stream (MASK_SEED, dose-independent), slot s is FIM iff u_s < dose. Dose
0.35's FIM slots are a strict superset of dose 0.20's etc., so the dose
family is monotone-paired: same underlying text pool, only the rendering
mix differs. Stream positions advance independently per stream (identical
across arms for a given stream-slot count), epoch permutations seeded
1273+epoch exactly like the parent PackedData.

Resumable per-arm (ckpt latest.pt every 200 steps + final.pt); telemetry
JSONL every 100 steps; dual held-out quick-eval every 200 steps (causal
stream BPB-proxy + PSM/FIM stream loss).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)                       # poc_twin/
sys.path.insert(0, POC)
from model import TinyGQA, model_config, chunked_ce  # noqa: E402
from muon import Muon                              # noqa: E402
import train as base                               # noqa: E402

TMP = base.TMP
LAD = os.path.join(TMP, "ladder")
LOGD = os.path.join(HERE, "logs")
MASK_SEED = 90210          # fixed across arms -> nested dose family


class StreamBlocks:
    """PackedData-style deterministic order over one block array."""

    def __init__(self, path, seed):
        self.blocks = np.load(path, mmap_mode="r")
        self.n = len(self.blocks)
        self.seed = seed
        self._cache = {}

    def _order(self, epoch):
        if epoch not in self._cache:
            self._cache = {epoch: np.random.RandomState(
                self.seed + epoch).permutation(self.n)}
        return self._cache[epoch]

    def take(self, start, k):
        """k consecutive blocks (in stream order) from stream position start."""
        out = np.empty((k, self.blocks.shape[1]), dtype=np.int32)
        got = 0
        while got < k:
            epoch = start // self.n
            within = start % self.n
            order = self._order(epoch)
            m = min(k - got, self.n - within)
            out[got:got + m] = self.blocks[order[within:within + m]]
            got += m
            start += m
        return out


class MixedStreams:
    """Nested-Bernoulli slot mask over the two streams, shared across arms."""

    def __init__(self, fim_path, causal_path, dose, seed, total_slots):
        self.fim = StreamBlocks(fim_path, seed)
        self.causal = StreamBlocks(causal_path, seed + 7_000_000)
        self.dose = dose
        self.total_slots = total_slots
        u = np.random.RandomState(MASK_SEED).random(total_slots)
        self.mask = u < dose                       # bool per global slot
        self.cumf = np.concatenate([[0], np.cumsum(self.mask)])  # FIM slots before pos

    def counts_before(self, pos):
        f = int(self.cumf[pos])
        return f, pos - f

    def batch(self, step, sps):
        s0, s1 = step * sps, (step + 1) * sps
        f0, c0 = self.counts_before(s0)
        m = self.mask[s0:s1]
        nf, nc = int(m.sum()), sps - int(m.sum())
        rows = np.empty((sps, self.fim.blocks.shape[1]), dtype=np.int32)
        if nf:
            rows[m] = self.fim.take(f0, nf)
        if nc:
            rows[~m] = self.causal.take(c0, nc)
        return rows


def build_optim(model, lr, lr_embed, wd):
    hidden, other = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in n:
            hidden.append(p)
        else:
            other.append(p)
    muon = Muon(hidden, lr=lr, momentum=0.95, ns_steps=5, weight_decay=wd)
    adam = torch.optim.AdamW(other, lr=lr_embed, betas=(0.9, 0.95),
                             eps=1e-8, weight_decay=wd, fused=True)
    return muon, [adam]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=float, required=True)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-embed", type=float, default=0.004)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=668)
    ap.add_argument("--tokens-per-step", type=int, default=524288)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1273)
    ap.add_argument("--fim-data", default=f"{TMP}/train_blocks.npy")
    ap.add_argument("--causal-data", default=f"{LAD}/train_blocks_causal.npy")
    ap.add_argument("--eval-fim", default=f"{TMP}/eval_blocks.npy")
    ap.add_argument("--eval-causal", default=f"{LAD}/eval_blocks_causal.npy")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--tau", type=float, default=100.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mem-frac", type=float, default=0.55)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--gate-min-free-mib", type=int, default=16 * 1024)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()
    tag = args.tag or f"ladder_fim{round(args.dose*100)}"
    os.makedirs(LOGD, exist_ok=True)

    if not args.no_gate:
        base.LaunchGate(min_free_mib=args.gate_min_free_mib).wait()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    dev = torch.cuda.current_device()

    cfg = model_config(max_seq=args.seq)
    model = TinyGQA(cfg).cuda()
    n_all = sum(p.numel() for p in model.parameters())
    opt, extra_opts = build_optim(model, args.lr, args.lr_embed, args.wd)
    opts = [opt] + extra_opts

    sps = args.tokens_per_step // args.seq
    accum = args.tokens_per_step // (args.seq * args.micro_bs)
    data = MixedStreams(args.fim_data, args.causal_data, args.dose,
                        args.seed, args.steps * sps + 64)

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
            print(f"[compile] FAILED ({type(e).__name__}: {e}); running eager",
                  flush=True)
            fwd_trunk = model.trunk

    log_path = os.path.join(LOGD, f"{tag}.jsonl")
    logf = open(log_path, "a")
    ckpt_dir = os.path.join(LAD, f"ckpt_{tag}")

    def log(rec):
        rec.update(tag=tag, dose=args.dose, ts=time.time())
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    print(f"[cfg] tag={tag} dose={args.dose} steps={args.steps} "
          f"tokens/step={args.tokens_per_step} micro_bs={args.micro_bs} accum={accum} "
          f"params={n_all/1e6:.1f}M fim_blocks={data.fim.n} causal_blocks={data.causal.n} "
          f"seed={args.seed} MASK_SEED={MASK_SEED}", flush=True)

    watchdog = base.GPUWatchdog()
    watchdog.start()

    t_start = time.time()
    tokens_seen = step0 * args.tokens_per_step
    win_loss, win_gn, win_qk, win_clip, win_t0 = [], [], -1.0, 0, time.time()
    yields = 0

    @torch.no_grad()
    def quick_eval(path, n_blocks=64, bs=8):
        eb = np.load(path, mmap_mode="r")
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
        if watchdog.event.is_set():
            base.save_ckpt(os.path.join(ckpt_dir, "yield.pt"),
                           model, opts, step, cfg, args)
            torch.cuda.empty_cache()
            yields += 1
            log(dict(event="yield", step=step, gpu_mib=watchdog.last_reading))
            waited = 0
            while watchdog.event.is_set():
                time.sleep(120)
                waited += 120
                if waited % 1800 == 0:
                    log(dict(event="still_yielding", step=step, waited_s=waited,
                             gpu_mib=watchdog.last_reading))
            log(dict(event="resume_after_yield", step=step, waited_s=waited,
                     gpu_mib=watchdog.last_reading))

        lr_now = base.lr_at(step, args.steps, args.lr)
        lr_emb_now = base.lr_at(step, args.steps, args.lr_embed)
        for g in opt.param_groups:
            g["lr"] = lr_now if g is opt.param_groups[0] else lr_emb_now
        for o in extra_opts:
            for g in o.param_groups:
                g["lr"] = lr_emb_now

        xb = data.batch(step, sps)
        micro_nats_sum = 0.0
        for o in opts:
            o.zero_grad(set_to_none=True)
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
            f_cnt, c_cnt = data.counts_before(step * sps)
            dt = time.time() - win_t0
            rec = dict(step=step, tokens=tokens_seen,
                       loss=float(np.mean(win_loss)), lr=lr_now,
                       lr_embed=lr_emb_now, grad_norm=float(np.mean(win_gn)),
                       qk_max=win_qk, qk_clipped_heads=win_clip,
                       tok_per_s=round(args.log_every * args.tokens_per_step / dt, 1),
                       fim_slots=int(f_cnt), causal_slots=int(c_cnt),
                       fim_frac=round(f_cnt / max(1, f_cnt + c_cnt), 4),
                       elapsed_s=round(time.time() - t_start, 1),
                       gpu_mib=watchdog.last_reading)
            if not math.isfinite(rec["loss"]):
                rec["event"] = "LOSS-NAN — aborting at checkpoint"
                log(rec)
                base.save_ckpt(os.path.join(ckpt_dir, "nan_abort.pt"),
                               model, opts, step, cfg, args)
                sys.exit(2)
            log(rec)
            win_loss, win_gn, win_qk, win_clip, win_t0 = [], [], -1.0, 0, time.time()

        if step % args.eval_every == 0:
            log(dict(event="eval", step=step, tokens=tokens_seen,
                     eval_loss_causal=round(quick_eval(args.eval_causal), 5),
                     eval_loss_fim=round(quick_eval(args.eval_fim), 5)))
        if step % args.ckpt_every == 0:
            base.save_ckpt(os.path.join(ckpt_dir, "latest.pt"),
                           model, opts, step, cfg, args)

    base.save_ckpt(os.path.join(ckpt_dir, "final.pt"), model, opts, step, cfg, args)
    log(dict(event="done", step=step, tokens=tokens_seen, dose=args.dose,
             total_s=round(time.time() - t_start, 1), yields=yields))
    watchdog.stop.set()


if __name__ == "__main__":
    main()
