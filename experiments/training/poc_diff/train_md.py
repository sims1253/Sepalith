"""Task 4: trainer for the masked-diffusion twin.

Reuses the poc_twin instrument exactly (import, not copy): LaunchGate +
GPUWatchdog + WSD lr_at + save_ckpt from ../poc_twin/train.py, the vendored
Muon from ../poc_twin/muon.py, the AR arm's winning recipe (Muon hidden lr
0.01 / AdamW embed+norms 4e-3 / wd 0.1 / grad clip 1.0 / QK-Clip tau=100
alpha=0.5 / seed 1273). Only the data plumbing and the loss differ:

  - data: /tmp/poc_diff/train_flat_ids.bin (memmap, RAM-lean) + row_lens;
    deterministic order RandomState(seed+epoch); each step consumes rows
    until ~tokens_per_step REAL (unpadded) tokens.
  - micro-batches: length-bucketed right-padded (ceil-to-128 buckets,
    <=MICRO_PAD_TOK padded tokens per micro), bool attn_mask hides pads.
  - loss: objective.span_batch_to_loss (timestep-free MDLM, span-only
    masking, 1/clamp(t,0.01) weight).
  - QK probe runs on the first micro of each step (parent discipline).

Checkpoints stage to /tmp/poc_diff/ckpt/ then rsync to
/mnt/h/sepalith/runs/poc_diff/ (drvfs ENOMEM flap — see design A2 §6.4).
2B tokens = 3815 steps x 524,288 real tokens (44.5M-token corpus =>
~45 epochs; the pre-registered budget on the frozen triple set).

Usage:
  smoke: uv run python -m experiments.training.poc_diff.train_md --steps 40 --smoke
  full:  uv run python -m experiments.training.poc_diff.train_md --steps 3815 --compile
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.training.poc_twin import train as base
from experiments.training.poc_twin.muon import Muon
from experiments.training.poc_diff import TMP
from experiments.training.poc_diff.model_md import MDGQA, model_config_md
from experiments.training.poc_diff import objective

POC_DIFF_DIR = os.path.dirname(os.path.abspath(__file__))
MICRO_PAD_TOK = 16_384
BUCKET = 128


class TripleData:
    """Flat-packed triples, deterministic (seed, epoch) row order, variable
    real-token consumption per step (parent PackedData discipline adapted
    to variable-length rows)."""

    def __init__(self, tmp=TMP, seed=1273):
        self.ids = np.memmap(os.path.join(tmp, "train_flat_ids.bin"),
                             dtype=np.int32, mode="r")
        lens = np.frombuffer(open(os.path.join(tmp, "train_row_lens.bin"),
                                  "rb").read(), dtype=np.int32).reshape(-1, 2)
        self.prompt_len = lens[:, 0]
        self.region_len = lens[:, 1]
        self.total_len = self.prompt_len + self.region_len
        self.offsets = np.concatenate([[0], np.cumsum(self.total_len)])
        self.n_rows = len(lens)
        self.tokens_per_epoch = int(self.total_len.sum())
        self.seed = seed
        self._order_cache = {}

    def _order(self, epoch):
        if epoch not in self._order_cache:
            self._order_cache = {epoch: np.random.RandomState(
                self.seed + epoch).permutation(self.n_rows)}
        return self._order_cache[epoch]

    def rows_for_step(self, step, tokens_per_step):
        """Row indices covering >= tokens_per_step real tokens, wrapping
        epochs like PackedData.batch. A row straddling a step or epoch
        boundary is emitted whole (its head may repeat the neighboring
        step — ~1 row in ~1300; deterministic by construction)."""
        pos = step * tokens_per_step
        epoch = pos // self.tokens_per_epoch
        within = pos % self.tokens_per_epoch
        rows, acc = [], 0
        while acc < tokens_per_step:
            order = self._order(epoch)
            if getattr(self, "_cum_epoch", None) != epoch:
                self._cum = np.concatenate(
                    [[0], np.cumsum(self.total_len[order])])
                self._cum_epoch = epoch
            i = int(np.searchsorted(self._cum, within, side="right") - 1)
            rows.append(int(order[i]))  # the (possibly straddled) first row
            acc += int(self.total_len[order[i]])
            i += 1
            while i < len(order) and acc < tokens_per_step:
                rows.append(int(order[i]))
                acc += int(self.total_len[order[i]])
                i += 1
            epoch += 1
            within = 0
        return rows, acc

    def epoch_float(self, step, tokens_per_step):
        return (step * tokens_per_step) / self.tokens_per_epoch


def micro_batches(rows, data):
    """Length-bucketed micro-batches: (row_idx_list, padded_len)."""
    micros, cur, cur_max = [], [], 0
    for r in rows:
        L = int(data.total_len[r])
        new_max = max(cur_max, L)
        if cur and (len(cur) + 1) * ((new_max + BUCKET - 1) // BUCKET * BUCKET) > MICRO_PAD_TOK:
            micros.append((cur, (cur_max + BUCKET - 1) // BUCKET * BUCKET))
            cur, cur_max = [r], L
        else:
            cur.append(r)
            cur_max = new_max
    if cur:
        micros.append((cur, (cur_max + BUCKET - 1) // BUCKET * BUCKET))
    return micros


def build_micro_tensors(micro, data, device):
    rows, pad_len = micro
    n = len(rows)
    x = torch.full((n, pad_len), 1, dtype=torch.long)
    span_pos = torch.zeros(n, pad_len, dtype=torch.bool)
    valid = torch.zeros(n, pad_len, dtype=torch.bool)
    for i, r in enumerate(rows):
        p, s = int(data.prompt_len[r]), int(data.region_len[r])
        ids = torch.from_numpy(
            np.asarray(data.ids[data.offsets[r]:data.offsets[r] + p + s],
                       dtype=np.int64))
        x[i, :p + s] = ids
        span_pos[i, p:p + s] = True
        valid[i, :p + s] = True
    return (x.to(device, non_blocking=True), span_pos.to(device),
            valid.to(device))


def rsync_out(src, dst):
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["rsync", "-a", src + "/", dst], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-embed", type=float, default=4e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=3815)
    ap.add_argument("--tokens-per-step", type=int, default=524288)
    ap.add_argument("--seed", type=int, default=1273)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--tau", type=float, default=100.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true",
                    help="40-ish step gate/telemetry check, no watchdog")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--gate-min-free-mib", type=int, default=16 * 1024)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--out", default="/mnt/h/sepalith/runs/poc_diff")
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
    n_all = sum(p.numel() for p in model.parameters())
    opt, extra_opts, _ = base.build_optim(
        "muon", model, args.lr, args.lr_embed, args.wd)
    opts = [opt] + extra_opts
    cuda_gen = torch.Generator(device=device).manual_seed(args.seed)

    data = TripleData(seed=args.seed)

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
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64  # per-bucket shapes
        try:
            fwd_trunk = torch.compile(model.trunk)
            x0 = torch.zeros(8, 512, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fwd_trunk(x0, probe=False)
            print("[compile] ok (bucketed shapes will compile lazily)",
                  flush=True)
        except Exception as e:
            print(f"[compile] FAILED ({type(e).__name__}: {e}); eager", flush=True)
            fwd_trunk = model.trunk

    ckpt_dir = os.path.join(TMP, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    def log(rec):
        rec.update(tag="md", ts=time.time())
        print(json.dumps(rec), flush=True)
        with open(os.path.join(POC_DIFF_DIR, "logs_md.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    n_ckpt = {"n": 0}

    def save(path, step):
        base.save_ckpt(path, model, opts, step, cfg, args)
        n_ckpt["n"] += 1
        if n_ckpt["n"] % 2 == 0 or "final" in os.path.basename(path):
            rsync_out(ckpt_dir, args.out)

    @torch.no_grad()
    def quick_eval(n_rows=128):
        """Held-out MDLM loss on eval triples (seeded t ~ U(0,1))."""
        rows = [json.loads(l) for l in
                open(os.path.join(TMP, "eval_triples.jsonl"))]
        rows = rows[:n_rows]
        from experiments.training.poc_diff import span_region
        losses = []
        for i in range(0, len(rows), 16):
            batch = rows[i:i + 16]
            prompts = [r["prompt_ids"] for r in batch]
            regions = [span_region(r["span_ids"]) for r in batch]
            pad = max(len(p) + len(s) for p, s in zip(prompts, regions))
            x = torch.full((len(batch), pad), 1, dtype=torch.long, device=device)
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
                h = fwd_trunk(x_in, probe=False,
                              attn_mask=vl[:, None, None, :])
                loss = objective.mdlm_loss(h, x, m, t, sp.sum(1),
                                           model.embed.weight)
            losses.append(float(loss))
        return float(np.mean(losses))

    print(f"[cfg] MDGQA {n_all/1e6:.1f}M params; steps={args.steps} "
          f"x {args.tokens_per_step} tok = "
          f"{args.steps*args.tokens_per_step/1e9:.2f}B; "
          f"rows={data.n_rows} ({data.tokens_per_epoch/1e6:.1f}M tok/epoch, "
          f"{args.steps*args.tokens_per_step/data.tokens_per_epoch:.1f} epochs)",
          flush=True)

    watchdog = None
    if not args.smoke:
        watchdog = base.GPUWatchdog()
        watchdog.start()

    t_start = time.time()
    tokens_seen = step0 * args.tokens_per_step
    win = {"loss": [], "gn": [], "t": [], "mask": [], "cov": [], "qk": -1.0,
           "clip": 0, "t0": time.time()}

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
            t = objective.sample_rates(x.size(0), generator=cuda_gen,
                                       device=device)
            m = objective.bernoulli_mask(span_pos, t, generator=cuda_gen)
            x_in = x.masked_fill(m, model.mask_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = fwd_trunk(x_in, probe=(mi == 0),
                              attn_mask=valid[:, None, None, :])
                loss = objective.mdlm_loss(h, x, m, t, span_pos.sum(1),
                                           model.embed.weight)
            (loss / len(micros)).backward()
            step_nats += float(loss)
            step_stats = dict(t_mean=float(t.mean()),
                              mask_rate=float(m.sum() / span_pos.sum()),
                              span_cov=float(span_pos.sum()) /
                              float(valid.sum()))
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
            win["cov"].append(step_stats["span_cov"])
        win["qk"] = max(win["qk"], qk_now)
        win["clip"] += n_clip

        if step % args.log_every == 0:
            dt = time.time() - win["t0"]
            rec = dict(step=step, tokens=tokens_seen,
                       loss=float(np.mean(win["loss"])), lr=lr_now,
                       grad_norm=float(np.mean(win["gn"])),
                       t_mean=float(np.mean(win["t"])),
                       mask_rate=float(np.mean(win["mask"])),
                       span_cov=float(np.mean(win["cov"])),
                       qk_max=win["qk"], qk_clipped_heads=win["clip"],
                       tok_per_s=round(args.log_every * args.tokens_per_step / dt, 1),
                       epoch=round(data.epoch_float(step, args.tokens_per_step), 2),
                       gpu_mib=watchdog.last_reading if watchdog else None)
            if not np.isfinite(rec["loss"]):
                rec["event"] = "LOSS-NAN — aborting"
                log(rec)
                save(os.path.join(ckpt_dir, "nan_abort.pt"), step)
                sys.exit(2)
            log(rec)
            win = {"loss": [], "gn": [], "t": [], "mask": [], "cov": [],
                   "qk": -1.0, "clip": 0, "t0": time.time()}

        if step % args.eval_every == 0:
            log(dict(event="eval", step=step,
                     eval_loss=round(quick_eval(), 5)))

        if step % args.ckpt_every == 0:
            save(os.path.join(ckpt_dir, "latest.pt"), step)

    save(os.path.join(ckpt_dir, "md_final.pt"), step)
    log(dict(event="done", step=step, tokens=tokens_seen,
             total_s=round(time.time() - t_start, 1)))
    if watchdog:
        watchdog.stop.set()


if __name__ == "__main__":
    main()
