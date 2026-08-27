#!/usr/bin/env python3
"""Full-step CUDA-graph capture for the POC-DDOT trainer ("megakernel").

Captures the ENTIRE micro-batch training step — trunk forward, OT-coupled
loss, and backward — as ONE CUDA graph per shape bucket, replayed with a
single launch. This removes the ~50-60 kernel enqueues per micro that kept
the live run at ~63% GPU utilization (every step is ~240 micros; the CPU
dispatch serialization between the big matmuls was the idle gap).

Design (each point is a hard-won lesson from this package's history):

* STATIC SHAPES by bucketing. B -> multiple of 4, T -> multiple of 128,
  packed span width N -> multiple of 64 (mirroring the Sinkhorn bucketing
  in objective_ot._sinkhorn), routed-pair budget R -> multiple of 256.
  Pad rows: valid=False (attn ignores them), span_pos=False, m=False,
  slots=0 -> they contribute EXACTLY 0 to the loss.
* REAL-ROW COUNT as a scalar tensor input (n_real): the loss normalizes
  by the number of real rows, not the padded B, so the estimator scale is
  exact and matches eager ot_step's `contrib.mean()` over the real batch.
* DENSE ROUTING replaces nonzero()/masked-select. The masked-row set is
  selected by a fixed-size top-R over a 0/1 score (all real masked rows
  always fit because R >= the actual masked count, known on the HOST
  since the mask is drawn there); non-selected rows get weight exactly 0
  and padded CE pairs contribute exactly 0. CE work waste is bounded by
  the R rounding (<256 extra pairs).
* RANDOMNESS OUTSIDE THE GRAPH. t, the Bernoulli mask draws, and the
  position noise are pre-drawn on the HOST (CPU torch.Generator — same
  distributions as objective.sample_rates/bernoulli_mask/noise_positions:
  U(0,1) rates, rand<t masks, randn noise) and copied into static input
  buffers. RNG capture inside CUDA graphs is only managed for the DEFAULT
  generator — the trainer's user-passed torch.Generator is not
  replay-safe, and host draws additionally give us the masked count
  without a GPU sync.
* SINKHORN EAGER INSIDE THE CAPTURE. objective_ot._sinkhorn's
  torch.compile(reduce-overhead) is a graph itself — nested capture fails.
  We call OT.sinkhorn_coupling directly; the manual capture graphs it
  anyway. With zero-mass marginals on the pads the resulting plan on the
  real region is numerically identical to the compiled path's (padded
  columns have log-marginal -1e9 -> plan entries exp(-5e7) = exactly 0,
  so logsumexp is unchanged).
* BACKWARD IN THE GRAPH. Canonical whole-network-capture pattern: warmup
  on a side stream, then capture forward + torch.autograd.grad(...) and
  accumulate the per-parameter gradients into SHARED static fp32 buffers
  (`acc`) inside the graph. Replays accumulate: the trainer zeroes `acc`
  once per step, replays the micros, then exports acc -> p.grad for the
  (outside-the-graph) optimizer step + clip. All graphs share one memory
  pool and write their outputs into non-pool static buffers, so replays
  in any order are safe and per-graph cost is only workspace, not
  activations-per-graph.
* PRECISION: trunk + CE run under bf16 autocast INSIDE the capture
  (identical kernels to the eager trainer path, which also runs ot_step
  under autocast); Sinkhorn, routing weights, and both loss reductions
  are fp32. Measured (tiny MDGQA, identical pre-drawn inputs, test
  test_graphed_step): loss/value/position rel-diff 0.0 (BIT-exact when
  shapes already match the buckets); grads agree to GEMM/scatter
  accumulation-order precision (~3e-3 rel on embed/trunk grads — the
  bucket-rounded zero-weight pad rows change cuBLAS's reduction tiling;
  per-pair terms are identical); padded-B/padded-T runs match the
  unpadded eager loss to <1e-4.
* GRAD TELEMETRY: grad-norm stays OUTSIDE (the trainer computes
  clip_grad_norm over acc once per step); the graph outputs only
  loss/value/position/plan_entropy as static tensors.

Trainer wiring (train_ot.py --full-graph): one GraphedOTStep owns all
buckets; unknown shapes beyond max_graphs fall back to the SAME math run
eagerly (correct, just ungraphed). Graph key = (Bp, Tp, Np, R); expect
~10-30 graphs over a full run, each a few hundred ms to warm up + capture.

Benchmark (RTX 5090, real 206M-param config, coexisting with the live
run): see __main__ / the commit message for measured ms/micro.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from . import ot_coupling as OT
except ImportError:                          # script/test path
    import ot_coupling as OT
try:
    from experiments.training.poc_diff.objective import T_MIN
except ImportError:                          # run as a bare script
    import sys
    sys.path.insert(0, __file__.rsplit("/experiments/", 1)[0])
    from experiments.training.poc_diff.objective import T_MIN


def _bucket(v, m):
    return max(m, ((int(v) + m - 1) // m) * m)


@dataclass
class _Static:
    key: tuple
    Np: int
    R: int
    bufs: dict
    out: dict


class GraphedOTStep:
    """Owns static buffers, warmup, capture, and replay for full-step
    CUDA graphs of the OT trainer's micro-batch. Use from the training
    loop as:

        gs = GraphedOTStep(model, pos_head)
        per step:   gs.zero_accumulators()
        per micro:  out = gs.run_micro(x, span_pos, valid, slots,
                                       span_nmax, gen=cpu_gen,
                                       scale=1/len(micros))
        after:      gs.export_grads(); opt.step(); ...

    `out` is a dict of static tensors (total, value, position, telemetry)
    that the NEXT replay overwrites — accumulate immediately.
    """

    def __init__(self, model, pos_head, trunk=None, eps=0.05, kappa=None,
                 lam=1.0, pair_topk=3, n_iters=50, chunk=4096,
                 b_mult=4, t_mult=128, n_mult=64, r_mult=256,
                 max_graphs=48, warmup=3, dtype=torch.bfloat16):
        self.model, self.pos_head = model, pos_head
        self.trunk = trunk if trunk is not None else model.trunk
        self.eps, self.kappa, self.lam = eps, kappa, lam
        self.pair_topk, self.n_iters, self.chunk = pair_topk, n_iters, chunk
        self.b_mult, self.t_mult = b_mult, t_mult
        self.n_mult, self.r_mult = n_mult, r_mult
        self.max_graphs, self.warmup = max_graphs, warmup
        self.dtype = dtype
        self.device = next(model.parameters()).device
        assert self.device.type == "cuda", "full-graph capture needs CUDA"
        self.params = ([p for p in model.parameters() if p.requires_grad]
                       + [p for p in pos_head.parameters() if p.requires_grad])
        self.acc = [torch.zeros_like(p) for p in self.params]
        self.pool = None            # shared across all graphs (see module doc)
        self.graphs = {}            # key -> torch.cuda.CUDAGraph
        self.statics = {}           # key -> _Static
        self.scale_buf = torch.ones(1, device=self.device)

    # ------------------------------------------------------------ public API

    def zero_accumulators(self):
        for a in self.acc:
            a.zero_()

    def export_grads(self):
        """Point p.grad at the accumulated buffers (read-only for the
        optimizer; re-zeroed next step via zero_accumulators)."""
        for p, a in zip(self.params, self.acc):
            p.grad = a

    def run(self, x, span_pos, valid, slots, t, m, noise, span_nmax,
            scale=1.0, accumulate=True):
        """One graphed step on PRE-DRAWN randomness.

        Args: x/span_pos/valid/slots (B,T) as in ot_step; t (B,) rates;
        m (B,T) bool mask draw; noise (B,T) standard-normal draw (the eps
        of noise_positions); span_nmax: HOST int, the max span length in
        this micro (the trainer knows it from the data row lens);
        scale: loss scale for gradient accumulation (len(micros) divisor).
        Returns the static output dict (overwritten by the next replay).
        """
        B_real, T_real = x.shape
        Bp = _bucket(B_real, self.b_mult)
        Tp = _bucket(T_real, self.t_mult)
        Np = _bucket(span_nmax, self.n_mult)
        n_masked = int(m.sum()) if m.device.type == "cpu" else int(m.sum().item())
        R = min(_bucket(n_masked, self.r_mult), Bp * Np)   # rows available
        key = (Bp, Tp, Np, R)
        st = self.statics.get(key)
        if st is None:
            st = self._make_statics(key)
        self._copy_in(st, x, span_pos, valid, slots, t, m, noise, B_real,
                      T_real)
        self.scale_buf.fill_(float(scale))
        g = self.graphs.get(key)
        if g is None:
            if len(self.graphs) < self.max_graphs:
                self._capture(key, st)
                g = self.graphs[key]
            else:
                self._exec(st, accumulate=accumulate)   # eager fallback
                return st.out
        g.replay()
        return st.out

    def run_micro(self, x, span_pos, valid, slots, span_nmax, gen=None,
                  scale=1.0, t=None, span_pos_cpu=None):
        """Convenience wrapper that DRAWS the randomness on the host (same
        distributions as ot_step's generator draws) and calls run().

        span_pos_cpu: the SAME span mask as `span_pos` but on the host —
        pass it when the caller knows the row geometry (the trainer does)
        to avoid the D2H sync of span_pos.cpu()."""
        B, T = x.shape
        if t is None:
            t = (torch.rand(B, generator=gen) if gen is not None
                 else torch.rand(B))
        draw = (torch.rand(B, T, generator=gen) if gen is not None
                 else torch.rand(B, T))
        sp = span_pos_cpu if span_pos_cpu is not None else span_pos.cpu()
        m = sp & (draw < t[:, None])
        noise = (torch.randn(B, T, generator=gen) if gen is not None
                 else torch.randn(B, T))
        return self.run(x, span_pos, valid, slots, t, m, noise, span_nmax,
                        scale=scale)

    # ------------------------------------------------------------ internals

    def _make_statics(self, key):
        Bp, Tp, Np, R = key
        d = self.device
        def z(*s, dt=torch.float32):
            return torch.zeros(*s, dtype=dt, device=d)
        bufs = dict(
            x=z(Bp, Tp, dt=torch.long), span_pos=z(Bp, Tp, dt=torch.bool),
            valid=z(Bp, Tp, dt=torch.bool), m=z(Bp, Tp, dt=torch.bool),
            slots=z(Bp, Tp), noise=z(Bp, Tp), t=z(Bp),
            n_real=z(()))
        out = dict(
            total=z(()), value=z(()), position=z(()),
            plan_entropy=z(()),
            iters=torch.full((), float(self.n_iters), device=d))
        out["telemetry"] = dict(plan_entropy=out["plan_entropy"],
                                iters_mean=out["iters"])
        st = _Static(key=key, Np=Np, R=R, bufs=bufs, out=out)
        self.statics[key] = st
        return st

    def _copy_in(self, st, x, span_pos, valid, slots, t, m, noise, B, T):
        b = st.bufs
        for name, src in (("x", x), ("span_pos", span_pos), ("valid", valid),
                          ("m", m), ("slots", slots), ("noise", noise)):
            b[name].zero_()
            b[name][:B, :T].copy_(src)
        b["t"].zero_()
        b["t"][:B].copy_(t)
        b["n_real"].fill_(float(B))

    def _capture(self, key, st):
        # stale autograd state (AccumulateGrad nodes created on the
        # default stream by earlier eager backward passes) breaks capture
        # with a legacy-stream dependency; drop it and sync first
        for p in self.params:
            p.grad = None
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                self._exec(st, accumulate=False)      # never touches acc
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        kw = {} if self.pool is None else dict(pool=self.pool)
        with torch.cuda.graph(g, **kw):
            self._exec(st, accumulate=True)
        if self.pool is None:
            self.pool = g.pool()
        self.graphs[key] = g

    def _exec(self, st, accumulate):
        """Forward + loss + backward on the static buffers. Runs eagerly
        during warmup/fallback and INSIDE the capture region otherwise —
        no .item()/sync/data-dependent shapes anywhere in this path."""
        with torch.autocast("cuda", dtype=self.dtype):
            total, value, position, entropy = self._loss(st)
        grads = torch.autograd.grad(total * self.scale_buf, self.params,
                                    allow_unused=True)
        if accumulate:
            for a, g in zip(self.acc, grads):
                if g is not None:
                    a.add_(g)
        st.out["total"].copy_(total.detach())
        st.out["value"].copy_(value.detach())
        st.out["position"].copy_(position.detach())
        st.out["plan_entropy"].copy_(entropy.detach())

    def _loss(self, st):
        """Graph-safe rewrite of ot_mdlm_loss on the padded static buffers
        (same estimator, same scale; see module docstring for the mapping
        from the dynamic ops to static ones)."""
        b = st.bufs
        model, pos_head, trunk = self.model, self.pos_head, self.trunk
        x, m, valid, span_pos = b["x"], b["m"], b["valid"], b["span_pos"]
        slots, t, noise = b["slots"], b["t"], b["noise"]
        B, Tp = x.shape
        Np, R = st.Np, st.R
        k = min(self.pair_topk, Np)
        dev = x.device

        # --- trunk + position head (bf16 autocast, same as ot_step) -------
        x_in = x.masked_fill(m, model.mask_id)
        h = trunk(x_in, probe=False, attn_mask=valid[:, None, None, :])
        pos_pred = pos_head(h).squeeze(-1)

        # --- noised positions from the PRE-DRAWN eps (noise_positions) ----
        m_sigma = m & (t > 0)[:, None]
        noised = torch.where(m_sigma, slots + noise * t[:, None], slots)

        # --- pack the span window to (B, Np) with STATIC ops --------------
        # (replaces span_pos.nonzero + data-dependent Nmax; scatter_reduce
        # amax is deterministic and duplicate-safe)
        tgrid = torch.arange(Tp, device=dev)
        lens = span_pos.sum(1)                                    # (B,)
        in_span = (torch.arange(Np, device=dev).unsqueeze(0)
                   < lens.unsqueeze(1))                           # (B,Np)
        sp_long = span_pos.long()
        rank = (sp_long.cumsum(1) - 1).clamp(min=0)
        t_of_k = torch.zeros(B, Np, dtype=torch.long, device=dev)
        t_of_k.scatter_reduce_(                                   # (B,Np)
            1, (rank * sp_long).clamp(max=Np - 1),
            torch.where(span_pos, tgrid + 1, torch.zeros_like(tgrid)),
            reduce="amax", include_self=True)
        t_of_k = ((t_of_k - 1).clamp(min=0)) * in_span            # pads -> 0
        x_px = x.gather(1, t_of_k)
        m_px = m.gather(1, t_of_k) & in_span
        zero = torch.zeros((), device=dev)
        src = torch.where(in_span, noised.gather(1, t_of_k), zero)
        dst = torch.where(in_span, slots.gather(1, t_of_k), zero)
        log_a = torch.where(
            in_span, -torch.log(lens.clamp(min=1).float()).unsqueeze(1),
            torch.full_like(lens.float().unsqueeze(1), -1e9))

        # --- Sinkhorn, EAGER inside the capture (see module docstring) ----
        with torch.no_grad():
            plan = OT.sinkhorn_coupling(
                src, dst, eps=self.eps, n_iters=self.n_iters,
                kappa=self.kappa, log_a=log_a, log_b=log_a).plan
        w = OT.row_weights(plan)                                  # (B,Np,Np)

        # plan entropy: masked mean over live rows (the eager path's
        # rows[W.sum(-1) > 0] mean, with a static shape)
        row_live = plan.sum(-1) > 0
        ent = -(w * (w + 1e-12).log()).sum(-1)
        entropy = (ent * row_live).sum() / row_live.sum().clamp(min=1)

        # --- top-k pair routing, DENSE: fixed R rows via top-R on a 0/1
        # score (all real masked rows always fit: R >= host-known count) --
        vals, idxs = w.topk(k, dim=-1)                            # (B,Np,k)
        w_row = vals / vals.sum(-1, keepdim=True).clamp_min(1e-12)
        score = m_px.reshape(-1).float()                          # (B*Np,)
        row_idx = score.topk(R).indices                           # (R,)
        sel = score[row_idx].unsqueeze(1)                         # 0/1
        b_sel = row_idx // Np
        w_sel = w_row.reshape(-1, k)[row_idx] * sel               # (R,k)
        j_sel = idxs.reshape(-1, k)[row_idx]                      # (R,k)
        b_f = b_sel.unsqueeze(1).expand(R, k).reshape(-1)         # (R*k,)
        h_px = h.gather(1, t_of_k.unsqueeze(-1).expand(B, Np, h.size(-1)))
        d = h.size(-1)
        h_rows = h_px.reshape(-1, d)[row_idx]                     # (R,d)
        h_sel = (h_rows.unsqueeze(1).expand(R, k, d).reshape(-1, d))
        tgt = x_px[b_sel.unsqueeze(1), j_sel].reshape(-1)         # (R*k,)
        w_pos = (w_sel * (1.0 / t.clamp(min=T_MIN))[b_sel].unsqueeze(1)).reshape(-1)

        # --- chunked+checkpointed CE (static chunk count per bucket) ------
        n = R * k

        def _ce(hc, wc, tc):
            return F.cross_entropy(F.linear(hc, model.embed.weight), tc,
                                   reduction="none") * wc

        parts = []
        for c in range(0, n, self.chunk):
            parts.append(checkpoint(_ce, h_sel[c:c + self.chunk],
                                    w_pos[c:c + self.chunk],
                                    tgt[c:c + self.chunk],
                                    use_reentrant=False,
                                    preserve_rng_state=False))
        per_pair = torch.cat(parts).float()
        ex_sum = torch.zeros(B, device=dev, dtype=torch.float32)
        ex_sum.scatter_add_(0, b_f, per_pair)
        # divide by the REAL row count (not padded B): exact estimator scale
        value = (ex_sum / lens.clamp(min=1).float()).sum() / b["n_real"]

        # --- position term (position_mse, dense form) ----------------------
        w_t = 1.0 / t.clamp(min=T_MIN)
        sq = ((pos_pred.float() - slots).pow(2) * m_sigma.float()
              * w_t.unsqueeze(1))
        position = (sq.sum(1) / lens.clamp(min=1).float()).sum() / b["n_real"]

        total = value + self.lam * position
        return total, value, position, entropy


if __name__ == "__main__":
    # Benchmark: eager ot_step+backward vs graphed .run() on the REAL
    # 206M-param config, coexisting with the live run (<3GB budget).
    import argparse
    import functools
    import os
    import time

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-frac", type=float, default=0.10)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    dev = torch.device("cuda")

    from experiments.training.poc_diff.model_md import MDGQA, model_config_md
    import sys
    sys.path.insert(0, "/home/m0hawk/Documents/Sepalith/experiments/"
                       "training/poc_ddot")
    import train_ot
    from objective_ot import ot_mdlm_loss
    from graphed_step import GraphedOTStep

    # chunk=1024 keeps the vocab-sized CE transient (~0.8GB/chunk at the
    # 130560 vocab) inside the coexistence budget; both paths get it so
    # the comparison stays apples-to-apples
    CHUNK = 512
    train_ot.ot_mdlm_loss = functools.partial(ot_mdlm_loss, chunk=CHUNK)
    ot_step = train_ot.ot_step

    cfg = model_config_md()
    model = MDGQA(cfg).to(dev)
    pos_head = torch.nn.Linear(cfg["d_model"], 1).to(dev)
    gen = torch.Generator().manual_seed(0)

    def make_micro(B, T, span):
        g = torch.Generator().manual_seed(42)
        x = torch.randint(3, cfg["base_vocab"], (B, T), generator=g).to(dev)
        span_pos = torch.zeros(B, T, dtype=torch.bool)
        span_pos[:, T - span:] = True
        valid = torch.ones(B, T, dtype=torch.bool)
        slots = torch.linspace(0, 1, span).unsqueeze(0).repeat(B, 1)
        slots = F.pad(slots, (0, T - span)).to(dev)
        return x, span_pos.to(dev), valid.to(dev), slots

    def bench(fn, iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    for B, T, span in ((8, 512, 128), (4, 768, 250)):
        x, sp, vl, sl = make_micro(B, T, span)
        # eager: ot_step + backward (the trainer's current per-micro work)
        cuda_gen = torch.Generator(device=dev).manual_seed(7)

        def eager():
            model.zero_grad(set_to_none=True)
            pos_head.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = ot_step(model, pos_head, x, sp, vl, sl,
                              generator=cuda_gen, pair_topk=3)
            out["total"].backward()

        eager()                                  # warmup/compile sinkhorn
        ms_e = bench(eager, args.iters)
        # free the eager .grad buffers before the graphed phase (the graph
        # keeps its own acc buffers; both are 830MB at this model size)
        model.zero_grad(set_to_none=True)
        pos_head.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        gs = GraphedOTStep(model, pos_head, model.trunk, pair_topk=3,
                           chunk=CHUNK, max_graphs=4)
        # fixed pre-drawn randomness -> a single (Bp,Tp,Np,R) bucket ->
        # ONE graph, steady-state replay timing (in training the bucket
        # varies per micro; each captures once and then replays like this)
        gcpu = torch.Generator().manual_seed(7)
        t_fix = torch.rand(B, generator=gcpu)
        m_fix = sp.cpu() & (torch.rand(B, T, generator=gcpu)
                            < t_fix[:, None])
        n_fix = torch.randn(B, T, generator=gcpu)

        def graphed():
            gs.run(x, sp, vl, sl, t_fix, m_fix, n_fix, span_nmax=span)

        graphed()                                # capture
        graphed()                                # first replay
        ms_g = bench(graphed, args.iters)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"B={B} T={T} span={span}: eager {ms_e:.2f} ms/micro, "
              f"graphed {ms_g:.2f} ms/micro, {ms_e / ms_g:.2f}x "
              f"(graphs cached: {len(gs.graphs)}, peak {peak:.2f} GiB)",
              flush=True)
        del gs
        torch.cuda.empty_cache()
