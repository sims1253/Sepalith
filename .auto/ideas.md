# Ideas backlog — final state (mission complete, PR #66)

All paths on the primary metric (edit-event TTFT p50 at Q8_0/8T) are
exhausted: the winner (eager full warm-up) sits at the decode-bound floor
(1 fresh prompt token + 1 decode step ≈ 64.6 ms). Remaining entries were
evaluated and closed:

- ~~`--cache-reuse 64`~~ — closed by analysis + the 256 measurement: the
  chunk matcher needs ≥N contiguous identical post-divergence tokens;
  window shifts and history insertions don't provide them at any threshold.
- ~~`--poll 0`~~ — potential is sub-millisecond (TTFT is ~55 ms decode +
  ~7 ms one-token prefill + ~2 ms HTTP); below measurement noise, not worth
  a 35-minute run.
- ~~multi-slot `--parallel 2`~~ — switch-back is already free via the
  default KV pool and warm-up covers switches; adds memory and complexity.
- ~~`--cache-ram 0` quantification~~ — report note only (defaults are
  right; cap to 2048 on laptops).
- ~~layout reordering retraining proposal~~ — measured negative (see
  layout_simulation.json); do not spend a training round on ordering.
- ~~history budget (cap events)~~ — no byte-prefix gain in simulation;
  quality risk without latency payoff now that warm-up exists.
- Open for OTHER missions: quant tier choice (Q6_K decode +31%, table in
  report §6), speculative decoding (decode floor is now the bottleneck),
  Vulkan iGPU prefill/decode on a real laptop (untestable here).
