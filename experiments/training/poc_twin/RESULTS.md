# Twin POC results — Muon(+QK-Clip) vs AdamW, from-scratch, RTX 5090

Night of 2026-08-20/21. First real training on our architecture shape.
The comparison is the deliverable, not the model.

## VERDICT vs the sweep's falsifier — CLEARED, decisively

Held-out BPB (500 astfim eval rows, 430,080 scored tokens, identical
data order / schedule / steps / QK-Clip / wd / grad-clip; only the
optimizer on hidden 2D weights differs):

| arm | BPB | nats/token | token ppl |
|---|---|---|---|
| **Muon(+QK-Clip), lr 0.01** | **0.7032** | 1.4837 | 4.41 |
| AdamW, lr 4e-3 | 0.7578 | 1.5990 | 4.95 |

**Muon is 7.2% BETTER on paired BPB** (rel = -0.072). The sweep's
falsifier ("Muon >= 1% worse paired R-BPB -> revert to AdamW-default") is
not merely cleared — the sign matches the Moonlight/K2 consensus at our
first crossed arm. At matched 682M-token budget on repeated data (2.78
epochs), Muon's advantage at the END (7.2%) is much smaller than its
early-token advantage (2+ nats at 52-105M tokens) — consistent with the
"2x compute efficiency" reading: AdamW closes as tokens accumulate, but
never catches up inside this budget.

## Instrument

- Model `TinyGQA` (model.py): d=768, 12L, GQA 16Q/2KV x hd64, SwiGLU 4x
  (3072x768 tall geometry = the 1.5B's 8192x2048 scaled down), pre-RMSNorm
  eps 1e-6, RoPE theta 500k from step 0, tied embeddings, MiniCPM5
  tokenizer (vocab 130,560), max_seq 1024. **206.5M total = 100.3M embed +
  106.2M hidden**. The spec band (100-150M) assumed a smaller vocab; the
  130k MiniCPM5 vocab makes the tied embedding alone 100M — the
  hidden-matrix budget (the Muon-relevant part, 106M) matches the spec's
  ~120M intent. Recorded as a deliberate overshoot.
- Data: astfim_v1 PSM-format R, 245.4M tokens train (276,206 docs), packed
  into 1025-token overlap-1 blocks with eos separators, deterministic
  order `RandomState(1273+epoch)` — **byte-identical token stream for both
  arms** (verified). Eval: eval.jsonl first 500 rows = 430,080 scored
  tokens / 1,309,158 UTF-8 bytes, packed identically.
- QK stabilization: NO QK-norm (sweep OPT-4a shape); **stateless QK-Clip
  tau=100, alpha=0.5, per-head, post-step, in BOTH arms**. GQA rule: the
  shared K head takes the strongest gamma of its Q-head group (verified
  exactly in CPU tests).
- Muon arm (muon.py — vendored Keller-Jordan single-file + the pinned
  adaptations): Moonlight/K2 RMS-matching update scale `0.2*sqrt(max(m,n))`,
  momentum 0.95 plain EMA, NS 5 iters (3.4445/-4.7750/2.0315) in bf16,
  Frobenius eps 1e-7, decoupled wd 0.1 (the load-bearing trick); side
  AdamW(lr 4e-3, wd 0.1) on embed + norms. AdamW arm: AdamW everything,
  betas (0.9, 0.95), wd 0.1. Grad clip 1.0 both.
- Schedule (both): WSD — 1.5% linear warmup, constant, linear decay to
  0.1x peak over the final 20%. **1300 steps x 524,288 tokens = 681.6M
  tokens/arm (2.78 epochs — the repeated-data regime is on-thesis).**
- LR selection (40-step probes x 262k tokens, ranked on 32 fixed eval
  blocks): AdamW {1e-3: 6.881, 2e-3: 7.082, **4e-3: 6.755**} -> 4e-3;
  Muon {**0.01: 5.529**, 0.02: 7.944, 0.04: 8.768} -> 0.01 (= RMS-matched
  2e-3-equivalent; 0.02/0.04 diverge-ish at this scale/batch).
- Precision: fp32 master + bf16 autocast; trunk torch.compile'd (4.4x
  speedup, eager 8.6k -> 60k tok/s), CE chunked+checkpointed on the eager
  fused kernel (inductor materializes fp32 vocab-logits and OOMs a 13.7GB
  budget; eager CE does not).

## Loss curves aligned by token count (train loss, 100-step windows)

| tokens | muon loss | adamw loss | diff | muon qk_max | adamw qk_max |
|---|---|---|---|---|---|
| 52M | 4.6134 | 6.7631 | -2.150 | 117.5 | 134.0 |
| 105M | 2.3587 | 4.6479 | -2.289 | 120.5 | 122.0 |
| 157M | 1.7781 | 3.8245 | -2.046 | 120.5 | 118.5 |
| 210M | 1.6154 | 3.0901 | -1.475 | 119.5 | 116.5 |
| 262M | 1.5080 | 2.4032 | -0.895 | 123.5 | 113.5 |
| 315M | 1.4251 | 2.0189 | -0.594 | 122.5 | 110.5 |
| 367M | 1.3875 | 1.8489 | -0.461 | 120.0 | 108.0 |
| 419M | 1.3473 | 1.6991 | -0.352 | 121.5 | 102.5 |
| 472M | 1.3138 | 1.5961 | -0.282 | 119.0 | 82.5 |
| 524M | 1.2341 | 1.4809 | -0.247 | 117.0 | 76.0 |
| 577M | 1.1933 | 1.4121 | -0.219 | 101.0 | 76.0 |
| 629M | 1.1221 | 1.3343 | -0.212 | 100.0 | 67.0 |
| 682M | 1.0405 | 1.2606 | -0.220 | 102.0 | 61.0 |

Held-out quick-eval (64 blocks, nats/token): Muon 1.572@131M -> 1.189@655M;
AdamW 3.772@131M -> 1.278@655M. The eval gap shrinks monotonically
(2.20 -> 0.09 nats) but never closes; the final full-500 BPB gap is 7.2%.

## QK-Clip telemetry (the clip was LIVE in both arms)

- Muon arm: max QK logit crossed tau=100 within the first 100 steps
  (117.5), stayed pinned at ~100-123.5 for the whole constant-LR phase,
  and clip frequency followed the exact K2 pattern — peak ~17 heads/step
  (1695 clips per 100-step window at step ~300), then monotone decay:
  1695 -> 1425 -> 1146 -> 905 -> 678 -> 410 -> 219 -> 65 -> ~0 by the
  decay phase. **Muon's spectral pressure keeps the logit AT the cap;
  QK-Clip is doing real work for the entire stable phase.**
- AdamW arm: a violent early burst (8563 clips/100 steps at step 100,
  ~86 heads/step — 30x Muon's early rate; its 4e-3 lr gives ~2x Muon's
  effective RMS on Q/K) then burns out completely: below tau by 472M
  tokens, finishing at 61 — AdamW's QK logits decay away on their own.
- Zero loss spikes in EITHER arm. No NaN, no divergence.
- Reading for the sweep: QK-Clip is not a Muon-only bandage at this
  scale — the AdamW arm needed it more, earlier (albeit at a higher
  probe-picked lr). Muon's sustained pin-at-tau is the mechanism the
  sweep's wiki note predicted (spectral inflation), visible at 206M.

## Throughput under GPU sharing (and after)

- Probes (shared with RL): 47-68k tok/s. Muon arm (shared, RL at
  10-19GB, 93-99% util): 53-61k tok/s, mean 54.6k. AdamW arm (sole GPU
  after 05:50): 69-74k tok/s, mean 72.5k.
- The 0.42 memory cap (~13.7GB) never bound: my process sat at 9-14.7GB
  total; peak with AdamW states 11.95GB in testing.
- Junior-job discipline: launch gate (>=16GB free before any CUDA
  context, coordinator mandate after the RL smoke held 31.3GB) paused the
  whole pipeline ~25 min (01:15-01:41); watchdog (yield if total >29GB)
  NEVER tripped — sharing stayed at 22-29GB total all night. **Zero
  yields, zero incidents.**

## Generation sanity (20 greedy samples, 10/arm, 200 tokens)

Both arms produce coherent PSM-format R: correct
`<|context|>path\ncode <|history|> <|cursor|><|suffix|>` scaffolding,
valid signatures (`function(x, vcov. = NULL, ...)`), idiomatic control
flow and comments. Qualitatively: AdamW samples show mild repetition
loops (duplicate "check if vcov. is a vector" blocks); Muon samples vary
repeated structure more. Neither degenerates. Full samples:
logs/final_eval.json (`generations`).

## What the 0.5BT arms should change / next steps

1. The POC validates the recipe mechanically (vendored Muon + RMS
   matching + wd + QK-Clip all behave per the papers at our shape).
   The 0.5B crossed arms can adopt this trainer nearly as-is: same
   packing, same paired-order discipline, same telemetry.
2. Muon lr for the bigger twin: keep the RMS-matched mapping
   (lr_muon ~= 0.5x the tuned AdamW lr in RMS terms came out optimal
   here: 0.01 vs 4e-3); do NOT assume the 0.02-0.05 speedrun lore maps
   over — at 0.5M-token batches 0.02/0.04 were clearly worse in probes.
3. AdamW probe non-monotonicity (2e-3 worse than 1e-3 and 4e-3) says
   40-step probes are noisy; the 0.5B LR pick should use 100+ steps.
4. Watch the AdamW early QK burst: 8.6k clips/100 steps in the first
   window — if the 0.5B AdamW control keeps QK-norm OFF, expect the
   same; with QK-norm ON (OPT-1 default candidate) the telemetry
   channel stays the readout.
5. Headroom: sole-GPU throughput was 72k tok/s at 206M params with the
   0.42 cap — the real 0.5B/0.5BT twins fit the 5090 comfortably
   (~0.15-0.2s/Mtok more at 0.5B, still hours not days).
6. Repeated-regime note (on-thesis): both arms saw 2.78 epochs; the
   per-epoch BPB slope readout (OPT-5's actual design) needs the
   {2,4,8}-epoch x {Muon, AdamW} cross — this POC's single-point
   2.78-epoch pair shows Muon ahead, not yet whether the SLOPE differs.

## Files

- Trainer seed (becomes the real pretrainer): train.py, model.py,
  muon.py, data_prep.py, evaluate.py, summarize.py, pick.py,
  run_night.sh; configs/{muon_arm,adamw}.json
- Logs: logs/{muon,adamw}.jsonl (per-100-step telemetry),
  logs/plan.json (probe evidence), logs/final_eval.json (BPB +
  generations), logs/orchestration.log, logs/gpu_trace.log
- Checkpoints: checkpoints/{muon,adamw}_{final,mid}.pt (bf16 weights +
  optimizer state, resume-able)
