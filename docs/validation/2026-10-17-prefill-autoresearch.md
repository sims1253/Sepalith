# Prefill autoresearch: cutting Sepalith's time-to-suggestion on CPU llama.cpp

Branch `research/prefill-autoresearch-20261017` (from `main` @ `6cc2d17`).
Report date: 2026-10-17. Mission brief:
`/home/m0hawk/sandbox/sepalith-prefill-autoresearch-brief.md`.

## TL;DR

The extension-equivalent server spends 2.5–5.0 s before the first suggestion
token on every non-typing event, because the PRM03 prompt's volatile middle
(edit history) and window shifts force llama.cpp's byte-prefix cache to
recompute 55–90% of a 0.3–2.4 k-token prompt. **A request-flow change —
prefill-as-you-settle warm-up (`n_predict: 0` requests issued right after
every document change, aborted instantly if the user pauses mid-warm-up) —
cuts every event type to the 64–65 ms floor** (1 fresh prompt token + the
first decode step) at 8 threads, **5.9× on the pooled median edit event and
up to 64× on cursor moves**, with no server-flag or model change. Output
token-identity vs a cold reference is 94.6% — but the shipped caching
behavior is itself only 95.5% cold-identical, because any prefix-cache split
reorders floating-point reductions; the mechanism and magnitude of drift are
the same as today's product.

## 1. Setup and hashes

| Item | Value |
| --- | --- |
| Host | AMD Ryzen 9 5900X (12C/24T), WSL2, ~47 GB RAM — a proxy for an analyst laptop; all runs `nice -n 10`, ≤ 8 threads, `CUDA_VISIBLE_DEVICES=""` (CPU build only) |
| llama.cpp | b10453, commit `3cb7ffb1a`, `~/.local/opt/llama.cpp-b10453/build-cpu/bin/llama-server` (read-only) |
| Model | `arm-a-Q8_0.gguf` sha256 `44b01828b21d6156eeda2fef67a4cf4ab4a9706f4ded543d9a844070de290fe` (2.49 GiB, 2.52 B params), staged read-only copy in `/tmp/prefill-models/` |
| Tokenizer | checkpoint-536 `tokenizer.json` sha256 `3e065a558a034185fe299917398685c1facd0169a9eea1e629eb30c171fed81`; prompts tokenized HF-side (`add_special_tokens=False`) with manual BOS 0 and sent as token IDs — byte-exact vs TRAIN `input_ids` (50/50 rows verified at freeze; every event re-verified against the manifest sha at bench load) |
| Workload | frozen 20-session/220-event PRM03 simulation built from TRAIN `context-sidecar.jsonl` (`DAT10-finish-source-repair-v4-braceindent`, sha in manifest); manifest `experiments/serving/prefill/manifest.json` (seed 20261018, per-event prompt sha256) |
| Requests | `/completion`, token-ID prompts, `temperature 0`, `top_k 1`, seed 3407, `return_tokens`, SSE streaming; TTFT = client wall to first streamed token |
| Passes | 2 server-restart passes per run (deterministic; pass0==pass1 outputs identical in every run) |
| Host load | recorded per run (`host_load1`); the night queue (C09 quant gate) shared the box until ~04:00 — matched-load comparisons only; the eager-vs-baseline headline pairs were measured at load 7.95/7.86 |

Event types per session: `open` (cold), `type` ×5 (typing bursts at the
cursor), `accept` (accepted suggestion → history grows mid-prompt),
`diag` (new diagnostic), `cursor_move` (same-document move — the region
becomes a later suffix line and the anchored file prefix grows, exactly as
in the product), `file_switch` (unrelated file), `switch_back`.

## 2. Time-to-suggestion by event type (8 threads, Q8_0)

Baseline = the extension's exact spawn args (`-m --alias --temp 0 --host
--port -c 16384 --parallel 1 -t 8 -ngl 0`; server defaults provide
`--cache-prompt on`, `-fa auto`, `-ub 512`, f16 KV).
Recommended = same flags + the warm-up request flow (§4). Medians over
2×(20 sessions); session-cluster bootstrap 95% CIs in the JSON.

| Event | n | Baseline TTFT p50 | Recommended TTFT p50 | Speed-up | Baseline cached frac | Recommended cached frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| type (typing burst) | 200 | 291.5 ms | 64.2 ms | 4.5× | 0.954 | 0.998 |
| accept | 40 | 3153.2 ms | 64.6 ms | 49× | 0.453 | 0.999 |
| diag | 40 | 2111.5 ms | 64.9 ms | 33× | 0.661 | 0.999 |
| cursor_move | 40 | 4175.5 ms | 64.9 ms | 64× | 0.233 | 0.999 |
| open (cold file) | 40 | 2544.9 ms | 62.8 ms | 41× | 0.177 | 0.997 |
| file_switch | 40 | 5010.3 ms | 63.7 ms | 79× | 0.080 | 0.999 |
| switch_back | 40 | 119.9 ms | 117.7 ms | 1.0× | 0.998 | 0.998 |
| **pooled edit events** | 320 | **382.8 ms** | **64.6 ms** | **5.9×** | — | — |
| pooled edit p95 | 320 | 8122.1 ms | 71.4 ms | 114× | — | — |

Notes:

* **Race bound (realism check):** the headline numbers assume the last
  document change's warm-up completed before the pause (a ≥150–300 ms
  debounce gap, typical when a human pauses). A deliberately harsher arm —
  one document change racing *every* warm-up, never refreshed — measures
  the opposite bound: typing 265 ms (one keystroke-burst delta ≈ 24
  tokens), while accept/cursor-move/file-switch fall back to their
  structural baseline costs (2.8/4.0/5.1 s) because a stale warm-up shares
  only the prefix; opens keep the 66 ms floor (an open's warm-up cannot be
  stale). Real typing pauses land between 64 and 265 ms depending on
  debounce; settle events hold the floor whenever the pause follows the
  settle by more than the debounce, and the abort-then-request rule keeps
  partial progress when it doesn't. This arm is in the artifacts as
  `warmup-eager-stale1`.

* `switch_back` is already nearly free on b10453 defaults: the default
  `--cache-ram 8192` unified-KV pool restores the previous file's context
  (~1 token recomputed). Nothing to fix there.
* After warm-up, TTFT is the decode-bound floor: 1 forced prompt token
  (~7 ms) + one decode step (~55 ms at 15–16 tok/s, Q8_0, 8 threads).
  Further TTFT gains now require faster decode (quant tier — §6 — or
  speculative decoding), not more caching.
* Full-response time is decode-dominated: with this harness's 64-token cap
  the median wall is ~3.9 s (the synthetic continuations often reach the
  cap); with the product's real output lengths (median ≈ 26 tokens on DEV)
  expect ≈ 0.2 s prefill-free + ~1.7 s decode. The DEV75 run below uses the
  real panel.
* 4-thread (laptop-like) row for the recommended flow: TTFT p50 67.0 ms
  (vs 64.6 at 8 threads — the floor survives), warm-up CPU +58%
  (661 s vs 418–441 s per 220-event pass), wall p50 2.15 s.

Warm-up CPU cost: 440.6 s per pass (200 warm-ups, ≈ 2.0 s average; every
prefill the baseline would have done at pause time, done earlier). The
gentler settle-only policy (warm everything except keystrokes) gives
edit p50 195.4 ms / type 229.6 ms for 392.8 s — per-keystroke warm-ups
cost only +6% CPU and buy 3.5× on typing events, so eager is the default
and settle-only is the battery-saving fallback.

## 3. Levers, measured

| Lever | Verdict | Evidence |
| --- | --- | --- |
| **Warm-up request flow (n_predict 0)** | **Adopt (eager)** | §2. Stable across two freeze versions, three runs, and 4/8 threads. |
| Server flags (`-t/-tb/-fa/-ub/--poll`) | Keep defaults; `-t` = all cores available | llama-bench sweep: pp512 87.7 (8T) vs 75.1 (6T) vs 61.9 (4T); `-fa on/off` within noise; `-ub 2048` +5–6% pp2048 only; decode best at 8T. `-ub` changes ubatch split → numerics, not worth 5%. |
| `--cache-reuse 256` (KV-shift chunk reuse) | Reject | ~10% apparent gain was host-load noise (baseline rerun at matched load reproduced it without the flag); cached fractions for accept/cursor unchanged (0.432/0.175); parity 0.9909. Identical-chunk requirement is not met by window shifts or history insertion. |
| KV cache type `-ctk/-ctv q8_0` | Reject on CPU | pp2048 61.8 vs 85.4 tok/s (f16) — quantize/dequant overhead dominates with 2 KV heads; also numerics-changing. |
| `--ctx-checkpoints` / KV pool | Keep defaults | switch_back already ~free via the default `--cache-ram` pool (checked: 0.998 cached). On RAM-tight laptops cap it (`--cache-ram 2048`). |
| Prompt-layout reordering (retraining proposal) | **Negative — do not spend a training round on ordering alone** | Token-level LCP simulation on the frozen sessions: pooled cached p50 0.801 (PRM03) vs 0.788 (stable-first: diagnostics-before-history hurts diag events) vs 0.801 (history capped to 1 event). Cursor moves stay 0.23 under every ordering: the anchored prefix itself grows, so divergence lands at the prefix end regardless of section order. The mid-prompt volatility problem is content change, not section order. |
| Warm-up overlap safety | Solved — no abort needed; optionally a second slot | Fresh-server probes (2438-token cold prompt): a client-side abort does **not** cancel an in-flight prefill nor free the slot (follow-up request queued 36.6 s; the warm-up runs to completion server-side, and its finished/aborted KV is what persists — the original "69 ms after abort" was the KV pool restoring an earlier full context). With `--parallel 2`: cross-slot pool restore gives a slot-0 request the slot-1-warmed prefix (91.6 ms, cache_n 2437); auto slot selection avoids the busy slot (195 ms); concurrent prefill costs CPU contention (~1.7 s) instead of queuing (30.9 s cold control). |

## 4. Recommended llama-server spawn args and request flow

For the extension's PRM03 port. Server side is unchanged from today except
explicitness; the win is the request flow.

```bash
llama-server \
  -m <model.gguf> \
  --alias sepalith \
  --temp 0 \
  --host 127.0.0.1 --port <port> \
  -c 16384 \
  --parallel 1 \
  -t <all-physical-cores> \
  -ngl 0 \
  --cache-ram 2048        # cap the KV pool on 16-32 GB laptops (default 8192)
```

* Do **not** add `--cache-reuse`, `--no-cache-prompt`, `-ub`, `-fa`, or
  KV-type flags: measured neutral or negative (§3).
* Keep `--parallel 1`: one slot, one file context, warm-ups and requests
  share it.

Request flow (the important part):

1. Send prompts as **token IDs** (`/completion`, `prompt: [ids]`,
   `return_tokens: true`, `temperature 0`, `top_k 1`, fixed seed, stream).
   Render with `render_prompt` + the pinned tokenizer (BOS 0 manual, no
   special parsing) so serving tokenization matches training exactly.
2. **Warm-up after every document change**: file open, cursor settle,
   accepted suggestion, manual edit, diagnostic arrival, and the stable
   heads of the other open editors (file-switch targets). Request body =
   the current prompt token IDs with `n_predict: 0` (prefill only, no
   decode, no output). During typing, debounce (~150–300 ms after the last
   keystroke) — a keystroke warm-up only pays the few-token delta.
   **Refresh matters:** skipping the refresh after a change re-exposes the
   structural prefill cost for that event (measured race bound, §2) — the
   debounce must be shorter than typical pause gaps for the floor to hold.
3. **On pause: just send the real request** — do not bother aborting an
   in-flight warm-up: llama.cpp does not cancel an underway prefill on
   client disconnect, and the queued real request lands at the floor as
   soon as the warm-up finishes (its KV is exactly the prompt you want).
   If pauses frequently race settle warm-ups, add `--parallel 2` (KV
   memory doubles, ~1.4 GiB) and keep at most one warm-up in flight: the
   pause request then takes the free slot immediately and the unified KV
   pool restores the warmed prefix across slots (measured 92 ms vs 30.9 s
   cold; concurrent prefill costs ~1.7 s of CPU contention worst case, vs
   queuing behind it).
4. Suggestion display streams from the first token (TTFT ≈ 64 ms); the full
   suggestion is decode-bound (~26 tokens ≈ 1.7 s at Q8_0/8T).

CPU/battery note: eager warm-up conserves total FLOPs (it is the same
prefill, moved into idle time) but concentrates CPU during typing bursts;
the settle-only variant (skip per-keystroke warm-ups) trades typing-event
TTFT 64→230 ms for −6% warm-up CPU if battery profile demands it.

## 5. Output parity

Reference semantics matter here: llama.cpp recomputes nothing for a cached
prefix, but the *split* of prefill across requests changes GEMM tiling, so
**any** cross-request prefix caching (including the shipped
`--cache-prompt` behavior) changes temp-0 outputs vs a cold, single-shot
reference:

| Run | Agreement vs cold ref | Agreement vs baseline session-flow |
| --- | ---: | ---: |
| Baseline session flow (today's effective behavior) | 0.9455 (freeze v2) | 1.0 |
| Recommended (eager warm-up) | 0.9136 (freeze v2) | 0.9455 |
| `--cache-reuse 256` (freeze v1) | — | 0.9909 |

Mismatching outputs are alternative near-tie continuations (verified by
decoding; both sides protocol-valid), not truncations or degeneration. The
recommended flow's added drift (−0.9 pp vs the session-flow baseline) is
the same mechanism the product already ships. Flag-level numerics are
untouched (identical spawn flags to the reference), so no gguf_dev_eval
confirmation slot was spent; if the team wants a hard gate anyway, run the
§6 DEV quality A/B once (one of the 3 allowed confirmations) comparing
default flags vs default flags + session-flow caching.

## 6. Quant-tier prefill/decode speed table (for the quant mission)

llama-bench, CPU b10453, 8 threads (4-thread rows marked), fa on, r=3,
host load ~4. tok/s.

| Tier | pp512 | pp2048 | tg32 |
| --- | ---: | ---: | ---: |
| bf16 | 131.4 | 125.3 | 9.22 |
| Q8_0 | 129.4 | 132.1 | 16.12 |
| imat-Q6_K | 120.5 | 113.5 | **21.12** |
| imat-Q5_K_M | 79.7 | 64.8 | 21.36 |
| imat-Q4_K_M | 116.8 | 106.1 | 15.80 |
| uncal-Q4_K_M | 111.5 | 106.1 | 18.07 |
| q4plus-Q4_K_M | 114.7 | 108.5 | 18.81 |
| attn-out-q8_0-Q4_K_M | 110.8 | 101.5 | 18.96 |
| head-q6_k-Q4_K_M | 116.6 | 108.8 | 20.61 |
| Q8_0 @4T | 63.2 | 55.4 | 12.28 |
| imat-Q4_K_M @4T | 75.2 | 67.7 | 15.51 |

Reading: prefill is compute-bound and roughly flat across 8-bit/4-bit
tiers (~106–132 tok/s; Q5_K_M is the outlier — slow dot path on this
build/CPU). Decode is bandwidth-bound but **Q4_K_M does not beat Q8_0 at
8 threads on this build** (15.8 vs 16.1 — the int8 paths are already
fast); **Q6_K is the decode winner (+31% over Q8_0)**. With the warm-up
flow, TTFT is decode-dominated, so the tier choice now moves the *floor*:
Q6_K would take the 64 ms floor to ~50 ms and full-suggestion decode from
~1.7 s to ~1.3 s at similar prompt speed. Fidelity calls stay with the
quant mission.

## 7. DEV75 final latency check (single allowed DEV latency use)

Usage log: DEV75 (`corrected-dev75-v1`, sha `7c144bcd…`, 75 cases) read
on 2026-10-17 for this latency pair only (two runs: baseline flags, then
warm-up flow; latency-only, no quality scoring). No other DEV reads by
this mission; DEV250 untouched; the 3 quality-confirmation slots unused.

Each case = one cold `open` event — the reviewer-walks-unrelated-files
worst case (no cross-request reuse possible), `n_predict` 96, 8 threads,
2 passes, host load ≈ 8:

| Config | TTFT p50 | TTFT p95 | wall p50 | prompt tokens p50 | generated p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline flags | 8183 ms | 17398 ms | 10097 ms | 1169 | ~26 tok (EOS-terminated) |
| + warm-up flow | **66.2 ms** | **72.9 ms** | **1724 ms** | 1169 | same |

**124× on the median cold open; 239× on p95.** Full suggestion in ~1.7 s
(decode-bound). Warm-up CPU: 320 s per pass for 75 prompts (≈ 4.3 s each,
these are big prompts under load) — all of it movable to idle time before
the pause. This is the number the PRM03 port should advertise: real DEV
prompts, no cross-file reuse assumed.

## 8. Proposals needing a user decision

1. **Prompt-layout retraining: recommend NOT for ordering alone.** The
   simulation (§3) shows section reordering buys ~nothing under byte-prefix
   caching; cursor-move recompute is caused by prefix growth, accept by
   history insertion — both content changes at the divergence point, which
   no ordering removes. A retraining round is better spent elsewhere.
2. **Prompt budget:** TTFT scales ≈ linearly with uncached prompt tokens
   (~7.7 ms/token at 8T, Q8_0). Halving `MAX_PREFIX_SUFFIX_CHARS` would
   halve cold/warm-up cost but changes what the model sees (quality risk,
   TRAIN-based estimate required — not measured here). With warm-up, the
   budget no longer affects pause latency at all; recommend leaving it
   until a quality-motivated change.
3. **llama.cpp upgrade:** b10453 already provides everything used here
   (prompt caching, KV pool, checkpoints). No upgrade needed for this win;
   revisit only if a future release ships faster Q6_K/Q4_K repack kernels
   (would move the decode floor).
4. **Vulkan/iGPU:** untestable on this box (CUDA GPU off-limits and not
   laptop-representative). Collect evidence on a real laptop; an iGPU
   could raise the decode floor independently of this work.
5. **Quant tier for serving:** hand §6 to the quant mission; Q6_K is the
   latency-favorable candidate if its gate passes.

## 9. Artifacts

* Code: `experiments/serving/prefill/` (build_sessions.py, bench.py,
  parity.py, probe_interrupt.py, layout_sim.py, build_dev_panel.py,
  flag_sweep.sh, quant_speed_table.sh, configs/).
* Frozen workload: `experiments/serving/prefill/sessions.jsonl` +
  `manifest.json` (seed 20261018; TRAIN-only sources, hashes pinned).
* Run records, summaries, server logs, sweep CSVs:
  `/mnt/e/sepalith/campaign-20260915/serving-work/prefill-20261017/`.
* Machine-readable results:
  `docs/validation/2026-10-17-prefill-autoresearch.json`.
* Parity references: `experiments/serving/prefill/reference_outputs.json`
  (session-flow) and `reference_outputs_cold.json` (per-event cold).
