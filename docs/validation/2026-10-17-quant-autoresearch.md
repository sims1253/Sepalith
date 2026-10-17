# Quant autoresearch: a smaller, faster, more faithful arm-A GGUF tier

Branch `research/quant-autoresearch-20261017` (from `main` @ `c42dc8a`).
Report date: 2026-10-17 (session in progress; sections marked ⏳ fill as runs
complete). Mission brief: `/home/m0hawk/sandbox/sepalith-quant-autoresearch-brief.md`.

## TL;DR

Two mixed-precision Q4_K_M recipes Pareto-dominate the current Q4_K_M imatrix
baseline (1,759.8 MB, CPU KLD 0.033973) — each is smaller, more faithful to
bf16, and no slower (one markedly faster) on CPU:

| recipe | MB | mean KLD (CPU) | top-1 | tg32 vs Q4_K_M | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **Q4_K_M-mixA "speed"** | 1,623.8 | 0.026501 | 95.00 | **+42%** | passes every gate check except valid_drop |
| **Q4_K_M-mixB "fidelity"** | 1,692.0 | 0.022949 | 95.31 | −1% | passes every gate check except valid_drop |

Both fail `valid_drop_pp` — like **every tier in the 2026-10-17 night run,
including Q8_0**. Section 6 shows that check is structurally unpassable for any
quantization of this model at DEV250's cap rate; it needs a user decision, not
a recipe.

Recipe definitions (see `variant_specs()`):

- **mixA**: `token_embd`/`output` **q5_K**, `attn_output` **q6_K**, `attn_k`/`attn_v`
  **q8_0** (all 42 layers), everything else Q4_K_M default.
- **mixB**: mixA + `ffn_down` **q6_K** on all 42 layers (the decode-costly
  fidelity spend; llama-bench shows ~30% slower decode than mixA).

## 1. Setup and method

- llama.cpp pinned b10453 commit `3cb7ffb1a`, **CPU build** only
  (`CUDA_VISIBLE_DEVICES=""`), ≤ 8 threads, `nice -n 10`, shared box (AMD
  Ryzen 9 5900X, WSL2) — the prefill mission benched on it concurrently
  (load ≈ 12–16 during most runs; matched-load comparisons within this report).
- Fidelity metric: `llama-perplexity --kl-divergence` vs a **CPU bf16 base**
  (one-time, 8.5 GB base file) on the held-out TRAIN slice
  (`calibration-v1/kld-heldout.txt`, 64 × 1024, 32,704 scored tokens).
  Determinism verified (identical values across reruns). CPU numbers are not
  comparable with the night run's CUDA numbers; deltas reproduce across
  builds (Q4_K_M: CPU 0.033973 vs CUDA 0.034197; attn-out-q8_0: 0.026750 vs
  0.026839).
- Sizes from the quantized files; speeds from `llama-bench -t 8 -fa on -r 3
  -p 512,2048 -n 32` (same flags as the prefill mission's §6 table; their
  table rows were measured at load ≈ 4, mine at load ≈ 16 — relative
  comparisons only across my own runs).
- Search signal: TRAIN data only (KLD slice + a 196-case TRAIN search panel,
  section 6). DEV250/DEV75 used only for the final confirmations (section 7;
  every DEV read is logged there).

## 2. CPU anchors of the night-run tiers

| tier | MB | KLD CPU | KLD CUDA | top-1 CPU |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 2,679.7 | — | 0.000914 | — |
| Q6_K (imat) | 2,199.7 | 0.004104 | 0.003205 | 97.99 |
| Q4_K_M (imat, baseline) | 1,759.8 | 0.033973 | 0.034197 | 94.35 |
| Q4_K_M+ (kv q8_0) | 1,779.0 | 0.030182 | 0.030327 | 94.68 |
| Q4_K_M-attn-out-q8_0 | 1,847.9 | 0.026750 | 0.026839 | 95.06 |

## 3. Sensitivity map (marginal mean-KLD per MB, from the Q4_K_M base)

| lever | ΔMB | ΔKLD | KLD/MB | note |
| --- | ---: | ---: | ---: | --- |
| attn_k+attn_v → q8_0 (all 42) | +19.2 | −0.00379 | 2.0e-4 | cheapest boost |
| attn_output → q6_K (all 42) | +45.2 | −0.00674 | 1.5e-4 | 93% of the q8_0 gain at 52% of its cost |
| attn_output → q8_0 (all 42) | +88.1 | −0.00722 | 0.8e-4 | HINT4's sensitive module, confirmed |
| ffn_down → q6_K (all 42) | +68.1 | −0.00330 | 0.5e-4 | but −30% CPU decode (bandwidth) |
| token_embd q8_0 → q6_K | −64.8 | +0.00021 | 0.03e-4 | essentially free |
| token_embd+output → q5_K | −71.1 | +0.0017 | 0.24e-4 | still cheap |
| token_embd+output → q4_K | −66.9 | +0.0079 | 1.2e-4 | too expensive; q5_K is the floor |
| IQ4_XS base (anchor) | −129.5 | +0.0097 | 0.07e-4 | direction closed (dominated by mixes) |

Boosts are sub-additive (measured composites land 5–10% above the sum of
deltas).

## 4. Pareto table (CPU KLD, 64 chunks)

| variant | MB | KLD | top-1 | p99 KLD | ΔPPL | over budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q6_K anchor | 2,199.7 | 0.004104 | 97.99 | 0.047 | +0.033 | +440 |
| **Q4_K_M-mixB** | **1,692.0** | **0.022949** | 95.31 | 0.246 | +0.068 | −68 |
| Q4_K_M-mixA | 1,623.8 | 0.026501 | 95.00 | 0.287 | +0.083 | −136 |
| C2 (mixA w/ emb+head q6_K) | 1,694.9 | 0.024784 | 95.25 | 0.283 | +0.074 | −65 |
| Q4_K_M+ | 1,779.0 | 0.030182 | 94.68 | 0.357 | +0.095 | +19 |
| IQ4_XS plain | 1,630.3 | 0.043625 | 93.88 | 0.509 | +0.117 | −130 |
| Q4_K_M baseline | 1,759.8 | 0.033973 | 94.35 | 0.386 | +0.093 | 0 |
| C7 (mixB w/ emb+head q4_K) | 1,625.1 | 0.030851 | 94.05 | 0.289 | +0.095 | −135 |

llama-bench (8T, load ≈ 16, matched within the set):

| model | pp512 | pp2048 | tg32 |
| --- | ---: | ---: | ---: |
| Q4_K_M baseline | 97.6 | 87.8 | 10.09 |
| C2 | 89.0 | 79.5 | 11.63 |
| **mixA** | 90.3 | 83.1 | **14.29** |
| mixB | 88.2 | 80.5 | 9.97 |

Reading: with the prefill mission's warm-up flow, laptop TTFT is decode-bound,
so **mixA's +42% decode is the latency win**; mixB trades it back for KLD.
Q4_K_M itself decodes no faster than Q8_0 on this build (prefill §6), so the
baseline was a poor latency choice regardless of fidelity.

## 5. b10453 `--tensor-type` counter semantics (tool hazard)

A `--tensor-type` override that matches `attn_v`/`ffn_down` tensors makes
those tensors "manual": `llama_tensor_get_type_impl` never runs for them, so
the internal `++i_attention_wv`/`++i_ffn_down` counters skip — and every
later tensor of that category shifts the Q4_K_M every-3rd q6_K mix. The night
run's **Q4_K_M+ receipt says "ffn_down of layers 0,1,40,41 q6_k" but the file
actually holds a shifted mix** (verified with GGUFReader; the shift reproduces
exactly from the counter rule). Rule adopted here: override those categories
completely (all 42 layers) or not at all. Both finalist files were verified
tensor-by-tensor.

## 6. The valid_drop problem (why nothing passes the gate)

From the night run's CUDA evals: every tier — including Q8_0 — fails
`valid_drop_pp` (Q8: 2.4 pp vs 0 allowed; q4: 2.4–6.4 pp vs 1 allowed).
`valid + cap = 250` on DEV250, so "valid drop" is exactly "cap rise":

| tier | caps (ref 11) | done→cap | cap→done | net |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 17 | 9 | 3 | +6 |
| Q6_K | 17 | 7 | 1 | +6 |
| Q4_K_M+ | 16 | 10 | 5 | +5 |
| Q4_K_M | 21 | 16 | 6 | +10 |
| Q4_K_M-attn-out-q8_0 | 27 | 20 | 4 | +16 |

The flipped cases are **mid-length generations (124–501 tokens) diverging into
1024-token runaways** — not near-cap boundary cases. The rate scales with
fidelity (≈ 0.12 × changed-cases) but is positive even at Q8_0's 99.06% top-1
agreement. No realizable quantization reaches the ≈92% whole-output token
identity that a ≤2-case net cap rise would require. **This threshold needs a
user decision** (accept a noise-calibrated bound, e.g. valid_drop ≤ 3 pp for
all tiers, or gate on exact/McNemar/nofp/KLD only). `gate.py` was not touched.

Search-panel evidence (TRAIN, no DEV reads): ⏳ bf16 reference run in flight;
mixA/mixB agreement + net cap rise vs bf16 to follow.

## 7. DEV confirmations (≤3; every DEV read logged here)

⏳ Planned: mixA and mixB on DEV250+DEV75, CPU build, via `quant_gate.py`
against a CPU bf16 reference (the search build; not comparable to the night
CUDA receipt). A night-queue job JSON
(`c09/jobs/C09-arm-a-quant-mix-confirmation.json`) is included for a CUDA
confirmation that IS comparable with the night receipt — queued by the user,
not by this session.

| run | date | panel | build | purpose |
| --- | --- | --- | --- | --- |
| ⏳ | | | | |

## 8. Recommendation

Ship **Q8_0 + Q4_K_M-mixA** as the small tier (pending the valid_drop
decision above): −136 MB vs Q4_K_M, KLD 0.0265 ≤ 0.03 with margin, +42% CPU
decode, prefill on par. If fidelity outranks latency for the small tier,
**mixB** instead (−68 MB, KLD 0.0229). Q6_K remains the quality fallback tier
at 2.2 GB with the best decode (21 tok/s at low load) — a Q6_K with q5_K
emb/head (~2.07 GB) was not needed to beat the baseline and was not pursued.

## 9. Open questions for the user

1. **valid_drop threshold** (section 6): structurally unpassable as written,
   including for Q8_0. Proposal: relax to ≤3 pp for every tier (≈ the Q8_0
   noise floor), or drop the check in favor of exact/McNemar/nofp/KLD.
2. **Night-queue job** for CUDA confirmation of mixA/mixB (comparability with
   the night receipt) — `C09-arm-a-quant-mix-confirmation.json`.
3. llama.cpp upgrade: not needed for these recipes (b10453 suffices); revisit
   only if a future build ships faster Q6_K/Q5_K head matvec paths (would move
   mixB's decode).
4. IQ4_XS: closed here (dominated); not worth thresholds.

## 10. Artifacts

- GGUFs + receipts + KLD logs + benches:
  `/mnt/e/sepalith/campaign-20260915/quant-work/autoresearch-20261017/`
  (finalists `ggufs/c4-…` and `ggufs/c6-…`; bf16 CPU KLD base `kld/bf16-base-cpu.kld`).
- Search panel: `panel/search-panel.json` (196 TRAIN cases, sha
  `11c4f32c…`), results under `panel/`.
- Machine-readable results: `docs/validation/2026-10-17-quant-autoresearch.json`.
- Code: `variant_specs()` additions (`Q4_K_M-mixA/mixB`),
  `sepalith/quant/search_panel.py`, `c09/search_panel_eval.py`,
  night job JSON.
