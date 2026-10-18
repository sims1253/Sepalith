# Quant autoresearch: a smaller, faster, more faithful arm-A GGUF tier

Branch `research/quant-autoresearch-20261017` (from `main` @ `c42dc8a`).
Report date: 2026-10-17/18 (DEV confirmations completed 06:45 on the 18th).
Mission brief: `/home/m0hawk/sandbox/sepalith-quant-autoresearch-brief.md`.

## TL;DR

Four mixed-precision Q4_K_M recipes Pareto-dominate the current Q4_K_M imatrix
baseline (1,759.8 MB, CPU KLD 0.033973) — each smaller, more faithful to bf16,
and decode-equal to the baseline within measurement noise. The two "e4"
variants take another ~33 MB off for ≈free (token_embd q4_K costs only
+0.0002 mean KLD; the q4_K penalty is ~98% the output head). **Q4_K_M-mixA-e4
passes the entire gate — the first tier in the campaign to do so** (including
`valid_drop`, by a 0.2 pp margin; see the honesty note in §7 on why that
margin is partly cap-flip luck).

| recipe | MB | mean KLD (CPU) | top-1 | DEV250 gate | exact Δ | nofp Δ |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| **Q4_K_M-mixA-e4 "ship"** | **1,590.4** | 0.026692 | 94.92 | **ALL CHECKS PASS** | +0.4 | −1.2 |
| Q4_K_M-mixA-e4-ao5 (TRAIN-only) | 1,567.0 | 0.028049 | 94.74 | gate-passing floor, no DEV slot | | |
| Q4_K_M-mixA | 1,623.8 | 0.026501 | 95.00 | all but valid_drop (2.4) | +0.8 | −1.6 |
| **Q4_K_M-mixB-e4 "fidelity"** | 1,658.5 | 0.023328 | 95.18 | TRAIN-only evidence | | |
| Q4_K_M-mixB | 1,692.0 | 0.022949 | 95.31 | all but valid_drop (2.4) | 0.0 | −0.8 |

`valid_drop_pp` fails for the confirmed tiers (2.4 pp) — like **every tier in
the 2026-10-17 night run, including Q8_0** (and the CPU-vs-CUDA bf16 references
differ by 5 cap cases with zero quantization). Section 6 shows that check is
structurally unpassable for any quantization of this model at DEV250's cap
rate; it needs a user decision, not a recipe.

**Speed honesty note (2026-10-18 correction):** an earlier revision claimed
"mixA +42% decode" from a bench set taken while two other missions' servers
saturated the box. Interleaved same-conditions reruns show the q4 family's
decode differences are **within shared-box contention noise** (run-to-run
flips of ±5–15%; the bytes-per-token mechanism bound is ±4%): one quiet-ish
interleaved set measured Q4_K_M 23.5 / mixA 24.4 / mixA-e4 24.3 tok/s, a later
set flipped orderings. The prefill mission's low-load table (their conditions)
still shows the head *type* matters (q8_0 head: Q4_K_M 15.8 vs head-q6_k 20.6
tg32 — a llama.cpp matvec-path effect, not bandwidth). The mixes' surviving
claims are the deterministic ones — **size and fidelity**; decode is at worst
baseline-equal.

Recipe definitions (see `variant_specs()`):

- **mixA**: `token_embd`/`output` **q5_K**, `attn_output` **q6_K**, `attn_k`/`attn_v`
  **q8_0** (all 42 layers), everything else Q4_K_M default.
- **mixB**: mixA + `ffn_down` **q6_K** on all 42 layers.
- **-e4**: the same with `token_embd` **q4_K** (output head stays q5_K — the
  head is where the q4_K penalty lives).

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
| **ffn_up → q6_K (all 42)** | +136.3 | **−0.00782** | **0.57e-4** | **surprise: the FFN sensitivity hotspot — inverts the usual down>up convention; powers mixC** |
| ffn_gate → q6_K (all 42) | +136.3 | −0.00524 | 0.38e-4 | 67% of ffn_up's gain; FFN order settled: up > down > gate |
| ffn_down → q6_K (all 42) | +68.1 | −0.00330 | 0.48e-4 | but −30% CPU decode (bandwidth) |
| token_embd q8_0 → q6_K | −64.8 | +0.00021 | 0.03e-4 | essentially free |
| token_embd+output → q5_K | −71.1 | +0.0017 | 0.24e-4 | still cheap |
| token_embd q5_K → q4_K | −33.4 | +0.00019 | 0.06e-4 | free (the q4 penalty is ~98% the head) |
| token_embd+output → q4_K | −66.9 | +0.0079 | 1.2e-4 | too expensive; q5_K is the floor |
| attn_output q6_K → q5_K | −23.4 | +0.00136 | 0.58e-4 | gate-passing floor lever (1567.0 MB) |
| IQ4_XS base (anchor) | −129.5 | +0.0097 | 0.07e-4 | direction closed (dominated by mixes) |
| ffn_gate+ffn_up → iq4_xs | −33.0 | +0.0035 | 1.1e-4 | fails gate; the FFN halves at iq4_xs are where the IQ4_XS base lost fidelity |
| ffn_down → q6_K, imatrix-top-11 of 21 | +35.7 | −0.00154 | 0.4e-4 | interior point; gain ~linear in layers, not importance-ranked |

Boosts are sub-additive (measured composites land 5–10% above the sum of
deltas).

## 4. Pareto table (CPU KLD, 64 chunks)

| variant | MB | KLD | top-1 | p99 KLD | ΔPPL | over budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q6_K anchor | 2,199.7 | 0.004104 | 97.99 | 0.047 | +0.033 | +440 |
| mixB-e4 + ffn_up q6 (map probe) | 1,794.8 | 0.015440 | 95.98 | 0.169 | +0.052 | +35 |
| **Q4_K_M-mixC** | **1,726.7** | **0.018876** | 95.67 | 0.211 | +0.064 | −33 |
| **Q4_K_M-mixB** | **1,692.0** | **0.022949** | 95.31 | 0.246 | +0.068 | −68 |
| Q4_K_M-mixA | 1,623.8 | 0.026501 | 95.00 | 0.287 | +0.083 | −136 |
| C2 (mixA w/ emb+head q6_K) | 1,694.9 | 0.024784 | 95.25 | 0.283 | +0.074 | −65 |
| Q4_K_M+ | 1,779.0 | 0.030182 | 94.68 | 0.357 | +0.095 | +19 |
| IQ4_XS plain | 1,630.3 | 0.043625 | 93.88 | 0.509 | +0.117 | −130 |
| Q4_K_M baseline | 1,759.8 | 0.033973 | 94.35 | 0.386 | +0.093 | 0 |
| C7 (mixB w/ emb+head q4_K) | 1,625.1 | 0.030851 | 94.05 | 0.289 | +0.095 | −135 |

(2026-10-18 additions: the e4/ao5/mixC rows — mixB-e4 1,658.5 @ 0.023328,
mixA-e4 1,590.4 @ 0.026692, mixA-e4-ao5 1,567.0 @ 0.028049, and mixC above.)

llama-bench (8T, `-fa on`, r≥2; **shared box — relative reading only; see the
speed honesty note in the TL;DR**):

| model | pp2048 (load ~16) | tg32 (load ~16) | tg32 interleaved (load ~11) |
| --- | ---: | ---: | ---: |
| Q4_K_M baseline | 87.8 | 10.09 | 22.2–23.6 |
| mixA | 83.1 | 14.29 | 17.0–24.5 |
| mixA-e4 | — | — | 20.8–24.6 |
| mixB | 80.5 | 9.97 | — |
| Q8_0 (context) | — | — | 13.4–15.1 |

Reading: q4-family decode differences do not separate from contention noise
on this host (orderings flip between interleaved reps); Q8_0 reads consistently
slower than every q4 mix in the interleaved sets; prefill is flat across the
q4 family. The prefill mission's §6 table (their low-load conditions) remains
the reference for tier-level comparisons.

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

Search-panel evidence (TRAIN, no DEV reads; 196 cases, 7 families × 28, greedy, cap 1,024,
bf16 CPU reference; `panel/agreement-*.json`):

| candidate | identical with bf16 | net cap rise | done→cap | cap→done | median Δlen | p90 Δlen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q4_K_M baseline (incumbent)** | 76.0% | **0** | 0 | 0 | 0 | 0 |
| Q4_K_M-mixA | **78.1%** | **+1** | 1 | 0 | 0 | +1 |
| Q4_K_M-mixB | 77.6% | **−2** | 0 | 2 | 0 | 0 |
| Q4_K_M-mixA-e4 (ship) | 76.0% | **0** | 1 | 1 | 0 | +2 |
| Q4_K_M-mixC (fidelity) | **78.1%** | **0** | 1 | 1 | 0 | +1 |

**Honesty finding (baseline panel, 2026-10-18):** the incumbent baseline scores
76.0% identity / net cap rise 0 on this panel — statistically indistinguishable
from mixA-e4 (76.0%/0; 44 of their 47 changed cases are even the same cases),
with the mixes at 77–78% (mixC tops the set at 78.1% with zero net cap rise
— the −44% KLD buys +2 pp greedy identity, modest but monotone). A −21% KLD improvement does **not** translate to
materially better greedy agreement at TRAIN-length generations. What the
recipes measurably buy is: the **KLD gate check** (baseline 0.0340 **fails**
0.03; every mix passes with 11–23% margin), **size** (−68 to −193 MB), the
exact/nofp/ship-bound checks (all pass, some improved), and — for mixA-e4 —
the first full gate pass. Cap behavior specifically is stochastic per
quantization (the baseline nets +10 caps on DEV250/CUDA but 0 on TRAIN).
Corroboration caveat updated accordingly: mixA-e4's zero net cap rise on TRAIN
matches the baseline's, so it neither strengthens nor weakens the §7 pass.

For scale: on the night run's DEV250 (CUDA), Q4_K_M was 51.2% identical with
bf16 (+10 net caps) and Q8_0 66.8% (+6). mixA's TRAIN-panel fidelity is far
above the baseline's, consistent with its KLD.

## 7. DEV confirmations (≤3; every DEV read logged here)

Gate runs on the **CPU build** (the search build) with a CPU bf16 reference,
via `quantize_and_gate.py`/`quant_gate.py`; receipt:
`quant-work/autoresearch-20261017/gate-receipt.json`. CPU numbers are not
comparable with the night CUDA receipt (the CUDA-confirmation night job is
included for that: `c09/jobs/C09-arm-a-quant-mix-confirmation.json`).

Every DEV read this session (all CPU, 2026-10-17/18, gate-eligible full
panels, cap 1,024): bf16 dev250+dev75 (reference), Q4_K_M-mixA dev250+dev75,
Q4_K_M-mixB dev250+dev75. **Two candidate confirmations of the three
allowed; the third slot is unused.** No other DEV access; no gold scoring.

CPU bf16 reference: exact 117, valid 234, caps 16 (the CUDA night reference
had 117/239/11 — same exacts; the cap boundary itself moves 5 cases between
backends, more evidence for §6).

| check | Q4_K_M-mixA | Q4_K_M-mixB | q4 threshold |
| --- | ---: | ---: | ---: |
| exact Δpp (lost/gained) | **+0.8** (2/4) | 0.0 (1/1) | ≥ −2.0 pass |
| McNemar p | 0.69 pass | 1.0 pass | fail only if p≤.05 and lost>gained |
| nofp rise pp | **−1.6** pass | −0.8 pass | ≤ 1 pass |
| valid_drop pp (caps 16→) | 2.4 (→22) **fail** | 2.4 (→21) **fail** | ≤ 1 |
| mean KLD (gate re-measure) | 0.026501 pass | 0.022949 pass | ≤ 0.03 pass |
| ship net-loss bound pp | 1.05 pass | 1.31 pass | ≤ 3 pass |
| token identity % | 52.4 recorded | **54.8** recorded | recorded |
| DEV75 parity | out (nofp +3, recorded) | within tolerance | recorded |

The single failing check for mixA/mixB is the one §6 shows no quantization can
pass reliably (Q8_0 itself fails it by the same margin). mixA-e4 landed on
the right side of it (§7). Notably, on the KLD/checks that are passable, the
mixes strictly dominate the Q4_K_M baseline profile (baseline CUDA: exact
0.0, nofp −0.4, KLD 0.0342 **fail**, identity 51.2).

**Q4_K_M-mixA-e4 (third and final slot, 2026-10-18): FULL GATE PASS —
verdict `pass`, shippable `True`.** exact +0.4 pp (117→118, 1 lost / 2
gained), McNemar 1.0, nofp −1.2 pp, **valid_drop 0.8 pp (caps 16→17, net
+1)**, mean KLD 0.026692, ship bound 1.06 pp, token identity 53.6%, DEV75
parity within tolerance.

Honesty note on that valid_drop pass: the margin (net +1 cap vs mixA's +6 on
the same reference) is inside the cap-flip noise band §6 quantifies; mixA has
slightly *lower* KLD than mixA-e4, so the pass is not attributable to the
embedding trim. The receipt is the receipt — but the threshold's
noise-dominance (and Q8_0's own failure) still warrants the user decision in
§7. The deterministic claims for mixA-e4 are: −169.4 MB vs baseline, KLD
−21% vs baseline and under the 0.03 gate with margin, exact/nofp/ship-bound
all passing.

| run | date | panel | build | purpose |
| --- | --- | --- | --- | --- |
| bf16 dev250+dev75 | 2026-10-17/18 | dev250, dev75 | cpu | gate reference |
| Q4_K_M-mixA dev250+dev75 | 2026-10-18 | dev250, dev75 | cpu | confirmation 1 of 3 |
| Q4_K_M-mixB dev250+dev75 | 2026-10-18 | dev250, dev75 | cpu | confirmation 2 of 3 |
| Q4_K_M-mixA-e4 dev250+dev75 | 2026-10-18 | dev250, dev75 | cpu | confirmation 3 of 3 — **full pass** |

## 8. Recommendation

Ship **Q8_0 + Q4_K_M-mixA-e4** as the small tier: **the only tier to pass the
entire gate** (DEV250 receipt in the work dir), −169.4 MB vs Q4_K_M, KLD
0.0267 ≤ 0.03 with 11% margin, exact +0.4, decode baseline-equal. If the
user prefers more fidelity headroom and accepts a TRAIN-evidenced (not
DEV-confirmed) variant, **mixC** (2026-10-18 late: 1,726.7 MB, −33 vs
baseline; KLD 0.0189, **−44% vs baseline** and 37% under the gate;
reproduced on slice-2 at 0.0182) supersedes mixB-e4 as the fidelity option —
it differs from the DEV-confirmed mixA-e4 by exactly one lever (ffn_up q6_K,
−0.0078 on both slices); mixB-e4 (−101.3 MB, 0.0233) remains the mid choice.
The emb-q5
variants (mixA/mixB) remain in `variant_specs()` as the DEV-confirmed forms;
Q6_K stays the quality fallback at 2.2 GB. Given the §7 honesty note, treat
mixA-e4's valid_drop pass as real-but-lucky; the standing §10 question about
that threshold remains worth deciding anyway.

## 9. Robustness: the frontier is not slice-specific

Every search decision above was made on the v1 held-out KLD slice. To rule
out slice-specific overfitting of the *search*, a **second, disjoint TRAIN
slice** (32 × 1,024, seed 20261018, same builder machinery, excluding all
calibration + v1-KLD + search-panel rows; `c09/build_kld_slice2.py`, llama-
tokenize round-trip verified) re-measured the frontier:

| variant | slice-1 KLD | slice-2 KLD | rank s1 → s2 |
| --- | ---: | ---: | --- |
| Q4_K_M baseline | 0.033973 | 0.033500 | 4 → 4 |
| mixA-e4-ao5 | 0.028049 | 0.027803 | 3 → 3 |
| mixA-e4 | 0.026692 | 0.026057 | 2 → 2 |
| mixB-e4 | 0.023328 | 0.022751 | 1 → 1 |
| **mixC** | **0.018876** | **0.018214** | rank 1 of the extended set, both slices |

Rank order and pairwise deltas reproduce (deltas within 8%; e.g. baseline →
mixA-e4: −21.4% vs −22.1%). The ship tier's fidelity margin is a property of
the recipes, not of the slice they were tuned on.

## 10. Open questions for the user

1. **valid_drop threshold** (section 6): structurally unpassable as written,
   including for Q8_0. Proposal: relax to ≤3 pp for every tier (≈ the Q8_0
   noise floor), or drop the check in favor of exact/McNemar/nofp/KLD.
2. **Night-queue job** for CUDA confirmation of mixA/mixB (comparability with
   the night receipt) — `C09-arm-a-quant-mix-confirmation.json`.
3. llama.cpp upgrade: not needed for these recipes (b10453 suffices); revisit
   only if a future build ships faster Q6_K/Q5_K head matvec paths (would move
   mixB's decode).
4. IQ4_XS: closed here (dominated); not worth thresholds.

## 11. Artifacts

- GGUFs + receipts + KLD logs + benches:
  `/mnt/e/sepalith/campaign-20260915/quant-work/autoresearch-20261017/`
  (finalists `ggufs/c4-…` and `ggufs/c6-…`; bf16 CPU KLD base `kld/bf16-base-cpu.kld`).
- Search panel: `panel/search-panel.json` (196 TRAIN cases, sha
  `11c4f32c…`), results under `panel/`.
- Machine-readable results: `docs/validation/2026-10-17-quant-autoresearch.json`.
- Code: `variant_specs()` additions (`Q4_K_M-mixA/mixB`),
  `sepalith/quant/search_panel.py`, `c09/search_panel_eval.py`,
  night job JSON.
