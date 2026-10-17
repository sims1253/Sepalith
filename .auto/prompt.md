# Autoresearch: Sepalith arm-A GGUF quantization recipe

## Objective
Find a GGUF quantization recipe for the arm-A model (2B params, 42 layers, vocab
130,560, untied embeddings) that is **smaller and/or faster on CPU than the Q4_K_M
imatrix baseline (1,759,769,728 B) at the same fidelity, or more faithful at the
same size**, measured against the bf16 GGUF. Fidelity = low mean KLD on the
held-out TRAIN slice + same greedy edits. Deliverables: tier specs in
`variant_specs()`, search-panel support, report + JSON in `docs/validation/`,
night-queue job JSON if GPU confirmation needed. PR against main, never merge.

Mission brief: `/home/m0hawk/sandbox/sepalith-quant-autoresearch-brief.md` (read
it — data rules, compute rules, hard stops). Branch: `research/quant-autoresearch-20261017`.

## Metrics
- **Primary**: `kld` — mean KL divergence vs bf16 on the held-out TRAIN KLD slice
  (`calibration-v1/kld-heldout.txt`, 64 chunks × 1024 ctx), **CPU build** with our
  own CPU bf16 base. Lower is better. NOT comparable to the night run's CUDA numbers.
- **Secondary**: `mb` (GGUF size, MB — hard budget for "same size": ≤ 1759.8 MB),
  `top1` (same-top-p %), `p99` (p99 KLD), `ppl_gap` (PPL(Q) − PPL(base)), later
  `pp2048`/`tg32` (llama-bench, from prefill mission's table + own runs for new
  recipes), search-panel greedy agreement with bf16.
- Gate thresholds (do not touch `gate.py`): q4 tier: exact ≥ −2.0 pp, nofp ≤ +1 pp,
  valid_drop ≤ 1 pp, mean KLD ≤ 0.03; q6_q5: −1.0/+1/+1/0.01; q8: −0.4/0/0/0.002.

## How to Run
`./.auto/measure.sh` — reads `.auto/candidate.json`, quantizes it (unless
`existing_gguf` reuses a file), measures KLD on CPU, emits `METRIC` lines.
Artifacts under `/mnt/e/sepalith/campaign-20260915/quant-work/autoresearch-20261017/`.
One iteration ≈ quantize (~2 min) + KLD (~6–10 min CPU, 64 chunks; ~3–5 min at 32).
`.auto/candidate.json` fields: label, tier, output_type, embedding_type,
tensor_types (list of PATTERN=TYPE regex overrides), imatrix (bool), kld_chunks,
existing_gguf (optional absolute path to reuse instead of quantizing).

## Files in Scope
- `.auto/*` — session files.
- `packages/sepalith/src/sepalith/quant/` — search-panel support (new module,
  must NOT loosen the DEV path in `gguf_dev.py`).
- `docs/training/plan-20260927/c09/` — new analysis scripts, `variant_specs()`
  additions in `quantize_and_gate.py` (existing tiers unchanged).
- `experiments/` — analysis scripts if needed.
- `docs/validation/2026-10-17-quant-autoresearch.{md,json}` — final report.
- `/mnt/e/sepalith/campaign-20260915/quant-work/autoresearch-20261017/` — artifacts.

## Off Limits
- `packages/sepalith/src/sepalith/quant/gate.py` (gate is fixed; threshold
  proposals go in the report).
- The night-run dir `/mnt/e/.../quant-work/arm-a/` and `calibration-v1/` (read-only).
- DEV250/DEV75 except ≤3 final confirmations (record every DEV read in the report).
- The sealed final set (never touch). The GPU (CPU build only, CUDA_VISIBLE_DEVICES="").
- `~/.local/state/sepalith/night-queue/`, `/home/m0hawk/sandbox/Sepalith`.
- `experiments/serving/prefill/` (prefill mission's PR owns it).

## Constraints
- CPU only, ≤ 8 threads, `nice -n 10`, llama.cpp pinned b10453 build-cpu
  (`~/.local/opt/llama.cpp-b10453/build-cpu/bin/`).
- Search signal from TRAIN data only (KLD slice; TRAIN search panel excluding
  calibration + KLD ids). If building a new imatrix: TRAIN only, same DEV exclusion.
- `python3 scripts/check_quality.py` (ruff + ty) must stay clean.
- Upload nothing. Kaggle only if truly needed (≤3 GPU-h, ask before any dataset).
- Disk on E: ≥ 100 GB free; clean up dead variants (keep finalist GGUFs).
- Machine is shared (prefill mission benches run here): check `uptime`/load before
  heavy runs; their latency benches need a quiet box — run our 8-thread jobs when
  load ≈ 0–2, or accept interleaving only for non-latency-critical work of theirs.

## Context (what the night run already established — CUDA numbers)
| tier | MB | mean KLD | top1 | exact Δpp | valid_drop pp | caps (ref 11) |
|---|---|---|---|---|---|---|
| Q8_0 | 2679.7 | 0.000914 | 99.06 | +0.8 | 2.4 ✗ | 17 |
| Q6_K | 2199.7 | 0.003205 | 98.24 | +1.2 | 2.4 ✗ | 17 |
| Q5_K_M | 1973.1 | 0.010786 ✗ | 96.90 | −0.4 | 4.0 ✗ | 21 |
| Q4_K_M (imat) | 1759.8 | 0.034197 ✗ | 94.43 | 0.0 | 4.0 ✗ | 21 |
| Q4_K_M-head-q6_k | 1695.0 | 0.034631 ✗ | 94.37 | 0.0 | 5.6 ✗ | 25 |
| Q4_K_M-uncal | 1759.8 | 0.042691 ✗ | 93.69 | +0.8 | 4.0 ✗ | 21 |
| Q4_K_M+ | 1778.9 | 0.030327 ✗ | 94.78 | +0.8 | 2.4 ✗ | 16 |
| Q4_K_M-attn-out-q8_0 | 1847.9 | 0.026839 ✓ | 95.13 | 0.0 | 6.4 ✗ | 27 |

- **valid_drop fails for every tier incl. Q8_0**: quantization flips mid-length
  generations (ref lengths 124–501) into 1024-token runaways; net cap rise ≈
  0.12 × changed-cases. Even Q8_0 nets +6 caps. Threshold (0/1 pp) looks
  structurally unpassable → document for the user, do NOT change the gate.
- All tiers ship with token_embd q8_0 + output q8_0 (night run flags).
- HINT4: attn_output is the sensitive module (restoring it alone helps most).
- Prefill mission (PR #66, CPU llama-bench 8T): pp2048 ≈ flat 101–132 for all
  q4/q6/q8 tiers (Q5_K_M outlier 64.8); tg32: Q8_0 16.12, Q6_K 21.12 (best),
  Q4_K_M 15.80 (worse than Q8_0!), head-q6_k-Q4 20.61, attn-out-q8_0 18.96.
  → decode-wise, Q4_K_M is a BAD laptop tier; head q6_k fixes it; Q6_K wins.
- Size levers: token_embd+output = 535M params (~570 MB at q8_0). head q6_k
  −65 MB (KLD-neutral); token_embd q6_k −65 MB (untested); attn_output q8_0
  +88 MB (KLD −0.0074).

## What's Been Tried
(Runs live in .auto/log.jsonl; summarize durable findings here.)

CPU KLD anchors (64 chunks, vs CPU bf16 base, deterministic):

| variant | KLD | MB | note |
|---|---|---|---|
| Q4_K_M imatrix (baseline) | 0.033973 | 1759.8 | CUDA 0.034197 |
| Q6_K | 0.004104 | 2199.7 | safe tier |
| Q4+attn_output q8_0 | 0.026750 | 1847.9 | CUDA 0.026839 |
| Q4_K_M+ (kv q8) | 0.030182 | 1779.0 | CUDA 0.030327 |
| probe attn_output q6_K | 0.027237 | 1805.2 | 93% of q8 gain, 52% cost |
| C1: ao6+kv8+head6, emb q8 | 0.024990 | 1759.6 | emb q6 costs +0.0002 |
| C2: ao6+kv8+emb6+head6 | 0.024784 | 1694.9 | ★ headline so far |
| C4: C2 with emb+head q5_K | 0.026501 | 1623.8 | passes gate, 136 MB under |

Module sensitivities (marginal KLD/MB from Q4_K_M): kv→q8 2.0e-4, ao→q6
1.5e-4, ao→q8 0.8e-4; trims: emb q6 +3.2e-6, emb+head q5 2.4e-5 (all tiny).

**b10453 override semantics discovered**: a --tensor-type override that
matches attn_v/ffn_down tensors skips `++i_attention_wv`/`++i_ffn_down`, so
all LATER tensors of that category shift the Q4_K_M every-3rd q6_K mix. The
night run's Q4_K_M+ ffn_down mix silently differs from its receipt. Rule:
override those categories completely (all 42) or not at all.

**valid_drop analysis (night CUDA eval)**: every tier incl. Q8_0 fails
valid_drop (0/1 pp); quantization flips mid-length generations (124–501 tok)
into 1024-cap runaways at ~0.12 × changed-cases; even Q8_0 nets +6 caps of
11. Threshold looks structurally unpassable → report to user, gate.py
untouched. Search panel measures net cap-rise vs bf16 as the proxy.

Search panel: 196 TRAIN cases (7 families × 28), excludes calibration + KLD
ids, `search-panel.json` built (sha 11c4f32c…). Results (vs bf16 CPU ref,
mean 75 gen tok): **mixA 78.1% identical, net cap rise +1; mixB 77.6%
identical, net cap rise −2** (2 recoveries). Baseline Q4_K_M on DEV250
(CUDA) was 51% identical / +10 caps for scale.

llama-bench (matched load 16, 8T): Q4_K_M 10.09 tg32; C2 11.63; **mixA
14.29 (+42%)**; mixB 9.97 (−1%). ffn_down q6 is a decode tax; head q5_K
is the decode win. Prefill ~equal across the set.

Finalists: **mixA** (speed) and **mixB** (fidelity). DEV250+DEV75 CPU
confirmation running (quantize_and_gate via --existing, gate receipt into
the work dir). Third DEV slot unused.

## Plan
1. CPU KLD anchors: Q4_K_M baseline, Q6_K (bf16 CPU base computed once).
2. Sensitivity map: from Q6_K base (emb+head q8_0), drop one module type to
   q4_K across all layers → KLD delta per module. Then layer bands for top modules.
3. Composites at ≤ 1759.8 MB: e.g. Q4_K_M + attn_output q8_0 + head q6_k +
   token_embd q6_k (≈1718 MB, target KLD ≤ 0.03); trim variants (attn_output
   q8_0 only on flagged layers; IQ4_XS on ffn_gate/up) if size needs it.
4. IQ4_XS anchor + mixes (has gate thresholds).
5. Search panel (TRAIN, ~150–200 rows, greedy agreement vs bf16 incl. cap-rise
   proxy) for top candidates; llama-bench for finalists.
6. ≤3 DEV250+DEV75 CPU confirmations via quant_gate.py; write report + PR.
