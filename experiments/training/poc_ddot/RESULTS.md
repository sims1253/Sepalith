# POC-DDOT results — OT position coupling vs CAL vs best arm

Three-way eval per the DDOT plan (pre-registered baselines and kill test). Position-MSE = spurious re-anchoring, (first-snapped-slot/window)^2 — fixed-window arms are 0 by construction; LOWER is better for every arm.

Eval rows: 216, window 256, steps 32.

## Metric table

| metric | base | cal | ot |
|---|---|---|---|
| span exact-match | 0.0741 | 0.0000 | 0.0000 |
| prefix-match | 0.1620 | 0.0694 | 0.0000 |
| position-match | 0.0741 | 0.0000 | 0.0000 |
| edit similarity | 0.4262 | 0.0973 | 0.0440 |
| length-MAE | nan | 190.6574 | 58.2454 |
| position-MSE | nan | 0.0000 | 0.0060 |
| [EMPTY] precision | nan | 0.0000 | 0.0000 |
| [EMPTY] recall (n_gt=%d) | nan | 0.0000 | 0.0000 |
| p50 latency ms | 781.2621 | 0.0000 | 6005.6538 |

## per-bucket — base

| bucket | n | exact | length-MAE |
|---|---|---|---|

## per-bucket — cal

| bucket | n | exact | length-MAE |
|---|---|---|---|
| 11-50 | 136 | 0.000 | 225.647 |
| 51-256 | 80 | 0.000 | 131.175 |

## per-bucket — ot

| bucket | n | exact | length-MAE |
|---|---|---|---|
| 11-50 | 136 | 0.000 | 32.243 |
| 51-256 | 80 | 0.000 | 102.450 |

## KILL TEST (DDOT plan, verbatim)

If OT-coupled diffusion fails to beat BOTH baselines on length-MAE AND position-MSE AND exact-match simultaneously, the OT mechanism adds nothing here — close family E, write the negative result. No rescue arm.

- [x] VERDICT: **KILLED** — family E closes with a negative result. No rescue arm (per plan).
- [x] numbers:
  - exact-match: OT 0.0000 vs base 0.0741 — FAILS (also fails vs CAL 0.0000 only by tie)
  - length-MAE: OT 58.25 — beats CAL (190.66, 3.3x) but FAILS vs base (0 by
    construction: the md arm samples at fixed GT length)
  - position-MSE: OT 0.0060 vs base/CAL 0 by construction (fixed windows) — FAILS
  - The test requires beating BOTH baselines on ALL three; OT beats neither
    baseline on any metric that the base arm actually contests.

## Reading (recorded before any post-hoc theorizing)

- The mechanism was LIVE, not degenerate: OT-plan entropy held ~2.99 vs the
  ln(256)=5.5 ceiling for all 45 epochs (never collapsed to identity), the
  position field converged (final position loss 0.0019), training was
  healthy (2B tokens, loss 3.72). The negative result is about the
  MECHANISM'S VALUE, not a failed run.
- The one partial signal: OT's length-MAE is 3.3x better than CAL's
  (58 vs 191; per-bucket 32 vs 226 on 11-50-token spans) — the position
  field does carry length information — but it is nowhere near
  GT-length-informed sampling, and it costs value quality (edit-sim 0.044
  vs base 0.426: the routing-trained values are markedly worse).
- CAL v1 (this harness's half-peak plateau rule, NOT the paper's full
  recipe) is itself a strong negative: 190-token MAE means first-step
  confidence is a poor length signal in this regime.
- OT p50 latency 6.0s vs base 0.78s: the sampler decodes the full
  256-window per row (no length prior); expected, noted for honesty.

