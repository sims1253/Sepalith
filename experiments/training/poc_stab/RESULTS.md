# P1 results — Muon-hygiene A/B (zcode-stabtok)

Pre-registered design + adoption rules: docs/research/2026-08-29-papers-recon-poc-plan.md (P1).
Arms: control / split (per-head Wq/Wk/Wv/Wo) / polar (PE schedule, NS-8, eps 1e-14) / nesterov (g + m*buf).
Each 480 steps x 524,288 tok = 0.25BT, dose 0.3, seed 1273, identical data order. Launched 2026-08-29.

## Scorer output (appended by run_p1_chain.sh on completion)

  path        /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_control.jsonl  (9 rows, 5 train, 4 skipped)
  spikes      0 / 4  (rate 0.00000, window 5)
  first/last  None/None
  grad p99.9  0.1859 = 0.19x clip(1.0)   max 0.19x
  path        /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_split.jsonl  (9 rows, 5 train, 4 skipped)
  spikes      0 / 4  (rate 0.00000, window 5)
  first/last  None/None
  grad p99.9  0.1596 = 0.16x clip(1.0)   max 0.16x
{
  "a": "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_control.jsonl",
  "b": "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_split.jsonl",
  "spike_count_a": 0,
  "spike_count_b": 0,
  "spike_margin": 0,
  "p999_frac_a": 0.18593343155235056,
  "p999_frac_b": 0.1595738583672047,
  "p999_frac_margin": 0.026359573185145868,
  "grad_axis": "present",
  "b_at_least_as_stable_as_a": true,
  "verdict": "B AT LEAST AS STABLE AS A"
}

  path        /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_control.jsonl  (9 rows, 5 train, 4 skipped)
  spikes      0 / 4  (rate 0.00000, window 5)
  first/last  None/None
  grad p99.9  0.1859 = 0.19x clip(1.0)   max 0.19x
  path        /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_polar.jsonl  (8 rows, 4 train, 4 skipped)
  spikes      0 / 3  (rate 0.00000, window 4)
  first/last  None/None
  grad p99.9  0.1835 = 0.18x clip(1.0)   max 0.18x
{
  "a": "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_control.jsonl",
  "b": "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_polar.jsonl",
  "spike_count_a": 0,
  "spike_count_b": 0,
  "spike_margin": 0,
  "p999_frac_a": 0.18593343155235056,
  "p999_frac_b": 0.18347730662994088,
  "p999_frac_margin": 0.0024561249224096804,
  "grad_axis": "present",
  "b_at_least_as_stable_as_a": true,
  "verdict": "B AT LEAST AS STABLE AS A"
}

  path        /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_control.jsonl  (9 rows, 5 train, 4 skipped)
  spikes      0 / 4  (rate 0.00000, window 5)
  first/last  None/None
  grad p99.9  0.1859 = 0.19x clip(1.0)   max 0.19x
  path        /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_nesterov.jsonl  (7 rows, 3 train, 4 skipped)
  spikes      0 / 2  (rate 0.00000, window 3)
  first/last  None/None
  grad p99.9  0.3776 = 0.38x clip(1.0)   max 0.38x
{
  "a": "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_control.jsonl",
  "b": "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/logs/p1_nesterov.jsonl",
  "spike_count_a": 0,
  "spike_count_b": 0,
  "spike_margin": 0,
  "p999_frac_a": 0.18593343155235056,
  "p999_frac_b": 0.3776003747609258,
  "p999_frac_margin": -0.19166694320857522,
  "grad_axis": "present",
  "b_at_least_as_stable_as_a": false,
  "verdict": "B LESS STABLE THAN A (veto)"
}
scorer failed for nesterov


## Verdict (2026-08-29T20:20+0200, zcode-stabtok) — P1 NEGATIVE: pinned recipe stands

Loss-side (held-out paired BPB, ladder/logs/bpb_eval_p1.json; identical eval
blocks for every arm):

| arm | causal BPB | FIM BPB | Δ vs control (causal/FIM) |
|-----|-----------|---------|---------------------------|
| control | 0.79078 | 0.77527 | — |
| split | 0.79654 | 0.78060 | +0.73% / +0.69% worse |
| polar | 0.79574 | 0.77970 | +0.63% / +0.57% worse |
| nesterov | 0.80312 | 0.78777 | +1.56% / +1.61% worse |

Stability (stress scorer; spikes = 0 on every arm; p99.9 pre-clip grad-frac
vs control 0.186): split 0.160 non-inferior; polar 0.183 non-inferior;
nesterov 0.378 VETO (2.0× control). Cost per step: split ≈ −4%, polar ≈ −2%,
nesterov ≈ parity.

Adoption rules applied: no arm is ≥0.3% better paired held-out BPB → nothing
enters the pinned recipe. Kill test tripped (B/C/D all ≤ pinned on both
slices) → the Muon-hygiene line is CLOSED with a negative result. The sweep's
PolarExpress-park and NVIDIA-Nesterov citations stand at our scale; Qwen's
§3.1 hygiene deltas do not transfer to a dense 206M / 0.25BT regime.

Caveats (pre-registered threat (a) applies): 0.25BT may under-power
NS-schedule effects (Qwen's evidence is 419BT–4T); with no winners there is
no 1BT scale-up per the rules. Residual: polar's stability axis was
non-inferior WITH margin at ~equal cost — if P2/MuonH stress testing ever
needs NS-side headroom, PE-8 remains the candidate lever; it is NOT a
quality win here.

Ops notes: v1 chain died at the step-400 quick_eval OOM (fragmentation under
the 17.5GB process cap; board 17:0x post) — v2 = expandable_segments +
memfrac 0.65 + resume from step-200 ckpts (position-deterministic data order
⇒ resume ≡ uninterrupted). The "scorer failed for nesterov" line above is
stress_metrics' veto EXIT CODE, not a scorer failure. Artifacts: control
final.pt archived to /mnt/h/sepalith/runs/p1_stabtok/; variant ckpts deleted
after the verdict was rendered (logs + bpb_eval_p1.json retain all evidence).
