# C19 results: hard-mix GRPO from C18b 300 — verdict ADOPT (2026-11-11)

Pre-registered final evaluation of card [C19](../cards/C19-hard-mix-grpo.md). Status:
[`status/C19.json`](../status/C19.json).

## Pilot end

`C19-grpo-exec-pilot-1` started from C18b checkpoint 300 (weights only) on the frozen mix of
1,375 HARD, 573 MIXED and 344 SOLVED tasks. It ran in four attempts (2026-11-03 to 2026-11-10;
two host-memory guard stops and one session stop, all resumed). It ended on the plateau rule at
update 300. Every gate from 50 to 300 was R6x continue.

| Update | Val all-pass | Val mean reward | CPT 2K NLL | DEV250 score250 |
| --- | --- | --- | --- | --- |
| start (C18b 300) | 0.3251 | 0.4363 | 0.9720 | 111 |
| 50 | 0.3127 | 0.4282 | 0.9758 | |
| 100 | 0.3151 | 0.4283 | 0.9756 | 112 (selection) |
| 150 | 0.3127 | 0.4240 | 0.9742 | 110 |
| 200 | 0.3127 | 0.4320 | 0.9732 | |
| 250 | 0.3102 | 0.4247 | 0.9715 | |
| 300 | 0.3077 | 0.4250 | 0.9710 | 112 |

Validation all-pass selected checkpoint 100. No accepted evaluation beat the start value, so
validation alone predicted no gain. Summary:
`/mnt/e/sepalith/campaign-20260915/checkpoints/C19-grpo-exec-pilot-1/pilot-summary.json`.

## Final hold-out evaluation

`C19-final-holdout`: frozen hold-out (3,803 tasks, 630 functions, gate `c631141`), greedy
decoding, checkpoint 300's pinned completions reused (sha256 matched). The first attempt was cut
at 2,880 of 3,803 completions by a WSL restart that a Windows Update install of WSL 3.0.5.0
caused (2026-11-10 05:31); the job resumed and completed 2026-11-11 00:54.

| Model | Mean reward (CI95) | All-pass (CI95) | Exact (CI95) | Protocol valid |
| --- | --- | --- | --- | --- |
| c18b-300 | 0.463 [0.442, 0.481] | 0.392 [0.373, 0.410] | 0.311 [0.294, 0.328] | 0.973 |
| c19-selected (checkpoint 100) | 0.476 [0.458, 0.496] | 0.399 [0.381, 0.418] | 0.321 [0.305, 0.338] | 0.981 |

Receipts: `~/.local/state/sepalith/campaign-20260915/evaluation-gates/C19-grpo-exec-pilot-1/final-step-100/`
(`receipt.json`, `receipt.md`, `metrics.json`). Verdict output:
[`receipts/C19-final-verdict.json`](../receipts/C19-final-verdict.json).

## Verdict

`python -m sepalith.training.grpo_exec.c19_jobs verdict`: **adopt = true**.

- **Primary (met):** hold-out all-pass difference +0.0076, CI95 [0.0034, 0.0122].
- **Secondaries:** mean reward +0.0132 [0.0086, 0.0180]; exact match +0.0100 [0.0055, 0.0147];
  protocol validity 0.973 → 0.981; DEV250 112 vs 111 (non-inferior); CPT 2K rise +0.0037 ≤ 0.03.

The gain is small and does not come from the hard buckets the mix targeted. finish_block
all-pass by target lines on the hold-out (descriptive): long 0.0033 → 0.0033, medium
0.0550 → 0.0503, short 0.3792 → 0.3798. All-wrong long blocks stay all-wrong.

## Consequence

C19 checkpoint 100
(`/mnt/e/sepalith/campaign-20260915/checkpoints/C19-grpo-exec-pilot-1/full/checkpoint-100`) is
adopted as the campaign's best model, superseding C18b checkpoint 300.

| Model | Hold-out all-pass |
| --- | --- |
| arm A 803 | 0.310 |
| C18 checkpoint 100 | 0.355 |
| C18b checkpoint 200 | 0.381 |
| C18b checkpoint 300 | 0.392 |
| C19 checkpoint 100 | 0.399 |

## Quant gate

`C19-quant-step-100` (same C09 gate and CUDA build as C18b 300; summary in
[`receipts/C19-quant-gate-summary.json`](../receipts/C19-quant-gate-summary.json)): Q6_K (2.20 GB)
is again the only tier that passes (score250 111, exact −0.4 pp, nofp +0.4 pp). Q8_0 fails only
nofp (+0.8 pp, 10 → 12). Q4_K_M+ fails exact (−3.2 pp), McNemar, nofp and KLD; it was a user
exception for C18b 300 and is worse here. The closest 4-bit tier is mixB-e4 (1.66 GB; exact
−0.8 pp, nofp +2.0 pp, failing only nofp).

Release tiers and publishing remain user decisions. Next-step design is out of scope for this note.
