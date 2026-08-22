# Twin POC arms C+D — Aurora (OPT-3) and Repetition cross (OPT-5)

Day of 2026-08-21 (follow-up twin agent; sole owner of the RTX 5090 —
the RL trial and the first twin POC both completed). Same instrument,
same paired discipline as the 2026-08-20/21 night run: both arms below
are compared against last night's `checkpoints/muon_final.pt`,
RE-EVALUATED FRESH THE SAME DAY with the same evaluate.py.

Arms (pre-registered in docs/research/optimizer-sweep-2026-08.md §5):

- **Arm C (OPT-3, AURORA)** — Muon's orthogonalization replaced by
  Aurora (arXiv:2606.27715, Alg. 3, K=2, beta=0.5) on the tall MLP
  up/gate projections Wg/Wu (3072x768) ONLY; every other matrix keeps
  the vendored Muon path. Identical everything else (lr 0.01 / embed
  4e-3, wd 0.1, 1300 steps x 524,288 tokens = 681.6M, seed 1273, WSD,
  QK-Clip tau=100 alpha=0.5, grad clip 1.0, identical data order).
- **Arm D (OPT-5, REPETITION CROSS)** — vendored Muon, identical
  optimizer/schedule/seed, but a SEEDED HALF corpus (138,103 of 276,206
  docs, RandomState(20260821)): 681.6M tokens = 5.55 epochs over 122.8M
  unique tokens vs the baseline's 2.78 epochs over 245.4M.

Status: [RUNNING — results filled in as arms complete]

## Arm C — Aurora vs Muon (paired)

[TO BE FILLED]

## Arm D — repetition cross (paired)

[TO BE FILLED]

## Files

- arms/aurora.py (Aurora optimizer; header documents exactly what is
  implemented vs skipped), arms/train_arm.py (trainer: parent train.py
  reused via import; deltas = arm set + tag-keyed checkpoint dirs),
  arms/data_prep_half.py (seeded half split), arms/neuron_census.py
  (momentum-leverage + activation dead-neuron criteria),
  arms/regurgitation.py (12-gram-index verbatim-span canary),
  arms/configs/{aurora_arm,muon_half_arm}.json
- logs/aurora.jsonl, logs/muon_half.jsonl (per-100-step telemetry),
  logs/arms_run/ (stdout + prep + census + canary outputs)
