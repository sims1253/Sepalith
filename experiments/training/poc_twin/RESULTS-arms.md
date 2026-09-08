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

Status: measurements complete; saved evidence reconciled 2026-09-08 below.

## Arm C — Aurora vs Muon (paired)

See the saved-result reconciliation below.

## Arm D — repetition cross (paired)

See the saved-result reconciliation below.

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

## Saved-result reconciliation — 2026-09-08 (W32)

The measurements finished in August; the RUNNING placeholder above was stale.
This pass recomputed saved JSON records and archived their source hashes without
inference or training. Evidence: `docs/validation/2026-09-08-w32.json` on
`t3code/queue-owner-handoff`; private archive `/mnt/h/sepalith/runs/w32-reconciliation-20260908`.

Arm C's saved compiled run improves eval_loss by 0.819%, 1.323% and 1.122% at
steps750/1000/1250. The eager repeat instead regresses by 0.672%, 1.028% and
0.400%. These are logged eval_loss values, not an independently recovered paired
R-BPB table. The saved census gives leverage-dead fractions 3.0463% vs0.2089%
(14.584x reduction), and near-dead activation fractions 1.7307% vs0.01085%
(159.50x reduction). This supports the measured utilization mechanism at the
saved geometry; it does not establish a reproducible loss improvement at scale.

The historical night note declared ADOPT using non-negative quality plus better
utilization. The registered optimizer-sweep rule was stricter: at least0.5%
better paired R-BPB AND at least5x lower dead-neuron fraction at equal BPB.
The eager repeat fails the corresponding loss improvement, and no paired
uncertainty analysis was recovered. Thus the dashboard's blanket adoption and
the later parked recipe are different decisions, not contradictory raw census
results. Current production optimizer behavior is unchanged; no new Aurora run
or optimizer reopening is authorized by this reconciliation.

Arm D's saved half-corpus run has eval_loss1.3148 vs1.18864 at step1250, a10.61%
regression, at the same logged token count. The regurgitation canary has train
max spans73 vs76 tokens, held-out max31 vs18; this is a small descriptive canary,
not a statistical population claim. The completed repetition measurement does
not need rerunning merely to fill this document.
