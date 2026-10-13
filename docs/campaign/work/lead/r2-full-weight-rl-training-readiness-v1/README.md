# Full-weight RL training readiness v1

This packet audits the existing RL and dense-training seams. It adds one small,
framework-free contract at the rollout-to-update boundary. It does not implement
or authorize optimizer training.

The executable signal pilot remains a diagnostic. Its 112 groups and 448
candidates can establish reward variance and parser health, but they cannot be
relabelled as policy updates: they do not bind a live policy step, policy
log-probabilities, dense optimizer/scheduler state, or a full-weight checkpoint.

The practical implementation route is a dynamic Trainer whose MRO is
`FullWeightOptimizerTrainerMixin, FixedIDGRPOTrainerMixin, GRPOTrainer`. The
existing fixed-ID mixin keeps generated prompts identical to the stored integer
prompts; inherited pinned TRL computes policy log-probabilities and BNPO GRPO
loss; `CampaignPRM03Reward` consumes those same completion IDs; the reviewed
full-weight mixin supplies the Aurora/Muon optimizer. A new runtime must call the
contract in this packet before loss/update, and use the reviewed dense
checkpoint path with `checkpoint_kind=full_weights`.

A checkpoint is valid only at an optimizer boundary after a whole rollout
buffer is consumed. Its sampler payload must bind source schedule, selected row
IDs, rollout group cursor, optimizer global step, generation geometry, and an
explicit empty rollout buffer. An interruption inside generation discards and
replays that uncommitted buffer; it must never resume from a partial group.

Run the CPU tests with bulk temporary files on E:

```bash
PKT=docs/campaign/work/lead/r2-full-weight-rl-training-readiness-v1
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/rl-fullweight-tests \
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s "$PKT/tests" -v
```

`readiness-review.json` lists the exact reusable paths, blockers, and resource
limits. `update-binding.template.json` is deliberately unbound. Root must select
a full-weight editing-SFT parent, objective recipe, LR, data schedule, and gates
before any launch.
