# Full-weight RL production driver v2

This packet connects the accepted fixed-ID PRM03 GRPO path to the reviewed
dense Aurora/Muon optimizer and `full_weights` checkpoint lifecycle. It remains
unbound and cannot launch.

`full_weight_rl_production.build_production_trainer` is the live construction
entry. It checks the selected dense parent/data/policy identities, rejects any
LoRA tensor, composes the production MRO, and installs the optimizer before
Trainer creates optimizer or scheduler state. `train_production` passes only
the full checkpoint path that the builder verified.

The contract runs after the real `CampaignPRM03Reward` callable and before
inherited TRL computes advantages and live-policy loss. It joins every candidate
by source row, stored prompt IDs, generated IDs, and reward output-ID hash. A
valid flat group is recorded as zero-advantage. Per-family signal admission is
bound separately and requires nonzero observed variance in each root-selected
family; a single flat update does not abort.

Run the CPU tests:

```bash
PKT=docs/campaign/work/lead/r2-full-weight-rl-production-driver-v2
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/rl-production-driver-tests \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  -m unittest discover -s "$PKT/tests" -v
```

After root supplies a selected target, admitted signal evidence, TRAIN-only
rollout files, optimizer identity, and recipe, run the CPU preflight command in
`root-commands.json`. The current inherited row loader remains bound to the
reviewed 2048-prompt/192-completion profile. It must not be presented as support
for the expanded long-output pool. No launch command is supplied until that
policy and a real G=4 memory/update/resume probe are admitted.
