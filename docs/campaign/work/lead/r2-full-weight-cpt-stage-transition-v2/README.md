# Full-weight CPT recursive stage transition v2

This preparation extends v1 so the source can be either an admitted
representative/full-corpus checkpoint or a full checkpoint created by a prior
stage-transition recipe. The latter must reproduce the prior recipe identity
and bind its stage-local sampler method, prior global offset, draw-schedule
hash, local cursor, global step, and dataset-managed resume flag.

A new transition sets `global_optimizer_step_offset` to the selected source
checkpoint's global step and starts the newly admitted destination schedule at
stage cursor zero. It loads the existing full model, hybrid FP32 optimizer,
scheduler, CPU/CUDA RNG and Trainer state. Optimizer configuration, learning
rate, constant-with-warmup scheduler, and original warmup steps cannot change;
there is no second warmup.

The actual Transformers 5.5 CPU regression covers source training, corpus A,
then a second corpus B transition. It compares the two-transition lane with a
single uninterrupted destination stream. Final model, optimizer and scheduler
states match exactly; B begins at its own row zero. Existing same-stage resume
and hybrid optimizer/RNG tests remain.

Run tests with temporary storage on E:

```bash
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/stage-transition-v2 \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B -m unittest discover \
-s docs/campaign/work/lead/r2-full-weight-cpt-stage-transition-v2/tests -v
```

This packet is preparation only. The existing checkpoint66 remains the initial
lineage source. A future backfill transition requires a new root data decision,
source-checkpoint decision, exact bound recipe, and launch admission.
