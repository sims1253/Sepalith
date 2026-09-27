# Full-weight CPT recursive stage transition v3

This packet preserves the accepted v2 recursive full-state transition and corrects one launch-blocking coordinate error. Bound recipes store global optimizer steps, while checkpoint cadence is defined within the fresh destination stage. The v2 loader incorrectly required `global_mandatory_stop % checkpoint_every == 0`; v3 requires `(global_mandatory_stop - global_optimizer_step_offset) % checkpoint_every == 0`. Thus representative source step 66 to destination stage step 128/global step 194 is valid.

The cadence audit confirms that `milestone_action` already computes its modulo in stage-local coordinates. Evaluation and retained milestone membership use global step values as intended. V3 also requires all evaluation/retention steps to remain inside the destination stage and requires the terminal global step in both lists. Model, optimizer, scheduler, RNG, streaming cursor, and scientific data semantics are unchanged. E output paths deliberately remain the accepted v2 template defaults.

Run tests on E with CPU only:

```bash
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/stage-transition-v3 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B -m unittest discover -s docs/campaign/work/lead/r2-full-weight-cpt-stage-transition-v3/tests -v
```

This source packet does not authorize launch. Root must issue exact data/stage admissions, bind a fresh recipe, pass actual CPU preflight, and separately launch under a reviewed CUDA and host-memory guard.
