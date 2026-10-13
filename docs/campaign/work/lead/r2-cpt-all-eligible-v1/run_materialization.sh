#!/bin/sh
set -eu
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
exec ionice -c3 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 docs/campaign/work/lead/r2-cpt-all-eligible-v1/materialize_all_eligible.py --output /mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1 >> docs/campaign/work/lead/r2-cpt-all-eligible-v1/materialization.log 2>&1
