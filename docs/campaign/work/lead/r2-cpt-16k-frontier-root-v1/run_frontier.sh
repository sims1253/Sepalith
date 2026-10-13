#!/bin/sh
set -eu
packet=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-16k-frontier-root-v1
output=/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-16k-frontier-v1
test ! -e "$output"
exec ionice -c3 nice -n10 taskset -c 0,2 python3 "$packet/source/lossless_rechunk.py" --input-manifest "$packet/input-manifest.json" --output "$output"
