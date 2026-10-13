# CPT expanded-stage root binding preparation v1

This packet prepares the accepted recursive stage-transition v2 runner for the
current terminal 16K corpus snapshot. It does not read the cache while the
cache builder is active. `assemble_root_admission.py` fails until the cache has
atomically published a `complete` manifest and root supplies a separate,
matching `DATA_ADMISSION` that explicitly authorizes stage-transition binding.
The included data-admission template is pending and cannot pass the assembler.

The immutable upstream metadata describes 183,084 unique rows, 177,190 complete
documents, 461,205,327 input tokens, 461,010,455 supervised tokens, and four
named tail replays. At effective batch 16 this is 11,443 destination updates.
The selected source is the full-state representative checkpoint at global step
66. The destination starts at its own cursor zero and ends at global step
11,509. It preserves the exact source optimizer (hidden LR 3e-6, side LR 3e-7),
scheduler state, completed two-step warmup, model, FP32 optimizer state, and
CPU/CUDA RNG. The first mandatory save/evaluation/root decision is stage step
128, global step 194; terminal evaluation and retention are stage step 11,443.

The scope is the current admitted terminal-union snapshot. It explicitly leaves
the 1,999-document global-cap frontier pending rehash and content dedup. It does
not claim that the source pool is closed. A later admitted frontier recovery can
use recursive stage-transition v2 from a root-selected full checkpoint with a
fresh destination cursor.

The assembler verifies only small immutable metadata and cache file sizes. It
does not hash the multi-gigabyte cache payload. The accepted v2 binder and
trainer perform their existing payload verification before training. The E
trainer/archive/output paths remain exactly those in the accepted v2 template.

Run the CPU tests with temporary storage on E:

```bash
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/r2-cpt-expanded-stage-launch-preparation-v1 \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -B docs/campaign/work/lead/r2-cpt-expanded-stage-launch-preparation-v1/tests.py -v
```

`root-commands.json` remains non-authorizing. Its proposed first-gate guard is
10,800 seconds with 14 GiB startup admission, 6 GiB soft floor, and the reviewed
guard's fixed 4 GiB hard floor. Root must review the terminal cache, issue the
DATA admission, run the assembler and v2 binder, complete preflight, and make a
separate launch decision.
