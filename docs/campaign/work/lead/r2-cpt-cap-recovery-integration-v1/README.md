# DAT-10 prefix-cap recovery integration

This packet verifies the completed v3 recovery as a candidate augmentation.
It is append-only and has no training-admission authority. The verifier reads
the 81 recovery group artifacts and the terminal union's document-provenance
metadata. It does not open the terminal union CPT rows or mutate the existing
cache, schedule, or union.

Run a cheap live check while the root controller is active:

```sh
PYTHONNOUSERSITE=1 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  docs/campaign/work/lead/r2-cpt-cap-recovery-integration-v1/integrate_recovery.py \
  --mode status
```

After the root terminal marker has `exit_code: 0`, run the bounded verifier:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES='' ionice -c3 nice -n 10 taskset -c 0,2 \
  /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  docs/campaign/work/lead/r2-cpt-cap-recovery-integration-v1/integrate_recovery.py \
  --mode verify \
  --accounting-output docs/campaign/work/lead/r2-cpt-cap-recovery-integration-v1/integration-accounting.json
```

The verifier requires the exact frontier, v3 receipt, source contracts,
global split, CPT partition, protected-parent registry, profile metadata, and
terminal document-provenance hashes. It checks source identity, TRAIN/CPT
partition, exact terminal dedup, every chunk's BOS/EOS and token range,
document token-stream conservation, and complete frontier state coverage.
The `progress.json` `payload_opened` field is initialization metadata from the
controller and is intentionally ignored; group artifacts are the evidence of
actual source consumption.

The optional augmentation build requires a root-supplied checkpoint JSON with
an explicit checkpoint step, source cursor, and checkpoint ID:

```sh
... integrate_recovery.py --mode build \
  --source-checkpoint-json /path/to/root-explicit-checkpoint.json \
  --effective-batch 16 \
  --augmentation-output /mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-cap-recovery-v3-integration-v1
```

It writes a fresh E-backed input containing each unique recovery row once,
then at most the named rows needed for batch alignment. The source checkpoint
cursor records the transition point; consumed rows are never replayed by this
packet. Until root supplies that checkpoint and performs its prompt/target,
license, and final provenance admission checks, the candidate remains
unadmitted.
