# R2 local DSpark profile against selected step-500 parent

This is an unarmed, CPU-prepared command packet for the official DeepSpec
DSpark profile. It binds the selected
`NATIVE/models/SFT11-task-global-b-500-merged` metadata, the repaired target
cache runtime, and the released public DSpark warm-start source. It does not
load either model, read TRAIN rows, start CUDA, contact a provider, or claim
profile success.

The selected parent is bound by the existing preparation manifest SHA
`92157a4a52a4ed928f76df9f1f77a6aa39235eefd6a859eb9c2b3d4118ee18db` and merged
weight SHA
`631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c`.
The original metadata manifest is [target-manifest.step500.json](target-manifest.step500.json).
The accepted-parent launch binding is [target-manifest.step500-accepted-wrapper.json](target-manifest.step500-accepted-wrapper.json); it preserves the same model and TRAIN identities while binding the accepted wrapper
`r2-step500-rl-gate-v2/parent-manifest.json` (SHA `05eac983…`). Its declared
model directory is the verified absolute path from the selected receipt; the
local profile must recheck all five canonical files before model load.

## Source overlay

[source/](source/) is a small, copied source closure from immutable provider
payload v6. Its only source change is replacing the v6 target runtime with the
cache-identity repair at SHA
`76c3d3ca89397a4a120f0c3e11fa24602123851fe9165872c7578d0ed00a644a`.
The local profile orchestrator has one profile-specific metadata change: its
target artifact pin is the selected step-500 SHA instead of v6's a-250 target
pin. The local profile config uses an explicit `step500_profile` experiment
name and `checkpointing_steps=8`; the upstream trainer's final save still
provides the `step_8` checkpoint gate. The refreshed inventory is at
[source/docs/campaign/work/r2-draft-cloud-profile-v2/source-inventory.json](source/docs/campaign/work/r2-draft-cloud-profile-v2/source-inventory.json).

The source closure retains DeepSpec revision
`005e03b81cec38b7da6399833d609ee89a2587f2`, the official target cache writer,
`prepare_teacher_cache`, `pretokenized_cache_bridge`, `teacher_rows`, the
corrected native CUDA fixture, and the existing TRAIN-only profile sources.
The TRAIN file is an explicit runtime input with admitted SHA
`85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e`; this
packet contains no TRAIN copy.

## CPU preflight

Run only the metadata/source preflight from the campaign worktree:

```sh
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python \
  docs/campaign/work/lead/r2-draft-local-profile-v1/cpu_preflight.py
```

The preflight checks the selected parent pins, canonical filenames, cache
identity expectation, the official profile settings, and every small source
hash in the refreshed inventory. It refuses to hash files over 1 MiB and does
not open target weights, public weights, or TRAIN.

## Plan-only command

The plan command prints the three fixed stage commands and performs no input
verification or execution:

```sh
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PACKET="$PWD/docs/campaign/work/lead/r2-draft-local-profile-v1"
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
RUN_DIR=/tmp/sepalith-r2-step500-profile-plan
"$PYTHON" "$PACKET/source/docs/campaign/work/r2-draft-cloud-profile-v2/profile_orchestrator.py" \
  --run-dir "$RUN_DIR" \
  --target-model-manifest "$PACKET/target-manifest.step500-accepted-wrapper.json" \
  --train-data-path "${SEPALITH_TRAIN_DATA:?root must bind the admitted TRAIN file}" \
  --source-root "$PACKET/source" \
  --python "$PYTHON"
```

`RUN_DIR` is only used to render command paths in plan-only mode. Do not use a
prior profile directory for execution.

## Root-gated execution command

After root promotes the selected parent manifest to the accepted
`merged_sft` contract, re-admits the exact TRAIN identity, acquires the global
CUDA lock/host guard, and binds fresh native-ext4 output paths, the bounded
profile command is:

```sh
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PACKET="$PWD/docs/campaign/work/lead/r2-draft-local-profile-v1"
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
RUN_DIR="${SEPALITH_DRAFT_PROFILE_RUN_DIR:?root must provide a fresh run directory}"
export SEPALITH_PUBLIC_DRAFT_WEIGHTS=/home/m0hawk/.local/state/sepalith/campaign-20260915/models/released-dspark-hf-v1/model.safetensors
export SEPALITH_TRAIN_DATA="${SEPALITH_TRAIN_DATA:?root must bind the hash-checked TRAIN file}"
SEPALITH_PROFILE_ADMITTED=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON" "$PACKET/source/docs/campaign/work/r2-draft-cloud-profile-v2/profile_orchestrator.py" \
  --execute \
  --run-dir "$RUN_DIR" \
  --target-model-manifest "$PACKET/target-manifest.step500-accepted-wrapper.json" \
  --train-data-path "$SEPALITH_TRAIN_DATA" \
  --source-root "$PACKET/source" \
  --python "$PYTHON"
```

The source orchestrator itself enforces `SEPALITH_PROFILE_ADMITTED=1`, a fresh
run directory, explicit cache roots, the cumulative 3,600-second profile
limit, and the 7,200-second outer-window reserve. Root's independent
watchdog remains required. The actual run hashes the selected 5,033,557,128
byte target model and the 647,558,522 byte public draft, so it is intentionally
outside this CPU preparation.

## Stage gates

The profile must pass all three stages in sequence:

1. **Teacher/cache, at most 1,200 seconds:** select exactly 8 TRAIN rows;
   preserve generated integer IDs and native EOG/cap/protocol statuses; require
   `cacheable_rows == written_rows == 8`, exact written-ID reconciliation,
   cache version 2, and `target_model_name_or_path` equal to the resolved
   selected step-500 model directory. The raw target cache stays ephemeral;
   only its manifest and teacher rows are persisted.
2. **Native CUDA smoke, at most 600 seconds:** require actual native Flex
   forward and backward on CUDA. Any eager CPU result is a reference and does
   not establish CUDA support.
3. **Warm-start trainer, at most 1,800 seconds:** require contiguous
   `step=1/8` through `step=8/8`, finite GPU telemetry and training metrics, a
   `step_8` checkpoint, and a warm-start receipt proving the public draft
   tensors loaded with the exact two frozen target matrices omitted from the
   public artifact. Do not resume a prior output.

The cumulative profile limit is 3,600 seconds, leaving at least 3,600 seconds
of the existing 7,200-second window for setup, persistence, cleanup, and a
root-controlled final refresh. A passed profile is still a readiness signal;
it does not promote the parent or release a model.

## Resource and persistence estimate

The selected target file is about 4.69 GiB on disk and the public draft is
about 0.60 GiB. The upstream trainer holds the bound target on CPU while the
trainable DSpark draft runs in BF16 on CUDA, so peak memory depends on the
actual Transformers/DeepSpec allocation and must be measured by the stage
telemetry. Reserve at least 16 GiB free host RAM and 12 GiB free device RAM
per participating GPU as a conservative admission floor, then abort on
measured pressure; this packet makes no claim that those floors are sufficient.
The expected wall budget is 1,200 + 600 + 1,800 seconds, plus the root-owned
setup/persistence reserve.

Persist source/target/teacher manifests, terminal receipts, finite telemetry,
and checkpoints under the fresh run's durable root. Exclude raw cache shards,
HF/Torch/Triton/temp roots, and other framework caches from durable upload.
Root should take the global lock before execution, check the host guard, keep
an independent watchdog, and release the device after the terminal readback.
