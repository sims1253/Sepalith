# R2 bounded cloud profile

This packet defines a portable, root-admitted profile for the target-cache
runtime, the pinned DeepSpec CUDA smoke, and the warm-start trainer. It does
not submit a provider job and has no uploader or watchdog credentials.

The profile is fixed at eight TRAIN rows and eight optimizer steps. The stages
run sequentially inside a cumulative 3,600-second profile deadline. The full
cloud window is 7,200 seconds, reserving the other 3,600 seconds for setup,
durable persistence, and final refresh. The local process deadlines are:

| Stage | Command shape | Limit |
| --- | --- | ---: |
| `teacher-cache-profile` | `target_runtime.py --limit 8 --device cuda:0 --dtype bfloat16` | 1,200 s |
| `native-cuda-smoke` | `native_cuda_smoke.py --device cuda:0` | 600 s |
| `warmstart-trainer-profile` | pinned `DeepSpec/train.py --config profile_config.py` | 1,800 s |

The first stage writes raw official cache shards under
`<run>/ephemeral-target-cache`. The orchestrator copies only its
`manifest.json` and `teacher_rows.jsonl` into
`<run>/durable/teacher-profile`; the raw cache is excluded from the durable
persistence manifest. Trainer checkpoints, stage terminal receipts, source
inventory, input hashes, and copied teacher metadata remain durable.

The target model manifest and TRAIN path are required command-line inputs.
The manifest must name `config.json` and `model.safetensors`, contain the
pinned DeepSpec revision and candidate merged-weights SHA, and point at a
directory containing the five canonical files pinned by
`source-inventory.json`. Alternative or sharded weight files are rejected. The
actual TRAIN file is hashed against the canonical admitted identity before any
model loader runs. `SEPALITH_PUBLIC_DRAFT_WEIGHTS` is required for the
warm-start profile; its released weight hash and sibling `config.json` hash
are both checked. No target manifest is discovered automatically.

Set `SEPALITH_PROFILE_SOURCE_ROOT` (or pass `--source-root`) to the staged
source root. The runtime sets these child-process paths inside the supplied
run directory. `HOME` remains unchanged; temporary and framework caches use
fresh run-owned roots:

```text
SEPALITH_DRAFT_CACHE
SEPALITH_DRAFT_OUTPUT_ROOT
SEPALITH_DRAFT_LOG_ROOT
SEPALITH_PROFILE_RUN_DIR
SEPALITH_PROFILE_TELEMETRY_PATH
TMPDIR / TEMP / TMP
HF_HOME / XDG_CACHE_HOME / TORCH_HOME / TRITON_CACHE_DIR
```

The DeepSpec and hardening roots are passed through
`SEPALITH_DEEPSPEC_ROOT` and `SEPALITH_HARDENING_ROOT`. The source inventory
contains only explicit source and requirements files and rejects `.git` and
`dependency-overlay` paths.

Inspect the command map without reading model or TRAIN content:

```text
python3 docs/campaign/work/r2-draft-cloud-profile-v2/profile_orchestrator.py \
  --run-dir <fresh-run-dir> \
  --target-model-manifest <explicit-target-model-manifest.json> \
  --train-data-path <explicit-relocated-TRAIN.jsonl>
```

Execution is a separate root-admitted action. It requires a fresh run
directory and `SEPALITH_PROFILE_ADMITTED=1`:

```text
SEPALITH_PROFILE_ADMITTED=1 \
SEPALITH_PUBLIC_DRAFT_WEIGHTS=<explicit-public-draft-model.safetensors> \
python3 docs/campaign/work/r2-draft-cloud-profile-v2/profile_orchestrator.py \
  --execute --run-dir <fresh-run-dir> \
  --target-model-manifest <explicit-target-model-manifest.json> \
  --train-data-path <explicit-relocated-TRAIN.jsonl>
```

Each stage writes `<run>/stages/<stage>.terminal.json`. The trainer stage also
writes owned GPU telemetry and requires finite samples, the contiguous
`step=1/8` through `step=8/8` sequence, and a `step_8` checkpoint. A failing or timed-out stage stops the sequence and
writes `<run>/profile-terminal.json` with a failure status.

This packet was only CPU-tested. Remaining entry gates are the root’s explicit
target manifest and local inputs, public draft plus sibling config
availability, cloud admission, native CUDA execution, target-cache
persistence validation, the eight-step trainer profile, and independent
provider readback/upload using the `durable/` handoff. No cloud or GPU launch
is implied by this packet.
