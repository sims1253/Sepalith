# Expanded SFT-B step-250 merge and native DEV commands

These commands are preparation only. The lead must first verify that training stopped at step 250, the full checkpoint exists, the B CUDA lease is released, and every destination below is absent. No command opens final data.

Set the common CPU environment for merge/export metadata work:

```sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-a/reviewed-source/packages/sepalith/src:/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-a/reviewed-source/experiments/training:/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/packet
```

Merge the exact full step-250 adapter into the selected merged SFT-500 parent under the existing host guard and global lease:

```sh
/usr/bin/flock -n /home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock \
  /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/host_memory_guard_v3.py \
  --command-json /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/merge-command.json \
  --output /mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-b-250-merge-supervision \
  --seconds 900 --minimum-free-mib 18432
```

Prepare the root-bound F16/Q8 export spec after the merge manifest exists:

```sh
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/packet/prepare_root_bindings.py \
  --parent-manifest /mnt/e/sepalith/campaign-20260915/models/SFT11-expanded-postsft500-b-250-merged/parent-manifest.preparation.json \
  --output-dir /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/root-bindings \
  --export-dir /mnt/e/sepalith/campaign-20260915/models/SFT11-expanded-postsft500-b-250-quant \
  --host-guard-output /mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-b-250-export-supervision
```

Place this argv in `export-command.json`, then execute it under the same global lock and host guard with a 900-second limit and 12 GiB minimum free memory:

```json
[
  "/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python",
  "-B",
  "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/packet/export_candidate.py",
  "--spec",
  "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/root-bindings/export-spec.json"
]
```

Review the resulting F16/Q8 structure and bind a new expanded profile:

```sh
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/packet/review_export.py \
  --export /mnt/e/sepalith/campaign-20260915/models/SFT11-expanded-postsft500-b-250-quant \
  --output /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/root-bindings/q8-integrity.json
```

Run `bind_native_profile.py` with the actual SHA256 values of the merge manifest and Q8 integrity receipt. Its output must be the currently absent `packet/native_evaluator/profile.json`:

```sh
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/packet/bind_native_profile.py \
  --parent-manifest /mnt/e/sepalith/campaign-20260915/models/SFT11-expanded-postsft500-b-250-merged/parent-manifest.preparation.json \
  --parent-manifest-sha256 ACTUAL_PARENT_MANIFEST_SHA256 \
  --integrity-receipt /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/root-bindings/q8-integrity.json \
  --integrity-receipt-sha256 ACTUAL_Q8_INTEGRITY_SHA256 \
  --output /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-native-selection-v1/packet/native_evaluator/profile.json
```

Then run the copied `native_controller/observe_client.py --observe-and-check` and `native_controller/build_closure.py`; review that the resulting source closure has no unresolved entries.

The lead fills the generated `native-approval.json` with fresh file hashes, lease/run IDs, times, and the new profile hash. Launch `packet/native_controller/root_controller.py --approval APPROVAL --run-root OUTPUT` only after the training and merge/export lanes release the global CUDA lock. The preserved native route is DEV75 only: 43 edit and 32 no-op cases, CUDA b10453, context 4096, batch/ubatch 256, parallel 1, output cap 192, GraphOpt 0. Existing SFT500 and calibrated quant artifacts remain untouched.
