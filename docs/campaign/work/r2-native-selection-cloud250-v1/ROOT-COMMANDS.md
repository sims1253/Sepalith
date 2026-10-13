This is a development-selection packet, not a selected model or final evaluator admission. Root must choose a completed task-SFT full checkpoint at 250, 500 or 1000 from source `8ec908a41904888af647de2ee8b40ba8f73837777509d91159051b25fc72dd3e`. The same source can have a fresh Midtrain-control parent or an explicitly merged-CPT parent. No checkpoint or model tensor was read during preparation.

Run the stages only after the current local CUDA owner is released. Root retains the manual resource ledger and the nonblocking global lock `/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock`; hold it through every native controller and its cleanup. CPU merge/export can also use that lock to prevent a concurrent training start. Do not change or unlink the lock inode.

Use a fresh root-owned copy of this packet, preserving the frozen bytes. Define these explicit paths:

```sh
PACKET=/ROOT/REVIEWED/COPY/r2-native-selection-preparation-v1
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
TASKSRC=/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-r2-task-v2/snapshots/8ec908a41904888af647de2ee8b40ba8f73837777509d91159051b25fc72dd3e/source
# CHECKPOINT_RECIPE: actual recipe retained with the chosen checkpoint.
# CHECKPOINT: restored and independently hashed full checkpoint, not a live directory.
# LOCAL_PARENT: explicit local parent with the same model/config/tokenizer hashes.
# MERGED, EXPORT, ROOT_WORK, RUN_ROOT: separate fresh absolute native-ext4 outputs.
```

For a downloaded cloud checkpoint, the actual recipe can name an ephemeral cloud parent directory. This binding changes only parent input paths, retains all hashes and the exact checkpoint identity, and records the original recipe hash. The merge later verifies the actual local parent bytes.

```sh
"$PY" -B "$PACKET/bind_merge_recipe.py" \
  --checkpoint-recipe "$CHECKPOINT_RECIPE" --local-parent "$LOCAL_PARENT" \
  --output "$ROOT_WORK/merge-recipe.json"

# Root runs this command inside the existing CPU host-memory/process guard.
# Reserve <=600s including TERM/KILL allowance, two CPU threads; no concurrent training.
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PACKET:$TASKSRC/experiments/training:$TASKSRC/packages/sepalith/src" \
  /usr/bin/timeout --signal=TERM --kill-after=30s 570s \
  "$PY" -B "$PACKET/merge_task_sft_cpu.py" --recipe "$ROOT_WORK/merge-recipe.json" \
  --checkpoint "$CHECKPOINT" --output "$MERGED"

"$PY" -B "$PACKET/prepare_root_bindings.py" \
  --parent-manifest "$MERGED/parent-manifest.preparation.json" \
  --export-dir "$EXPORT" --output-dir "$ROOT_WORK/export-preparation"
```

Root reviews the merge receipt and verifies all actual merged files. The helper retains the accepted 588 adapter-tensor comparisons, 294 independent FP32 matrix additions with one final BF16 rounding, finite weights and no LoRA remnants. Original `tokenizer.json` and `tokenizer_config.json` bytes are copied back exactly. The manifest is newly named `sepalith.r2-task-sft.parent-manifest.v1`; it records full checkpoint identity, prior parent, source cursor, original/relocated recipe lineage and output hashes.

Bind `export-spec.json` to the fresh host-guard preflight and set its status to `root_admitted`. It contains exact commands and pinned b10453 converter/quantizer inputs. Run the following directly beneath the accepted `host_memory_guard_v3.py`, with CUDA hidden and offline Python settings. The export enforces direct parent guard identity and at least 18 GiB Windows free memory at admission. Preserve a 12 GiB runtime soft floor and 4 GiB hard floor. Reserve 900s outer time (each conversion command has a 240s cap plus cleanup); keep 112 GiB disk headroom as in the accepted export route. Peak memory is not newly measured.

```sh
"$PY" -B "$PACKET/export_candidate.py" \
  --spec "$ROOT_WORK/export-preparation/export-spec.json"
# Run this completed-file metadata review under the same bounded CPU host policy.
"$PY" -B "$PACKET/review_export.py" --export "$EXPORT" \
  --output "$ROOT_WORK/q8-integrity.json"
```

The two commands are F16 conversion followed by Q8_0 quantization with both output and embedding tensors protected at Q8_0. The review verifies hashes, 381 tensor names/shapes, Q8 types 296 plus F32 types 85, exact tokenizer metadata equality to F16, and contiguous file extents. This is integrity evidence, not native inference or tensor-value equivalence. Converter/gguf Python sources and quantizer bundle dependencies are pinned in `conversion-source-pins.json`; installed Python packages/system loader dependencies remain the accepted environment plus root launch validation, not a newly proven universal closure.

After root independently verifies the new parent and integrity receipt hashes:

```sh
"$PY" -B "$PACKET/bind_native_profile.py" \
  --parent-manifest "$MERGED/parent-manifest.preparation.json" \
  --parent-manifest-sha256 "$PARENT_MANIFEST_SHA" \
  --integrity-receipt "$ROOT_WORK/q8-integrity.json" \
  --integrity-receipt-sha256 "$Q8_INTEGRITY_SHA" \
  --output "$PACKET/native_evaluator/profile.json"

# Fresh CPU tokenizer/import observations, one core, <=90s and <=1.5GiB each.
# These read tokenizer/config only and synthetic text; they do not start native or read DEV.
"$PY" -B "$PACKET/native_controller/observe_client.py"
"$PY" -B "$PACKET/native_controller/build_closure.py"
"$PY" -B "$PACKET/native_controller/observe_client.py" --check-policy-only
```

Root must review the changed origin comparison and all unresolved entries. Do not relabel the retained v3 graph as complete for new paths. The builder reuses the accepted interpreter/factory policy but binds the actual new source paths/profile and fresh tokenizer-only observations. Root hashes the completed graph and full native source/loader inputs before filling the approval. The accepted native mapped-code allowlist and ELF loader metadata are unchanged; the actual fresh server still must match them.

Fill `native-approval.json` with actual run/lease IDs, UTC interval, profile canonical digest, completed source-policy input hashes and selected tokenizer/config file hashes. Keep the exact corrected DEV panel path/SHA from the accepted v3 approval; its fixed denominator is 75 unique DEV cases, 43 edits and 32 no-ops. The private controller now also checks those edit/no-op counts. Native profile stays CUDA b10453, GraphOpt0, context4096, batch256, ubatch256, one slot, PRM03, output192 and five seconds per case.

```sh
# Outer lock remains held until the controller and its cleanup return.
/usr/bin/flock -n /home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock \
  /usr/bin/timeout --signal=TERM --kill-after=30s 810s \
  "$PY" -I -S -B "$PACKET/native_controller/root_controller.py" \
  --approval "$ROOT_WORK/native-approval.json" --run-root "$RUN_ROOT"
```

The existing inner limits remain host780s, controller720s, DEV client600s, native load60s, client RSS1.5GiB, server RSS8GiB and host MemAvailable8GiB. Actual PID/starttick, bundle aliases and resolved file bytes, CUDA offload, argv, model file mappings, props, GraphOpt0, exclusive-client lock and before/after native hashes remain enforced. The root must independently verify process/port release and pair all75 per-case outputs with the current Q8 baseline (26 edit exact,25 no-op,5 false suggestions); do not select from aggregate rewards or partial output.

Remaining bindings are the checkpoint and its original recipe, local parent bytes, merged artifact, Q8 artifact and integrity hash, profile, fresh source-origin replay and graph, root approval/deadline/lease. The untouched transport/scorer/final-row helper files retain their legacy standalone defaults; the DEV controller and evaluator pass the selected model/tokenizer paths explicitly, with no theta0 fallback. No final constructor, final rows, final time gate or final admission was changed or exercised.
