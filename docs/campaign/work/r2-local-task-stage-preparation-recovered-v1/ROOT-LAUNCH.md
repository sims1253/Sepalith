Root owns selection, budget admission, resource leases and dispatch. This packet prepares the first task-SFT stage on a completed, merged CPT parent. It does not select a checkpoint or authorize a launch.

Use the original CPT merge helper for source 5149a406, or the separately reviewed global CPT merge helper for source aaacb632. Verify the completed checkpoint, exact source identity, optimizer cursor, merge receipt and all merged model files. The binder reads metadata only. Supply the actual merge helper SHA256; the parent manifest must contain the same hash. Root must review that helper before trusting its hash. For the global helper, also review its upstream parent validation receipt.

Set these shell variables to actual absolute paths and values. Keep the existing global CUDA lock inode. Acquire it through the accepted root supervisor and hold it through cleanup. Confirm that the previous guard, trainer and checkpoint writer have exited. Recompute the remaining task-SFT plus DEV budget and UTC deadline. The provisional first attempt cap is 3300 seconds plus 60 seconds for the outer guard.

```sh
PACKET=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-local-task-stage-preparation-recovered-v1
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
TASKSRC=/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-r2-task-v2/snapshots/8ec908a41904888af647de2ee8b40ba8f73837777509d91159051b25fc72dd3e/source
# ROOT_WORK, MANIFEST, MANIFEST_SHA, MERGE_HELPER_SHA, OUTPUT, ARCHIVE,
# DEADLINE, ATTEMPT_SECONDS and JOB_ID are actual root-reviewed bindings.
"$PY" -B "$PACKET/bind_task_parent.py" \
  --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
  --merge-helper-sha256 "$MERGE_HELPER_SHA" \
  --output-dir "$OUTPUT" --archive-dir "$ARCHIVE" \
  --deadline "$DEADLINE" --seconds "$ATTEMPT_SECONDS" --job-id "$JOB_ID" \
  --output-recipe "$ROOT_WORK/recipe.json"

env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$TASKSRC/experiments/training:$TASKSRC/packages/sepalith/src" \
  /usr/bin/timeout --signal=TERM --kill-after=30s 180s \
  "$PY" -B "$TASKSRC/experiments/training/campaign_task_sft.py" \
  "$ROOT_WORK/recipe.json" --preflight-only
```

The task preflight has no `--verify-model-files` flag: its reviewed preflight verifies declared inputs. Run it after host-memory admission. Preserve its output. Root then changes only `launch_authorized` to true in the new recipe after all admission checks, and records the final recipe hash. The binder leaves this flag false.

```sh
"$PY" -B "$PACKET/make_runner_recipe.py" \
  --recipe "$ROOT_WORK/recipe.json" --output "$ROOT_WORK/runner-recipe.json"
```

The runner recipe uses the four pinned broad-c guard helpers and source 8ec908a4. It routes campaign_launch to campaign_task_sft. Root uses Runner at `/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-r2-task-v2`: verify_snapshot, plan, record admission, enqueue the reviewed runner recipe, and run_next from a new root supervisor. Copy the accepted broad-c run_owned.py into ROOT_WORK, changing only its work directory and Runner root. Verify its hash and exact paths before starting it. Do not run the original CPT supervisor unchanged.

The stage uses 8526 admitted rows, the 16000-draw schedule with 25% no-ops, effective batch16, microbatch2, accumulation8, context4096, rank32/alpha64, LR0.0002, target-only loss, and the unchanged 1000-step cosine horizon. It creates fresh LoRA and optimizer state on the merged parent. It never resumes the CPT optimizer. Full checkpoints occur every250 steps; DEV milestones are250/500/1000. The first mandatory stop and decision are250.

Verify actual CUDA load, finite target-only first-gate values, actual optimizer updates and host telemetry. At250 verify the terminal guard, full checkpoint files and identity, source cursor4000, all per-step losses and actual runtime. HF DEV is diagnostic only. Use the reviewed `r2-native-selection-preparation-v1/ROOT-COMMANDS.md` merge, Q8 export and native DEV route for selection: fixed43 edits/32 no-ops, matched PRM03, cap192, GraphOpt0. Root decides whether to continue to500. An exact task-SFT continuation must preserve identity and the1000-step horizon, use full250/500 and explicit resume_milestone, and receive a fresh lease/deadline. This packet prepares only the initial stage.

Unresolved at preparation: selected CPT checkpoint, actual CPU merge and model hashes, host-memory admission, remaining time, first CUDA target gate and throughput, native quantization quality, and final release choice. No final-evaluation contents were read.
