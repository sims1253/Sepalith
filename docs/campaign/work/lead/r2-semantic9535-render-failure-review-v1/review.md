# DAT-10: Semantic9535 render failure review

This is a bounded review of the failed `render-16k-v1` controller. No full renderer, tokenizer/model, optimizer, or campaign rerun was started. The only data-content access was the authorized streaming of shards 0027 and 0028 through their two named failing rows.

## Evidence

The controller record at `docs/campaign/work/lead/r2-semantic9535-render16-root-v1/terminal.json` is:

```json
{"at":"2026-09-15T03:47:28.605362+00:00","status":"failed_no_retry","error":"","training_admitted":false}
```

Its `lane0.log` and `lane1.log` are empty. The useful evidence is in the two shard terminal records and their bounded log tails under:

`/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-16k-v1`

| shard | core | elapsed | status/exit | output rows | failing row |
| ---: | ---: | ---: | --- | ---: | --- |
| 0027 | 8 | 348.345 s | failed / 1 | 371 | `b15446c8d06df1c77e9caac6` |
| 0028 | 10 | 348.220 s | failed / 1 | 371 | `b777001bc9e4d6462cb2a8c1` |

Both tails end at the same renderer throw:

```text
Error: provider_infrastructure:<row_id>:prediction_namespace_infrastructure:Error
```

The two `.jsonl` files are therefore partial outputs. Their terminal records have `status: "failed"`; they are not valid completed shards. The exact output directory contains no shard 29–40 files, and the original files must remain untouched.

## Diagnosis

The confirmed failure locus is provider namespace-evidence resolution on input row 372 of each independently running shard. In the frozen provider source, `render_shard.ts` calls `resolvePredictionNamespaceEvidence`; any reason beginning `prediction_namespace_infrastructure:` is escalated to the `provider_infrastructure` exception shown above.

In `source/namespace_runtime.ts`, that reason is produced by the `catch` around the namespace and source `Rscript` calls, temporary-output reads, and JSON parsing. The current catch records only `error.name`, so the observed suffix `Error` loses the child process code, signal, timeout flag, and stderr. The evidence proves a namespace helper infrastructure result reached the renderer. It does not prove whether the child timed out at the 2,000 ms default, exited nonzero, or produced unreadable output.

The copied helper-only diagnostic then streamed each shard only through its named failing row. Each replay processed 371 preceding rows with zero infrastructure or outer errors, and each target returned `status: "supported"`, empty dependencies/reasons, and the same NAMESPACE hash as the original metadata. The fresh one-row controls also passed. This rules out a deterministic row-content failure and a cumulative namespace-helper failure. The original error needs a full-renderer condition that the helper-only probe excludes, such as concurrent renderer resource pressure or another component's interaction with the helper. The lower-level OS error is unavailable until a source snapshot with structured catch diagnostics is admitted.

This is not the outer controller timeout: the per-shard wrapper allows 1,800 seconds and the controller allows 9,000 seconds. It is also not a lane-assignment failure. `run_lane.sh` assigns manifest positions 0,2,… to core 8 and positions 1,3,… to core 10, then `set -e` stops each lane at its first failed shard. Thus 29–40 were never attempted after 27 and 28 failed.

The empty controller error has a separate, confirmed cause. `launch.py` raises a descriptive error only while a failed child is still observed as running. Both lane children can finish with exit 1 between the two-second polls; then the loop exits and bare `assert all(c.returncode==0 for c in children)` raises `AssertionError`, whose string is empty. Replace that assertion with an explicit exception containing every lane exit code. This is the minimal safe production fix supported by the evidence and preserves the original failure information.

The minimal controller change is:

```python
codes = [c.returncode for c in children]
if any(code != 0 for code in codes):
    raise RuntimeError("render lanes failed:" + ",".join(f"{i}:{code}" for i, code in enumerate(codes)))
```

The renderer-side prevention fix remains unproven. The safe next step is to bind the copied diagnostic runtime into a new source snapshot and capture the original helper error under the same full-renderer resource profile. A serialized-lane retry is a conservative workaround if resource pressure is confirmed; it has not been measured here.

## Bounded diagnostic

`diagnostic/diagnose_one_row.ts` and `diagnostic/diagnose_prefix_to_row.ts` import only the copied runtime and provider module. The prefix commands were run as two concurrent, nice-priority Node processes pinned to cores 8 and 10, with the existing helper paths and the runtime's 2,000 ms child timeout. They had a 600-second process bound, stopped after the exact target row, and wrote target-free JSON metadata only. They did not start the tokenizer bridge or read any row after the target.

The copied runtime at `diagnostic/source/namespace_runtime.ts` retains phase, message, code, signal, killed, stderr, and stdout fields in its unresolved result. It passes Node syntax checking. It has not been substituted into the frozen provider source, and no source manifest or production output was changed.

The isolated and prefix result files are `diagnostic/results/shard-0027.json`, `shard-0028.json`, `prefix-0027.json`, and `prefix-0028.json`; all report success at the target. `synthetic-enoent-0027.json` verifies the instrumentation: it captures `phase: "namespace_helper"`, `code: 2`, `killed: false`, the command message, and Rscript stderr when the helper path is intentionally invalid. No complete 9,534-row rerender was launched.

## Safe retry packet

Use a new output root. `run_shard.sh` has a fresh-path guard and will refuse a shard when either an output or terminal file already exists. Do not delete, truncate, rename, or reuse `render-16k-v1`; it is the failure evidence. The current root has no completed valid shard, so a complete retry needs 0027–0040 in a new root.

The following command is a prepared command only; it was not run during this review. It keeps the original source/input roots and parameters, uses the original alternating core assignment, and continues a lane after an individual failure so one bad shard cannot hide later missing shards. It refuses a pre-existing retry root and checks the recorded source/input manifest hashes before creating it.

```bash
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PROVIDER="$PLAN/docs/campaign/work/lead/r2-semantic9535-provider-preparation-v1"
INPUT_ROOT=/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-inputs-v1
RETRY_ROOT=/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-16k-retry-v1

test ! -e "$RETRY_ROOT" || { echo "refusing existing retry root: $RETRY_ROOT" >&2; exit 2; }
test "$(sha256sum "$PROVIDER/source-manifest.json" | awk '{print $1}')" = \
  8c1231d49066dcc6335c4b6a9c244249231874e31895d2799bba327a5672d9dc || exit 3
test "$(sha256sum "$INPUT_ROOT/manifest.json" | awk '{print $1}')" = \
  db1dd374d1d029695c9b8daef59151272d0db0a912276f8073b24b0df4f8e9a0 || exit 4
mkdir "$RETRY_ROOT"

run_lane_retry() {
  local core="$1"; shift
  local failed=0 rc shard
  for shard in "$@"; do
    if bash "$PROVIDER/run_shard.sh" "$shard" "$core" 16384 2048 "$INPUT_ROOT" "$RETRY_ROOT"; then
      rc=0
    else
      rc=$?
      printf 'shard %s failed (rc=%s); continuing lane\n' "$shard" "$rc" >&2
      failed=1
    fi
  done
  return "$failed"
}

run_lane_retry 8 0027 0029 0031 0033 0035 0037 0039 & lane0=$!
run_lane_retry 10 0028 0030 0032 0034 0036 0038 0040 & lane1=$!
if wait "$lane0"; then rc0=0; else rc0=$?; fi
if wait "$lane1"; then rc1=0; else rc1=$?; fi
test "$rc0" -eq 0 -a "$rc1" -eq 0
```

Do not substitute the existing `run_lane.sh` for this wrapper: its `set -e` behavior is the reason each original lane stopped before its later shards. If any retry shard fails, retain that new root and use another fresh suffix for its rerun. Promote or merge outputs only after every terminal record is complete and input/output hashes and row counts match the input manifest.

To expose the lower-level helper failure before spending a full retry, use the copied diagnostic runtime at `diagnostic/source/namespace_runtime.ts` with a fresh provider source snapshot. Recompute and bind a new source manifest; never edit the frozen `r2-semantic9535-provider-preparation-v1` source in place. The copied runtime adds the helper phase and child message/code/signal/killed/stderr/stdout fields to the infrastructure result. It is a diagnostic preparation, not a launch, and it does not make the current failed outputs valid.

## Inputs and integrity

The review used these bounded files: the controller `terminal.json`, `launch.json`, `input-root-review.json`, empty lane logs, the provider `run_lane.sh`, `run_shard.sh`, `render_shard.ts`, `source/namespace_runtime.ts`, `source/namespace_evidence.R`, `source/source_imports.R`, and the two shard terminal records plus log tails. Their SHA-256 values are recorded in `DAT-10-semantic9535-render-failure-review.json`. No source/data tree search was performed.
