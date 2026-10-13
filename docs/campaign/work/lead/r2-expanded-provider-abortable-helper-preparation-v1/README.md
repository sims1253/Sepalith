# Abortable expanded-provider helper preparation v1

This is an isolated source candidate for a future expanded-model editor path. It does not modify or activate the current E750 selector, b4, a VSIX, or any service.

`owned_helper.ts` starts each R helper as its own process group, captures the Linux PID start tick, and shares one monotonic `RequestHelperBudget` across the serial NAMESPACE and current-buffer source-import calls. An editor abort, document-version invalidation, output overflow, or budget expiry terminates only the matching owned process group. It sends TERM, waits a bounded grace period, verifies the same PID/start tick, and sends KILL only when that exact owned group remains. The helper promise settles after cleanup. Partial output and nonzero exits are infrastructure failures.

The revised `namespace_runtime.ts` passes the request signal and shared budget to both helpers, invokes `Rscript --vanilla`, limits stdout/stderr, and preserves exact parse-unavailable versus infrastructure semantics. It never invents namespace evidence and has no fallback to the current E750 selector.

`abortable_namespace_adapter.ts` is the coordinator-facing candidate. It links the editor signal, polls the VS Code document version during helper work, rechecks version plus content hash afterward, and verifies the observed NAMESPACE path/hash against the coordinator cache identity. A mutation aborts the live owned helper. This file is not yet integrated into `extension.ts`.

Tests use actual short-lived R children for normal exit, timeout, and abort; verify the matching PIDs are absent and delayed marker files are not written; test one shared two-call budget; reject a missing executable; and cover adapter version invalidation and external cancellation. The exact authorized TRAIN controls preserve the reviewed result for `e677ee6a8da38436f4bdb6b6` (explicit source-import evidence unavailable) and `43b24d15b32aae89fdf72245` (supported inventory). No generated R or package code is executed; only the fixed parse-only helper scripts and test fixtures run.

Root integration should copy this closure into a fresh extension snapshot, construct one coordinator resolver with the extension cancellation token bridged to `AbortSignal`, and pass an exact NAMESPACE path/hash cache identity. Then run an editor-shaped cancellation test that mutates the live document while an actual helper is active, followed by context-mode/range/prompt parity and latency checks for the root-selected expanded target. Keep the current E750 path unchanged until those gates pass.
