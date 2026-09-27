# Final-target serving refresh v4

This target-null packet prepares the TRAIN-only imatrix, Q8/Q4_K_M/IQ3_M audits, paired 2K/8K speculative probes, and bounded server lifecycle. It does not authorize conversion, quantization, CUDA use, server execution, evaluation, or promotion.

Root binds one selected dense target, one measured quant, a fresh root admission, a loopback port, and all artifact receipts. The acceptance gate remains exact greedy token/text/termination parity, point speedup at least 1.4, paired trace-cluster bootstrap 95% lower bound above 1.0, and no candidate p95 regression. Draft traces retain both repetitions inside each 2K/8K bootstrap cluster.

`record_imatrix_completion.py` ignores launcher-asserted chunk and coverage counters. It requires a pinned successful terminal envelope and log, parses the exact 64-chunk compute/save messages, then reads the GGUF imatrix metadata and `.in_sum2`/`.counts` tensors. Every non-embedding matrix in the bound F16 GGUF must have a complete positive-count pair.

`record_prelaunch_verification.py` performs the expensive full checkpoint, selected quant, available draft, and required-HF hashes once before any server starts. Each lifecycle receives that immutable receipt separately and checks its binding, paths, identities, and sizes without rereading optimizer or model payloads while the server is live.

`server_lifecycle_v4.py` requires the numeric fd of the root's already-held stable CUDA lease. It passes that fd to the server, probe wrapper, and paired probe. Children remain in the outer supervisor process group and also receive Linux parent-death TERM. SIGTERM/SIGINT and startup failures enter a single cleanup path. Cleanup targets only the recorded child PID, using its `/proc` start tick after capture; an unreaped `Popen` object provides exact ownership during the shorter pre-capture interval.

Root command order (place all generated output on E):

```bash
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
P=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-final-target-serving-refresh-v4
$PY -B "$P/prepare_refresh_v4.py" --binding "$BINDING" --stage export --out "$RUN/export.json"
$PY -B "$P/record_imatrix_completion.py" --binding "$BINDING" --execution "$ROOT_IMATRIX_TERMINAL" --out "$RUN/imatrix-completion.json"
$PY -B "$P/record_prelaunch_verification.py" --binding "$BINDING" --out "$RUN/prelaunch-verification.json"
# Root computes PRELAUNCH_VERIFICATION_SHA256, retains the existing CUDA lock fd, then runs each lifecycle exclusively.
$PY -B "$P/server_lifecycle_v4.py" --binding "$BINDING" --verification "$RUN/prelaunch-verification.json" --verification-sha256 "$PRELAUNCH_VERIFICATION_SHA256" --cuda-lock-fd "$CUDA_LOCK_FD" --profile draft --mode ordinary --run-dir "$RUN/baseline"
$PY -B "$P/server_lifecycle_v4.py" --binding "$BINDING" --verification "$RUN/prelaunch-verification.json" --verification-sha256 "$PRELAUNCH_VERIFICATION_SHA256" --cuda-lock-fd "$CUDA_LOCK_FD" --profile draft --mode released_dspark --run-dir "$RUN/candidate"
$PY -B "$P/analyze_refresh_v4.py" --binding "$BINDING" --profile draft --baseline "$RUN/baseline/probe.json" --candidate "$RUN/candidate/probe.json" --candidate-mode released_dspark --out "$RUN/comparison.json"
```
