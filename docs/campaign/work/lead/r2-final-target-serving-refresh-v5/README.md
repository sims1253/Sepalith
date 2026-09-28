# Final-target serving refresh v5

This target-null packet prepares the TRAIN-only imatrix, Q8/Q4_K_M/IQ3_M audits, paired 2K/8K speculative probes, and bounded server lifecycle. It authorizes no model conversion, CUDA use, server, evaluation, or promotion.

The current calibration tokenizes to 104,448 tokens. The pinned llama.cpp computes `min(--chunks, floor(tokens/context))`, so `--chunks 64` requests a maximum and the exact complete result is 51 chunks of 2,048 tokens. Two prior accepted artifacts independently bind 51 chunks, count 104,448, and 295 covered matrices. V5 validates the actual log and GGUF metadata at 51; it does not repeat or pad the corpus to assert 64. Every rank-2/3 non-embedding F16 matrix must have a compatible `.in_sum2`/`.counts` pair, finite nonnegative sum2, finite positive counts, and the expected dense count.

`record_prelaunch_verification.py` performs the expensive full checkpoint, selected quant, and available draft hashes once before any server starts. It snapshots device, inode, size, mtime_ns, and ctime_ns before and after hashing. Each live lifecycle rechecks those seals and hashes only small HF metadata, so the 17 GiB checkpoint/optimizer and served model are not reread during latency measurements. Same-size replacement fails the seal.

`server_lifecycle_v5.py` accepts only an fd for the exact campaign `resource-locks/cuda0.lock`. A separate nonblocking flock attempt must prove that the exclusive lease is already held. The fd passes to the server, probe wrapper, and paired probe. Children stay in the outer supervisor group and receive Linux parent-death TERM. SIGTERM/SIGINT and startup failures enter one cleanup path; exact owned PIDs are never confused with unrelated processes. An unreaped pre-start-tick child remains owned through TERM and KILL fallback.

Root command order (generated output belongs on E):

```bash
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
P=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-final-target-serving-refresh-v5
$PY -B "$P/prepare_refresh_v5.py" --binding "$BINDING" --stage export --out "$RUN/export.json"
$PY -B "$P/record_imatrix_completion.py" --binding "$BINDING" --execution "$ROOT_IMATRIX_TERMINAL" --out "$RUN/imatrix-completion.json"
$PY -B "$P/record_prelaunch_verification.py" --binding "$BINDING" --out "$RUN/prelaunch-verification.json"
$PY -B "$P/server_lifecycle_v5.py" --binding "$BINDING" --verification "$RUN/prelaunch-verification.json" --verification-sha256 "$PRELAUNCH_VERIFICATION_SHA256" --cuda-lock-fd "$CUDA_LOCK_FD" --profile draft --mode ordinary --run-dir "$RUN/baseline"
$PY -B "$P/server_lifecycle_v5.py" --binding "$BINDING" --verification "$RUN/prelaunch-verification.json" --verification-sha256 "$PRELAUNCH_VERIFICATION_SHA256" --cuda-lock-fd "$CUDA_LOCK_FD" --profile draft --mode released_dspark --run-dir "$RUN/candidate"
$PY -B "$P/analyze_refresh_v5.py" --binding "$BINDING" --profile draft --baseline "$RUN/baseline/probe.json" --candidate "$RUN/candidate/probe.json" --candidate-mode released_dspark --out "$RUN/comparison.json"
```
