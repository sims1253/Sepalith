# Final-target serving refresh v3

This target-null preparation preserves the v2 data panels and converter closure. It does not authorize export, quantization, server execution, evaluation, or promotion.

Root must bind one selected dense target and one measured quant key from `q8`, `q4_k_m`, or `iq3_m`. Every served quant requires its own exact artifact hash and passing saved-norm audit. Quantization also requires an imatrix completion receipt bound to the exact F16, TRAIN-only calibration, binary, successful exit, all 64 chunks, and zero non-embedding coverage mismatches.

`analyze_refresh_v3.py` treats each trace as the bootstrap cluster. It samples trace IDs within the 2K and 8K strata and retains both repetitions for a repeated draft trace. Acceptance requires exact greedy output and termination parity, median point speedup at least 1.4, paired cluster-bootstrap 95% lower bound above 1.0, and candidate p95 no worse than baseline. A complete result may honestly report no winner.

`server_lifecycle_v3.py` owns one server process group and one paired probe. It records the exact PID/start tick, selected loopback port, health transition, bounded deadline, event and server-log hashes, TERM/KILL actions, and verified process disappearance. Root supplies an admitted binding and fresh run directory. The wrapper does not confer CUDA or serving authorization.

Preparation commands:

```bash
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
P=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-final-target-serving-refresh-v3
$PY -B "$P/prepare_refresh_v3.py" --binding "$BINDING" --stage export --out "$RUN/export.json"
$PY -B "$P/prepare_refresh_v3.py" --binding "$BINDING" --stage audit-f16 --out "$RUN/audit-f16-command.json"
$PY -B "$P/prepare_refresh_v3.py" --binding "$BINDING" --stage imatrix --out "$RUN/imatrix-command.json"
$PY -B "$P/record_imatrix_completion.py" --binding "$BINDING" --execution "$ROOT_IMATRIX_TERMINAL" --out "$RUN/imatrix-completion.json"
$PY -B "$P/prepare_refresh_v3.py" --binding "$BINDING" --stage quantize --out "$RUN/quantize.json"
$PY -B "$P/prepare_refresh_v3.py" --binding "$BINDING" --stage audit-quants --out "$RUN/audit-quants.json"
$PY -B "$P/server_lifecycle_v3.py" --binding "$BINDING" --profile draft --mode ordinary --run-dir "$RUN/baseline"
$PY -B "$P/server_lifecycle_v3.py" --binding "$BINDING" --profile draft --mode released_dspark --run-dir "$RUN/candidate"
$PY -B "$P/analyze_refresh_v3.py" --binding "$BINDING" --profile draft --baseline "$RUN/baseline/probe.json" --candidate "$RUN/candidate/probe.json" --candidate-mode released_dspark --out "$RUN/comparison.json"
```
