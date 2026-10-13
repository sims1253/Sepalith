# RUN-03 b4 notebook endpoint baseline v2

This isolated packet prepares a root-owned loopback measurement of the
protected b4 Q8 fallback. It does not start a server, load weights, contact an
editor, read DEV/final rows, or make a quality or promotion claim. The original
`b4-notebook-baseline-preparation` packet is unchanged. The v2 client is
`run03_b4_notebook_baseline_v2.py` and owns only HTTP requests, the pinned
legacy renderer, native tokenization, parsing, and an in-memory fixture
record.

The two hand-authored fixtures are target-free synthetic endpoint fixtures.
Their trailing context keeps each visible document above the extension's
30-character provider-eligibility floor. They preserve the legacy
`run_eval.render_zeta2` format, seven extension stops, 8192 context, and
`n_predict=320` cap. The request body is independent of fixture operation and
family labels. The in-memory record includes the extension's typed-partial
range rule as geometry evidence, but does not score the synthetic target or
claim no-op correctness.

## Separate artifact identities

Authority data is nested as separate model and runtime objects. The protected
model is `packaging_b4-Q8_0.gguf`, 2,012,011,904 bytes, SHA256
`e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d`. The
accepted local CPU runtime is build 10453, commit `3cb7ffb1a`, SHA256
`123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804`.

The old preparation's flattened `local_metadata_check` reported the 17,896
byte runtime metadata as `bytes` beside the runtime hash. v2 never places that
value in the model object. `local_artifact_metadata()` reports model and
runtime paths and byte counts separately and leaves both hashes
`not_checked`; only the root-owned live verifier may stream and hash the
protected model and executing runtime. A notebook runtime is a separate live
identity and must be recorded by root rather than copied from the local CPU
runtime.

## Request sequence and deadlines

The server is preflighted once (`/health` must report `status: ok`, then
`/props` and `/v1/models` are checked). The four cycle labels are:

1. `cold`: the first request after that single preflight;
2. `repeat`: the same first fixture request again;
3. `different_prompt`: the first request for the second fixture;
4. `repeat`: the second fixture request again.

`repeat` uses `cache_prompt=false`; it carries no cache-hit or cache-state
claim. The second fixture is never labelled `cold`.

Each foreground cycle has an absolute five-second deadline measured from cycle
start through local render, `/tokenize`, `/completion`, and parsing. Every HTTP
call receives the residual deadline, so tokenization consumes the time it
actually takes and completion cannot receive a fresh five-second budget. The
overall run stops at `soft_deadline - reserve` (defaults 600 - 30 seconds),
including preflight. `--diagnostic --diagnostic-timeout-s 60` is an explicit,
separate 60-second diagnostic mode; its records are labelled
`diagnostic_60s` and cannot be used as foreground latency evidence.

## Root-owned invocation

After root starts the separately supervised notebook server and records its
actual model/runtime identities, the foreground client can be run with:

```sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
/usr/bin/python3 docs/campaign/work/b4-notebook-baseline-v2/run03_b4_notebook_baseline_v2.py \
  --url http://127.0.0.1:18099 \
  --model /home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf \
  --runtime /home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server \
  --repo-root /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb \
  --cycle-deadline-s 5 \
  --soft-deadline-s 600 \
  --reserve-s 30 \
  --output /home/m0hawk/.local/state/sepalith-campaign-20260915/RUN-03-b4-notebook-baseline-v2/per_cycle.jsonl \
  --summary /home/m0hawk/.local/state/sepalith-campaign-20260915/RUN-03-b4-notebook-baseline-v2/summary.json
```

The optional diagnostic run must use a fresh output directory and explicit
`--diagnostic --diagnostic-timeout-s 60`. The client refuses non-loopback URLs,
missing or changed local artifacts, wrong server model paths, unhealthy
servers, and wrong context size. Each cycle is fsynced before the next one;
summary and directory fsync failures are fatal.

## Validation and limits

Run the model-free tests with:

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
  docs/campaign/work/b4-notebook-baseline-v2/test_run03_b4_notebook_baseline_v2.py
```

The 14 tests load the actual pinned renderer and receipts, verify separate
identity metadata, prove only the first request is `cold`, exercise parser and
typed-partial geometry, reject unhealthy/wrong preflight data, and inject a
slow tokenizer to verify completion receives only its residual budget. They do
not read or hash model weights, start a server, use SSH/GPU, or contact an
editor.

This remains endpoint/parser preparation. Root must supply notebook runtime
and model evidence, actual editor application and cancellation observations,
server process ownership and cleanup, and host resource measurements before
any live baseline is interpreted.
