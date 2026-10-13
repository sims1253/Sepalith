# RUN-03 b4 notebook baseline preparation

This packet prepares a root-owned measurement of the protected b4 Q8 fallback
on the notebook. It does not start a server, load a model, contact an editor,
or read the DEV/final case population. The client is
`run03_b4_notebook_baseline.py`; it is a loopback-only HTTP client and never
owns the server process.

The fixture has two hand-authored cases: one replacement and one no-op. They
are target-free synthetic rows, so they do not reuse PRM-03 prompts or make a
quality claim. Both are rendered by the pinned
`run_eval.render_zeta2` implementation. Each is sent once as `cold` and once
as `warm`, where warm means an identical immediate replay with
`cache_prompt=false`. Request settings are fixed before the labels are read:
`n_predict=320`, temperature 0, stream false, cache_prompt false, and the
seven extension stops. Native `/tokenize` uses `add_special=true`,
`parse_special=true`, and `with_pieces=false`.

Every live cycle retains the full native tokenizer/completion response,
render time, native-tokenizer round-trip, completion round-trip, parser time,
and `parsed_suggestion_latency_ms`. The latter runs from cycle start through
the pinned parser, so it is a parsed-result latency and not TTFT. A simulated
in-memory application plan is retained with zero writes. Cancellation and
actual editor application are marked `not_exercised`; root must collect those
events during a real editor request. The client closes its HTTP contexts but
does not report server cleanup; root must supply process ownership and
survivor evidence.

## Pinned identities

The complete model, runtime, renderer, parser, stop list, and request values
are read from the accepted
`docs/campaign/receipts/SFT-08-b4-dev-preparation.json` receipt. The client
cross-checks model/runtime/renderer identity against
`docs/campaign/receipts/PRE-08-fallback-ledger.json` and fails closed on drift.
The protected model is `packaging_b4-Q8_0.gguf`, 2,012,011,904 bytes,
SHA256 `e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d`.
The accepted CPU runtime is build 10453, commit `3cb7ffb1a`, with SHA256
`123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804`.

A targeted read-only SSH metadata check on `m0hawk@192.168.178.40` found the
protected model absent at these explicit notebook candidates:

* `/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf`
* `/home/m0hawk/.local/share/sepalith-campaign-20260915/models/packaging_b4-Q8_0.gguf`
* `/home/m0hawk/.local/share/sepalith-campaign-20260915/models/b4/packaging_b4-Q8_0.gguf`

It found a 16,000-byte executable at
`/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server`.
That remote binary was not hashed, and no remote model search or transfer was
performed. Its runtime identity and the protected model's presence on the
notebook therefore remain live gates. The local protected model exists with
the receipt byte count; it was metadata checked only in this preparation.

## Root-owned live sequence

Root should first start the protected CPU server under its own existing
notebook supervisor, with one slot and the accepted b4 context/CPU settings.
The client must be run through a loopback URL, for example:

```sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
/usr/bin/python3 docs/campaign/work/b4-notebook-baseline-preparation/run03_b4_notebook_baseline.py \
  --url http://127.0.0.1:18099 \
  --model /home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf \
  --runtime /home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server \
  --repo-root /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb \
  --output /home/m0hawk/.local/state/sepalith-campaign-20260915/RUN-03-b4-notebook-baseline-v1/per_cycle.jsonl \
  --summary /home/m0hawk/.local/state/sepalith-campaign-20260915/RUN-03-b4-notebook-baseline-v1/summary.json
```

The server's `/props.model_path` must exactly equal the model path supplied to
the client and `default_generation_settings.n_ctx` must be 8192. If a
forwarded notebook server reports a remote path, root may pass that exact
reported path with `--server-model-path` only after separately recording that
the protected model hash is present on the server host. The client refuses
non-loopback URLs, missing local protected artifacts, changed renderer/parser
sources, wrong server model paths, and wrong context sizes.

The output directory and both files must be fresh. Each JSONL row is flushed
and `fsync`ed before the next cycle; summary and directory `fsync` errors are
fatal. No archive is made by this client. Root should retain the server
supervisor launch/terminal, host load and memory, cancellation/application
events, and a final archive identity alongside the two client artifacts.

## Validation

The framework-free test command is:

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
  docs/campaign/work/b4-notebook-baseline-preparation/test_run03_b4_notebook_baseline.py
```

The tests load the actual pinned renderer and accepted receipts, exercise
both synthetic phases through mocked HTTP responses, check the native stop
and tokenizer contract, prove operation/family labels cannot alter request
settings, reject wrong model/context preflight, validate the parser's no-op
path, and verify fresh-output and loopback/deadline guards. They do not start
a server and do not hash or load model weights.

This packet closes preparation only. It supplies no notebook latency,
cancellation, editor-application, host-load, server-cleanup, or model-presence
evidence until root completes the live run.
