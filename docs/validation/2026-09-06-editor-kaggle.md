# Editor and Kaggle readiness, 2026-09-06

The VS Code integration tests pass on 1.85.2 and 1.104.3. One private Kaggle GPU
smoke completed and its outputs passed local verification. The legacy-to-runner
migration remains pending the S1 drain and coordinated follow-on hold.

## Editor evidence

The first extension-host test failed during activation: the provider's display and
partial-accept callbacks required `inlineCompletionsAdditions`, an experimental API.
The fix uses the stable item command for full acceptance and labels the counters
as offered and accepted. No experimental API flag is required.

Further tests cover cancellation, same-prompt retry after failure, rejecting a late
response after editing, actual replacement/accept/undo commands, acceptance cooldown,
switching files while a response is pending, preserving an external server, and
reconnecting after stop. The cooldown assertion first failed with an extra request;
recording the accepted document version now suppresses that request until an edit.

Both version runs passed all eight reported groups. `scripts/check_product.py`
also passed lint, TypeScript compilation, bundling, context/completion/runtime/process
checks, 20 packaging tests and five quant-export tests. Native Zed, Positron and
platform-runtime validation remains on the [release checklist](../EDITOR-ACCEPTANCE.md).

## Remote smoke evidence

- Private job: `m0hawk/sepalith-smoke-9359ae641b544233/1`.
- State: `~/.local/state/sepalith/kaggle-smoke-20260906/state.json`.
- Verified outputs: the adjacent `artifacts/` directory.
- Uploaded source SHA-256: `fc02d444caae549ab71f550f8cbec216febda29e0bc040ec0cbb8eaa3c4f8ac8`.
- Receipt SHA-256: `de4b5498bfcbb21ea72f8985894c17542c0f07c488e7fe56f3ba557e2c51992c`.
- Environment: `/usr/bin/python3.12`, Python 3.12.13, PyTorch 2.10.0+cu128,
  CUDA 12.8, Tesla T4 (capability 7.5). The managed image was observed, not pinned.
- Limits: one submission, 600-second server timeout, 120-second code alarm,
  64 training steps, no internet or external datasets.
- Result: loss 2.35446 to 0.000203179; restored-checkpoint MSE 0.000172976.
  Job code took 11.685 seconds. All numerical acceptance checks passed.
- Operational status: verified. Scientific verdict: not applicable.

The quota API reported 29.25/30 GPU hours remaining before and immediately after
the smoke, with 20/20 TPU hours remaining. These rounded observations do not establish
zero billed usage. No TPU or Benchmark model credit was used.

Eight offline controller tests passed: source mutation, quota rejection, exclusion,
uncertain launch recovery, crash-before-push blocking, terminal failure handling,
corrupt output recovery, and repeat collection without duplicate submission.
The real run tested successful submission and collection. Remote failure/recovery
paths used fake jobs; they have not all been induced against Kaggle itself.
The completed run predates the controller's added timestamp fields; its original
state is preserved without invented submission or verification times.

## Archive

The [artifact index](2026-09-06-editor-kaggle-artifacts.json) lists the exact source,
metadata, receipt, metrics, checkpoint, state and test logs in the private
`scholzmx/sepalith-raw` bucket. Every indexed object was uploaded, downloaded and
compared byte for byte. Bucket keys contain the SHA-256; the bucket itself permits
mutation. Verify against the Git index when retrieving an object.

For example, with an authenticated Hugging Face CLI:

```sh
hf buckets cp hf://buckets/scholzmx/sepalith-raw/sha256/de4b5498bfcbb21ea72f8985894c17542c0f07c488e7fe56f3ba557e2c51992c/receipt.json /tmp/sepalith-smoke-receipt.json
sha256sum /tmp/sepalith-smoke-receipt.json
```

## Queue reconciliation

The X5 owner posted full evaluation, verdict and archival completion at 16:20/16:21.
Its primary verdict is NO-WIN; the secondary findings do not change that verdict.
The queue-manager closeout also records the completed Anyscale LR sweep and production
plan v1. Production LR stays at 2e-4. No repeat sweep or parked authorization was
activated here.

A fresh process check still found the S1 supervisor, benchmark and separate-session
server. The historical handoff's S1/S2/V1c plan is not proof of current dispatch state.
The migration still requires the benchmark's evaluation/verdict/archive, explicit
legacy follow-on hold, candidate scope agreement and a fresh process audit. This
remote smoke is not the first real migrated local experiment.
