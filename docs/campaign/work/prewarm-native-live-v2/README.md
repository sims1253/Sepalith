# RUN-04 native prewarm A/B harness v2

This packet is a launchable measurement harness for one pinned TRAIN fixture
row per process. It is a preparation artifact; it does not start a server,
load a model, or claim a prewarm speedup.

The accepted v1 exact adapter is copied under `src/`. The v2 copy rejects a
native response with `truncated: true` for both the cap-1 warm request and the
foreground request. A cap stop is admissible for warm evidence only when the
server reports `truncated: false`; the warm token is never parsed or published
as an edit.

`fixture_loader.ts` verifies the 38,525-byte fixture SHA, manifest SHA, TRAIN
split, renderer/tokenizer identity, row ID/order, sidecar prompt SHA, exact
context rendering, replacement identity, and prompt token prefix. It streams
the context sidecar until the requested row. The returned packet contains only
the prompt prefix IDs; target arrays, target text, family and operation never
enter the adapter. Every snapshot calls `prepare(..., false)`, including the
fixture row whose source family is `no_op`.

The live harness sends the same BOS-inclusive integer prompt to the two arms.
`--off` performs one foreground `/completion` request with `n_predict=192`.
`--on` schedules one cap-1 request and then performs the same foreground
request after the real debounce timer. The scheduler owns one slot, and the
foreground has a five-second wall deadline including its wait for that slot.
Each request records monotonic and epoch timestamps, client-side queue wait,
raw completion, stop metadata, generated IDs, prompt-ID digest and elapsed
time. Server queue time is explicitly `null` because it requires correlated
server logs. An `AbortController` is attached to the foreground deadline.

The URL must be an HTTP loopback origin. The root launcher owns the native
server and must restart it or otherwise establish the requested cache state
between arms. This harness has no server lifecycle controller. Supply a new
`--run-id` and an exclusive output directory for each case; an existing result
file is never overwritten.

## Root launch commands

Run one command in one process for each arm/case. The server must already be
ready on `127.0.0.1:18403` and the root must capture its startup/device logs
separately. Replace the run ID and output directory with fresh values for every
invocation.

```sh
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
FIXTURE="$PLAN/docs/campaign/work/lead/prewarm-fixture-inputs/native-probe-train-fixture.jsonl"
MANIFEST="$PLAN/docs/campaign/work/serving-readiness/native-probe-train-fixture.manifest.json"
SIDECAR=/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/context-sidecar.jsonl
OUT="$PLAN/docs/campaign/work/prewarm-native-live-v2/results/off-case3-$(date -u +%Y%m%dT%H%M%SZ)"
PYTHONDONTWRITEBYTECODE=1 node --experimental-strip-types \
  "$PLAN/docs/campaign/work/prewarm-native-live-v2/harness/native_live_ab_harness.ts" \
  --arm off --fixture-index 3 --fixture "$FIXTURE" --manifest "$MANIFEST" \
  --sidecar "$SIDECAR" --base-url http://127.0.0.1:18403 \
  --run-id off-case3-unique --output-dir "$OUT"
```

The matching prewarm arm changes only `--arm on`, `--fixture-index`,
`--run-id` and `--output-dir`; it must use a fresh server/cache run. Cases 0,
1, 2 and 3 are the four exact TRAIN rows in manifest order. A result with a
native error is still written for diagnosis and exits with status 2. The
result status is evidence of protocol/transport completion, not a quality or
latency promotion.

## CPU transport tests

```sh
PYTHONDONTWRITEBYTECODE=1 node --experimental-strip-types --test \
  "$PLAN/docs/campaign/work/prewarm-native-live-v2/tests/native_live_v2.test.ts"
```

The tests bind a local ephemeral HTTP server only. They exercise all four
fixture joins, off-arm nonzero request count, on-arm warm/foreground order and
identical IDs, one outstanding request, abort propagation, and rejection of
`truncated: true` for both request kinds. They do not exercise native model
quality, cache reuse, or server-side queue timing.
