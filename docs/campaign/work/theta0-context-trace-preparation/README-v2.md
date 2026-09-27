# Theta0 context-aware native transition trace, v2 API correction

This is a narrow derivative of the accepted synthetic RUN-04 transition replay. It preserves the nine event identities, renderer, history, tokenization, request body and 192-token output cap from the original fixture. The original files remain unchanged:

- replay source SHA-256 `341a96dd370a67bcbccfb9d59d50d146dd4be8b5b34c819edb8181d4c055c739`
- fixture SHA-256 `e6aaf98c888cdcc3c3908d98e0e817ae65d8c94c5a17c966a9019a9d8e2bc8ea`

The derivative accepts only `--context 2048`, `--context 4096`, or `--context 8192`. It performs a read-only **GET** `/props` before the first event and requires the actual native field `props.default_generation_settings.n_ctx` to equal the selected profile. A top-level-only `n_ctx` is rejected. This follows the saved b10453 response shape in `prewarm-recovery-a-evidence/health.json` (SHA-256 `545b18cabead8207cd33929dd23d63a2bc8cd7888a1a76b631f00c08217b627b`). A prompt is rejected explicitly when its token count plus the fixed 192-token output budget exceeds that context; the row records `contextOverflowRejected` and does not dispatch `/completion`.

Profile meaning is fixed by contract:

| Context | Use | Deadline |
|---:|---|---:|
| 2048 | native diagnostic trace only | 5 seconds normally, 60 seconds with `--diagnostic` |
| 4096 | primary editor profile | 5 seconds normally, 60 seconds with `--diagnostic` |
| 8192 | stress/latency trace only | 5 seconds normally, 60 seconds with `--diagnostic` |

The v2 event `transportCalls` now retains the complete native response object, including returned token IDs, `stop_type`, `stopping_word`, timings, usage and cache-related counters. The existing projected metrics remain alongside this raw record. The trace is protocol, applicability, overflow and latency evidence; it makes no quality or promotion claim.

Run the synthetic CPU checks without a server:

~~~sh
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types \
  docs/campaign/work/theta0-context-trace-preparation/replay-context-trace-v2.ts --check
~~~

A root-owned server run uses a fresh endpoint whose command binds the selected context, fixed `-t 6 -tb 6 --threads-http 2 --parallel 1 -b 256 -ub 256`, the selected theta0 model/runtime identity, and no unrelated cache. After the server is already running, root invokes one profile at a time:

~~~sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types \
  docs/campaign/work/theta0-context-trace-preparation/replay-context-trace-v2.ts \
  --fixture docs/campaign/work/serving-transition-panel/transition-fixture.json \
  --server http://127.0.0.1:PORT --context 2048 --diagnostic \
  --mode serial --out TRACE-2048.json

node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types \
  docs/campaign/work/theta0-context-trace-preparation/replay-context-trace-v2.ts \
  --fixture docs/campaign/work/serving-transition-panel/transition-fixture.json \
  --server http://127.0.0.1:PORT --context 4096 \
  --mode serial --out TRACE-4096.json
~~~

Use `--context 8192` only for a separately labeled stress trace. Do not treat it as editor support. The duplicate-client path receives the same requested context and 192-token policy as the serial/cancel client; the underlying `NativeCampaignClient` context gate therefore applies consistently. The `--diagnostic` flag changes only the operational deadline from 5 seconds to 60 seconds; it does not alter source, prompt, tokenization, cap, or event order. A real `/props` mismatch, missing `n_ctx`, prompt overflow, timeout, cancellation, noncanonical stop, or stale identity remains visible in the trace and is excluded from any quality denominator.

CPU checks are synthetic and launch-free. This worker did not start a server, model, network operation, SSH session, or GPU process. Live profile support, endpoint availability, raw native behavior, and latency remain root-owned gates.
