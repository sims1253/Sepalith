# PRM-05 request lifecycle fixes

**Observed:** 2026-09-12T07:03:06+02:00. **Owner:** worker-runtime. **Status:** CPU/offline implementation complete; live host and serving checks remain pending.

The accepted lifecycle gaps are fixed in the released execution files. A configuration event now increments a runtime generation and invalidates active work. Changes to managed binding settings (`modelPath`, `serverPath`, `manifestUrl`, `port`, `backend`, `gpuLayers`, `threads`, or `contextSize`) mark the managed sidecar as restart-required. Suggestions refuse while that flag is set. Only an explicit managed stop/start that reaches `ready` with the current binding clears it; configuration changes never launch or restart a server automatically. Harmless settings, including `requestTimeoutMs` and `scopeContext`, invalidate active work but do not require a sidecar restart.

Each shared request now owns one deadline beginning when the first keyed request is created. Same-key joiners reuse that deadline and cannot extend it. The default is 5000 ms; configuration validation accepts finite values from 100 through 60000 ms. On expiry the helper marks a `TimeoutError`, aborts the shared signal, invokes provider invalidation to clear caches and the request generation, and settles every caller even when the mocked transport ignores abort. A late result fails the generation check and cannot populate the cache. A timed-out key can retry.

The legacy `/v1/completions` route and primary `/tokenize` plus `/completion` route both run inside this shared deadline. Existing independent caller cancellation, legacy `await` lifetime, one-active semantics, and cursor/document stale checks remain intact.

## Evidence

The corrected actual-provider fixture uses a VS Code-shaped configuration event and passes nine scenarios:

```text
node extensions/vscode-sepalith/scripts/check-provider-requests.cjs /home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith/node_modules/esbuild
check-provider-requests: 9 actual provider lifecycle scenarios passed; no server or model started
```

The scenarios cover same-key sharing with independent cancellation, out-of-order LSP scope, legacy lease lifetime, abort-ignoring stale responses, transient retry, configuration-event invalidation, refusal until explicit restart, primary timeout plus late-result/cache protection, legacy timeout, invalid timeout configuration, and harmless timeout-setting change without restart. The short timeout fixture uses 100 ms only as a test value.

The shared helper fixture passes five lifecycle scenarios and five validation assertions:

```text
node --experimental-strip-types --no-warnings extensions/vscode-sepalith/scripts/check-campaign-requests.ts
check-campaign-requests: 5 lifecycle scenarios + 5 timeout-validation assertions passed
```

The fifth lifecycle scenario proves that same-key joining at 25 ms with a 60000 ms requested timeout still expires at the original 100 ms owner deadline, aborts the provider signal, settles both callers with `TimeoutError`, and permits a fresh retry.

The full pinned TypeScript compiler passes:

```text
/home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith/node_modules/typescript/bin/tsc --noEmit -p extensions/vscode-sepalith/tsconfig.json --typeRoots /home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith/node_modules/@types
```

The earlier faulty-behavior review probe remains retained at [PRM05-request-review-probes.cjs](PRM05-request-review-probes.cjs). It still demonstrates the pre-event mutation and no-deadline behavior when no VS Code configuration event is delivered; the corrected fixture exercises the actual event path and short deadline.

## Source hashes

The reviewed execution HEAD is `a79d6de38890355ec66d7227c974140160ab314e`.

| File | SHA256 |
| --- | --- |
| `extensions/vscode-sepalith/src/extension.ts` | `86367f6615e02e59bed425dea8b33f0c210aaf1c0be6a9c6d91e840111bf5cd9` |
| `extensions/vscode-sepalith/src/campaign_requests.ts` | `4fdc6a881b93ae8beb9614c2b210f80fd5b0a2d5592ae02c8174345612a17535` |
| `extensions/vscode-sepalith/src/campaign_client.ts` | `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333` |
| `extensions/vscode-sepalith/package.json` | `3bcaa631b43076bce5f6913bd3ccf252401172698c669ec0b31dc739c80900fa` |
| `extensions/vscode-sepalith/scripts/check-campaign-requests.ts` | `67ceacf0177bf15d6bc69a0b21691d2518fa51323dbf265fbf39a586992d7541` |
| `extensions/vscode-sepalith/scripts/check-provider-requests.cjs` | `b45bdc64391365293b0da26fa112bd790d85f329f28a59286c9635103bf51d14` |
| `docs/campaign/work/runtime/PRM05-request-review-probes.cjs` | `c2bff9789fa1a55e295b7323ecaa8531f55a8834a917b00831bb9f4315336308` |

## Limits

No server, model, CUDA, cloud job, host configuration, real HTTP endpoint, or live VS Code host was used. The 5000 ms default and 100 ms fixture value are policy candidates, not latency optima. A live check must verify that a managed child refuses stale settings until an explicit restart, that native and legacy transports settle under the configured deadline, and that the selected timeout is appropriate for the final profile.
