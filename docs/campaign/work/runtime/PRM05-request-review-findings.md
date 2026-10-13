# PRM-05 request lifecycle independent review

**Observed:** 2026-09-12T06:41:00+02:00. **Owner:** worker-runtime. **Status:** reviewed with two reproducible lifecycle gaps and the declared one-active policy verified.

This was a read-only review of the execution worktree. No extension source, model, server, host configuration, data, or final-set artifact was changed or started. The retained probe is an offline mocked-editor fixture; it bundles the current execution `extension.ts` and forbids process/network operations.

## Inputs and hashes

The reviewed execution HEAD was `a79d6de38890355ec66d7227c974140160ab314e`.

| Input | SHA256 |
| --- | --- |
| `extensions/vscode-sepalith/src/campaign_requests.ts` | `a502c85b15e35d76a4a90be1e0ad5b7c76008c5f969ad27c2be49123a93d7b68` |
| `extensions/vscode-sepalith/src/extension.ts` | `ee2f561a90930d88abb224e74a7000618d5cfcbb136aad0d31053d7870b74cc4` |
| `extensions/vscode-sepalith/src/campaign_client.ts` | `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333` |
| `extensions/vscode-sepalith/scripts/check-campaign-requests.ts` | `e4af1e10243f20120e0a6d17ab455a064f0fdc9046fb12c687347b18ca93f5cd` |
| `extensions/vscode-sepalith/scripts/check-provider-requests.cjs` | `0acd95f7de5ed5dc44364bcb9950c01d865d53502c27712a2415bfb6efcdb6fd` |
| `docs/campaign/work/runtime/PRM05-request-review-probes.cjs` | `23ec4f83ca27e7738bf2c894f83ad3041192c057a625188f2ea95c4a7a4bc064` |

The `campaign_client.ts` hash is recorded for completeness because primary transport was inspected; the four source hashes requested for the lifecycle harness are also recorded in the JSON receipt.

## Reproduced gaps

### PRM05-RQ-001 — settings changes do not invalidate an active response

**Trigger:** Start a primary request, let it wait inside `client.complete`, change `sepalith.modelPath`, then resolve the old completion before any new provider invocation.

**Observed:** The old request returned one `InlineCompletionItem` after the model path changed. The exact retained probe output is:

```text
config-race: modelPath changed during request; old response published (1 item)
```

The provider key snapshots `cfg()` at invocation (`extension.ts:763-779`), but `providePrimary` accepts a response using only request generation, VS Code cancellation, URI, version, and full-buffer hash (`extension.ts:676-683`). The activation wiring has no `workspace.onDidChangeConfiguration` subscription (`extension.ts:1010-1126`). `Sidecar.setState` invalidates only when the state value changes (`extension.ts:227-232`), so a settings mutation alone leaves the old lease current. A later provider invocation with a changed configuration key would supersede it, but the setting event itself does not.

This applies to model/profile-affecting settings such as `modelPath`, `manifestUrl`, `port`, `backend`, `threads`, `gpuLayers`, and `contextSize`. For `port` or a managed model path, a new request can also observe a configuration that no longer describes the already-running child.

**Minimal fix proposal:** On relevant configuration changes, invalidate the provider before another response can publish. Bind each lease to a runtime/configuration generation or captured profile fingerprint and compare it at every response/cache publication. Settings that change the managed child binding need an explicit stop/start or a refusal until the child is restarted; invalidation alone prevents stale display but does not reconfigure the child.

### PRM05-RQ-002 — completion requests have no bounded deadline

**Trigger:** Route through the external legacy path and make the mocked `/v1/completions` transport never settle while the caller, document, and sidecar remain live.

**Observed:** The provider remained pending after 75 ms. The retained probe output is:

```text
timeout: non-resolving transport remained pending after 75 ms
```

The legacy request calls `postCompletion` with the lease signal but no deadline (`extension.ts:120-130`, `extension.ts:858-926`). The primary default transport also passes only the lease signal to `/tokenize` and `/completion` (`campaign_client.ts:85-100`, `campaign_client.ts:140-175`). The existing 500 ms scope race and sidecar readiness timeouts do not bound model completion.

This is a responsiveness and resource-lifetime gap. An editor caller can remain pending until cancellation, edit, close, state invalidation, or a server response. If a transport ignores abort, cancellation returns the caller but can leave the old provider promise retained indefinitely.

**Minimal fix proposal:** Add a profile or request-policy deadline to both legacy and native completion paths. Race the operation against the deadline so the provider promise settles even when a transport ignores `AbortSignal`, abort the underlying signal, and classify the result as transient/aborted. Keep the actual deadline value as a lead/profile decision rather than inventing it in this review.

## Verified lifecycle behavior

The direct request helper fixture passed all four scenarios:

```text
node --experimental-strip-types --no-warnings extensions/vscode-sepalith/scripts/check-campaign-requests.ts
check-campaign-requests: 4 lifecycle scenarios passed
```

It covers same-key subscribers with independent cancellation, different-key supersession before and after provider start, last-subscriber cancellation with a fresh retry, and rejection releasing the owner.

The actual provider mock fixture passed all four scenarios:

```text
node extensions/vscode-sepalith/scripts/check-provider-requests.cjs /home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith/node_modules/esbuild
check-provider-requests: 4 actual provider lifecycle scenarios passed; no server or model started
```

Those cases verify same-snapshot sharing with one caller cancelled, out-of-order LSP scope resolution, a legacy request whose lease remains live through HTTP and parsing, and an abort-ignoring primary response that cannot overwrite the newer cache. A transient primary error remains retryable. The retained independent probe also passed three scenarios: the two gaps above and two distinct documents under the global one-active policy:

```text
node docs/campaign/work/runtime/PRM05-request-review-probes.cjs /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912 /home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith/node_modules/esbuild
config-race: modelPath changed during request; old response published (1 item)
timeout: non-resolving transport remained pending after 75 ms
cross-document: global one-active policy superseded A; B published (starts=2)
PRM05 request-review probes: 3 scenarios passed
```

The cross-document result is consistent with the product specification's “Only one request in flight; cancel the previous” rule (`extensions/vscode-sepalith/SPEC.md:85-87`). `SuggestionRequests` keeps one active key (`campaign_requests.ts:17-47`); a different document or cursor supersedes the previous request, while same-key callers retain independent cancellation (`campaign_requests.ts:49-74`). This review found no same-key ownership or stale cursor/document publication bug in the tested paths.

The pinned TypeScript compiler also passed:

```text
/home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith/node_modules/typescript/bin/tsc --noEmit -p extensions/vscode-sepalith/tsconfig.json --typeRoots /home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith/node_modules/@types
```

## Bounds and remaining checks

The mocks do not provide a live VS Code host, real `CancellationToken` delivery, real LSP transport, real HTTP timing, managed child, or native model. They therefore do not establish host responsiveness, a suitable deadline value, or model/profile identity after a settings change. No external Opus result was used. Parent review remains responsible for choosing the config-generation/restart behavior and the latency budget before live host validation.
