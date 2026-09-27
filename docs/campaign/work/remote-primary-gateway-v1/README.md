# Remote primary gateway preparation

This is a root-run candidate. Preparation used synthetic HTTP servers and fake process files only. No production gateway, native server, GPU, SSH session, model read, model hash, or final-data access occurred.

The production entrypoint is `gateway.mjs`. It fixes its listener to desktop `127.0.0.1:18423` and its backend to desktop `127.0.0.1:18403`. Root owns the existing CUDA server and the SSH reverse forward from notebook `127.0.0.1:18403` to the gateway. It must bind the reverse forward to notebook loopback. The gateway does not establish a tunnel, launch native processes, change authentication, or kill the backend.

Root integration:

1. Complete full artifact preflight and lease admission. Supply the full validated primary manifest with exactly one Linux CUDA bundle. The pinned existing validator requires eight common library roles plus the CUDA role (nine total); reconcile any older eight-library inventory before use.
2. Write native metadata with exactly `pid` (integer), `startTick` (decimal string), `uid` (integer), and `modelPath` (absolute path). Write admission metadata with exactly `leaseId` and `preflightReceiptSha256`. The latter identifies root's preflight receipt; the gateway does not independently verify that receipt or hash model/server assets.
3. Run the metadata-only helper below to create a new directory. Give its unchanged `binding.json` to the extension's remote binding configuration. The helper reuses the existing manifest validator and generates one shared UUID.
4. Under root's bounded serving lease, run the production command below. The receipt path must not exist. Root manages earlier lease deadlines and backend termination. Gateway's additional absolute lifetime cap is 30 minutes.

```sh
node make_config.mjs --manifest ROOT_MANIFEST.json --native-identity ROOT_NATIVE.json --admission ROOT_ADMISSION.json --out NEW_CONFIG_DIRECTORY
node gateway.mjs --config NEW_CONFIG_DIRECTORY/gateway-config.json --receipt NEW_GATEWAY_RECEIPT.jsonl
```

The config contains `{schema:1,binding,native,admission}`. The extension binding is exactly `{schema:1,endpoint:'http://127.0.0.1:18403',instanceId,manifest,backend:'cuda'}`. Port and inspector overrides are exported synthetic test seams; the production CLI accepts no upstream, port, or inspector override.

At startup, before native forwarding, and before response publication, the gateway verifies the supplied Linux process epoch and owner. It reads only that PID's stat, environment, TCP table, and descriptor links. It requires one `GGML_CUDA_GRAPH_OPT=0` entry and ownership of the exact backend loopback listening socket. It rejects exited/zombie/reused processes. Native props must match `model_path`, `default_generation_settings.n_ctx=4096`, and `total_slots=1`. Identity failure remains latched. A 100 ms process monitor aborts active transport on process identity loss. These checks bind to root's admitted process and paths; they do not attest loaded tensor bytes or independently establish GPU offload.

`GET /sepalith/runtime` returns exactly schema 1, instanceId, modelSha256, serverSha256, backend `cuda`, contextSize 4096, maxOutputTokens 192, renderer `zeta2-prm03-v1`, and cudaGraphOptimization 0. `GET /props` preserves the original native JSON bytes and fields. Other forwarded routes are GET `/health` and POST `/tokenize`, `/completion`, `/detokenize`. Other paths, query variants, methods, upgrades and CONNECT are rejected. Native request bodies and response bodies/statuses are preserved within limits; hop-by-hop headers are removed. Every HTTP response gets `X-Sepalith-Instance-Id` and a gateway-generated `X-Sepalith-Request-Id`.

The root-authorized `GET /sepalith/observation` is a gateway diagnostic. It checks process identity and makes no native HTTP call. Its exact body is:

```json
{"schema":1,"completionId":null,"phase":"idle","utc":"2026-09-13T12:00:00.000Z","dispatchSequence":0}
```

`phase` is `idle`, `preparing`, or `dispatched`. On completion dispatch, `completionId` is its gateway request UUID and `dispatchSequence` increases. Dispatch means the HTTP request bytes were written to the owned backend socket. It does not prove native task acceptance. On backend transport release, the current ID clears, phase becomes idle, and sequence remains unchanged. The harness must require a sequence newer than its initial observation and a current dispatched ID before editing. A fast completion missed by polling leaves in-flight cancellation evidence pending. The response has `Cache-Control: no-store` and the required instance header.

The JSONL ledger records UTC, desktop gateway monotonic nanoseconds, sequence numbers, request IDs, body byte counts/SHA-256, dispatch, cancellation, HTTP transport release, publication, and shutdown. It records no bodies, token values, arbitrary request URLs, request headers, or environment values. The SHA-256 of a request body supports root's exact prompt/request correlation. `elapsedMs` uses monotonic nanoseconds divided by one million. Timing scope is the desktop gateway; it is neither notebook editor latency nor a cross-host monotonic clock. The diagnostic reveals only current ID, phase, timestamp and dispatch sequence.

Only one completion is admitted until its HTTP transport settles and its final identity check ends. A requester disconnect or deadline destroys the backend request/socket promptly. Receipts explicitly mark native task acceptance/release as unproven: transport closure alone cannot establish when native inference stopped. Root must collect native task cancellation/release logs and enforce exclusive backend use through the serving lease. The gateway's advertised output cap comes from the validated primary manifest; the existing primary client enforces the 192-token request protocol. This proxy preserves request bytes and does not rewrite completion parameters.

Bounds: 1 MiB request body, 4 MiB response body, 16 KiB request headers, eight active requests, sixteen client sockets, one unsettled completion, 15-second total request deadline, 1.5-second props deadline, and 32 MiB receipt. Responses are buffered within the cap, so this route does not provide streaming publication. Receipt failure closes the gateway and aborts its owned sockets. SIGINT/SIGTERM and the lifetime limit close owned sockets with a two-second transport-drain bound; the backend remains untouched. Root must treat absent terminal receipts as incomplete shutdown evidence.

Verification: `node --test --test-concurrency=1 gateway.test.mjs`. Tests use ephemeral loopback ports and fake `/proc` readers; they never launch the production entrypoint. Node 22.18+ is needed for the pinned TypeScript validator import; this run used Node 26.6.0. `vendor/runtime.ts` and `vendor/campaign_protocol.ts` are unchanged snapshots of the accepted extension sources, imported for manifest validation only. Source and test hashes are in `source-test-manifest.json`.

This candidate supplies no final-data admission mechanism. Root's existing final-evaluation gates remain required: both 2026-09-14T10:00Z and an explicit weight/harness freeze receipt. No final evaluation is authorized by this preparation.
