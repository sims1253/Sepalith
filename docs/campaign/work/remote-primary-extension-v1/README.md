# Remote primary extension candidate

The private extension adds application-scoped `sepalith.remoteBindingPath`, an
absolute JSON file path. An empty setting preserves the existing managed and
legacy routes. A nonempty setting selects only the explicit remote route;
invalid files, identities, or properties fail without trying another route.
No production source, runtime manifest validator, tokenizer gate, or serving
profile was changed. The admitted CUDA profile remains batch/ubatch 256/256,
context 4096, output cap 192 and CUDA graph optimization 0.

The binding contains exactly `schema: 1`, `endpoint`, `instanceId`, `manifest`,
and `backend: "cuda"`. Endpoint must be the literal configured
`http://127.0.0.1:<sepalith.port>` with no path, credentials, alternate hostname,
or remote IP. `instanceId` must be a UUID. The complete manifest must pass the
existing validator and contain the primary renderer and exactly one CUDA
bundle. The source contract has no remote platform selector, so multiple CUDA
bundles are rejected. All nine Linux CUDA runtime library roles remain required.
Root supplies the actual selected Q8 model and CUDA bundle identities in that
manifest; the test fixture is synthetic and must not be deployed.

Startup reads the binding and validates two identity-tagged responses:

* GET `/sepalith/runtime`: exactly schema, instanceId, modelSha256, serverSha256,
  backend, contextSize, maxOutputTokens, renderer and cudaGraphOptimization,
  equal to the validated binding/profile (CUDA and graph optimization 0).
* GET `/props`: actual native `default_generation_settings.n_ctx === 4096`
  and `total_slots === 1`. Those fields exist in b10453 server-context.cpp.

Each startup GET and every native `/tokenize` and `/completion` POST requires
an exact `X-Sepalith-Instance-Id`. Redirects fail. A missing/mismatched header
invalidates the connection and fails the sidecar state. A returned request ID
header is permitted; this candidate does not yet copy it to editor telemetry.
The gateway must preserve the native bodies and supply identity headers on
errors too. The extension trusts the explicit binding and root-controlled
SSH/gateway ownership checks; the HTTP header is not independent proof of
remote file hashes or GPU execution.

Remote clients reuse the unchanged NativeCampaignClient tokenization, integer
prompt, context, EOS, token accounting and output parser gates. Requests carry
the caller's abort signal and a five-second transport bound. The existing
shared request deadline still spans context selection and all serving calls.
A late response is checked after its body is read, so cancellation prevents it
from becoming a successful result even if a test transport ignores abort.

Stop/deactivate abort startup and active remote HTTP, disconnect, invalidate
requests and clear the primary profile. They never spawn or terminate a remote
process. Binding-related setting changes disconnect the active remote session
and require stop/start; they do not silently fall back to the legacy client.
A change during startup cannot publish a ready binding. A different binding file
at the same path is picked up on explicit stop/start; no file watcher is added.

## Review and CPU evidence

`source.diff` contains the three production candidate file changes relative to
EXEC: package.json, src/extension.ts and new src/remote_primary.ts. The complete
private source and its bundle are in `extension/`. `source-baseline.json` pins
the source used, and `artifact-manifest.json` pins the handoff files. Existing
node_modules is a read-only symlink to EXEC dependencies; all bundle outputs
are private. There is no VSIX or installation in this packet.

Run `python3 run_cpu_checks.py` from this directory. It uses one CPU and existing
dependencies, with no package download. The recorded 12 commands passed:
typecheck, extension bundle, 65 synthetic binding/transport assertions, 14 actual
extension lifecycle scenarios under mocked editor/files/HTTP/TCP, the original
10 provider lifecycle scenarios, acceptance identity tests, primary runtime
profile tests, and 38 PRM-05 assertions. The synthetic tests verify rejection
of wrong identity/properties, both POST response headers, preserved integer
prompt/token gates, abort/deadline and late-response handling, stop/start,
settings invalidation, and default legacy/managed-primary behavior. They prove
CPU control flow only; no HTTP connection, model or native process was used.

## Root integration route and acceptance limits

Root reviews source.diff and combines this candidate with the separately
reviewed gateway capsule. Root writes the actual binding after verifying model,
CUDA bundle, process, environment, and native props, then establishes its SSH
loopback forward on the configured port. In the isolated editor use this
private extension bundle, set the application binding path and configured port,
and invoke `sepalith.startServer`. The current work does not run that command.

Live acceptance still requires the selected Q8 response, the verified runtime
and props, actual visible ghost text/first-visible latency, and cancelled/stale
nonpublication in the editor. A client abort or gateway transport release is
not proof of native task release. Root must correlate native evidence if that
property is required. LAN operation remains subject to the reviewed gateway
and SSH lifecycle; this packet never launches or manages those processes.

Rollback is stop/disconnect, restore the previous extension artifact and clear
the application binding setting. Existing primary manifests still reject an
unidentified occupied sidecar. No promotion or final acceptance is claimed.
