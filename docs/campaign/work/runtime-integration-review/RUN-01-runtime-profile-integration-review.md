# RUN-01/RUN-02/RUN-09 runtime profile integration review

**Observed:** 2026-09-12T16:53:06Z. **Scope:** CPU-only source and receipt
review; no build, install, model read, server, GUI, SSH, CUDA, Vulkan, or
state operation was performed by this review.

**Decision:** the current 0.0.7 extension is source/build ready for review, but
it is not ready to launch an adapted PRM-03 runtime through the extension under
the frozen profile. The native request client is largely contract-correct. The
child process and release manifest do not carry enough runtime information to
bind the required batch, HTTP, backend, graph, library, quantization, and model
identity settings.

## Inputs and evidence class

| Input | Identity / evidence class |
| --- | --- |
| EXEC extension | `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`, detached HEAD `a7345e35219ceb624957115c39b869101026a817`; current package/source hashes are recorded below. Source evidence only. |
| Build packet | `docs/campaign/receipts/RUN-01-editor-build-preparation.json`, SHA-256 `3133f7520195eda86d545eeade692496495312690eef524e85ad7fde3dde0d50`; its VSIX is `36517` bytes, SHA-256 `a37097d0d8bcdd08b2b6acf404e48be107f7f9e0c870d4fb4830a4d9ec42f910`. Build/protocol/runtime/process checks passed; GUI, model load, server readiness and editor acceptance remain pending. |
| Frozen profile addendum | `docs/campaign/work/editor-build/RUN-01-editor-build-profile-v2.md`, SHA-256 `9c2c874b84565b67a4e49a890a89dc2f19aff38f739f871b9f0b7fe04cf367b8`; context `4096`, batch/ubatch `256/256`, parallel `1`, six generation and batch threads, and host-specific GPU layers. |
| Pinned llama.cpp | `/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453`, commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`, tree `f9a9f82f92eb23b6dbc05494e542ddb1f907a0c4`, tracked archive SHA-256 `f2c0528a047e084862dd9c34103c7cdb2de394526dbd0ae62929d85123cf6f34`. Source evidence only. |
| Existing runtime build | `RUN-01-runtime-preparation.json`, SHA-256 `8add5fec981244227970d7a55dde82b4b9a3c19bddd38c781972a784a1452a27`; m0pad CPU AVX2 binary is version `0.1.0-dev (build 10453, commit 3cb7ffb1a1)` with binary SHA-256 `e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6`. This is a historical build receipt, not a live check here. |
| Existing Vulkan profile | `RUN-09-q8-vulkan-profile-admission.json`, SHA-256 `d8fa023b5096ded043a0cbef1e8e8667c69a8bbc0106c2a86468ad266e4070a9`; historical/profile evidence uses b10453, `-b 256 -ub 256 -c 4096 -t 6 -tb 6 -ngl 99`, model SHA-256 `f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256`. It does not prove extension integration or the current SFT model. |
| Vulkan build artifact | `docs/campaign/work/lead/RUN-01-vulkan-build-terminal.json`, SHA-256 `6f4b5c3064e74cd25013e142417aa6cc71aa66a7591d442a978d74f5a75ba7ed`; b10453 Vulkan server `/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-vulkan-avx2/bin/llama-server`, SHA-256 `92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f`. Build-only evidence. |

Current EXEC source hashes, read at the review timestamp:

| File | SHA-256 |
| --- | --- |
| `extensions/vscode-sepalith/package.json` | `3bcaa631b43076bce5f6913bd3ccf252401172698c669ec0b31dc739c80900fa` |
| `extensions/vscode-sepalith/src/extension.ts` | `86367f6615e02e59bed425dea8b33f0c210aaf1c0be6a9c6d91e840111bf5cd9` |
| `extensions/vscode-sepalith/src/runtime.ts` | `3a26fb1677eb97488fa03491ba795ec0cca264db24af8f5f7db983bf46e5bfbf` |
| `extensions/vscode-sepalith/src/campaign_client.ts` | `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333` |
| `extensions/vscode-sepalith/src/campaign_protocol.ts` | `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824` |
| `extensions/vscode-sepalith/src/campaign_requests.ts` | `4fdc6a881b93ae8beb9614c2b210f80fd5b0a2d5592ae02c8174345612a17535` |

## Findings

### R-01 — launch flags are not profile-bound (release blocker)

`Config` has only `threads` and `contextSize` for launch geometry
(`src/extension.ts:31-46`), and the package exposes no batch, ubatch, batch
threads, HTTP threads, seed, or graph setting (`package.json:105-121`). The
managed child receives only `-c`, `--parallel 1`, `-t`, and `-ngl`
(`src/extension.ts:374-387`). It does not receive `-b 256`, `-ub 256`,
`-tb 6`, `--threads-http 2`, or a reproducible seed.

The pinned server accepts all of these options: `-t/-tb` at
`common/arg.cpp:1514-1533`, `-b/-ub` at `common/arg.cpp:1658-1671`, and
`--threads-http` at `common/arg.cpp:3508-3514`. Omitting HTTP threads is
material: b10453 defaults `n_threads_http` to `-1` (`common/common.h:609-612`)
and expands it to `max(n_parallel + 4, hardware_concurrency - 1)`
(`tools/server/server-http.cpp:311-315`). On the 12-logical-CPU m0pad this is
11 HTTP threads rather than the frozen 2. The server's batch and ubatch defaults
are also left to the binary/model configuration.

The launch path also has no graph setting. In b10453, CUDA graph optimization
is enabled only when `GGML_CUDA_GRAPH_OPT=1`, is read once per process, and
requires an active CUDA graph and one device (`ggml/src/ggml-cuda/ggml-cuda.cu:4318-4343`).
Vulkan graph optimization is enabled by default and disabled by the presence of
`GGML_VK_DISABLE_GRAPH_OPTIMIZE` (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:6186-6187,17410-17418`).
The extension inherits the VS Code process environment without recording either
choice.

**Smallest implementation proposal:** make the primary manifest's selected
bundle carry a required, strict launch profile containing `contextSize`,
`batchSize`, `ubatchSize`, `parallel`, `threads`, `threadsBatch`,
`httpThreads`, `gpuLayers`, `seed`, graph environment, and b10453 source/build
identity. Return that profile from `provision()` and construct the managed
argument array from it. Set graph environment explicitly in the child
`env`. Keep legacy profiles separate and reject a primary launch with missing
profile fields. A wrapper with the exact command below is an interim lead-owned
preflight, but the current extension cannot claim that it bound those flags.

### R-02 — desktop CUDA is not representable by the extension (release blocker
for that route)

The runtime backend union is `cpu | vulkan | metal`
(`src/runtime.ts:48`), the package setting has the same enum
(`package.json:154-164`), and `detectBackend()` never returns CUDA
(`src/runtime.ts:175-195`). `Bundle.backend` and `selectBundle()` therefore
cannot select a desktop CUDA bundle. A manually launched CUDA server cannot be
attached as a primary sidecar: a configured `serverPath` bypasses managed
manifest loading, leaves `primary` unset, and routes requests through the
legacy text endpoint.

The notebook Vulkan route is representable, subject to R-01 and R-04. The
reviewed profile documents `ngl=43` for m0pad Vulkan while the current assignment
and the RUN-09 profile receipt use `ngl=99`. Root must resolve this host-profile
conflict and record one value before launch; this review does not silently pick
between them.

If desktop CUDA is selected, the minimal source change is to add a strict
`cuda` backend through the manifest union, selection/detection, and child
profile, plus explicit `GGML_CUDA_GRAPH_OPT`. Otherwise mark CUDA unsupported
for the extension and use the notebook Vulkan profile only.

### R-03 — model, quant, renderer and runtime provenance are only partially
bound (release blocker)

`validateManifest()` verifies the model asset's URL, size and SHA-256 and pins
the PRM-03 renderer/tokenizer/protocol constants
(`src/runtime.ts:102-140`). That is useful protocol identity evidence. It does
not verify the model's GGUF architecture or quantization, and `modelRevision`
is only required to be a nonempty string (`src/runtime.ts:114-118`). The
primary profile has no model SHA field that must equal `m.model.sha256`, no
quantization field, no llama.cpp source commit, no binary/library hashes, and
no graph/backend build identity. The server bundle has only platform, arch,
backend, executable name and assets (`src/runtime.ts:49-50,145-155`).

The installer places assets in a cache directory
(`src/runtime.ts:293-319`) but does not tie the selected binary to the declared
model profile or verify the selected backend's runtime behavior. `selectBundle`
falls back from a requested non-CPU backend to a CPU bundle
(`src/runtime.ts:197-202`), so an unavailable Vulkan request can silently
become CPU unless the caller records the selected bundle.

**Smallest implementation proposal:** require a primary `modelSha256` equal to
the top-level model asset SHA, require a host bundle launch profile with the
pinned b10453 commit/build and library hashes, and require an explicit
architecture/quantization admission field supplied by the model owner. Reject
an explicitly requested backend when its bundle is absent; report an `auto`
fallback as a distinct selected backend. The extension still needs the owner
model admission to verify the GGUF header/embedded tokenizer; the current
manifest cannot perform that check itself.

### R-04 — managed library paths are not bound (release blocker for packaged
runtimes)

The CPU build receipt records an absolute ELF RUNPATH at the original build
`bin` directory (`RUN-01-runtime-preparation.json:123-155`). `provision()`
installs the server and sibling files under a new cache runtime directory, but
`Sidecar.spawn()` supplies no loader environment (`src/runtime.ts:304-314`;
`src/extension.ts:374-388`). A managed installation can therefore fail to find
its packaged `.so` files or resolve stale libraries at the old absolute path.
The same manifest shape has no backend-library path or hash roles for CUDA or
Vulkan.

**Smallest implementation proposal:** rebuild the server with a verified
`$ORIGIN` RUNPATH and assert it with `readelf -d`, or return the installed
runtime directory and set `LD_LIBRARY_PATH` to that directory in the managed
child environment. Include every sibling library hash in the primary bundle
record and perform one clean-cache dynamic-load smoke. The current direct
m0pad build is usable as a host artifact only because its original siblings
are present; that is not packaged-runtime evidence.

### R-05 — a manual configuration can bypass the primary profile (release
blocker for adapted weights)

Manifest preflight occurs only when `!c.serverPath && c.manifestUrl`
(`src/extension.ts:309-320`), and `primary` is assigned only after managed
`provision()` (`src/extension.ts:343-353`). Setting both `serverPath` and
`manifestUrl` skips the primary manifest entirely. The provider then uses the
legacy `/v1/completions` route with `max_tokens: 320`
(`src/extension.ts:923-970`), while the package defaults remain context `8192`,
threads `8`, and GPU layers `0` (`src/extension.ts:48-65`; `package.json:105-115,
166-172`). This permits a PRM-03 model to be paired with the b4 protocol and
legacy defaults.

**Smallest implementation proposal:** when `manifestUrl` is set, always load
and classify it before accepting manual paths; reject a primary profile with a
manual `serverPath`, or require a separately admitted manual launch profile.
For the current managed route, root must leave `serverPath` empty, set
`backend=vulkan`, `contextSize=4096`, and `threads=6`, and use the primary
manifest. The package default `8192/8` remains an unaccepted development
fallback.

### R-06 — native request path is source-compatible, with two RUN-02 gaps

The native client performs the required sequence:

1. `/tokenize` receives text with `add_special:false`, `parse_special:false`
and `with_pieces:false` (`src/campaign_client.ts:140-147`).
2. It prepends exactly BOS ID `0` (`src/campaign_client.ts:157-166`).
3. `/completion` receives integer `prompt`, `n_predict` from the validated
profile cap, `temperature:0`, `stream:false`, `return_tokens:true`, and
currently `cache_prompt:true` (`src/campaign_client.ts:168-175`).
4. It rejects over-cap output, truncation, early/native control IDs, a
noncanonical terminal, non-`eos` stop type, and parser-invalid text
(`src/campaign_client.ts:186-219`).

The pinned server appends every sampled ID, including EOS, when
`return_tokens` is true (`tools/server/server-context.cpp:1766-1774`) and
returns the non-stream final token list, `tokens_evaluated`, `stop_type`, and
truncation fields (`tools/server/server-context.cpp:2035-2054`; `tools/server/server-task.cpp:342-360`).
The client currently treats `tokens_evaluated` as optional, although b10453
returns it; a strict primary client should require equality to the full
manual-BOS prompt length.

The five-second request deadline is genuinely combined at the extension
request-manager level: `SuggestionRequests` starts its timer before invoking
`performInlineCompletionItems` (`src/extension.ts:840-864`; timer/abort logic
`src/campaign_requests.ts:54-83`). Scope work, tokenization and completion all
run under that shared signal. This is a client cancellation deadline. The
server source documents that a timed-out prefill may continue briefly and
change slot/cache state; it is not proof of a hard server stop.

For RUN-02, the extension client hardcodes `cache_prompt:true`, so the editor
path has no fresh (`false`) arm. The server supports both modes and common-prefix
reuse (`tools/server/server-context.cpp:3123-3137`). Use the separate native
fixture harness for the fresh/cache comparison, or add a test-only client
parameter with default `true`; do not call the quality evaluator's
`cache_prompt:false` policy a serving result. Also ensure the primary manifest
sets `maxOutputTokens=192` for the editor/serving profile. The same field is
used by `nativeClient()` (`src/extension.ts:270-281`), so a DEV profile carrying
`512` would send `n_predict=512` and violate the serving target.

The startup readiness probe remains generic `/v1/completions` text `"x"`
(`src/extension.ts:419-439`). It proves an OpenAI-compatible response, not the
managed native `/tokenize` plus integer `/completion` contract. Add a native
readiness probe after profile provisioning or have the lead run the native
smoke before GUI acceptance.

### R-07 — primary runtime test coverage and lint attribution are incomplete

The build packet records `npm run build`, protocol, runtime, process and
request checks as passing, but `npm run lint` failed with 79 anti-slop errors
and 3 warnings. The receipt calls them “pre-existing dirty source/helper
files”; it supplies no file-level diagnostics. The EXEC checkout was already
dirty across `extension.ts`, `runtime.ts`, campaign source, helper scripts and
other campaign work. Therefore the 79 are known at the packet's start, with
no errors introduced by this review, but they cannot be classified individually
as historical or campaign-source findings from the stored evidence. Treat lint
as an unattributed open quality gate; the compile/bundle pass is not a lint
pass.

The 58 `check-runtime` fixtures are also legacy-heavy: a read-only count found
56 `zeta2-v1` profiles, one missing profile, and one `zeta3`; there is no
`zeta2-prm03-v1` primary manifest fixture. `scripts/check-runtime.ts:9-25`
tests a legacy `zeta2-v1` manifest. The new primary validation/provision branch
therefore has no fixture-backed test in the recorded build check. Add one
frozen primary manifest fixture and assertions for runtime-profile fields,
backend selection, primary/manual rejection and library environment before
calling the package release-ready.

## Lead-only preflight recipe

This is a recipe, not an execution record. Run it only after the current merged
SFT GGUF, model SHA, tokenizer/renderer admission, and host profile are
accepted. It binds the latest assigned notebook Vulkan value `ngl=99`; the
profile-v2 document's `ngl=43` discrepancy must be resolved in the receipt
before the command is accepted.

~~~sh
set -eu
BASE=/home/m0hawk/.local/share/sepalith-campaign-20260915
BIN=$BASE/build-b10453-vulkan-avx2/bin/llama-server
MODEL=<hash-admitted-prm03-gguf>
RUN_DIR=$BASE/runtime-integration/<fresh-run-id>
mkdir -p "$RUN_DIR"
sha256sum "$BIN" "$MODEL" > "$RUN_DIR/inputs.sha256"
# Vulkan graph optimization is enabled by default; do not set the disable env.
env -u GGML_VK_DISABLE_GRAPH_OPTIMIZE \
  "$BIN" -m "$MODEL" --alias sepalith --temp 0 \
  --host 127.0.0.1 --port 18099 \
  -c 4096 -b 256 -ub 256 --parallel 1 --threads-http 2 \
  -t 6 -tb 6 -ngl 99 --seed 20260905 \
  >"$RUN_DIR/server.stdout" 2>"$RUN_DIR/server.stderr"
~~~

For a desktop CUDA arm, current 0.0.7 cannot bind the backend. A lead-owned
CUDA wrapper would need a CUDA-enabled b10453 binary and an explicit graph arm,
for example `env GGML_CUDA_GRAPH_OPT=0 ...` or `env GGML_CUDA_GRAPH_OPT=1 ...`,
with the same `-c/-b/-ub/-np/-t/-tb/--threads-http` values. The wrapper must
record binary, library, model, graph environment, and exact argv hashes.

For an extension smoke after source/profile integration, use a primary manifest
with `serverPath=""`, `modelPath=""` (or a SHA-verified model override),
`backend="vulkan"`, `contextSize=4096`, `threads=6`, and
`requestTimeoutMs=5000`. Confirm the child command contains every required
flag, the loader resolves the installed sibling libraries, then issue one
native tokenize/completion request before any GUI request. Keep the current
extension port and any native benchmark port separate.

## Acceptance state

| Gate | State | Reason |
| --- | --- | --- |
| b10453 source identity | source-backed pass | Commit/tree/archive are pinned. |
| Native BOS/EOS/return-token protocol | source-backed pass | Client and pinned server agree; no live current-model check. |
| Combined 5-second request deadline | source-backed pass | Shared request timer covers scope, tokenize and completion; cancellation is not hard-stop proof. |
| Lifecycle ownership/termination | source-backed capability | Existing process-lifecycle source and checks pass; no termination was exercised here. |
| Context/parallel binding | partial | `-c` and `--parallel 1` are passed, but managed primary startup rejects default context 8192 unless root configures 4096. |
| Batch/ubatch/HTTP/tb/seed/graph binding | fail | Spawn and manifest omit them. |
| Backend choice | fail for desktop CUDA; partial for Vulkan | CUDA is absent from the extension union; Vulkan selection needs strict profile and no fallback ambiguity. |
| Model/quant/tokenizer compatibility | partial | Protocol identity is pinned, but model SHA/architecture/quant/tokenizer embedding are not checked by the runtime code. |
| Packaged library resolution | unresolved/blocking | Historical CPU binary has absolute RUNPATH; managed cache loader environment is absent. |
| RUN-02 cache/fresh fixture | partial | Client always sends `cache_prompt:true`; separate harness or parameterization is required for fresh mode. |
| Primary manifest test coverage | fail | Recorded runtime fixtures contain no primary PRM-03 manifest. |
| GUI/editor/live current SFT acceptance | unresolved | Lead-owned and intentionally unrun. |

**Next concrete action:** root resolves `ngl=43` versus `ngl=99`, admits the
current model/quant/tokenizer and a strict host bundle, then either applies the
small profile/loader integration or records a lead-owned wrapper as the only
serving path. The first live check should inspect the actual child argv and
loader paths, run native `/tokenize` plus integer `/completion` with cap 192,
and persist the exact model/binary/profile hashes before GUI activation.
