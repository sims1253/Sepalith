# RUN-06 prefill and serving options

Prepared 2026-09-12T14:24:08Z for RUN-06 and the RUN-01 serving handoff. This
is a read-only source and receipt audit. It did not start a server, access
SSH, load a model, build a backend, run CUDA, or change a profile.

## Decision for the lead

The first useful screen is three sequential, one-server arms using the same
SFT step-500 Q8 model, native integer-token client, four accepted TRAIN rows,
`cap=192`, `ctx=4096`, `-np 1`, and the same cold/warm request order:

1. **CPU anchor:** existing AVX2/OpenMP build, `-t 6 -tb 6 -b 256 -ub 256
   -ngl 0`. This is the current reference and needs no rebuild.
2. **OpenBLAS:** a new pinned AVX2/OpenMP build with `GGML_BLAS=ON` and
   `GGML_BLAS_VENDOR=OpenBLAS`, otherwise the CPU profile is unchanged.
3. **Vulkan:** a new pinned AVX2/OpenMP build with `GGML_VULKAN=ON`, then one
   bounded `-ngl all` load/health/protocol screen on the Radeon iGPU. A device
   that cannot load or answer the short fixture is a stop, not a fallback
   timing result.

Do not run these servers concurrently. Stop and verify the tracked PID and
port before changing arms. Keep `-fa auto`, `--repack` default, KV offload
default, seed `20260905`, and all renderer/parser/client fields fixed. The
current 5-second result is a feasibility signal, not a passing baseline:
root's live notebook receipt has 8/8 exact prompt IDs, one accepted warm
no-op (prompt 345, TTFT 248.595 ms, wall 976.571 ms, 12 IDs, cache 345 and
one prompt token processed), and seven requests timing out before their first
token. The receipt is [RUN-01-primary500-notebook-baseline-5s.json](../../work/lead/RUN-01-primary500-notebook-baseline-5s.json)
(SHA-256 `fd53fd5f6b3c08d3e51c9a12390209376b5fa248af1da2bdad748a60329ac652`).
Root separately reported an approximately 61 token/s long-prefill and
approximately 14 token/s warm-generation diagnostic; those figures are
current root observations and require the lead's diagnostic artifact before
they can support a promotion decision.

The existing four-row manifest is
`native-probe-train-fixture.manifest.json`, SHA-256
`0b2195b87b5f6eabc892164124c25b876833a00ba6add0d79af6826413f5d0b3`. It
contains two short prompts (345 and 886 prompt tokens before the explicit BOS)
and two long prompts (1,464 and 1,902 before BOS), all from TRAIN/context-only
rows. There is no fabricated 8K case: 8K stress remains unsupported/pending,
and training admission is 4096 while RL contexts are 2048.

## What the pinned server actually does

The source is llama.cpp b10453, commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`, tree
`f9a9f82f92eb23b6dbc05494e542ddb1f907a0c4`, archive SHA-256
`f2c0528a047e084862dd9c34103c7cdb2de394526dbd0ae62929d85123cf6f34`.

* In `tools/server/server-context.cpp:3123-3125`, `cache_prompt=true`
  computes the common integer-token prefix in the slot. With
  `cache_prompt=false`, that common-prefix branch is skipped; it is the right
  cold control but does not prove that a prior cancelled task has stopped.
* At `:3309-3315`, an exact full-prefix hit is reduced by one token so every
  active slot evaluates at least one prompt token. Thus a 346-ID prompt can
  legitimately report `cache_n=345`, `prompt_n=1`, and
  `tokens_evaluated=346`. Compare `timings.cache_n` and `timings.prompt_n`,
  not `tokens_evaluated`, when measuring warm work.
* `tools/server/server-common.cpp:67-80` emits `cache_n`, `prompt_n`,
  `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, and
  `predicted_per_second`. The server README at `:1347-1362` defines context
  occupancy as `cache_n + prompt_n + predicted_n`. Record these fields for
  every accepted request; TTFT includes prompt work, the first decode, and
  client transport overhead.
* `--cache-reuse N` is a different, optional KV-shift path. The server checks
  `llama_memory_can_shift` at `server-context.cpp:1174-1184` and again at
  `:3133-3144`; if the loaded context cannot shift, it logs that cache reuse
  is unsupported and disables the setting. The actual chunk matching and
  shifting are at `:3155-3179`. Ordinary RoPE Llama is a plausible candidate
  for this path, but no current SFT Q8 load-time result proves it. Test
  `--cache-reuse 256` only after ordinary `cache_prompt=true` is measured and
  only on prompts with a changed suffix after a common prefix of at least 256
  IDs. It is not a third arm in this screen.
* `return_tokens` appends each sampled ID, including EOS, at
  `server-context.cpp:1766-1774`; streaming partial responses carry one ID at
  `:1798-1810`, and EOS is classified after the partial response at
  `:1885-1887`. In stream mode the final response intentionally has empty
  `content` and `tokens` at `:2035-2041`. The client must collect partial IDs
  and require terminal EOS ID 1 under the native contract.
* The HTTP layer marks a closed connection at `server-http.cpp:589-600` and
  `:637-647`; `server-stream.cpp:617-629` turns that into `should_stop`. The
  model result poll in `server-models.cpp:1726-1737` wakes at most about once
  per second. Therefore a client 5000 ms timeout is a client deadline and
  cancellation request, not a measured server hard stop. A timed-out prefill
  may continue briefly and may leave slot/cache state. Do not retry it or
  count a later cache hit as a cold result. Cancellation latency remains
  unmeasured in this packet.

The source hashes used for this review are:

| path | SHA-256 |
| --- | --- |
| `tools/server/server-context.cpp` | `26f130b76c27be72e4674943754575cf5efa14b6a6325591be07df57f651e681` |
| `tools/server/server-common.cpp` | `067f3f5db9a0bb72c5c834938eb9469d3d81085d9358b85e3be914f39d42f584` |
| `tools/server/server-http.cpp` | `babd79770e94194a47542089721f8b6dec2e8f5fe027dc2600139e248062347d` |
| `tools/server/server-stream.cpp` | `96cea95271afcb6e359fd5666c3de93e1dc9c94b35b7d961ca3e9d9854575b1a` |
| `tools/server/server-models.cpp` | `20c1373cacd1295993f90e2bef4a796efb353f96746d925f7f6f5798802b90ba` |
| `tools/server/README.md` | `bdbc982fbfecc4cdae490a5062f9d5de05275c28485a89ef8946b526d0f107e2` |

## Exact build and launch recipes

These commands are for root to run on `m0pad`; they do not run in this
packet. The source archive already transferred by RUN-01 is used so the build
identity remains the pinned b10453 tree. Each build directory is independent.

Common variables:

```bash
set -eu
SRC=/home/m0hawk/.local/share/sepalith-campaign-20260915/llama.cpp-b10453
ROOT=/home/m0hawk/.local/share/sepalith-campaign-20260915
MODEL=/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step500-runtime-gguf/model-Q8_0.gguf
```

The model used for all arms is Q8_0 SHA-256
`f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256`. The
root checkpoint merge identity is
`5f62d58e0736fe559a8f0d27fa641a4abebfd6333b2002ee0f938d4a435124ca`.

### Arm 1: existing CPU anchor

Use the already verified binary
`$ROOT/build-b10453-avx2/bin/llama-server` (SHA-256
`e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6`). The
launch shape is:

```bash
BIN=$ROOT/build-b10453-avx2/bin/llama-server
"$BIN" -m "$MODEL" --alias sepalith --host 127.0.0.1 --port 18401 \
  --temp 0 --seed 20260905 -t 6 -tb 6 -c 4096 \
  -b 256 -ub 256 -np 1 -ngl 0 -lv 4 --no-ui \
  >$ROOT/run06-cpu-anchor.log 2>&1
```

The current root server is already this profile. Do not launch a second copy
until its tracked hard stop and port-close check pass.

### Arm 2: OpenBLAS build

Root observed the host prerequisites `openblas 0.3.34-1.1`, CMake, Ninja,
pkg-config, and GCC. This is a host observation, not a worker installation
or independent SSH check. First record what CMake will resolve, then
configure and build with at most two compiler workers:

```bash
pkg-config --modversion openblas
BUILD=$ROOT/build-b10453-openblas-avx2
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF \
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON \
  -DGGML_OPENMP=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_MUSA=OFF \
  -DGGML_METAL=OFF -DGGML_RPC=OFF -DGGML_SYCL=OFF \
  -DLLAMA_BUILD_NUMBER=10453 \
  -DLLAMA_BUILD_COMMIT=3cb7ffb1a
cmake --build "$BUILD" --target llama-server llama-cli --parallel 2
```

Use the exact same runtime profile on port 18402, with OpenBLAS and OpenMP
thread counts bounded to the six physical cores:

```bash
export OPENBLAS_NUM_THREADS=6
export OMP_NUM_THREADS=6
BIN=$ROOT/build-b10453-openblas-avx2/bin/llama-server
"$BIN" -m "$MODEL" --alias sepalith --host 127.0.0.1 --port 18402 \
  --temp 0 --seed 20260905 -t 6 -tb 6 -c 4096 \
  -b 256 -ub 256 -np 1 -ngl 0 -lv 4 --no-ui \
  >$ROOT/run06-openblas.log 2>&1
```

The pinned BLAS backend only supports the larger `MUL_MAT` shapes after
checking contiguous buffers, F32 activations, and dimensions at least 32;
see `ggml/src/ggml-blas/ggml-blas.cpp:394-431`. For quantized weights it
dequantizes into a float work buffer before `cblas_sgemm` at `:31-149`, so
OpenBLAS may lose to the AVX2 Q8 kernels. `ggml-backend-reg.cpp:159-173`
registers BLAS before CPU, and `ggml-backend.cpp:883-887,1193-1233` selects
the first higher-priority backend that supports the buffer and operation.
Record the build log's BLAS detection and runtime logs' backend/device lines;
the build alone does not prove that the measured prefill used BLAS.

### Arm 3: Vulkan build and bounded iGPU smoke

Root observed `vulkaninfo`, `glslc`, the Radeon Vulkan package, and a
world-readable `/dev/dri/renderD128`. The pinned source requires both Vulkan
with `glslc` and `SPIRV-Headers` at
`ggml/src/ggml-vulkan/CMakeLists.txt:9-15`; shader feature probes run at
`:35-53` and generated shaders are part of the build at `:177-243`.

```bash
vulkaninfo --summary
glslc --version
BUILD=$ROOT/build-b10453-vulkan-avx2
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF \
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON \
  -DGGML_OPENMP=ON -DGGML_BLAS=OFF -DGGML_VULKAN=ON \
  -DGGML_VULKAN_CHECK_RESULTS=OFF -DGGML_VULKAN_VALIDATE=OFF \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_MUSA=OFF -DGGML_METAL=OFF \
  -DGGML_RPC=OFF -DGGML_SYCL=OFF \
  -DLLAMA_BUILD_NUMBER=10453 \
  -DLLAMA_BUILD_COMMIT=3cb7ffb1a
cmake --build "$BUILD" --target llama-server llama-cli --parallel 2
BIN=$BUILD/bin/llama-server
"$BIN" --list-devices >$ROOT/run06-vulkan-devices.txt 2>&1
"$BIN" -m "$MODEL" --alias sepalith --host 127.0.0.1 --port 18403 \
  --temp 0 --seed 20260905 -t 6 -tb 6 -c 4096 \
  -b 256 -ub 256 -np 1 -ngl all -lv 4 --no-ui \
  >$ROOT/run06-vulkan.log 2>&1
```

`-ngl all` is intentionally a single bounded trial, not an assumption that
the shared-memory Vega device will be faster or fit every allocation. If the
binary cannot enumerate a device, the model cannot load, `/health` is not
HTTP 200, the short native fixture does not return protocol-valid EOS output,
or the process reports device loss/allocation failure, stop the arm and retain
the CPU anchor. If it loads, run the same four-row cold/warm client once. Do
not set `GGML_VK_PREFER_HOST_MEMORY`, `GGML_VK_ALLOW_SYSMEM_FALLBACK`, or
other tuning environment variables in this first comparison; those would
change the arm's memory path. The Vulkan source hash is
`c73e8f7980cd416f7fd30860c43f85e4ecacb7db84c0bf057cbb505aaf90142f` and its
CMake hash is `95b77526cbede60b669abf1bc13657956284d0b20835096e64f24682a5a1a0e2`.

For both new builds, preserve the sibling-library RUNPATH and capture the
CMake cache, build log, server binary, and all output hashes. A successful
compile is a preparation result; only the paired native run can decide the
arm.

## Measurement and stop rules

The root client should execute one request per row with
`cache_prompt=false`, then the same row with `cache_prompt=true`; no retry,
second repetition, or post-timeout replay is allowed in this bounded screen.
For each request retain the prompt-ID hash, returned partial token IDs,
terminal EOS/stop type, parser status, HTTP status, TTFT, combined wall,
`cache_n`, `prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`,
`predicted_ms`, and timeout/cancellation status. Report cold and warm rows
separately by short/long band. The user deadline is the combined tokenize plus
completion budget of 5000 ms; a longer diagnostic request may be used only as
a separately labeled diagnostic with a maximum of 60 seconds.

Reject an arm on any prompt-ID mismatch, malformed SSE, duplicated or missing
stream token, missing EOS ID 1, parser/marker mismatch, context overflow,
unexpected server error, device loss, or untracked process. A timeout before
the first token is a timeout observation, not a zero-token speed result.
Promotion requires all four rows to pass the protocol contract and a paired
latency improvement against the same anchor; the existing RUN-06 speed gate
remains at least 1.4x end-to-end with bootstrap lower bound above 1.0 and no
p95 regression. The current 1/8 accepted result cannot establish that gate.

## Other options and unresolved gates

* Increasing `-b/-ub` is a direct prefill sensitivity check because the server
  documents them as logical and physical batch ceilings. It is deferred from
  this three-arm screen to keep the BLAS and Vulkan alternatives available;
  root may substitute `-b 512 -ub 512` for the CPU anchor only if the lead
  explicitly records that as a different arm and retains the same hashes and
  stop rules.
* Q6_K and Q5_K_M are supported by the pinned quantizer (`tools/quantize`
  README `:45-67,158-171`) and could reduce weight bandwidth, but no admitted
  SFT Q6/Q5 artifact or quality/parity result exists. A future candidate must
  be quantized from the verified F16 artifact with the pinned quantizer, then
  receive a new model hash and exact output/quality check. It is not part of
  this runtime screen.
* `ngram-mod` remains the first speculation scout from the accepted RUN-06
  audit, with flags `--spec-type ngram-mod --spec-draft-n-max 64
  --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48
  --spec-ngram-mod-n-max 64`. It can reduce decode work after a first token;
  it cannot remove initial prompt prefill and is therefore deferred until the
  ordinary baseline returns protocol-valid first tokens. Released DSpark,
  lighter AR, MTP, EAGLE-3, and DFlash statuses remain exactly those in
  `RUN-06-speculative-paths-audit.json`.
* The extension UI/dependencies and final editor acceptance remain outside
  this CPU source audit. No current measurement proves Vulkan, OpenBLAS,
  cache-reuse, or cancellation benefit on the loaded SFT Q8 model. No 8K
  stress support is claimed.

Estimated root work is 5–15 minutes for the OpenBLAS configure/build, 10–25
minutes for Vulkan configure/shader build, and 10–20 minutes per loaded
four-row native screen. These are estimates, not observed spend or latency;
paid compute remains USD 0.
