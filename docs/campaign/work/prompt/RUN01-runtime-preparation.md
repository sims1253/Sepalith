# RUN-01/RUN-06 target runtime preparation

Prepared 2026-09-12 on the user-authorized CPU target `m0hawk@192.168.178.40`.
This packet records target intake, a source-only transfer, and a CPU AVX2
llama.cpp build. It does not admit a model, start a server, run a benchmark,
or establish editor acceptance.

## Source and reviewed-patch identity

The pinned local llama.cpp source is
`/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453`.
The observed Git identity is:

- commit: `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`
- commit subject: `model : remove some ggml_concat (#27176)`
- commit time: `2026-08-16T14:12:55+02:00`
- Git tree: `f9a9f82f92eb23b6dbc05494e542ddb1f907a0c4`
- tracked files: `3425`
- deterministic archive command: `git -C /home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453 archive --format=tar --prefix=llama.cpp-b10453/ HEAD`
- archive bytes/SHA-256: `171663360` / `f2c0528a047e084862dd9c34103c7cdb2de394526dbd0ae62929d85123cf6f34`

`git diff --quiet` passed. The only local untracked path was
`BUILD_RECEIPT.md`; it was excluded from the tracked archive and was not
transferred. No reviewed source patch or dirty tracked source was required.
The remote source snapshot is therefore a source archive of the pinned tree,
not a Git checkout; build metadata is supplied explicitly with
`LLAMA_BUILD_NUMBER=10453` and `LLAMA_BUILD_COMMIT=3cb7ffb1a`.

Selected source hashes matched after transfer:

| path | SHA-256 |
| --- | --- |
| `CMakeLists.txt` | `883da9b4f38614ec92b4222e5fe08457330ee4543a7e27ce530c4374342b6e83` |
| `common/speculative.cpp` | `81248dbf9b755f02f200a92ee613b0c32f999c27bd192b5d3bd9b997030818d3` |
| `tools/server/server-context.cpp` | `26f130b76c27be72e4674943754575cf5efa14b6a6325591be07df57f651e681` |
| `src/llama-model.cpp` | `518506c4aaf12a8cbb44f9f12ce3d75e876753f8131d94d6507515cd3a57ff5a` |

## Target observations

SSH used `BatchMode=yes`, `ConnectTimeout=8`, and
`StrictHostKeyChecking=accept-new`. Public-key authentication succeeded. The
negotiated host key was `ssh-ed25519` with fingerprint
`SHA256:Tn5oE3/sIruL5la/jz4RivzqBO5Wb3dKWt4RD0crq6w`.

The host reports CachyOS rolling, kernel
`7.2.3-1-cachyos`, hostname `m0pad`, and an HP EliteBook 845 G8 Notebook PC.
The CPU is an observed AMD Ryzen 5 PRO 5650U with Radeon Graphics (6 physical,
12 logical cores); RAM is 15,656,516 KiB total and 11,916,584 KiB available in
the intake sample. The root filesystem has 354,898,172 KiB available at 64%
use. AVX2 is present and AVX512 was not observed.

The bounded tool check found `tmux 3.7c`, GCC/G++ `16.2.1`, Clang `22.1.8`,
CMake `4.4.3`, Ninja `1.13.2`, GNU Make `4.4.1`, Git `2.55.0`, Node `v26.8.1`,
NPM/NPX `12.0.2`, and VS Code CLI `1.137.0`. `code-server`, `Xvfb`,
`xvfb-run`, `xdotool`, and `weston` were absent. `vulkaninfo` exists, but no
Vulkan or GPU query was run.

The canonical extension source directory exists at
`/home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith`, with package
version `0.0.7`, but its `node_modules` and `dist/extension.js` were absent.
No bounded target runtime path contained `llama-server` or `llama-cli` before
this preparation.

A pre-existing `bench_starling` process was observed during the build and
changed between intake samples (one command used a `build-bench-vk` library).
It was not started, inspected beyond bounded process metadata, paused, or
terminated by this task. Its CPU and memory contention invalidate any timing
interpretation here. No `llama-server` or `llama-cli` process was running
after the build.

## Source-only target build

The tracked archive was unpacked under the only newly authorized remote root:

`/home/m0hawk/.local/share/sepalith-campaign-20260915/llama.cpp-b10453`

Remote inventory found 3425 files and the selected source hashes above. The
build wrapper is
`/home/m0hawk/.local/share/sepalith-campaign-20260915/build_cpu_avx2.sh`, with
mode 700, and its log is
`/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2.log`.
The build directory is
`/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2`.

The exact build command is the wrapper's CMake configure followed by:

```text
export CMAKE_BUILD_PARALLEL_LEVEL=2
cmake --build /home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2 \
  --target llama-server llama-cli --parallel 2
```

CMake used Release/Ninja, `/usr/bin/gcc`, `/usr/bin/g++`, explicit build number
10453 and commit prefix `3cb7ffb1a`, `LLAMA_BUILD_SERVER=ON`,
`LLAMA_BUILD_TOOLS=ON`, and `GGML_OPENMP=ON`. CUDA, HIP, MUSA, Vulkan, Metal,
OpenCL, SYCL, RPC, WebGPU, BLAS, LLAMAFILE, native CPU tuning, AVX512 and
tests/examples were off; AVX, AVX2 and BMI2 were on. No package installation,
sudo, source mutation, model/data transfer, server launch, or benchmark launch
was done.

Build session: `sepalith-run01-runtime` in tmux. It ran from
`2026-09-12T09:38:24Z` through `2026-09-12T09:51:24Z`, used two compiler
workers, reached `326/326`, and exited normally with `BUILD_STATUS=complete`. The observed
tmux/cmake/ninja process IDs were 28308/28440/28441 (pane shell 28309); all
were gone at completion.
The configure warning that the source had no `.git` is expected for the
archive; explicit build metadata made the resulting version identity stable.
The build also reported no UI assets and produced a headless server without an
embedded llama.cpp web UI. The extension communicates with the HTTP API and
still requires its own extension build/GUI check.

Built artifacts:

| artifact | bytes | SHA-256 | version |
| --- | ---: | --- | --- |
| `bin/llama-server` | 16000 | `e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6` | `0.1.0-dev (build 10453, commit 3cb7ffb1a)`; GNU 16.2.1, Linux x86_64 |
| `bin/llama-cli` | 15992 | `2546ca42ca7d31b8e9ea1ad9f43cc6e4542778e2db4458c527e385869d3af323` | `0.1.0-dev (build 10453, commit 3cb7ffb1a)`; GNU 16.2.1, Linux x86_64 |

`CMakeCache.txt` SHA-256 is
`49f50e7660a04ce539cfa885c373a0cda6f6c168cec515e11c6b3aee8fcb48b5`.
Help-only invocation on the built server exposed `ngram-mod`,
`--spec-ngram-mod-n-match`, `--spec-ngram-mod-n-min`,
`--spec-ngram-mod-n-max`, `--spec-draft-n-max`, `--ctx-size`, `--parallel`,
`--batch-size`, and `--ubatch-size`. This loaded the binary only; it did not
bind a port or load a model.

The executable RUNPATH is the absolute build `bin` directory. A later package
must retain this sibling-library location or perform a separately reviewed
repackaging step. The required runtime libraries are `libllama-server-impl`,
`libllama-cli-impl`, `libllama-common`, `libmtmd`, `libllama`, `libggml`,
`libggml-cpu`, `libggml-base`, `libstdc++`, `libm`, `libgcc_s`, `libc`, the
Linux loader, and `libgomp`; the first seven are built siblings.

## Smallest TRAIN-only runtime check after lead admission

No model or trace was transferred, and no benchmark was run in this packet.
The lead must first provide a hashed, admitted TRAIN-only model and a hashed
TRAIN-only trace fixture. The existing S1 evaluation trace set and final/TU3
content remain excluded from this target preparation.

The smallest mechanical check is one frozen TRAIN trace in each 2K/8K class,
run once through a baseline and once through `ngram-mod`, with exact output
comparison and no timing claim. If that passes, the smallest timing screen is
20 frozen TRAIN 2K plus 20 frozen TRAIN 8K traces, one cold and one warm pass
per trace, baseline first and candidate second, on one quiet window. Use one
tracked server process at a time on a free loopback port, for example:

```text
SERVER=/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server
$SERVER -m "$TRAIN_APPROVED_MODEL" --host 127.0.0.1 --port 18401 \
  -t 2 -tb 2 --threads-http 2 --parallel 1 -c 10240 \
  -b 256 -ub 256 -ngl 0
```

The candidate adds:

```text
--spec-type ngram-mod --spec-ngram-mod-n-match 24 \
--spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
```

For each selected trace, both arms receive the same native `/completion`
request with `stream: true`, `temperature: 0`, `n_predict: 64`,
`stop: [">>>>>>> UPDATED"]`, and the same prompt. Record model/server/trace
hashes, host load, cold and warm TTFT, decode rate, full-cycle wall time,
stop-hit, and exact candidate-vs-baseline text. The promotion gate from
`SPECULATIVE-PATHS.md` remains exact output equality, at least 1.4x end-to-end
speed, bootstrap lower bound above 1.0, and no p95 regression. A contended
host is a wiring result only, not a latency result. The lead owns model
transfer, server admission, trace selection, and any later held-out evaluation
run.

## Remaining editor gate

SSH shell access cannot establish extension activation or InlineCompletion
behavior. A later authorized GUI/Remote-SSH window must build the existing
extension checkout and verify activation, loopback child ownership, one R
same-line completion, no-op, cancellation, and shutdown. The target has no
Xvfb or other bounded headless display prerequisite, and the extension
`node_modules`/`dist/extension.js` are currently absent, so this gate remains
open. No editor result is claimed here.
