# RUN-09 theta0 notebook CPU DEV comparator

This is a root-owned launch capsule. The worker did not start a server, load
the Q8 model, use SSH, use Vulkan/CUDA, or access final data. It reuses the
accepted corrected DEV client and panel and adapts only the already-reviewed
notebook lifecycle to the CPU binary.

The pinned runtime is the existing b10453 AVX2 CPU artifact:

- binary:
  /home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server
  SHA-256 e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6;
- dependency list and per-file hashes:
  docs/campaign/work/lead/notebook-b4-cpu-runtime-identity.json,
  SHA-256 0569e1707d79bb3f8f5f39a9a9e63d1fcb1f6f5cf613e8a5f07bbae8ddb1968e;
- selected theta0 Q8 model:
  models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf,
  SHA-256 22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559;
- corrected DEV panel SHA-256
  7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035;
- client adapter SHA-256
  4c9caecaacfdf88f2124f9c8be9f6c8727d150ccee0886f29f544c6e01c12cb0.

The CPU profile is context 4096, generation cap 192, server threads 6,
batch threads 6, HTTP threads 2, parallel 1, batch 256, ubatch 256, and
ngl 0. The environment clears all GGML backend overrides and
SEPALITH_VK_TRACE, sets CUDA_VISIBLE_DEVICES to the empty string, and sets
LD_LIBRARY_PATH to the pinned binary directory. The child audit fails if
Vulkan library mappings or DRI file descriptors appear; no Vulkan DRI is
required for this comparator.

Root launch command, after checking the runtime and model identities:

~~~sh
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
REMOTE=/home/m0hawk/.local/share/sepalith-campaign-20260915
RUN=/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-09-theta0-q8-cpu-dev-a
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  timeout --foreground --signal=TERM --kill-after=20s 1200s \
  python3 "$PLAN/docs/campaign/work/theta0-notebook-cpu-dev-preparation/run_theta0_notebook_cpu_dev.py" \
  --run-root "$RUN" --port 18413
~~~

The capsule verifies the 75-row corrected panel and model/runtime hashes,
rejects an existing output directory, reserves a fresh loopback port, keeps
the server and client in separate process groups, and fsyncs launch,
preflight, terminal, and client outputs. The server has a 1150-second hard
timeout; the quality client has a 1000-second deadline and 60-second reserve;
the capsule also enforces an absolute 1200-second ceiling and a 2 GiB
MemAvailable floor.

The resulting quality.json is directly comparable by panel IDs and the
existing scorer's denominators with the CUDA Q8 and notebook Vulkan runs.
The terminal identity records loaded CPU dependency mappings and the absence
of Vulkan/DRI use. This packet makes no quality, latency, or model-promotion
claim. Root must inspect the completed quality artifact and compare the
existing CUDA Q8 result (26/43 exact edits, 25/32 correct no-ops, 5 false
positives) and notebook Vulkan result (26 exact edits, 23/32 no-ops, 7 false
positives) using identical case IDs and protocol denominators.
