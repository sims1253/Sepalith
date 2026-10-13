# RUN-05 Q6 native baseline/candidate A/B capsule

This packet is a root-owned execution capsule. The worker did not build either
binary, start a server, load Q6 weights, invoke Vulkan/CUDA, or use SSH. The
capsule imports the pinned framework-free runtime_native_probe.py only when
the root executes it.

The build closure is the accepted unpacked llama.cpp-b10453 snapshot:

- commit 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70;
- source manifest SHA-256
  b0e6fbb71b2528924480c6da34f9bd90c87f7b847c504b7e3ee8132fb446bc7e;
- reviewed Q6 patch SHA-256
  f9f57597ae119e433095c52e3de63b83a0cdbf472f982a123389256a01763310;
- selected model
  models/SFT-primary-step1000-quant-candidates-c/model-Q6_K.gguf,
  SHA-256
  7ea2ddfd45dce35d5016af8f1d1ba14408d29764df5a90f3a0dfe6f886ecec3578.

The build evidence directory must contain completed baseline and candidate
compiled-only terminal/source/artifact receipts. The wrapper binds each
explicit binary path to its artifact receipt and rehashes the binary before
creating the run root. It also checks the fixture and native probe hashes,
the four selected TRAIN IDs, and that the requested run root and four ports
are fresh.

The required server profile is fixed at context 4096, generation cap 192,
threads 6, HTTP threads 2, parallel 1, batch 256, ubatch 256, and ngl 99.
Both arms receive an empty CUDA_VISIBLE_DEVICES, six-thread CPU variables,
and a removed graph/poison/MMVQ override set. The trace phase sets only
SEPALITH_VK_TRACE=1; the timing phase removes it and records timing_claim.
Trace results are evidence for dispatch inspection, not latency. The native
probe generic server_profile_expected field describes its older m0pad profile;
the capsule launch.json server_profile is the authoritative Q6 profile.

Use an outer GNU timeout as a second guard. This command is illustrative and
must be run only after root reviews the build receipts and accepts the Q6
quality gate:

~~~sh
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
REMOTE=/home/m0hawk/.local/share/sepalith-campaign-20260915
EVID="$PLAN/docs/campaign/work/lead/q6-kernel-build-b-evidence"
MODEL="$REMOTE/models/SFT-primary-step1000-quant-candidates-c/model-Q6_K.gguf"
FIXTURE="$PLAN/docs/campaign/work/serving-readiness/native-probe-train-fixture.jsonl"
MANIFEST="$PLAN/docs/campaign/work/serving-readiness/native-probe-train-fixture.manifest.json"
PROBE="$PLAN/docs/campaign/work/serving-readiness/runtime_native_probe.py"
RUN="$REMOTE/runs/q6-kernel-native-ab-a"
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  timeout --signal=TERM --kill-after=20s 1200s \
  python3 "$PLAN/docs/campaign/work/q6-kernel-native-ab-preparation/run_q6_native_ab.py" \
  --baseline-binary "$REMOTE/runs/q6-kernel-build-b/baseline-build/bin/llama-server" \
  --candidate-binary "$REMOTE/runs/q6-kernel-build-b/candidate-build/bin/llama-server" \
  --baseline-artifacts "$EVID/baseline-artifacts.json" \
  --candidate-artifacts "$EVID/candidate-artifacts.json" \
  --build-evidence-root "$EVID" \
  --model "$MODEL" --model-sha256 7ea2ddfd45dce35d5016af8f1d1ba14408d29764df5a90f3a0dfe6f886ecec3578 \
  --fixture "$FIXTURE" --manifest "$MANIFEST" --runtime-probe "$PROBE" \
  --run-root "$RUN" --base-port 18401
~~~

If root decides the Q6 quality regression makes timing unnecessary, use the
same command with --trace-only. That mode still runs one trace probe per arm,
uses separate arm/phase directories and ports 18401 and 18403, and writes a
truthful terminal record with timing omitted. A complete run uses baseline
trace 18401, baseline timing 18402, candidate trace 18403, and candidate
timing 18404. It sends one cold and one warm request for each of the four
selected TRAIN rows per phase, with no retry.

probe-cli.json is preserved exactly as emitted by the existing helper.
probe.json adds the explicit capsule arm name: the candidate maps from the
helper's permitted ngram-mod CLI alias. comparison-trace.json and
comparison-timing.json require exact output text/token IDs and stable
protocol, cap, stop, EOS, prompt, and cache fields.

Each phase writes an fsynced launch, server log, probe client log, raw and
normalized probe output, and terminal record. The parent supervises process
groups, checks the 2 GiB MemAvailable floor, and stops on process failure,
floor breach, phase deadline, or the absolute 1200-second ceiling. It never
retries a failed case or continues after a failed arm.

The current build evidence proves compile completion only. Root must inspect
SEPALITH_VK_TRACE=1 logs for the ordinary Q6 DMMV pipeline, effective
MMVQ/integer-dot settings, workgroup/specialization and shape-guard reach.
This packet cannot establish GPU dispatch, SPIR-V numerical correctness,
latency, model quality, or promotion. The selected four-row fixture also
does not provide the unsupported greater-than-2048-token/8K stress case.
