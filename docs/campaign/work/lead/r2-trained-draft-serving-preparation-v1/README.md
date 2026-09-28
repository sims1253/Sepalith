# R2 trained DSpark serving preparation

This packet binds the serving follow-up to the selected step-500 target and to
the fresh eight-step warm-start output reported by the local profile. The
checkpoint path is:

```text
/home/m0hawk/.local/state/sepalith/dp-b/checkpoints/sepalith-r2/dspark_minicpm5_2b_step500_profile/step_8
```

Root's readback bound the checkpoint model safetensors to
`6eed8dfafd7a2b419950d9bafe3865fbe93885061bbebae1da70041f321875b8` and its
config to
`dd328e3ea202509d032d2b9de020c84724082ee41e41c5cd41516fb5df5b2435`; eight
optimizer states, 58 changed public tensors, and the two target matrices were
present. This worker did not run conversion, bind CUDA, or start a provider.

The selected target is the accepted step-500 wrapper
`docs/campaign/work/lead/r2-step500-rl-gate-v2/parent-manifest.json`, with
merged-weight pin
`631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c`. The
target GGUF already exported for the native screen is
`/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf`
with the selected R2 export pin
`d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db`.
Those are metadata bindings from prior receipts; root must independently
read back the target and candidate files before serving.

## Conversion gate

Run [conversion-command.txt](conversion-command.txt) only after root has
verified the checkpoint inventory and the target directory. The command
requires `config.json`, `model.safetensors`, and `train_config.py` in `step_8`,
plus `sepalith-warmstart.json` at the checkpoint root. It creates a fresh
empty output directory and refuses to overwrite one.

The b10453 converter dispatches `Qwen3DSparkModel` to the `DSparkModel` GGUF
writer (`MODEL_ARCH.DFLASH`). The DSpark writer requires
`--target-model-dir`; it obtains target tokenizer/vocabulary metadata and
converts each configured target layer `i` to extraction metadata `i + 1`.
For this profile the expected extracted IDs are `[2, 11, 21, 31, 40]`, from
draft taps `[1, 10, 20, 30, 39]`. The conversion command deliberately does not
pass `--dspark`: b10453 reserves that switch for `DeepseekV4ForCausalLM`, while
Qwen3 DSpark is selected by the checkpoint architecture registry.

After conversion, root must inspect both GGUF headers without accepting the
artifact on file creation alone. Require `general.architecture=dflash`,
`dflash.block_size=7`, `tokenizer.ggml.mask_token_id=75982`, extracted target layers
`[2,11,21,31,40]`, vocabulary `130560`, hidden size `2048`, and Markov tensors
including `markov_w1.weight` and `markov_w2.weight`. Verify that the target
tokenizer/config pins are the selected step-500 target and that all expected
draft tensors are present. Quantize only after the F16 header passes, using
the pinned b10453 `Q8_0` command in the file.

Root's completed F16 conversion is staged at
`/home/m0hawk/.local/state/sepalith/campaign-20260915/models/R2-step500-dspark-step8-gguf/model-F16.gguf`;
the corresponding Q8 path is the same directory's `model-Q8_0.gguf`, subject
to the Q8 terminal/header readback.

The target matrices are part of the target-pair contract. `--target-model-dir`
must point at the selected HF target directory, never at the target GGUF and
never at a public base-model directory. If the checkpoint's warm-start
manifest or GGUF audit shows a missing or mismatched target matrix, stop and
report the exact key; do not fill it from an unrelated target.

## Four-arm native screen

Run [pair-benchmark-command.txt](pair-benchmark-command.txt) with one tracked
b10453 CUDA server at a time. It uses the existing four-row TRAIN-only fixture
`docs/campaign/work/serving-readiness/native-probe-train-fixture.jsonl` and
its manifest. The fixture has four rows, all `split=train`, with one short
no-op, one short replacement, and two long replacements. It contains no DEV,
S1, final, or authored evaluation data. Each arm gets one cold and one warm
request per row, `n_predict=192`, context `4096`, manual BOS `0`, and the
canonical terminal EOS `1`; native EOG `130073` remains a recognized control
ID. The client must send stored prompt IDs and preserve the exact tokenizer
and output protocol.

The arms are:

* `ordinary-baseline`: selected step-500 target only.
* `model-free-ngram`: the same target with b10453 `ngram-mod` settings.
* `released-dspark`: a separately staged released MiniCPM5 DSpark GGUF,
  only if root has a concrete file and header/hash manifest. The currently
  available root candidate is BF16, while the trained candidate is Q8; either
  compare the released file at BF16 with that precision caveat or quantize it
  first and bind the resulting Q8 hash. The public release is a conditional
  comparator, not evidence about the trained draft.
* `trained-targetmatched-dspark`: the Q8 conversion from `step_8`, paired with
  the selected step-500 target.

The small screen is a protocol and systems-readiness gate. Record the exact
returned IDs and raw text for every row/phase, then require candidate output
IDs and raw text to match the ordinary baseline exactly. A mismatch, malformed
SSE stream, noncanonical terminal, context violation, or load error fails the
arm regardless of speed. The adapter in this packet retains the server's
`draft_n` and `draft_n_accepted` final fields; the analyzer reports acceptance
as `sum(draft_n_accepted) / sum(draft_n)` and reports `missing` if a build does
not expose those fields. Do not infer acceptance from output length.

For each arm report cold and warm end-to-end wall time, TTFT, prompt and decode
timings, p50/p95, request failures, drafted-token denominator, accepted-token
numerator, and acceptance by short/long band. Compare latency against the
ordinary baseline from the same run. The four-row screen is too small for a
promotion claim; a useful candidate needs the registered larger short/long
panel, repeated measurements, exact greedy parity, and a lower confidence
bound above the agreed speed threshold. No quality or speed win is assumed by
this packet.

The existing `runtime_native_probe.py` parser exposes only `baseline` and
`ngram-mod`. `dspark_probe.py` is a narrow packet-local adapter: it delegates
all request, token, EOS, cap, and timeout checks to that existing client and
adds only a model-free/DSpark arm label plus the final draft counters. This
avoids changing the shared probe. It does not generate tokens or alter
sampling.

## Source and runtime pins

The static checker records the source hashes and markers used by this packet.
The relevant b10453 source is
`/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453` at
commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` (the source tree's tracked
revision used by the prior receipts). The server binary expected by the
one-GPU root session is
`/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server`;
its recorded binary SHA is
`e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee`.
The serving profile uses `-t 6 -tb 6 --threads-http 2 --parallel 1 -c 4096
-b 256 -ub 256 -ngl 99 -ngld 99 -lv 4`, `CUDA_VISIBLE_DEVICES=0`, and
`GGML_CUDA_GRAPH_OPT=0`.

The static contract result is written to `contract-check.json`; its PASS
means source/config/fixture metadata agrees with this packet. It does not
prove a checkpoint conversion, GGUF tensor load, CUDA launch, target/draft
greedy parity, acceptance, or latency. Those remain root-owned gates.

The four-row screen is a bounded readout. If it passes, root may prepare a
larger TRAIN-only screen in a new run directory. Keep each arm sequential,
retain server logs and `/health` plus `/props` responses, and verify the
tracked server PID and all children are gone before starting the next arm.
The follow-up panel should include identical 2K and 4K prompt traces and a
selected 8K stress arm. The target profile has a 4096-token context cap, so an
8K request must be explicitly rejected or run under a separately admitted
context configuration; it must never be silently truncated or treated as a
daily 4096-token result. The four simple fixture rows cannot supply a
representative 50th/95th-percentile claim for that panel.
