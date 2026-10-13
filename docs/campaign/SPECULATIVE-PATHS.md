# Speculative paths for the Tuesday model

**Decision memo — 2026-09-12, Europe/Berlin.** Speculation is a latency layer:
at a fixed target checkpoint and greedy policy, an exact verifier preserves
quality and changes time-to-output. Latency can change the product decision
when a faster profile is needed for the editor budget. Keep the quality path as
`MiniCPM5-2B-Midtrain` → frozen R-edit SFT → RL, with b4/O2 as rollback. Do not
add pretraining. Draft training is post-training distillation from a target
snapshot.

## Checkpoint answer

The pinned `openbmb/MiniCPM5-2B-Midtrain` revision
`8dc5f6055b90fe4b9422340810b270b9569f37f3` is documented as a standard
`LlamaForCausalLM`: 42 layers, hidden size 2048, vocabulary 130560, and no
MTP, `nextn`, draft, or speculator fields. Its file listing has one
`model.safetensors` and no weight index. This is strong metadata evidence of
an ordinary target, but not tensor-level proof. After download, inspect
`state_dict` keys for `mtp.*`, `nextn.*`, or equivalent before export.

`MiniCPM5-2B-DSpark` is a separate draft trained for exact pairing with the
released post-trained `MiniCPM5-2B`, not a Midtrain head. Its card gives five
draft layers, 323,776,001 parameters, seven draft tokens per pass, target taps
`[1, 10, 20, 30, 39]`, and aggregate acceptance 5.5174 at temperature zero.
Its config has a `Qwen3DSparkModel`, confidence/Markov heads, block size 7,
and a 42-layer target. Shared shape makes an early test reasonable; the
published acceptance does not transfer to our SFT/RL weights. Sources: [Midtrain card](https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain),
[pinned config](https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain/blob/8dc5f6055b90fe4b9422340810b270b9569f37f3/config.json),
[DSpark card](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark), and
[DSpark config](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark/blob/main/config.json).

## Decision table

| Path | What the evidence says | Decision in this campaign |
| --- | --- | --- |
| `ngram-simple` | Model-free and lossless when verified. Broad CPU results were near 1×; a tiny loaded smoke was faster, so host load matters. “ngram-copy” is not a verified alias for this arm name. | Baseline comparator; no broad sweep. |
| `ngram-mod` | Model-free, roughly 16 MB state, variable draft length, shared hash pool; no Sepalith result. Use `--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`. | Highest-value scout on the separate CPU host during CUDA training. |
| `ngram-map-k4v` | Model-free and explicitly experimental. | Add one small screen only if `ngram-mod` leaves time. |
| Released MiniCPM DSpark | 324M draft trained against the released final with exact target/tokenizer pairing. It may remain usable after adaptation, but acceptance is open. | Test against first `theta0`; verify again on final. |
| Adapted DSpark or compatible lighter draft | Plausible: DeepSpec trains from target caches but has Qwen3/Gemma4 classes, no ready MiniCPM recipe. The 0.8B AR stand-in had high acceptance but lower end-to-end throughput under contention. | Early readiness study; train only for clear benefit within shared budget. |
| MTP head | Midtrain has no advertised MTP metadata. An untrained upstream-head graft accepted 0/90 smoke drafts; that is a lower bound, not a target-trained result. | Reserve; require a ready target-specific path within the shared cap. |
| EAGLE-3 or DFlash | `llama.cpp` supports them, but drafts are target-specific; DeepSpec has no ready MiniCPM recipe. | Reserve behind the same gate; no automatic graft or port. |

See the [72-hour plan](../72-HOUR-MODEL-PLAN.md), [compute/source review](../research/72h-model-compute-sources.md),
[new-base addendum](../research/72h-new-base-challenge.md),
[spec_bench.py](../../experiments/eval/spec_bench.py), and [S1 runbook](../../experiments/eval/S1_RIG.md).
`spec_bench.py` deliberately rejects DSpark; adding it is reviewable work.

## Parallel sequence

**A. First `theta0`, while RL runs — readiness.** Give one worker at most 90
engineer-minutes for source and architecture feasibility. Inspect the target
config/state-dict contract, tokenizer and special IDs, the DSpark five tap
layers, pinned b10453 `--help`, converter assumptions, and the sidecar path.
Check whether the released DSpark or a lighter existing draft can
load against this exact target. Any draft training or acceptance sample uses
the **TRAIN split only**. The existing S1 speculative trace set is evaluation
data and remains strictly held out; never build a draft cache from it.

The worker returns a go/no-go memo with code files, artifact sizes, conversion
command, hashes to record, and a one-smoke estimate. Go means tokenizer/config
match, hidden-state taps, proven b10453 flags, and a draft that loads or trains
without changing the target. No training job is required for this packet.

**B. Conditional intermediate pair.** If A finds a ready path, policy checks
pass, and a short smoke shows a credible speed opportunity, the lead may
reassign at most **two GPU-hours** from the already planned four-hour optional
consolidation, or use
the existing approved cloud ceiling. This is a reassignment, not a new
allocation. Freeze the first target snapshot, generate target responses and
hidden states from TRAIN only, train/export the draft, and measure exact greedy
parity plus end-to-end time. The target and draft can be developed in parallel
through snapshots; they do not need a shared optimizer or joint backprop.

DeepSpec consumes target hidden states and applies CE, L1, and confidence
losses. Its [workflow](https://github.com/deepseek-ai/DeepSpec) is target cache
→ draft training → evaluation; its [DSpark trainer](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/trainer/dspark_trainer.py)
has Qwen3 and Gemma4 implementations. A MiniCPM port is ready only when the
packet identifies exact changes and a smoke fits the reassigned cap.

**C. Final pair.** After the target weights are selected, use the same draft
first and measure it before refreshing anything. A target change does not
automatically invalidate a draft; acceptance and latency decide whether a
refresh is worth its cost. If refreshed, regenerate the TRAIN-only cache from
the final target and charge training/export to the same two-GPU-hour
reassignment or existing cloud ceiling. Do not let a draft refresh displace
the final target evaluation, packaging, or rollback.

The speculative envelope is the existing two-hour CPU scout plus up to
four-hour confirmation. DSpark is one candidate in that envelope, not an extra
screen. Target and draft share one serving job; CUDA serving cannot overlap
CUDA training. The CPU n-gram scout may run on its separate host.

## Agent packets and gates

**`SPEC-READINESS-90M`.** Read the local rig, DeepSpec source, official
configs, and b10453 help. Do not download weights, build a cache, or launch a
job. Return the compatibility matrix for released DSpark, adapted DSpark, a
lighter AR draft, MTP, EAGLE-3, and DFlash; list exact code changes and the
smallest TRAIN-only smoke. Completion is a bounded go/no-go memo.

**`SPEC-CPU-NGRAM`.** Run `ngram-mod` with a paired baseline on 20 frozen 2K
and 20 frozen 8K evaluation traces, one pass, unchanged stop marker,
temperature zero, 64-token cap, context 10240, and one request slot. Record
the actual b10453 flags, `-b 256` profile, host load, binary/trace/target
hashes, per-request output parity, acceptance, TTFT, and wall rate. Promote
only with exact output, at least 1.4× end-to-end speed, bootstrap lower bound
above 1.0, and no p95 regression.

**`SPEC-DRAFT-PAIR`.** If readiness is green, run baseline first and the
candidate on 12 fresh 2K plus 12 fresh 8K evaluation traces, two repetitions.
For DSpark use target `-m`, draft `-md`, and the pinned
`--spec-type draft-dspark --spec-draft-n-max 7` flags only after `--help`
confirms them. Store acceptance by band, exact output checks, target/draft
hashes, converter metadata, and runtime hash. The pair passes only the same
1.4×/p95 gate and complete lossless parity. A failed intermediate screen
closes that draft path unless new evidence changes the readiness estimate.

The [official llama.cpp speculative documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
describes n-gram modes as model-free and DSpark/DFlash as target-specific;
pinned b10453 and local S1 receipts govern Sepalith flags. This memo records
config/source inspection only: no weights were downloaded and no training or
serving job was launched.
