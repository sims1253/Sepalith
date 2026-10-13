# PRM-07 profile preparation

**Observed:** 2026-09-12T07:40:55+02:00. **Owner:** worker-runtime. **Status:** CPU input/preflight and fake-model measurement contract complete; live GPU profile and RL rollout probe remain pending the lead-owned device window.

The execution worktree now contains [campaign_profile.py](../../../../tuesday-execution-20260912/experiments/training/campaign_profile.py) and [test_campaign_profile.py](../../../../tuesday-execution-20260912/experiments/training/test_campaign_profile.py). The command has two explicit modes:

```text
--preflight  hash and validate inputs and pinned artifacts; imports no torch/Unsloth/Transformers/TRL
--profile    repeat preflight, guard live GPU/process state, then load the disposable model and measure
```

The profile input is deliberately explicit. `--candidate-file` must be a hashed UTF-8 JSONL registry of complete PRM-03 training rows, either bare rows or the DAT-04 `status: "tokenizer_candidate_only"` envelope. The envelope must carry `admitted_for_training: false`, a `source_ref.split` of `train_group`, and audited lengths that match the nested token arrays. `--selected-train-ids` must be a second hashed JSON object with exactly `schema_version`, `split: "train"`, and a unique ordered `row_ids` array. Every candidate row is validated against the frozen protocol, and every row in the registry must be `split: "train"`; selected IDs alone determine the measured order. No directory walk, dev/final discovery, or model-call candidate admission exists in this path.

Natural buckets use the stored complete `input_ids` length: `short <= 2048` and `long 2049..4096`. Rows above 4096 fail closed. Dynamic batches pad with ID 1 only to form tensors; `attention_mask=0` and `labels=-100` at every padding position. Denominators report actual sequence labels, causal objective positions, prompt tokens without manual BOS, and target tokens including the protocol EOS. No target or evidence truncation and no fabricated useful padding occur.

The live loader checks the exact staged MiniCPM path and hashes before framework import, then rechecks the current SFT policy from `campaign_sft.TARGET_MODULES`: LoRA rank 32, alpha 64, seven target modules, 294 attachments, 50,233,344 expected trainable parameters, BF16, full-text labels, and pad/EOS ID 1. The `--profile` path runs an `nvidia-smi` single-GPU/active-process/VRAM guard before importing Unsloth or torch. A signal-backed outer wall bound produces a durable `complete`, `timeout`, `guard_failed`, `input_failed`, `oom`, or `failed` receipt.

The RL surface is intentionally limited to `validate_rl_candidate_records`: it requires a manual BOS prompt, caps prompt/completion IDs at 2048/192, and accounts for canonical EOS, noncanonical EOG 130073, cap hits, and CONTROL IDs before terminal. No real generation or recomputed policy-logprob forward/backward was run in this CPU-only packet. The missing live step must use concurrent generation on one real selected short row, then recompute policy log probabilities from returned integer IDs before any RL admission.

## Preflight evidence

An actual `--preflight` run verified the staged artifacts and the root's 520-row DAT-04 candidate-only pilot (263 packages, six families):

```text
candidate SHA256 bfb1071e8c86bee02902e317c14b8f00408aa69f0c5e174dc6067c98a0fa83cb
selected IDs SHA256 67d160681c2004e55917abfe8fb2ee2314801005dc5e27e643eec6924384921d
status preflight_pass; CUDA_started false; framework_imports []
candidate rows=520; selected rows=520; tokenizer_candidate_only=520
natural buckets short=398, long=122, out_of_profile=0
actual sequence tokens short=487075, long=267147
prompt tokens without BOS short=470012, long=263100
target tokens including protocol EOS short=16665, long=3925
```

The staged identity matched revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`, weights SHA256 `38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad` and 5,033,557,128 bytes, tokenizer JSON SHA256 `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, tokenizer config SHA256 `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`, config SHA256 `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180`, and generation config SHA256 `9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1`.

## Verification and next launch

```text
python3 -m unittest -v experiments.training.test_campaign_profile
10 tests passed
python3 -m py_compile experiments/training/campaign_profile.py experiments/training/test_campaign_profile.py
passed
```

The lead can run the guarded SFT profile after materializing the real hashed train registry and selected-ID manifest:

```text
python3 experiments/training/campaign_profile.py --preflight \
  --candidate-file /ABS/candidate-token-rows.jsonl --candidate-sha256 SHA256 \
  --selected-train-ids /ABS/selected-train-ids.json --selected-ids-sha256 SHA256 \
  --receipt /ABS/prm-07-preflight.json

python3 experiments/training/campaign_profile.py --profile \
  --candidate-file /ABS/candidate-token-rows.jsonl --candidate-sha256 SHA256 \
  --selected-train-ids /ABS/selected-train-ids.json --selected-ids-sha256 SHA256 \
  --receipt /ABS/prm-07-profile.json --warmup-steps 1 --timed-steps 3 \
  --batch-size 1 --wall-timeout-seconds 1800 --max-existing-vram-mib 1024
```

The commands must be serialized after the active CUDA owner exits. The live result must report both natural buckets where selected data supports them, actual peak allocated/reserved bytes, synchronized forward/backward/optimizer wall time, and actual denominators. The proposed 80/20 mixture, 2048/4096 limits, and 1024 MiB guard are profiling policy inputs; none is a measured quality or latency optimum.
