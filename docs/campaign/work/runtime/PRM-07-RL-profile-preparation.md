# PRM-07/RL-01 policy-logprob profile preparation

**Observed:** 2026-09-12T07:59:37+02:00. **Owner:** worker-runtime. **Status:** CPU preparation and fake-tensor mechanism checks complete; the guarded live probe remains lead-owned and was not launched.

The execution worktree contains [campaign_rl_profile.py](../../../../tuesday-execution-20260912/experiments/training/campaign_rl_profile.py) and [test_campaign_rl_profile.py](../../../../tuesday-execution-20260912/experiments/training/test_campaign_rl_profile.py). The command has two explicit modes:

```text
--preflight  verify explicit hashed candidate/ID inputs and pinned MiniCPM without model-framework imports
--profile    repeat those checks, guard the lead-owned GPU, load a disposable r16/a16 adapter, and run the probe
```

The input contract is the same PRM-07 candidate-only contract. `--candidate-file` is a SHA256-pinned UTF-8 JSONL file of complete PRM-03 train rows, including the DAT-04 `tokenizer_candidate_only` envelope with audited lengths. `--selected-train-ids` is a second SHA256-pinned object containing an ordered unique `split: train` ID list. The probe reads `input_ids[:target_start]` from those rows directly; it does not re-render prompt text, discover directories, admit model-call candidates, or read dev/final rows. Prompt selection requires the stored manual BOS ID 0 and a prompt of at most 2048 IDs. The managed prompt plus completion geometry is capped at 2048 + 192 IDs.

After the process/VRAM guard, the live loader verifies the pinned MiniCPM path and tokenizer identity, attaches the seven campaign modules with LoRA rank 16 and alpha 16, and requires 294 attachments and 25,116,672 trainable parameters. Generation runs each explicit prompt with bounded concurrent candidate batches, default sizes 2 and 4, `max_new_tokens=192`, and native EOG IDs `[1, 130073]`. Returned integer sequences are trimmed only at the first native terminal; generated IDs are retained for the policy pass. The accounting records canonical EOS 1, noncanonical EOG 130073, cap hits, CONTROL IDs before terminal, and framework padding after an early terminal.

The policy pass right-pads only for tensor shape, sets attention to zero on padding, and masks exactly the generated response positions. It recomputes token log probabilities from the model logits with `use_cache=False`, forms a finite mean negative log probability from the actual returned IDs, and backpropagates it. The probe requires finite, nonzero gradients on trainable adapter parameters. Generation is enclosed by the existing `campaign_sft.training_configuration_guard`; its `FastLanguageModel.for_training` restoration is invoked before the differentiable pass. No optimizer update, GRPO objective, reward, or quality claim is made.

Each candidate-count arm reports synchronized generation, recompute, and backward seconds; allocated/reserved and peak CUDA memory; exact prompt/generated token denominators; terminal/control accounting; response-mask geometry; and gradient checks. The durable receipt distinguishes `preflight_pass`, `complete`, `guard_failed`, `input_failed`, `timeout`, `oom`, and `failed`. A live run is serialized after the current CUDA owner exits and must use a fresh receipt path.

## CPU verification

```text
python3 -m unittest -v experiments.training.test_campaign_rl_profile
8 tests passed
python3 -m py_compile experiments/training/campaign_rl_profile.py experiments/training/test_campaign_rl_profile.py
passed
```

The actual 520-row DAT-04 pilot was also preflighted with candidate SHA256 `bfb1071e8c86bee02902e317c14b8f00408aa69f0c5e174dc6067c98a0fa83cb`, selected-ID SHA256 `67d160681c2004e55917abfe8fb2ee2314801005dc5e27e643eec6924384921d`, and the pinned MiniCPM artifacts. It passed with 520 candidate rows, 520 selected train IDs, 420 prompts under the prompt-only cap, and 404 rows under the combined natural prompt/response 2048/192 geometry reported by the source audit. The first deterministic prompt ID is recorded in the preflight receipt. The pilot is candidate-only evidence and does not grant training admission.

The pinned identity is revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`, weights SHA256 `38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad` and 5,033,557,128 bytes, tokenizer JSON SHA256 `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, tokenizer config SHA256 `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`, config SHA256 `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180`, and generation config SHA256 `9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1`.

The brief's old 480 prompt / 170 target filters and the old RL 512 prompt / 192 completion limits remain historical comparison points. This probe uses the proposed 2048 prompt / 192 completion RL bound and never truncates a stored target or evidence row. The missing live work is GPU memory/throughput and differentiable policy-logprob evidence; the CPU fake model verifies masks and finite nonzero gradient plumbing only.

## Lead launch command

```text
python3 experiments/training/campaign_rl_profile.py --profile \
  --candidate-file /ABS/candidate-token-rows.jsonl --candidate-sha256 SHA256 \
  --selected-train-ids /ABS/selected-train-ids.json --selected-ids-sha256 SHA256 \
  --receipt /ABS/prm-07-rl-profile.json --prompt-count 1 \
  --generation-candidates 2,4 --wall-timeout-seconds 900 \
  --max-existing-vram-mib 1024
```

The lead must provide a final admitted train registry and selected IDs, close the active CUDA owner, verify the single-GPU guard, and retain the complete receipt. Any noncanonical EOG, preterminal CONTROL ID, non-finite loss, missing gradient, cap/timeout, or OOM is evidence for the mechanism gate and is not silently converted into an RL success.
