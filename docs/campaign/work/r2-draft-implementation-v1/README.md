# R2 draft implementation packet

This packet makes the smallest CPU-testable bridge for a target-matched MiniCPM5 DSpark run. It contains no model weights, does not launch a GPU or cloud job, and does not alter campaign data. A cloud launch still needs root's target binding and admission under the existing `2 GPU-hours` / `$20 Anyscale` reservation.

## Pinned inputs

The local target config is:

`/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native/config.json`

Its recorded SHA-256 is `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180`. It is a 42-layer Llama config with hidden size 2048, intermediate size 6144, 16 attention heads, 2 KV heads, head dimension 128, vocabulary 130560, BOS 0, PAD 1, and native EOG `[1, 130073]`. Its tokenizer SHA-256 is `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`.

The TRAIN source rows are the exact admitted file:

`docs/campaign/work/r2-task-mixture-v1/verified-corrected-short-token-rows.jsonl`

The recorded SHA-256 is `85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e`. Those rows contain authored screening tails. They are input prompts for regeneration only; they are not a target cache and must not be copied into training records.

DeepSpec is pinned to GitHub `deepseek-ai/DeepSpec` revision `005e03b81cec38b7da6399833d609ee89a2587f2` (the merge of PR #62). The source inventory used here is:

- [`deepspec/modeling/dspark/qwen3/config.py`](https://raw.githubusercontent.com/deepseek-ai/DeepSpec/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/config.py), whose builder deep-copies the target config and sets the DSpark geometry;
- [`deepspec/modeling/dspark/qwen3/modeling.py`](https://raw.githubusercontent.com/deepseek-ai/DeepSpec/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py), whose `Qwen3DSparkAttention` reads `config.sliding_window` and whose model owns `embed_tokens`, `layers`, `fc`, `hidden_norm`, `lm_head`, Markov, and confidence modules;
- [`scripts/data/prepare_target_cache.py`](https://raw.githubusercontent.com/deepseek-ai/DeepSpec/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/prepare_target_cache.py), the target-hook/cache writer path; and
- [`deepspec/data/target_cache_dataset.py`](https://raw.githubusercontent.com/deepseek-ai/DeepSpec/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/data/target_cache_dataset.py), the v2 reader/collator path.

The one-field compatibility patch is applied immediately after the upstream target-config deepcopy and before `Qwen3DSparkModel` construction:

```python
draft_config.sliding_window = getattr(target_config, "sliding_window", None)
```

`minicpm5_dspark_config.py` provides both this object-level helper and a JSON-compatible builder. It also uses the released MiniCPM5 DSpark mask ID `75982`, which is inside the target vocabulary. This is an internal masked-token embedding ID; generated serving stops remain `[1, 130073]`.

## Warm-start candidate

The public `openbmb/MiniCPM5-2B-DSpark` checkpoint is a viable warm-start candidate and should be considered before scratch initialization. Its HF `main` revision is pinned here as `114a20fdbf53220712c7fbdd7dccddbf1dedebb4`; the config blob OID is `5f2d826e2d137bc3e77a4b0263a9ce7942cd1018`. The public card/config reports 5 draft layers, block size 7, target taps `[1, 10, 20, 30, 39]`, BF16, and 323,776,001 parameters. The single safetensors file is 647,558,522 bytes with SHA-256 `ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97` and Xet hash `9d964ccffe1dc34b319be30a72f82e26328ac401c593736f589a343901e33bde`.

Its config matches the local target dimensions and EOG IDs, so a cloud metadata gate can try loading it against the bound target before spending scratch-training steps. The warm start is not evidence of campaign quality: the public draft was trained on a broad mixture and still needs the new target-generated TRAIN cache and target identity check. The source-derived tensor-prefix inventory is recorded in `public_dspark_warm_start.py`: embeddings, five draft layers, normalization, `fc`, `hidden_norm`, `lm_head`, Markov, and confidence heads. Exact key/shape enumeration should happen after an admitted cloud download; this packet did not inspect the weight payload.

## Teacher and cache bridge

`prepare_teacher_cache.py` selects a deterministic bounded subset by row-ID hash and calls an injected target generator with only `row["input_ids"][:row["target_start"]]`. The generator must return `generated_ids`, or a token sequence. A returned mapping that contains only a full `input_ids` sequence is rejected so an authored tail cannot enter by accident.

`teacher_rows.py` then records the exact generated IDs, prompt prefix, status, and explicit mask:

- `input_ids = prompt_prefix + target_generated_ids`;
- `loss_mask = 0` on the prompt and `1` on every exact generated token;
- native EOG is preserved if generated (`1` or `130073`);
- a cap hit without EOG is retained with `cap_hit=true`;
- an invalid edit protocol is retained with `protocol_status="invalid"`;
- empty output or out-of-vocabulary IDs are row-level cache rejections; and
- no path synthesizes EOS, trims a generated prefix, or substitutes the authored tail.

The stage continues after cap hits and protocol-invalid rows. The summary records their counts so the 256-row run can be audited instead of ending on the first imperfect suggestion.

`pretokenized_cache_bridge.py` maps these records to the v2 writer fields. Its `PretokenizedTargetCollator` is the drop-in replacement for the upstream conversation collator and returns `input_ids`, `attention_mask`, and `loss_mask` tensors without applying a chat template. It uses int32 IDs, uint8 attention/loss masks, BF16 target hidden states shaped `[T, 5H]`, and BF16 target last hidden states shaped `[T, H]`. The index tuple is the upstream 56-byte `<QIIQQQQQ` record (`sample_id, shard_id, seq_len`, then five offsets). The v2 writer does not require EOS, so a non-empty cap-hit generation is cacheable as-is; the bridge marks EOS injection as forbidden.

For H=2048 and five taps, payload cost is `24,582 * T` bytes. The prior CPU estimate for 256 selected rows is about 7.46 GiB. Keep that cache ephemeral on the cloud host. Persist the manifest, teacher rows, source/config/tokenizer pins, finite profile metrics, and resumable draft checkpoint; do not send the bulk cache through the existing artifact uploader or its 600-second upload cap. A 128-row cache is the bounded fallback if host memory or disk pressure requires it.

## Training, profile, and export

`dspark_minicpm5_2b.py` mirrors the field names in the pinned upstream DSpark config (`confidence_head_alpha`, `loss_decay_gamma`, `ce_loss_alpha`, and `l1_loss_alpha`) and restores the production geometry: five layers, block 7, taps `[1, 10, 20, 30, 39]`, mask 75982, 512 anchors, Markov rank 256, and confidence head. It sets a bounded eight-step BF16 profile with `torch_compile=False`; the root launch may override cache path, target path, batch, and step count.

Before a long run, profile 8 rows and include target generation, cache-hook time, first backward/optimizer step, and asynchronous cache queue memory. Record nonzero target-label and anchor coverage, including short/no-op responses. If 512 anchors exceed the A10G memory budget, profile with 32 then 64 anchors and measure the first backward before deciding; restore 512 for the admitted objective unless the root receipt explicitly changes it. The final refresh, if useful, remains inside the same two-GPU-hour reservation.

The pinned upstream launcher is `python train.py --config ... --opts ...` and spawns workers for visible GPUs. The intended cloud command shape is:

```text
python train.py \
  --config config/dspark/dspark_minicpm5_2b.py \
  --opts data.target_cache_path=<ephemeral-cache> \
  --opts model.target_model_name_or_path=<bound-target> \
  --opts train.max_train_steps=8
```

No launch is admitted by this packet. Root must bind the actual control-250/frozen target, verify private persistence, and decide whether the public warm start or scratch initialization is appropriate.

`export_header_gate.py` is a small pre-export gate. It requires GGUF format, DSpark/Qwen3 geometry, target dimensions, mask 75982, taps, pinned target/tokenizer/DeepSpec identities, required source-derived tensors, target identity, and serving stops exactly `[1, 130073]`. It rejects synthesized EOS and authored-tail substitution. The normal HF-to-GGUF conversion route should be used after this gate; the DeepseekV4-only `--dspark` converter switch is not implied by this Qwen3 DSpark model.

## CPU checks

Run from this directory:

```text
python3 test_contract.py
python3 tiny_dspark_smoke.py
python3 export_header_gate.py
```

The contract suite currently runs 12 tests. It covers the missing `sliding_window` seam, a real tiny Transformers `LlamaForCausalLM` forward, DSpark-shaped CE/L1 loss and finite gradients, cap-hit/invalid-protocol retention, no-EOS cache handling, v2 index geometry, public warm-start structural matching, and positive/negative export headers. The tiny run uses a randomly initialized 5-layer, 16-hidden-dimension model and reports `target_forward=true`, `draft_forward=true`, finite CE/L1/loss/gradients, `compat_sliding_window=null`, and `campaign_weights_read=false`.

Known gaps are intentionally explicit: this worktree has no DeepSpec checkout, no HF draft payload, no regenerated target outputs, and no target cache. The packet proves the compatibility and data contracts on CPU; it does not claim trained quality, acceptance, latency, or final export readiness.
