# R2 actual upstream DSpark smoke

This packet contains a sparse checkout of DeepSpec and a CPU-only proof using
the pinned upstream `Qwen3DSparkModel`, its upstream loss, and its official
target-cache writer/reader. It contains no campaign weights, target rows, or
GPU/cloud launch.

The source is `deepseek-ai/DeepSpec` at revision
`005e03b81cec38b7da6399833d609ee89a2587f2`. The exact source files loaded by
the smoke are listed in `upstream_dspark_smoke.py` and their SHA-256 values
are recorded in the receipt. The environment used for the run was
`torch 2.11.0+cu130`, `transformers 5.13.0.dev0`, Python 3.10.

The smoke constructs the real upstream `Qwen3DSparkModel` from a tiny
`transformers.LlamaConfig`, with the one compatibility assignment required by
the pinned source:

```python
draft_config.sliding_window = getattr(target_config, "sliding_window", None)
```

That field is absent from the local target config. The model forward and
`compute_dspark_loss` run with finite values and finite gradients on CPU. CPU
eager attention uses a dense representation of the upstream DSpark mask only
because eager attention cannot consume the upstream `FlexAttention` block
mask. The model class and loss remain the official upstream classes.

The native CPU FlexAttention probe also completes a no-grad forward, but its
training path raises the exact PyTorch limitation
`NotImplementedError: FlexAttention does not support backward on CPU. Please
set the input requires_grad to False or use another device.` A real cloud
smoke should use the native Flex path and one CUDA device:

```text
CUDA_VISIBLE_DEVICES=0 python3 docs/campaign/work/r2-draft-upstream-smoke-v1/upstream_dspark_smoke.py --device cuda:0
```

That command is a launch shape only; this packet did not run it. The script
keeps its CPU official-cache roundtrip even when the model device is changed.

The cache proof writes one tiny sample with
`LocalTargetCacheWriter`, finalizes the official v2 index and manifest, and
reads it through `CacheDataset`. It verifies values, shapes, dtypes, the 56
byte `<QIIQQQQQ` index record, and v2 metadata. The official writer does not
require EOS; the generated sequence policy must still preserve exact target
prefixes, native EOG `[1, 130073]`, cap/protocol status, and explicit loss
masks. The reader intentionally does not return the writer's attention mask;
sequence length is carried by the index and padding is represented by the
collator's attention mask.

The released `openbmb/MiniCPM5-2B-DSpark` warm start remains a candidate for an
admitted cloud run. Its public safetensors header has 62 draft keys and omits
`embed_tokens` and `lm_head`; the bound target must supply those frozen tensors.
The export/load gate should allow exactly those two omissions and reject every
other missing or mismatched key. The public checkpoint was not downloaded by
this packet.

## Run

```text
PYTHONPATH=docs/campaign/work/r2-draft-upstream-smoke-v1/vendor/deepspec \
  python3 docs/campaign/work/r2-draft-upstream-smoke-v1/upstream_dspark_smoke.py
```

The resulting receipt is
`docs/campaign/receipts/RUN-08-r2-draft-upstream-smoke.json`.

## Findings from the teacher/cache review

The existing adapter tests cover authored-tail substitution prevention,
native EOG preservation, cap/protocol retention, padding, and the compact
index. The external review identified production hardening work that is not
silently claimed by this upstream proof:

- `_generation_parts` must require an unambiguous `generated_ids` mapping and
  verify a returned full sequence has the exact prompt prefix before taking a
  suffix. A bare HF `generate` sequence can otherwise duplicate the prompt.
- `classify_generated_tail` currently records tokens after the first native
  EOG and accepts over-cap or length-terminated rows. Production code must
  truncate at the first native EOG, reject over-cap rows, and apply the
  campaign decision on length termination while retaining the row outcome.
- Generator exceptions need a distinct count and a fail-closed stage policy;
  a systemic adapter error must not look like a successful zero-row stage.
- The writer must derive manifest sample counts from rows actually written,
  and validate target tensors are BF16 before byte packing. A padded row that
  ends in PAD/EOG ID 1 must retain its indexed sequence length through the
  attention mask.
- The admitted TRAIN JSONL is the exact TRAIN source path and its corpus
  receipt provides the split boundary, but the regeneration function itself
  only records `train_only: true`; it does not inspect a per-row split field.
  A production launch should bind the allowlisted TRAIN file/hash before
  regeneration and fail closed on any other source.

These are adapter readiness items for the owner of the prior implementation
packet. They do not invalidate the actual upstream model/cache smoke above.
