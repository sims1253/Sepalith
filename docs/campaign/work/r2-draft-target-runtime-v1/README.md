# R2 target regeneration and official cache runtime

This packet adds the smallest target runtime after the adapter hardening
packet. It is a separate copy and does not edit the earlier packets, trainer,
warm-start, or campaign files. It contains no campaign model weights. The
runtime requires an explicit local model directory plus config and weight
SHA-256 values before calling any HF loader.

`target_runtime.py` uses a real `AutoModelForCausalLM` for deterministic
greedy generation over pretokenized prompt IDs. It returns both
`generated_ids` and the full `input_ids` sequence, stops only on native EOG
`[1, 130073]`, and caps generation at 192 tokens. It has no chat-template,
authored-tail, or EOS-synthesis path. The hardened adapter rejects an
unverified suffix after EOG, over-cap output, and short non-EOG output while
retaining valid cap-hit and invalid-edit rows.

For target features, it calls the pinned upstream
`scripts.data.prepare_target_cache.run_target_forward_with_hooks` on the
CausalLM backbone (`target_model.model`). The official layer hooks capture
the outputs at the configured taps, and the runtime compares their
concatenation with `hidden_states[tap + 1]`. It separately verifies the
normalized `last_hidden_state`. The captured tensors are converted to BF16 and
written through the official `LocalTargetCacheWriter`; the official index,
manifest, and `CacheDataset` are then finalized and read back. Manifest and
index counts use the IDs actually written.

The default CLI limit is an eight-row profile. Limits 9–256 require the
explicit `--allow-bounded-run` flag, so root can measure the profile before a
larger run. The required source binding remains:

```text
docs/campaign/work/r2-task-mixture-v1/verified-corrected-short-token-rows.jsonl
sha256 85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e
```

The CPU integration uses a six-layer randomly initialized HF Llama CausalLM,
taps `[0, 1, 2, 3, 4]`, and a calibrated native EOG first prediction. It
verifies the exact prompt-only prefix, loss mask `[0, 0, 0, 1]`, hook/tap
parity, target identity hash gate, official v2 cache, 56-byte index, and
written row ID. It loads only the temporary tiny test weights and deletes the
temporary model/cache before returning.

Run the CPU integration:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=docs/campaign/work/r2-draft-target-runtime-v1:docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec \
  python3 -m unittest discover \
  -s docs/campaign/work/r2-draft-target-runtime-v1 -p 'test_target_runtime.py'
```

The default profile CLI shape after root supplies a verified model manifest is:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=docs/campaign/work/r2-draft-target-runtime-v1:docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec \
  python3 docs/campaign/work/r2-draft-target-runtime-v1/target_runtime.py \
  --model-manifest <verified-model-manifest.json> \
  --train-data-path docs/campaign/work/r2-task-mixture-v1/verified-corrected-short-token-rows.jsonl \
  --output-dir <ephemeral-target-cache> \
  --limit 8
```

The package used for this local run was `/usr/bin/python3` 3.10.12 with
`transformers 5.13.0.dev0` from
`/home/m0hawk/.local/lib/python3.10/site-packages/transformers` and
`torch 2.11.0+cu130` from `/home/m0hawk/.local/lib/python3.10/site-packages/torch`.
Root separately reran the upstream smoke in the canonical `transformers
5.5.0` environment; this packet did not mutate or load that training
environment.

The receipt is
`docs/campaign/receipts/RUN-08-r2-draft-target-runtime.json`. Remaining gaps
are the actual private target directory/hash manifest, target generation over
the admitted rows, CUDA/native FlexAttention and A10G memory profiling,
trainer/warm-start binding, resumable checkpoint persistence, quality gates,
and root’s existing budget/admission decision.
