# R2 teacher/cache adapter hardening

This packet is a hardened copy of `r2-draft-implementation-v1`; the original
packet is unchanged. It contains no target or draft weights and performs no
GPU/cloud work. The upstream model/cache proof remains in
`r2-draft-upstream-smoke-v1`.

The adapter now requires the exact admitted TRAIN source binding:

```text
docs/campaign/work/r2-task-mixture-v1/verified-corrected-short-token-rows.jsonl
sha256 85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e
```

It validates optional row split fields as `train` or `cpt_train`, requires
prompt BOS `0`, and rejects prompt or cache IDs outside vocabulary `130560`.
Missing or duplicate row IDs fail before deterministic selection.

Generation adapters must return a mapping with explicit `generated_ids`. If a
full `input_ids` sequence is supplied, it must equal the exact prompt prefix
followed by that generated suffix. Bare sequences and prompt duplication are
rejected. Native EOG is preserved through the first `[1, 130073]` stop; any
post-EOG suffix is rejected and excluded from labels. A future batched adapter
must provide an explicit valid length and prove removed values are PAD `1`.
Over-cap output is rejected, while a
valid exact cap hit and an invalid edit protocol remain cacheable when all
retained tokens are valid. A short generation without EOG is marked
`length_terminated` and rejected. No path synthesizes EOS or copies an
authored tail.

Target-service exceptions propagate as systemic failures. Malformed rows and
malformed generator mappings become explicit `generation_error` records, and
the summary counts them. Cache manifests can carry the IDs actually written;
their `num_samples` and written-ID digest are derived from that list. The
collator fails on misalignment or silent low-target rows instead of dropping
them. Target hidden tensors must be `torch.bfloat16` before byte packing.

The focused suite includes the prior compatibility and tiny-model checks plus
tests for prompt duplication, post-EOG rejection, over-cap and
length-terminated output, systemic errors, source/split/BOS/vocabulary gates,
BF16 validation, written-ID reconciliation, and the official DeepSpec
`LocalTargetCacheWriter` → v2 index/manifest → `CacheDataset` roundtrip. The
roundtrip ends in native EOG/PAD ID `1` and verifies indexed length `4` and
mask `[1, 1, 1, 1]`; PAD remains an ID contract and length is mask/index based.

Run the CPU checks with two threads:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=docs/campaign/work/r2-draft-adapter-hardening-v1 \
  python3 -m unittest discover \
  -s docs/campaign/work/r2-draft-adapter-hardening-v1 -p 'test_contract.py'
```

The test receipt is
`docs/campaign/receipts/RUN-08-r2-draft-adapter-hardening.json`.

Remaining launch gates are target selection and private target persistence,
official target-layer hooks/cache generation, public warm-start header loading
with frozen target `embed_tokens`/`lm_head`, native CUDA FlexAttention smoke,
finite eight-row generation/cache/memory profile, resumable checkpointing, and
the existing budget/admission decision. This packet does not claim trained
quality or serving benefit.
