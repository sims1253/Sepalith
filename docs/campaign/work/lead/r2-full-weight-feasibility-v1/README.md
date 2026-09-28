# SFT-11 full-weight feasibility

This packet answers whether the merged SFT-500 MiniCPM parent can be trained
with every parameter on the available RTX 5090. It contains CPU-only arithmetic
and a lead-owned CUDA resource smoke. The worker did not load the model on CUDA
or start a training job.

The inspected model has 2,516,756,480 parameters. Its base tensor header is
all BF16, with 5,033,512,960 parameter bytes (4.688 GiB). The exact partition
used for the hybrid estimate is:

| group | parameters | role |
|---|---:|---|
| hidden 2-D matrices | 1,981,808,640 | Muon or Aurora momentum |
| embedding, `lm_head`, norms and other side parameters | 534,947,840 | AdamW side group |
| total | 2,516,756,480 | full model |

The persistent core arithmetic is BF16 weights plus BF16 gradients, before
activations, temporary buffers, allocator reserve, or checkpoint output:

| optimizer/state assumption | bytes | GiB |
|---|---:|---:|
| weights + gradients | 10,067,025,920 | 9.376 |
| AdamW with two BF16 moments | 20,134,051,840 | 18.751 |
| AdamW with two FP32 moments | 30,201,077,760 | 28.127 |
| AdamW with FP32 moments plus FP32 master weights | 40,268,103,680 | 37.503 |
| paged AdamW 8-bit estimate | 15,105,454,424–15,110,369,960 | 14.068–14.073 |
| Muon/Aurora BF16 hidden momentum + BF16 AdamW side moments | 16,170,434,560 | 15.060 |
| same hybrid with FP32 side moments | 18,310,225,920 | 17.053 |
| current composite implementation: FP32 hidden momentum + FP32 side moments | 22,273,843,200 | 20.744 |

The current `FullWeightCompositeOptimizer` preparation deliberately uses FP32
hidden momentum and FP32 side moments, so its measured target is the last row
of the table (no FP32 master weights). The smaller BF16 hybrid rows describe
the vendored POC's lower state policy and remain useful only if a separately
reviewed BF16 state implementation is selected.

The two BF16 moment result is supported by a CPU PyTorch probe. The paged
8-bit range assumes two uint8 moments and two FP32 scale values per 4096 or
2048 element block. CUDA state dtypes and allocator peaks remain acceptance
checks. Muon/Aurora has one hidden-matrix momentum, and its FP32 row norm is a
transient operation buffer; its MiniCPM dispatch and checkpoint integration
remain a separate implementation gate.

At batch one and 4096 tokens, the 42 layer-boundary BF16 hidden-state lower
bound is 704,643,072 bytes (0.656 GiB). A materialized BF16 logits tensor would
add 1,069,547,520 bytes (0.996 GiB) before backward copies. The smoke therefore
passes labels through the Unsloth loss path with `UNSLOTH_RETURN_LOGITS=0` and
records real peak allocation. These values are references, not an activation
upper bound.

The supported loader path is:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MERGED_SFT500,
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=True,
    float32_mixed_precision=False,
    fast_inference=False,
    trust_remote_code=False,
    use_gradient_checkpointing=True,
)
```

After loading, the copied long-run trainer must set `model.config.use_cache =
False`, call `FastLanguageModel.for_training`, skip `get_peft_model`, and assert
that every one of the 2,516,756,480 parameters is trainable. It must preserve
the target-only collator and fused/chunked cross entropy used by the reviewed
expanded source. The current expanded data contract remains
`max_sequence_tokens=4096` and `train_max_target_tokens=1024`; rows above those
limits stay in a measured long-context queue until a context-specific source
materialization is reviewed. No target is silently truncated.

The lead smoke uses eight exact 2048-token TRAIN rows from eight distinct
groups, only to exercise resource behavior. It initializes a fresh optimizer
from the merged SFT-500 weights, performs two one-row forward/backward steps,
checks finite loss and gradient coverage, records optimizer state bytes and
CUDA peaks, and saves no checkpoint. The prepared command is recorded in
`full-weight-smoke-recipe.json`; the lead may use `--optimizer
torch_adamw_bf16` after the paged AdamW arm if a second state measurement is
useful.

Root separately ran the copied no-optimizer arm on the same merged parent and
one 2048-token row. All 2,516,756,480 parameters were trainable across 381
tensors; all 381 gradient tensors were finite and nonzero. Peak allocation was
11,435,378,176 bytes (10.650 GiB), peak reserve was 11,830,034,432 bytes
(11.018 GiB), load took 3.91 seconds, and the forward/backward took 19.21
seconds. The external report is
`/mnt/e/sepalith/campaign-20260915/training/SFT11-full-weight-smoke-a/report.json`
(SHA256 `cf12448c2a832d34a1fe7ee4d1943e5ada7d3f611349f6919829f41be995f4c1`).
This establishes that the model and checkpointed forward/backward fit the
device. It does not establish optimizer-state memory.

A long run requires a new full-weight checkpoint writer. The accepted adapter
checkpoint callback assumes `adapter_model.safetensors` and
`adapter_config.json`, so it cannot seal this stage. The new writer must save
full `model.safetensors`, optimizer, scheduler, RNG, tokenizer identity, and
the fresh `sft_merged` parent binding at steps 250, 500, and 1000. It must also
partition Muon/Aurora explicitly so `lm_head` is never accidentally placed in
the matrix optimizer.

## Evidence pins

- Model config: `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-merged/config.json`, SHA256 `f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991`.
- Merged weights: `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-merged/model.safetensors`, SHA256 `631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c`.
- Parent manifest: `parent-manifest.preparation.json`, SHA256 `92157a4a52a4ed928f76df9f1f77a6aa39235eefd6a859eb9c2b3d4118ee18db`.
- Base tensor header: `/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain/tensor-header.json`, SHA256 `aa6110a153c82de57987fbeaaa11bdff3614f363b660389bea6f5817f40295a0`.
- Unsloth package versions used for source inspection: `unsloth 2026.8.18`, `torch 2.11.0`, `transformers 5.5.0`, `trl 0.24.0`, `bitsandbytes 0.50.1`.
- Existing LoRA profile: `docs/campaign/receipts/SFT-11-expanded-target-profile-root-readout.json`, SHA256 `710a356480b2d21a55db656027fc8cd1eca2ba328af295add3b27594f0587530`; it measured batch 2/4 finite gradients at sequence 3064, but is not full-weight evidence.
