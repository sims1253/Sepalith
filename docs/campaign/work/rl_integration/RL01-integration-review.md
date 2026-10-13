# RL-01 integration review

Observed against the execution worktree on 2026-09-12. The RL-02 context
sidecar is now the preferred data input. Its rows carry `source_identity` and
`selection_geometry` beside the target-free `PromptContext`; the loader binds
the complete sidecar byte hash and an ordered per-row identity hash. Rows may
refer to different static source documents. A common source snapshot remains
accepted only for the legacy `schema_version`/`capture` fixture shape.

The pinned TRL 0.24.0 geometry supports both measured 32-row arms:

| arm | policy microbatch | gradient accumulation / SPS | generated rows per update | generation groups |
| --- | ---: | ---: | ---: | ---: |
| selected default | 8 | 4 | 32 | G=4, eight groups |
| memory alternate | 4 | 8 | 32 | G=4, eight groups |

`num_iterations=1`, and the dynamic trainer rejects a mismatched generation
batch, accumulation/SPS pair, or shuffle. The fixed-ID adapter calls
`model.generate` once per contiguous G-sized repeated-prompt group, retains the
stored prompt IDs/manual BOS and native EOG output, checks the returned prompt
prefix, and records group order/accounting. The generation guard surrounds the
whole buffer and restores once after the buffer completes.

TRL's dataloader consumes one generation-sized batch per accumulation step. A
full optimizer checkpoint therefore advances the sampler stream by
`generation_batch_size * steps_per_generation` (128 rows for 8x4 or 256 rows
for 4x8), while the generation buffer itself contains 32 completions. The
checkpoint sampler payload records both counts and reconstructs the next
buffer only at a generation/update boundary.

The TRL import boundary normalizes its missing optional-dependency tuple flags
before loading `GRPOTrainer`; the actual pinned subclass and constructor now
work in the canonical environment without installing mergekit or llm_blender.
The reward path remains PRM03 exact semantic reward plus 0.2 line-F1 shaping,
with canonical EOS, raw native CONTROL, no-op/delete, and target-outside-prompt
gates intact.

Live CUDA generation, optimizer/backward, merged-SFT parent loading, sustained
resume interruption, and promotion remain lead-owned gates.
