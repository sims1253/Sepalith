# RL-06 mixed-memory review

## Scope and verdict

This is a CPU-only, read-only diagnosis of `RL-primary-p2-mixed-a`. The frozen source snapshot is `be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9`; the source file present in that snapshot is `experiments/training/campaign_rl_train.py` (there is no `campaign_rl_trainer.py` in the snapshot). No framework, model, CUDA device, or campaign state was loaded or changed.

The run completed its first 32-row rollout and reward calculation, then failed in the first backward call. The practical memory arm is the already-supported `per_device_train_batch_size=4`, `gradient_accumulation_steps=8` shape. It needs a new recipe/identity and a schedule metadata copy with `buffer_reuse=8`; no trainer-source patch is required. This is a memory mitigation and a distinct optimization lineage. It does not preserve exact optimizer or future RNG equivalence with the 8x4 arm.

## Observed failure

`RL-primary-p2-mixed-a-host-supervision/process.log` records:

- `Batch size per device = 8`, `Gradient accumulation steps = 4`, total batch `32`.
- four generation calls of eight rows each; the output generation JSONL has 32 records, eight G4 groups, and 32 reward records.
- `torch.OutOfMemoryError` during `trainer.train` -> Unsloth `_unsloth_training_step` -> Accelerate `backward` -> PyTorch autograd -> Unsloth Zoo gradient-checkpoint backward. The allocator requested 5.25 GiB with 10.16 GiB free, 15.96 GiB allocated, 3.94 GiB reserved, and a 23.88 GiB (75%) process allowance.
- child exit code 1 at `2026-09-12T19:20:17.857005+00:00`; no gradient or checkpoint record was written.

The asynchronous autograd traceback identifies the failure path and allocation request. It does not prove that the named gradient-checkpoint line is the original kernel fault. The mixed run had host paging pressure in preflight (`PageReadsPersec=681`, `PagesInputPersec=10851`, output 0), so the next arm must retain the existing host guard. This review makes no claim that 4x8 guarantees a fit.

## Exact geometry effect

`resolve_trl_geometry` at frozen `campaign_rl_train.py:536-614` requires the 32-row shape to be 8x4 or 4x8 and derives `generation_batch_size=32` and `steps_per_generation=gradient_accumulation_steps`. `CampaignRepeatSampler` at `:731-840` repeats each source group `steps_per_generation` times; its state at `:842-905` records both generation rows and sampler rows. The live trainer at `:1459-1497` enforces one generation buffer per update.

| field | current 8x4 | proposed 4x8 | effect |
|---|---:|---:|---|
| candidate count G | 4 | 4 | unchanged |
| source groups per update | 8 | 8 | unchanged |
| logical completions per update | 32 | 32 | unchanged |
| generation batch | 32 | 32 | unchanged |
| generation calls at P2 | 4 x 8 rows | 4 x 8 rows | unchanged |
| microbatch rows | 8 (two G groups) | 4 (one G group) | lower backward peak |
| steps per generation / accumulation | 4 | 8 | changed |
| sampler repeat count / buffer reuse | 4 | 8 | changed |
| sampler rows per optimizer update | 128 | 256 | changed bookkeeping, same 8 source draws |
| source draw cursor per optimizer update | +8 | +8 | unchanged |

The sampler is ordered and shuffle-free. With the same `row_ids` sequence, the eight source IDs used by an update and their four-candidate group boundaries remain the same. The 4x8 arm repeats that same 32-row generation buffer eight times for policy microsteps, rather than creating 64 new completions. The generation/reward denominator remains 32; the sampler denominator is 256 and must not be confused with rollout rows.

Because the schedule loader at `:416-510` rejects a `buffer_reuse` or `steps_per_generation` mismatch, v5's schedule metadata (`buffer_reuse=4`, `gradient_accumulation_steps=4`, `steps_per_generation=4`) cannot be reused unchanged. Keep the exact sequence and sequence SHA if desired, regenerate the self-describing schedule metadata with `buffer_reuse=8`, and bind its new full-file schedule SHA in the new identity. Keep source draws per update 8, candidate count 4, and all data/prompt/generation limits unchanged.

## BNPO and RNG semantics

The installed pinned TRL source (`trl/trainer/grpo_trainer.py:1488-1509,1730-1735`) and the production compiled Unsloth source (`unsloth_compiled_cache/UnslothGRPOTrainer.py:4838-4859,846-848`) compute group advantages over the complete reward buffer: `rewards.view(-1, num_generations)` with G=4, then apply the group standard deviation. This remains eight groups and 32 rewards in either arm.

The same production paths normalize BNPO loss per *current microbatch* token mask and then divide by `current_gradient_accumulation_steps` (`trl:1733-1735`; compiled: `:846-848` and `:2090-2092`). Therefore 4x8 is not objective-identical to 8x4 when completion lengths differ. In 8x4 each loss call contains two G4 groups and each call receives one quarter weight; in 4x8 each call contains one G4 group and receives one eighth weight. The per-group reward advantages and reward denominators stay fixed, but the aggregate gradient weights and floating-point reduction order change. This is an intentional, documented optimization semantic change, not a source-data or reward-contract change.

The ordered sampler uses no shuffle RNG. The four generation calls and 32 sampled rows per update can retain the same pre-generation RNG layout if the new arm starts from the same theta0 and all generation settings remain fixed. After generation, eight policy forwards/backwards instead of four consume a different model RNG stream and produce different post-update RNG, optimizer, and scheduler state. A 4x8 checkpoint must therefore have a new identity and must not resume an 8x4 checkpoint or claim bytewise continuation equality.

## Recommended correction and live gate

Use root's prepared 4x8 arm (`primary-mixed-mb4-a`, recipe SHA `b7b97beb92afb2b47a5b33226b3334b18367134773b8235645807624d4571df4`, identity SHA `dee63a052efeb9dd1ddeb2fbbf7e623c0b3cc8dc61cefcd8d5526df7ddbe8384`, v6 schedule SHA `2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132`). Keep the existing 0.75 CUDA fraction, single-device owner, 8 GiB host floor, model context/generation caps, P2 generation call policy, and frozen source sequence. The first live acceptance must show:

1. successful generation of exactly 32 records in four calls and 32 rewards in eight G4 groups;
2. first backward finite/nonzero LoRA gradient telemetry and no allocator failure;
3. sampler state with `repeat_count=8`, `sampler_rows_per_update=256`, source cursor +8, and the exact v6 schedule/sequence hashes;
4. full checkpoint at the requested boundary, with the new identity; and
5. no claim of 8x4 optimizer or RNG equality. A later resume check is within the new 4x8 lineage only.

If the 4x8 backward still fails, the next reduction would change the 32-row update contract and needs a separate scientific decision; do not silently lower rollout rows or candidate/group counts.

## Required CPU checks

The existing constructors should pass for 8x4 and 4x8 and reject product mismatches, non-4/8 32-row shapes, world size other than one, and schedule `buffer_reuse` mismatches. Add or run a bounded fixture that checks the sampler index stream and state at step 1 for both arms, confirms the same eight source IDs and 32 logical rows, and distinguishes generation rows (32) from sampler rows (128/256). A BNPO fixture with unequal completion lengths should assert the documented 8x4 versus 4x8 aggregate weighting difference. The live arm must additionally verify one gradient record, generation/reward joins, checkpoint identity, and no OOM; CPU evidence cannot close that gate.

## Evidence and limits

Known artifact hashes and sizes are recorded in the receipt. The 52 MiB telemetry JSONL and multi-megabyte identity/preflight files were not parsed. Small generation/reward JSONL and the 7,396-byte supervision process log were read. No model weights, tensors, framework imports, processes, SSH, or GPU operations were used.
