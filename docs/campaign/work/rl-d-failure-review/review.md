# RL-08 full5-d failure-window audit

Status: **failed during update 104 backward**. This review uses only compact run records, host-guard metadata, the frozen source schedule, and targeted source line reads. It does not read model weights, raw targets, or raw source bodies.

## Observed failure

The generation window is `global_step=103`, `global_step_before_update=103`, `trl_microstep=24`; it contains 32 candidates in 8 G=4 groups and 4 generation calls (2 groups per call). Generation and reward accounting completed: 31 canonical EOS, one completion cap hit, zero invalid/noncanonical terminals, 34,436 prompt tokens, 1,436 generated tokens, and 1,428 post-terminal padding tokens. Reward records join one-to-one with all 32 generated records. The eight source IDs are schedule rows 824:832 of v6; their compact list hash is in the JSON receipt.

Three gradient records are durable for steps/global steps 101, 102, and 103; each is finite with 588 present and 588 nonzero LoRA tensors. There is no gradient record for the subsequent backward/update 104. The failure JSON reports `CUDA out of memory: Tried to allocate 3.56 GiB`, with 31.84 GiB total, 9.58 GiB free, 23.88 GiB allowed, 16.19 GiB allocated, and 4.28 GiB reserved but unallocated. The traceback reaches the production trainer backward through Accelerate and Unsloth gradient checkpointing.

## Length geometry

The eight consecutive record-order optimization microbatches (B=4, accumulation=8) are captured as prompt lengths, generated lengths, and per-row prompt+generated lengths in `failure-window-summary.json`. Their local padded-width envelopes (`max prompt + max generated`) are 1876, 622, 993, 1918, 296, 1145, 2027, and 173 tokens for groups 0 through 7. The global window has max prompt 2002 and max completion 192, giving a 2194-token prompt-plus-completion envelope under the fixed 2240 context cap. These are derived from persisted row lengths; the collator's actual per-forward tensor width was not persisted.

## Logits allocation finding

The pinned Unsloth loss path sets `logits_to_keep` from completion width, calls `grpo_accumulated_loss`, and invokes the model output/selective log-softmax path. The traceback does not record a tensor shape. For sizing only, 3.56 GiB equals about a full FP32 `[4, 1830, 130560]` tensor; BF16 would require about 3660 sequence positions. A full `[4, 2194, 130560]` tensor is larger than the requested allocation, while multiple hidden/logit/gradient buffers can sum to it. Therefore the exact largest allocation is unresolved, and the separate recompute-policy-logprob profile cannot be treated as proof of the production peak.

## Checkpoint and host state

`archive/` and `output/` contain no `checkpoint-*` files, so no post-100 durable checkpoint exists for this failed attempt. The host guard recorded child exit 1 at `2026-09-12T23:13:58.453794+00:00` after 385.887740 seconds; launch PIDs 3711815 and 3711854 were gone when checked. Preflight had an 8192 MiB floor, 15,025 MiB initial Windows available memory, zero page reads/input/output, and no driver events; a low-memory sample still reported 40,992,020 kB Linux MemAvailable and no swap. This supports a CUDA allocator/peak-pressure failure and gives no evidence of a data or host paging fault.

The run used the admitted 0.75 per-process CUDA fraction and `expandable_segments:True`. Raising the cap to 0.80 is not established as a repair by this failure: it increases the permitted footprint and needs a separately admitted safety check.

Source evidence: `campaign_rl_train.py:1464-1497` enforces B=4/accumulation=8/G=4 geometry; `campaign_rl_train.py:1499-1521` persists sampler state at checkpoint boundaries; `campaign_rl_entry.py:1109-1116` writes failure evidence and re-raises.
