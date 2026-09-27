# Production training handoff — September 27, 2026

## Current decision

The user selected **CPT checkpoint 11,586** as the parent for editing SFT.
CPT completed all 11,649 planned steps on September 26 at 08:06 Berlin time.
Editing SFT and RL have not started. No production release was promoted.

Use the weights and tokenizer from checkpoint 11,586 to initialize the new
editing objective. Its CPT optimizer and sampler state are not an editing-SFT
resume state. Preserve checkpoint 11,649 as the completed-CPT alternative.

The [selection receipt](SFT-11-editing-parent-selected-11586.json) binds the
checkpoint, model and tokenizer hashes. Both retained full checkpoints passed
payload-hash and sampler-cursor verification during the comparison.

## Model

The checkpoint is a dense `LlamaForCausalLM` from the MiniCPM5-2B lineage. It
is not the Qwen3.5 gated-DeltaNet hybrid used in the earlier LoRA experiments.
Its `config.json` lists 42 layers, hidden size 2048, 16 attention heads with 2
key-value heads, a 130,560-token vocabulary, untied embeddings and
end-of-generation ids 1 and 130073. Training updates all 2,516,756,480
parameters, and the full-weight optimizer state takes 12.2 GB.

The Qwen3.5-specific dtype failures in the
[Kaggle notes](../../research/2026-09-06-kaggle-compute-integration.md) do not
apply to this model. Kaggle's 16 GB T4 GPUs still cannot hold its full-weight
training state.

## Checkpoint comparison

Lower held-out causal loss is better. Both checkpoints used identical cases,
evaluation code and settings; only the model binding differed.

| Context | Checkpoint 11,586 | Checkpoint 11,649 | Cases favoring 11,586 |
| --- | ---: | ---: | ---: |
| 2K | 0.94196984 | 0.94434960 | 364 / 499 |
| 8K | 0.71110394 | 0.71168698 | 12 / 20 |
| 16K | 0.76016675 | 0.76103432 | 6 / 6 |

The advantage is small: 0.08–0.25% lower loss. The 2K panel covers 50 packages;
the 8K and 16K panels cover only eight and three packages. The package-bootstrap
interval for the 8K difference includes zero. These results support the parent
choice but do not establish better editing quality after SFT.

See the [paired comparison](comparison.json),
[comparison receipt](SFT-11-cpt-parent-comparison-20260927.json), and
[complete evaluation history](cpt-evaluation-history.json).
The sealed final editing evaluation was not accessed.

## Next stage

The [post-CPT plan](../plan-20260927/README.md) supersedes the preparation
notes below where they conflict. It audits these packets, fixes their known
problems and splits the remaining work into hand-off cards.

The prepared full-weight editing-SFT packet needs its parent and training recipe
bound to this selection. The reviewed baseline contains 15,006 examples with
complete targets and no truncation. A separate reviewed queue of 10,017 roxygen
candidates still needs a training-admission decision under the all-eligible-data
policy. Do not silently discard that queue or treat it as already admitted.

The prepared development gate uses 75 editing/no-op cases and a candidate
1,024-token generation budget. It requires a standalone evaluator resource
proof and checkpoint-bound results before continuation. Confirm the latest
local packets before choosing the schedule, learning rate and gate cadence:

- [Editing-SFT preparation](SFT-11-full-weight-edit-SFT-preparation.json)
- [Development-evaluation gate preparation](SFT-11-full-weight-edit-SFT-eval-gate-preparation.json)

The earlier 48-hour unrestricted GPU window ended September 26 at 12:51 Berlin
time. The supervisor completed CPT within that window and is stopped. This
handoff does not extend that window or launch SFT.

For later RL, retain GRPO as the baseline and prioritize useful learning signal
per rollout. KLPO remains an unproven comparison candidate for this task. The
[research receipt](RL-11-new-resources-review-20260922.json) records the local
review and qualifications; it does not admit RL training.

## Artifact locations and recovery

The evidence in this directory is portable metadata. Absolute paths in the
receipts describe the training host; they are not files supplied by a clone.
Model weights, optimizer payloads, datasets, credentials and full runtime logs
remain outside Git.

- Durable checkpoints: `/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-{11586,11649}`
- Final native checkpoint: `/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-cpt450-cadence64-to706-v1/runtime/checkpoint-11649`
- Runtime and supervisors: `/home/m0hawk/.local/state/sepalith/resume-20260921`
- Full campaign plan, receipts and prepared sources: `/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign`
- Prior execution source: `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`

The September 22 stop was a host-memory guard intervention during checkpoint
publication. Recovery added targeted clean-file-cache release without lowering
the guards. A later September 23 stop was the disk free-space guard. A subsequent
recovery continued from checkpoint 4,930. The final 48-hour continuation completed
normally. Do not interpret the first recovery supervisor's failure log as the
current runtime status; use the included [terminal status](status.json).

The runtime retained Windows soft/hard thresholds of 8/4 GiB, a 16 GiB admission
threshold, a 6 GiB Linux available-memory floor, a 95% CUDA allocator cap, and a
70 GiB disk reserve after the next expected checkpoint. These are stop conditions,
not guarantees against every possible resource failure.

## Repository and worktree reconciliation

The September 27 inventory found 22 registered worktrees. Two worktrees had no
tracked changes, untracked files, ignored files or unpublished commits and were
removed without force. Their branches remain:

- `t3code/evaluate-colab-sft-rl`
- `t3code/check-paper-presence`

Twenty worktrees remain. The planning worktree alone had 25,790 untracked files,
and several other worktrees contain uncommitted changes or ignored artifacts.
They were preserved. The planning branch also has nine commits absent from the
fetched remote main; this focused handoff does not merge or publish that larger
backlog. Review it separately before consolidation, particularly its data
ledgers and host-path dependencies. See the
[pre-cleanup inventory](worktree-inventory-before-cleanup.json).
