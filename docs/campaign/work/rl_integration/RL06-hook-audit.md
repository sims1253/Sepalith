# RL-06 checkpoint and telemetry hook audit

Observed 2026-09-12 in the execution worktree
`/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`.  This was a
CPU-only, read-only audit of the RL trainer, supervised entry, launcher,
checkpoint/control helpers, and the installed TRL/Transformers sources.  No
implementation file, model weight, CUDA process, server, cloud resource, or
campaign state was changed.

The source snapshot was:

| surface | SHA256 |
| --- | --- |
| `experiments/training/campaign_rl_train.py` | `794348ccf5976133fec76ca8bf706fcabe0f7f4fde7fc2325bff226e748f87a0` |
| `experiments/training/campaign_rl_entry.py` | `793a282570be4c90f1804ab43bd7320519fbc6f91f5eaaa4b87ce41bb99096ee` |
| `experiments/training/campaign_rl_launch.py` | `fa245326052bfae2997d91d80bbf8f45c055d35ffcd0b4b2b7d4440045ffb36a` |
| `experiments/training/campaign_checkpoint.py` | `64202f4e5b88c94a74794442be3a4955f07d195079c9d64bbcff2850dbadd0ba` |
| `experiments/training/campaign_control.py` | `9a5b0614f134ecf6b804d2f0d5e37a4fecf2b1dc53589b92f17580678f9dbdac` |
| installed `trl/trainer/grpo_trainer.py` | `bb6905182acf2ec426f1dc9ffc37e5210d58ee47d2764f7cf12dbb90a4a28954` |
| installed `transformers/trainer.py` | `17e212935057f58b52efe79c3f952bf044bbada8afd4f13c8e905142b1d2ef0a` |

## Findings

The structural full-checkpoint path is **PASS**.  `make_grpo_config` pins
step saves, `save_only_model=False`, and `ignore_data_skip=False`
(`campaign_rl_train.py:387-430`).  The checkpoint callback repeats those
checks at `on_train_begin` and rejects a non-step save strategy
(`campaign_checkpoint.py:170-200`).  `FULL_STATE_FILES` requires
`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, and `trainer_state.json`, and
`seal_checkpoint` checks their presence, the saved global step, and nonempty
sampler metadata before writing the inventory manifest
(`campaign_checkpoint.py:19-20,89-111`).  The pinned Transformers save path
indeed writes optimizer/scheduler, scaler, RNG, and trainer state
(`transformers/trainer.py:3032-3086`), while its load path restores model state,
trainer state, optimizer/scheduler, and RNG
(`transformers/trainer.py:1407-1414,3139-3187,3559-3635`).

The deterministic sampler identity is **PASS in source and CPU fixtures**.
`CampaignRepeatSampler` rejects shuffle, expands contiguous candidate groups,
and records seed, cursor, source schedule hashes, and update geometry
(`campaign_rl_train.py:469-665`).  The dynamic trainer computes the consumed
stream at an optimizer boundary as
`global_step * generation_batch_size * steps_per_generation` and stores the
loaded data identity (`campaign_rl_train.py:1179-1201`).  The existing test
reconstructs the first buffer, its repeated accumulation slices, and the next
buffer at 128 rows for the 8x4 geometry and 256 rows for the 4x8 constructor
fixture (`test_campaign_rl_train.py:470-525,836-886`).

The RL-specific generated-buffer resume proof is **PENDING**.  Installed TRL
initializes `_step=0` and `_buffered_inputs=None`
(`trl/trainer/grpo_trainer.py:390-398`).  On resume, `_prepare_inputs` observes
the empty buffer, calls `_generate_and_score_completions`, splits the generated
batch, and stores it only in process memory
(`trl/trainer/grpo_trainer.py:982-1015`).  No custom stateful callback saves
`_step`, `_buffered_inputs`, completion IDs, rewards, advantages, or old log
probabilities.  The current geometry (`steps_per_generation ==
gradient_accumulation_steps`, `num_iterations == 1`) means an optimizer-boundary
checkpoint should have consumed the whole buffer, so deterministic RNG restore
and source-schedule reconstruction can regenerate the next buffer in theory.
That claim has not been proven with an actual `CampaignGRPOTrainer` interruption
and resume comparison.  `verify_checkpoint` checks the sealed bytes and top-level
identity, but does not parse or independently validate the sampler payload in
`campaign-state.json` (`campaign_checkpoint.py:114-123`).

The required adapter cadence is **PENDING recipe binding**.  The callback
supports a light cadence that divides the full cadence and writes adapter plus
tokenizer snapshots at `on_step_end`; full checkpoints are archived at
`on_save` (`campaign_checkpoint.py:201-248`).  However, the RL defaults are
`light=10` and `full=50` (`campaign_rl_train.py:80-87`), and the entry defaults
to those same values (`campaign_rl_entry.py:368-373`).  RL-06 requires
`light_save_steps=100` and `full_save_steps=500`; the future recipe must set
those fields explicitly and use a `max_steps` divisible by 500.  No current
test asserts the required 100/500 values, and no live archive was produced in
this CPU audit.  This is configuration readiness, not evidence that the
required cadence has run.

Reward and trainer telemetry is **PARTIAL / PENDING live validation**.  The
per-completion reward record contains family/package, generated length,
canonical/noncanonical termination, control-token count, cap, protocol status,
operation, line-F1 shaping, reward, failure reason, and output-ID hash
(`campaign_rl_train.py:750-841`).  The entry installs an fsynced per-attempt
JSONL sink (`campaign_rl_entry.py:669-672`).  Installed TRL supplies completion
length and clipped-ratio metrics (`trl/trainer/grpo_trainer.py:1318-1346`),
reward mean/std and `frac_reward_zero_std`
(`trl/trainer/grpo_trainer.py:1519-1527`), and Transformers supplies
`grad_norm` in ordinary logging (`transformers/trainer.py:2052-2074`), which
the control callback persists as `trainer_metrics`
(`campaign_control.py:98-100`).

There are two material telemetry gaps.  The generation adapter records
`terminal_reason` and accounting in `_campaign_last_generation`, but that object
is in memory only (`campaign_rl_train.py:1022-1037,1088-1104`); the reward sink
does not persist terminal reason, generation/update step, or a per-generation
accounting record.  Also, `grad_norm` is logged when the Trainer emits metrics,
but there is no RL-specific finite-and-nonzero adapter-gradient assertion.  A
live first-update receipt must therefore supply the gradient check and prove the
family, cap, termination, zero-variance, and length fields are present in the
durable artifacts.

Same-owner development evaluation and mode restoration are **PASS in source;
PENDING live exercise**.  The checkpoint callback invokes the evaluator
synchronously from `on_save`, in the process that owns the trainer/model
(`campaign_checkpoint.py:222-248`).  `preserve_random_state` enters eval mode
under `torch.no_grad()` and restores model mode plus Python/NumPy/CPU/CUDA RNG
state even on failure (`campaign_checkpoint.py:149-167`).  The RL entry passes
the existing supported Unsloth training-configuration guard as
`evaluation_context` (`campaign_rl_entry.py:660-695`), and the guard removes
generation markers, restores checkpointing/config/tokenizer state, and leaves
the exception visible (`campaign_sft.py:171-195`).  The CPU test exercises
failure cleanup and RNG/mode restoration (`test_campaign_checkpoint.py:83-119`).
There is no live RL evaluator invocation, and an arbitrary future evaluator
factory is not independently prevented from spawning work; root must keep the
factory local to the single CUDA owner.

The 4K development-panel path is **PENDING an explicit context-capacity gate**.
The existing `campaign_eval:development_evaluator` factory requires an SFT-shaped
evaluation wrapper: `renderer_id`, a verified `development_panel`, predeclared
`development_case_ids`, `development_max_new_tokens`, and
`parameters.max_sequence_tokens` (`campaign_eval.py:121-143`).  It builds the
full teacher-forced row and then generates inside the same
`case_evaluation_guard`; both the prompt plus the requested cap and the full
reference must fit the declared capacity (`campaign_eval.py:145-180`).  The
admitted 75-case DEV panel has observed maximum prompt length 2619 with a 512
token generation cap, so its evaluator capacity is 4096.  The panel identity is
`/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl`
with SHA256
`b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`.

The RL entry currently loads the model with Unsloth `max_seq_length=2240`
(`campaign_rl_entry.py:548-585`), while the fixed-ID GRPO runtime enforces the
separate admitted policy geometry of prompt 2048, completion 192, context 2240
(`campaign_rl_train.py:884-905,1040-1053`).  Passing the RL recipe directly to
the development factory is also structurally invalid because the factory reads
the SFT `parameters` and `development_*` fields.  A model compiled/loaded only
for 2240 cannot be accepted as evidence for the 4096-row evaluator: the
teacher-forced pass or generation may reject, truncate, or exceed the managed
context.  The current source has no explicit load-capacity field that binds
these two contracts.

The minimal consistent live design is to load the one verified base model with
an explicit `model_load_max_seq_length=4096` identity field, keep the RL
rollout/policy caps frozen at 2048/192/2240, and give the evaluator a small
SFT-shaped wrapper containing `parameters.max_sequence_tokens=4096` plus its
verified panel and 512-token cap.  A 4096 load capacity does not by itself
raise the GRPO caps enforced above, but it can change compiled memory and peak
allocation.  Root must measure one RL-length probe and one longest DEV
teacher-forced/generation probe under the same CUDA owner, verify the mode and
tokenizer are restored, and include the 4K peak in the checkpoint reserve.  If
the 4096 load cannot fit the admitted lease, the gate must stop with that
measured result; it must not silently exclude the long DEV cases or reduce the
DEV cap.  The existing prefill evidence records the intended split as
`training_context_admission=4096` and `rl_context_admission=2048`
(`docs/campaign/receipts/RUN-06-prefill-options.json:78-80`).

The policy log-probability path is **PENDING live memory confirmation with a
known source mismatch**.  Installed TRL does not pass `labels` in
`_get_per_token_logps_and_entropies`; it optionally passes `logits_to_keep + 1`,
drops the final prediction position, retains the trailing padded completion
width, divides by temperature, and applies `selective_log_softmax`
(`trl/trainer/grpo_trainer.py:786-857`).  The loss later masks padded completion
positions (`trl/trainer/grpo_trainer.py:1652-1735`).  This agrees with the
accepted `labels=None` and response-logit-only direction and does not admit the
full `[8,~2240,130560]` FP32 conversion.  It differs from the measured profile's
boolean response-position selection before FP32 conversion: TRL retains padded
completion columns, and if MiniCPM does not expose `logits_to_keep`, the model
may produce full sequence logits before TRL slices them.  The real MiniCPM
signature, dtype, peak memory, finite loss, and adapter gradients remain a
root-owned CUDA gate.

Parent/config identity and supervision are **PASS in CPU preflight/tests**.
The entry verifies the accepted merged-SFT manifest, exact `config.json` and
`generation_config.json` bytes, tokenizer files/IDs, and every weight inventory
entry before model imports (`campaign_rl_entry.py:215-350`).  It loads BF16,
non-4-bit, `trust_remote_code=False` only after the gates
(`campaign_rl_entry.py:548-585`), checks r16/a16 attachment counts, and binds
the loaded parent identity (`campaign_rl_entry.py:588-610,1506-1515`).  The
launcher writes an immutable supervision receipt and passes a distinct
entry-result receipt (`campaign_rl_launch.py:70-115`); entry success and failure
tests preserve the supervision bytes (`test_campaign_rl_entry.py:324-366`).

One wiring issue should be resolved before accepting a recipe with omitted
evaluation steps.  `_validate_schedule` supplies a default `[max_steps]`
(`campaign_rl_entry.py:374-383`), but `build_live_trainer` passes an empty tuple
when the recipe omits the field (`campaign_rl_train.py:1571-1582`).  The
checkpoint callback rejects an explicitly empty evaluation set
(`campaign_checkpoint.py:185-188`).  Root can avoid this on the immediate run
by declaring `evaluation_steps` explicitly; the durable fix is to pass the
validated schedule through the builder or reject omission during entry
validation.

## Required root-owned dry-run gate

Before releasing the CUDA lane to RL-05, root should perform this disposable
theta0 check:

1. Export an accepted merged-SFT `theta0` parent manifest with exact
   `config.json`/`generation_config.json` hashes, complete weight inventory,
   tokenizer IDs/hashes, base revision, and accepted SFT identity.  Put its
   manifest hash and weight identity in the RL identity.
2. Materialize a recipe with the admitted RL-02 sidecar and lead-order selected
   IDs, `G=4`, 8x4 generation/update geometry, BNPO/group/beta=0, explicit
   `light_save_steps=100`, `full_save_steps=500`, fresh native-ext4 output,
   archive, and telemetry paths, and an evaluator factory with predeclared
   development IDs/denominators.
3. Run framework-free preflight, then a single-owner CUDA attempt with a
   disposable `max_steps=1000` hook recipe.  Stop at the full checkpoint at
   step 500.  Verify adapter snapshots at 100, 200, 300, 400, and 500 and a
   full archive containing adapter files, optimizer, scheduler, RNG, trainer,
   and campaign state/inventory files.
4. Start a fresh supervised attempt from the sealed step-500 checkpoint and
   resume to step 1000.  Compare its post-resume source-draw cursor,
   generation-buffer boundary, output/reward identities, optimizer/scheduler
   states, RNG state, and adapter weights against an uninterrupted 1000-step
   reference.  A file-presence check is insufficient for RL because TRL's
   generation buffer is regenerated in memory.
5. Require the first RL update to show finite loss, finite nonzero trainable
   adapter gradients, reward/family/length/cap/termination and zero-variance
   telemetry, and a same-process development evaluation whose model/config/
   tokenizer mode is restored before the next update.  Verify the launcher
   supervision receipt remains byte-identical and the entry-result receipt is
   distinct.  Keep the known PEFT/Unsloth `_flag_for_generation` transition
   gate fail-closed; do not synthesize private attributes.
6. Bind `model_load_max_seq_length=4096` and run one longest admitted DEV case
   through the SFT-shaped evaluator wrapper while the same process still owns
   the model.  In the same disposable attempt, run an admitted RL prompt at
   the frozen 2240 context limit.  Record both peak-memory observations,
   generation lengths, and post-evaluation mode/tokenizer restoration.  The
   RL source-selection and rollout caps remain 2048/192/2240 throughout this
   probe.

## Proposed remedial edits (not applied here)

* Add an RL checkpoint boundary assertion/validator in the dynamic trainer or
  checkpoint callback that requires `global_step` to be at a complete
  `steps_per_generation` boundary and validates the sealed sampler payload
  against current source-schedule/data identity on resume.  Keep the explicit
  one-buffer-per-update regeneration contract, or persist the generated token
  IDs/rewards/advantages/old log-probs as a stateful trainer artifact if a
  future recipe permits mid-buffer saves.
* Emit a durable generation accounting event beside `reward-records.jsonl`
  containing step, source/generation identity, terminal reasons, canonical and
  noncanonical EOG counts, control count, cap count, and length denominators.
  Add a first-update finite/nonzero trainable-gradient guard and a focused test
  for the required telemetry keys.
* Make the 100/500 cadence explicit in the admitted recipe schema or change the
  RL defaults to the campaign cadence, and test the exact values.  Wire the
  validated default evaluation list into `build_live_trainer` so preflight and
  callback construction cannot disagree.
* Add a root-owned live probe that records whether MiniCPM accepts
  `logits_to_keep`, confirms `labels=None`, records response-logit shape/dtype
  and peak memory, and rejects a full FP32 vocabulary conversion.
* Add an explicit model-load context field to the admitted RL identity and
  entry manifest (4096 for the current DEV panel), then construct the existing
  development evaluator from an SFT-shaped wrapper.  Keep this field separate
  from the RL prompt/completion/context caps and add a live two-length memory
  and mode-restoration probe before accepting same-owner evaluation.

The CPU result is therefore **preflight and structural-hook pass, live RL hook
acceptance pending**.  No claim of theta0 loading, RL training, generated-buffer
resume parity, live gradient health, CUDA memory safety, or scientific model
quality is made by this audit.
