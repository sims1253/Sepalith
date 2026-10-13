# SFT-04 independent integration review

Review date: 2026-09-12, source owner commit `a79d6de38890355ec66d7227c974140160ab314e` in `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`.

This is a read-only review. It did not load MiniCPM, initialize CUDA, start a
trainer, or change execution source. The installed reference was
`.venv-sft`: Unsloth `2026.8.18`, TRL `0.24.0`, Transformers `5.5.0`, Torch
`2.11.0`.

## Findings

### F01 — full-text training logits require an explicit launch gate (blocking)

The data path does construct full positional labels. `campaign_sft_data.py:81-95`
copies every real input position into `labels`, including the terminal ID 1,
and masks only right padding. `campaign_sft.py:269-278` passes
`completion_only_loss=False`, `packing=False`, and the custom collator. The
evaluation path also partitions the shifted loss correctly:
`campaign_eval.py:15-33` counts `target_start:` through the terminal/EOS ID.

The pinned Unsloth Llama forward has a hidden compatibility seam. At
`unsloth/models/llama.py:1464-1470`, `UNSLOTH_RETURN_LOGITS` defaults to false;
with labels present, `:1487-1513` computes fused CE and returns the
`EMPTY_LOGITS` sentinel instead of a logits tensor. TRL SFT then unconditionally
passes `outputs.logits` to entropy and token-accuracy code at
`trl/trainer/sft_trainer.py:1096-1105` and `:1136-1158`. The sentinel raises
`NotImplementedError` when its `shape` is read (`unsloth/models/_utils.py:3567-3595`).
The campaign source and its launch wrapper do not set this variable;
`campaign_launch.py:61-63` only copies the ambient environment. A pinned
source probe calling TRL entropy on the sentinel produced
`NotImplementedError:EMPTY_LOGITS`.

The no-label evaluator forward is structurally different and is suitable for
teacher-forced NLL: `campaign_eval.py:95-99` calls the model with
`use_cache=False` and no `labels`; Unsloth's no-label branch reaches the full
`lm_head` at `llama.py:1514-1517`. This is source evidence only. No actual
MiniCPM forward was run in this review. Before a 50-step launch, the lead must
prove on the real model that the trainer receives a tensor with shape
`[batch, sequence, vocab]` while full positional EOS loss remains enabled.
Setting `UNSLOTH_RETURN_LOGITS=1` is the pinned library's documented route, but
its memory and step-time cost must be measured in the lead's GPU smoke.

### F02 — generation state is restored after the evaluator, but not between cases (blocking)

At `campaign_checkpoint.py:241-244`, the callback wraps the complete evaluator
in `preserve_random_state` and the supplied
`training_configuration_guard`. The guard calls the supported
`FastLanguageModel.for_training` cleanup at `campaign_sft.py:107-116`, so the
whole-evaluator success and exception paths have an explicit cleanup route.
The CPU fixture exercises the generic guard and RNG restoration, but it does
not invoke the pinned Unsloth model.

The per-case loop at `campaign_eval.py:94-110` performs a full forward then
`model.generate` for each case without a cleanup boundary. The pinned
`unsloth_fast_generate` implementation snapshots `self.training` at
`unsloth/models/llama.py:2157-2167`, enters `for_inference`, and calls
`for_training` only when that snapshot was true (`:2243-2249`). The outer
callback deliberately called `model.eval()` first (`campaign_checkpoint.py:156-160`),
so generation starts with `self.training=False` and leaves inference markers
until the entire panel finishes. `for_inference` sets
`_flag_for_generation` and clears gradient checkpointing at `llama.py:3818-3857`;
`for_training` removes the flag and fast LoRA state at `:3861-3917`.

The next case's teacher-forced forward explicitly uses `use_cache=False`, and
the decoder's fast branch is gated by `use_cache and _flag_for_generation`
(`llama.py:800-823`), so the output path appears to avoid that one fast branch.
That does not prove equivalent LoRA/module state or complete logits after a
preceding generation. Add a real MiniCPM instrumentation check or an explicit
supported `for_training`/`for_inference` boundary between cases before calling
the evaluator complete. Do not substitute `model.train()` alone; the pinned
source says it leaves generation cleanup state behind.

### F03 — checkpoint/resume mechanisms are present; faithful SFT resume remains unproven

The admitted recipe sets `max_steps=params["max_steps"]`, effective batch 16,
`ignore_data_skip=False`, and `save_only_model=False`
(`campaign_sft.py:269-278`). It uses a `SequentialSampler`
(`campaign_sft.py:119-128`), and the callback records the deterministic schedule
and `global_step * 16` consumed draws (`:261-264`). The recipe identity requires
the complete schedule (`campaign_sft.py:131-153`), while resume preflight checks
the checkpoint identity (`:178-179`; `campaign_checkpoint.py:114-123`).

The pinned Transformers implementation loads `trainer_state.json` and checks
arguments at `trainer.py:1407-1415` and `:1540-1553`, restores optimizer and
scheduler at `:1634-1644`, skips already-consumed sequential batches and reloads
RNG at `:1674-1709`, and restores Python/NumPy/CPU/CUDA states at
`:3514-3553`. The campaign's full archive requires optimizer, scheduler, RNG,
and trainer state (`campaign_checkpoint.py:19-20, 89-111`).

Two limits remain. Transformers' argument comparison only warns for logging,
evaluation, save cadence, and batch size (`trainer_utils.py:1131-1167`), so the
campaign identity/preflight check, rather than Transformers, is what protects
the 50-step schedule. Also, `restore_callback_states_from_checkpoint` defaults
to false (`training_args.py:533-535, 1229-1234`); the campaign callbacks are
not stateful, so this is safe only because the sampler is intentionally
reconstructed from the immutable sequential draw schedule. The existing CPU
foundation receipt reports a successful plain HF interrupted-resume fixture,
but this review did not run it and it does not exercise MiniCPM/Unsloth.

Therefore F03 is an evidence gap rather than a source-level failure. The lead
still needs the actual interrupted-at-50-step smoke: archive a full checkpoint
before termination, resume with the same recipe and `max_steps=50`, show the
next row IDs equal the uninterrupted schedule, and compare optimizer,
scheduler, RNG/sampler evidence and final LoRA tensors. The command must also
show that an attempted schedule change is rejected by identity/preflight.

## Checks run

All checks were CPU/read-only and used no model load or CUDA context.

* `PYTHONPATH=experiments/training:packages/sepalith/src /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -m unittest -v experiments.training.test_campaign_eval` — 3 tests passed in 0.067 s.
* `PYTHONPATH=experiments/training:packages/sepalith/src /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -m unittest -v experiments.training.test_campaign_sft_data.SFTDataTests.test_shared_pad_eos_id_has_distinct_loss_and_gradient_behavior experiments.training.test_campaign_sft_data.SFTDataTests.test_provenance_schedule_and_token_guards_fail_closed` — 2 tests passed in 0.006 s.
* `PYTHONPATH=experiments/training:packages/sepalith/src /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -m py_compile experiments/training/campaign_sft.py experiments/training/campaign_eval.py experiments/training/campaign_checkpoint.py experiments/training/campaign_control.py experiments/training/campaign_sft_data.py` — passed.
* A source-only `SFTConfig` probe yielded `max_steps=50`, effective batch `16`, `ignore_data_skip=False`, `save_only_model=False`, `restore_callback_states_from_checkpoint=False`, `completion_only_loss=False`, and `packing=False`.
* A source-only TRL entropy probe on an `EMPTY_LOGITS`-equivalent object yielded `NotImplementedError:EMPTY_LOGITS`, matching the pinned Unsloth sentinel implementation.

No test in this packet proves real MiniCPM quality, real teacher-forced logits,
GPU timing, or faithful Unsloth interrupted resume.
