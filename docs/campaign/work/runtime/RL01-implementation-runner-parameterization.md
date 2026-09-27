# RL-01 fixed-ID runner preparation

Status: CPU implementation complete. The runner is ready for a lead-owned
admitted-data and merged-SFT build, but it has not loaded a model or launched
training. The implementation is in the execution worktree
`/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`.

The new [campaign_rl_data.py](../../../../tuesday-execution-20260912/experiments/training/campaign_rl_data.py)
joins three explicit, byte-hashed inputs: canonical PRM-03 train rows, a
pre-edit `PromptContext` sidecar, and an ordered selected-ID document. It
requires `admission_status=admitted`, rejects candidate-only envelopes, checks
the row with `validate_training_row`, verifies that `render_prompt(context)` is
the stored prompt text, and binds every context to its URI, document version,
content digest, and source-snapshot digest. It preserves selected order and
emits a nested `prompt={text,ids}` envelope plus out-of-band reward metadata.
There is no directory discovery, admission call, target-derived prompt, or
dev/final fallback.

[campaign_rl_train.py](../../../../tuesday-execution-20260912/experiments/training/campaign_rl_train.py)
provides the fixed-ID trainer adapter, reward, sampler, recipe validation, and
framework-free preflight. `CampaignGRPOTrainer` is a dynamic subclass of the
pinned TRL `GRPOTrainer`; framework imports occur only in the live config or
trainer builder. The builder accepts an already loaded model/tokenizer and
requires the lead to pass the exact admitted merged-SFT manifest and merged
weight hashes. The Midtrain MiniCPM artifact is retained as a profile input and
cannot satisfy that parent gate.

TRL 0.24.0 rejects a configuration that sets both `generation_batch_size` and
`steps_per_generation`. The resolver therefore sets `generation_batch_size`
to the declared completion rows per optimizer update, sets the per-device
batch to that same value, leaves `steps_per_generation` unset for TRL to derive
`1`, and keeps gradient accumulation and `num_iterations` at `1`. The default
is 32 completion rows per update: G=4 gives eight prompt groups and G=2 gives
16 groups. A lead memory arm can set `rollout_rows_per_update=G` for one group
per update. This preserves the legacy 32-row update exposure by default while
changing its microbatch shape; memory and throughput must be measured before
choosing the live arm. The ordered sampler emits contiguous G copies of each
selected prompt, drops incomplete prompt chunks, never shuffles, and records
the seed, epoch, stream cursor, selected-ID identity, and batch geometry for a
full-resume checkpoint.

The generation override consumes `input_ids[:target_start]` directly. It does
not call tokenizer encoding, apply a template, or truncate a prompt. It left
pads the stored IDs with PAD/EOS 1, calls `generate` with `max_new_tokens=192`,
the identity-bearing sampling policy (`do_sample=true`, temperature 0.7,
top-p 0.95, repetition penalty 1.0), and native EOG IDs `[1,130073]`, then
strips the exact padded prompt and retains the first terminal. The injected
generation guard and optional post-generation restoration are the seams for
the lead's verified Unsloth `for_training` transition. The adapter does not
invent or mutate private Unsloth flags.

`CampaignPRM03Reward` decodes raw IDs before the final token with special-token
skipping and cleanup disabled, parses the exact PRM-03 wire output, maps
`no_op` to the captured old region, `delete` to an empty region, and
`replace` to the labelled body lines, then computes the legacy exact + 0.2 ×
line-F1 score. Missing/duplicate/premature EOS, terminal 130073, native
CONTROL before the canonical terminal, out-of-range IDs, cap overflow, and
parser failures receive zero. It records protocol, terminal, control, cap,
semantic, family, package, row ID, and bounded output-hash fields. The object
exposes the callable name required by TRL 0.24.0.

The live builder wires `campaign_control.control_callback` and
`campaign_checkpoint.checkpoint_callback`. It requires full-state save cadence,
non-model-only checkpoints, deterministic data skip, a fresh telemetry path,
an explicit deadline/reserve, and nominated development evaluation steps. The
checkpoint closure obtains the sampler state only after trainer construction;
the callback can therefore seal adapter, optimizer, scheduler, RNG, trainer,
campaign-state, and inventory files through the existing shared implementation.
The builder constructs but does not call `train()`.

Validation used no model, weights, CUDA, server, cloud, registry, or final-set
access. The owned test file contains protocol reward fixtures for replacement,
whitespace mismatch, `[NO_EDIT]`, unchanged full-copy no-op, deletion,
empty-range no-op, invalid terminals, CONTROL, and duplicate markers. It also
checks stale context snapshots, selected-order and nested Dataset metadata,
G=2/G=4 geometry, actual pinned TRL config post-init, the callable reward
name, deterministic sampler state, and the actual fixed-ID `generate` call with
a fake CPU model whose tokenizer is never invoked.

The pinned venv test cannot construct the complete `GRPOTrainer` subclass
because this environment lacks the optional `mergekit` package. Its actual
`GRPOConfig` test and `trl.models.unwrap_model_for_generation` fixed-ID test
pass. A real interruption/resume Trainer test therefore remains a lead-owned
live gate; the callback and sampler state are wired, but no file-presence or
CPU fake result is presented as faithful resume evidence.

Remaining gates are the lead's admitted train registry and fresh context
sidecar, the future merged-SFT parent identity, post-construction tokenizer
contract, PEFT/Unsloth generation-to-training transition (including the known
`_flag_for_generation` failure), actual r16/a16 attachment and parameter
counts, G=2 versus G=4 memory/throughput, and a full checkpoint interruption
and resume comparison. Candidate-only DAT-04 rows and the Midtrain base do not
authorize a training launch.
