# RL-03 generation call packing proposal

The current fixed-ID adapter emits one `model.generate` call for each logical
G-sized prompt group.  For the admitted G=4, 32 completion rows per update
shape, that is eight calls for eight source prompt groups.  This preparation
adds an identity-bound `generation_groups_per_call` arm with the finite values
`1`, `2`, `4`, and `8`.  It packs adjacent complete groups into one rectangular
call while preserving the source row order, one-G reward groups, 32 completion
rows per optimizer update, batch8/acc4 training geometry, and all prompt and
completion caps.

The proposed patch is saved as
`campaign_rl_train_generation_groups_per_call.patch`.  It is generated against
the immutable f051ebb9 source trainer and changes only the isolated copy in
this preparation directory.  The recipe generator patch is saved as
`prepare_rl03_live_recipes_generation_groups_per_call.patch`; it adds the same
value to the top-level recipe and `identity.policy`.  The current live recipe
is unchanged and intentionally lacks this field.

## Observed baseline

Root's completed continuous smoke measured the original P=1 path at one
generation step: eight serial calls, 32 candidate rows, 3,037 generated
tokens, 24 canonical EOS rows, and eight cap rows.  Generation took 93.73 s;
peak allocated/reserved memory was 8.675/9.362 GB.  The smoke stopped
gracefully at step one because the 600 s attempt plus checkpoint reserve could
not fit the remaining deadline.  It had a finite nonzero gradient norm
0.19590027 with 294/588 nonzero tensors and no new driver fault.  These are
observations of P=1; they do not measure or promote another arm.

For the same eight logical groups, expected call counts are:

| arm | rows per model call | calls/update | logical groups/update | completion rows/update |
| ---: | ---: | ---: | ---: | ---: |
| P=1 | 4 | 8 | 8 | 32 |
| P=2 | 8 | 4 | 8 | 32 |
| P=4 | 16 | 2 | 8 | 32 |
| P=8 | 32 | 1 | 8 | 32 |

The call counts are arithmetic, not throughput or memory measurements.  P=2
is the first profile candidate.  P=4 is conditional on the live memory gate;
P=8 should remain a guarded candidate because an earlier actual policy
microbatch profile reached 14.983/21.550 GB allocated/reserved.  Training
batch8/acc4 remains fixed.  No claim is made about speed, quality, or safe
32-row activation memory until root measures the selected identity.

## Adapter changes

`FixedIDGRPOTrainerMixin.configure_campaign_runtime` accepts and validates the
new value.  The validator admits only `(1, 2, 4, 8)`, rejects missing, bool,
zero, arbitrary, and out-of-buffer values, and is called from both recipe
validation and the live builder.  The top-level recipe value must equal
`identity.policy.generation_groups_per_call`.

`_generate_prompt_batch` validates each inner group as exactly G repeated
stored ID arrays, flattens only adjacent groups, and calls the existing
unpinned seam with the existing EOS=1 left padding and explicit attention
mask.  It checks the returned row count against the flattened call and keeps
the exact padded prompt prefix check and `trim_generated_sequence` terminal
handling.  `_generate_prompt_group` remains as a one-group compatibility
wrapper, so P=1 retains the existing fixture path.

`_generate_single_turn` partitions the unchanged prompt list into calls of
`P*G` rows.  It restores each row's logical `group_index` and
`group_row_index`, records `call_index`, and stores call count and P in the
last-generation and durable telemetry geometry.  The reward/accounting input
order is still the original flattened order.  The source sampler cursor and
checkpoint identity are untouched; a resumed run uses the same P identity and
starts at the same source group.  A cursor that falls inside a packed call
must start a fresh call at that logical group, never reuse a partial model
call.

The recipe generator has one explicit `GENERATION_GROUPS_PER_CALL` constant,
defaulting to P=1.  Root can prepare a separate P=2 or P=4 candidate by
changing that constant and regenerating a new recipe, which necessarily
changes the policy and recipe identity.  Existing P=1 recipe bytes are not
rewritten.

## Resume and acceptance gates

The CPU tests in `test_generation_groups_per_call.py` cover all four G=4
arms, exact flattened order, mocked output row-to-group mapping, exact suffix
reconstruction at a source-group boundary, recipe top-level/policy equality,
missing and out-of-set values, mixed prompt groups, and wrong model row
counts.  An additional isolated fake-model check against the patched source
copy exercised P=1/2/4/8 through `_generate_single_turn`: it observed
8/4/2/1 calls, 32 returned rows, eight logical groups, and matching row
indices for each arm.

Before root freezes any candidate, the live gate should verify, with the
selected new identity:

1. The generated recipe contains the same finite P in its top level and
   `identity.policy`, and its trainer source hash is the patched source hash.
2. Every call contains only adjacent complete G groups; generated rows and
   terminal trimming retain exact prompt prefixes and row order.
3. One update still consumes 8 source groups, 32 completions, and 128
   dataloader rows under G=4/batch8/acc4.  The unique selected-ID and source
   draw identities remain unchanged.
4. The live allocator stays within `.75`, the attempt/deadline gate, and root's
   current reserved-memory ceiling.  P=4/P=8 are rejected if the measured
   gate fails; P=1 remains the fallback.
5. A same-P split/resume run reproduces the exact source schedule cursor and
   row prefix/suffix.  Cross-P output equality is not an acceptance
   requirement because P is part of the new identity and sampled generation
   can have shape-dependent RNG consumption.

## Installed TRL/Unsloth evidence

The installed environment reports TRL 0.24.0 and Unsloth 2026.8.18.  TRL's
regular path (`trl/trainer/grpo_trainer.py`, lines 1271-1307) uses one
rectangular `unwrapped_model.generate` call after left padding and explicit
attention preparation.  Its paged path calls `generate_batch` (lines
1247-1263), but that path uses the model's paged attention API and is not a
drop-in replacement for stored integer-ID envelopes.  The proposal therefore
uses the existing regular seam.

Unsloth's GRPO replacement only removes TRL's tokenizer truncation kwargs and
adds multimodal fixes (`unsloth/models/rl_replacements.py`, lines 1056-1103);
it does not provide adjacent fixed-ID group packing.  `unsloth_zoo.vllm_utils`
has `generate_batches`, but it is a vLLM-only OOM splitter and is outside this
native regular-generation path.  PrefixGrouper in the RL replacements is a
training log-probability optimization, not a generation-call API.  These
source facts establish structural compatibility, not a measured throughput
benefit.  Memory and speed for P=2/P=4/P=8 remain unmeasured hypotheses.

No model, server, CUDA workload, SSH, cloud service, or Unsloth runtime
profile was launched by this preparation.
