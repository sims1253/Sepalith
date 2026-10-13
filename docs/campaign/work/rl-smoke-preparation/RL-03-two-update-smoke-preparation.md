# RL-03 two-update smoke preparation

This packet prepares a bounded mechanism smoke for the PRM-03 RL entry. It
does not choose a parent, admit the RL-02 pool, load a model, reserve CUDA,
start a server, start a launcher, or create campaign state. Recipe generation
is deliberately closed until root supplies both an accepted merged-SFT parent
manifest and a positive RL-02 admission receipt.

The generator is
`prepare_rl_two_update_smoke.py`. Its `--inspect-only` phase has already read
the frozen DEV panels with the canonical tokenizer. It copies the exact
14-case bytes when the longest case is present, and appends that case only if a
future 14-case input omits it. The current result is therefore a 14-case
panel, not an artificial 15-case panel:

- 14-case source: `/mnt/e/sepalith/campaign-20260915/data-work/SFT-smoke-dev-14-v1.jsonl`
  SHA256 `9192bd278c0342ce94d029ee5b1b9964b3fce0048ed6b99e4bcc0d97aaf06b59`.
- 75-case source: `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl`
  SHA256 `b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`.
- The maximal 75-case prompt is
  `dat07-existing-719cd49683667d0fb86fb2fa`, with 2,619 prompt tokens and
  2,631 prompt-plus-target tokens. It is already one of the 14 cases. The
  target fields are used only for this audit count; they never enter the
  generated request or the RL training prompt.
- The exact inspected panel and audit are under
  `generated-inspect-v2/development-panel-small.jsonl` and
  `generated-inspect-v2/development-panel-audit.json`.

The smoke keeps the accepted v4 source schedule intact. The schedule is
`docs/campaign/work/rl-pool/source-row-draw-sequence-v4.json`, SHA256
`892991f89d063544d85902eba04c7607f3867897749d3eee4f4e2ac6f407ee48`, with
ordered draw sequence SHA256
`d44a076d0b850fd607e7c613caa096d04ed352292ceeeb2e17f736dc20a089f2` and
24,000 source draws. The selected train order is the admitted RL-02 lead
order, 8,440 unique IDs, SHA256
`24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d`.
The prefix audit is in
`generated-inspect-v2/source-prefix-audit.json`; it records the first 16
source draws as two groups of eight. It does not write a shortened schedule,
change quotas, or change the identity-bound source draw count.

The fixed smoke geometry is G=4, 32 completion rows per optimizer update,
per-device completion batch 8, gradient accumulation 4, one generation
buffer per update, and `num_iterations=1`. Thus each update consumes 8 source
groups and 128 sampler rows (`32 * 4`), while the first two updates audit 16
source draws, 64 completion rows, and 256 sampler rows. The full schedule
remains the sampler input, so the checkpoint cursor must be 8 after step 1 and
16 after step 2. The main schedule ceiling remains 3,000 updates; this smoke
does not make a 1,000-update disposable treatment.

## Recipe contract

When root inputs are supplied, the generator emits three JSON recipes with one
identical seven-field identity (parent, tokenizer, renderer, data, source,
policy, schedule):

1. `rl-two-update-uninterrupted.recipe.json`: `max_steps=2`,
   `full_save_steps=1`, `light_save_steps=1`, explicit
   `evaluation_steps=[1,2]`, and no decision stop.
2. `rl-two-update-split-first.recipe.json`: the same identity and geometry,
   with `decision_steps=[1]`. The expected entry terminal status is
   `lead_decision` at step 1, followed by a full checkpoint at
   `archive/full/checkpoint-1`.
3. `rl-two-update-split-resume.recipe.json`: the same identity, a fresh output,
   archive, and telemetry path, with `resume_from` set to the split-first
   `archive/full/checkpoint-1` path. It has no decision stop and must finish at
   step 2 with `schedule_complete`.

Every recipe binds `model_load_max_seq_length=4096` while retaining the RL
caps prompt 2,048, completion 192, and managed context 2,240. The evaluator
wrapper is the existing `campaign_eval:development_evaluator` with
`renderer_id=zeta2-prm03-v1`, `parameters.max_sequence_tokens=4096`, the exact
small-panel path and SHA, its exact case ID list, and
`development_max_new_tokens=512`. The generated recipes also carry the
75-case path and SHA as `main_development_panel.required_for_main=true`.
The 14/15-case panel is only for the two-update mechanism smoke; the long RL
recipe must use all 75 DEV cases and must retain its full denominator.

The policy identity records rank 16, alpha 16, all seven target modules,
294 adapter attachments, 25,116,672 trainable parameters, BNPO/group/beta=0,
and sampling `do_sample=true`, temperature 0.7, top-p 0.95, repetition
penalty 1.0. The tokenizer and renderer identities retain native EOG IDs
`[1,130073]`, canonical EOS/PAD 1, manual BOS 0, and the exact PRM-03
tokenization policy. Recipe generation fails if the inspected tokenizer bytes
differ from the accepted parent tokenizer bytes.

The root admission receipt has an intentionally narrow proposed contract. It
must be a fresh, hashed JSON receipt with `launch_admitted=true` or an explicit
admitted status (`admitted`, `accepted`, `rl02_admitted`, `source_admitted`,
or `launch_admitted`), and it must recursively bind the rows, sidecar,
selected-ID, source-schedule, and source-sequence SHA256 values supplied to
the generator. The current
`RL-02-v4-pool-lead-review.json` has `launch_admitted=false` and candidate
status, so the generator rejects it. No candidate receipt is promoted by
changing a field locally.

The parent argument is likewise only a wrapper of the form
`{"path": "/absolute/accepted-parent.json", "sha256": "<64 lowercase hex>"}`.
The current entry helper verifies `status=accepted`, `kind=merged_sft`, the
complete sorted weight inventory, every weight byte/hash, the aggregate
weight identity, tokenizer files and IDs, `config.json`, and
`generation_config.json`. The generator does not substitute the Midtrain
model or a disposable theta0/profile fixture for this argument.

## Preparation command

Run this after the separate theta0 gate and RL-02 admission have produced the
root-owned inputs. Choose a new output directory under the PLAN worktree and
a new run root on native local ext4. The generator does not create the run
root or any output/archive/telemetry path used by training.

```bash
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
EXEC=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
GEN="$PLAN/docs/campaign/work/rl-smoke-preparation/prepare_rl_two_update_smoke.py"
OUT="$PLAN/docs/campaign/work/rl-smoke-preparation/generated-rl03-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT=/native/ext4/rl03-two-update-smoke-20260912T<UTC>
PARENT=/absolute/accepted/merged-sft-parent.json
ADMISSION=/absolute/root/receipts/RL-02-admission.json

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" "$GEN" \
  --execution-root "$EXEC" \
  --output-dir "$OUT" \
  --parent-manifest "$PARENT" \
  --parent-manifest-sha256 "<PARENT_SHA256>" \
  --rl02-admission-receipt "$ADMISSION" \
  --rl02-admission-receipt-sha256 "<RL02_ADMISSION_SHA256>" \
  --run-root "$RUN_ROOT" \
  --deadline "<UTC_DEADLINE>" \
  --protocol-sha256 5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156 \
  --trainer-sha256 77bcc7fbe4ef6b6a8ba497d22f6948215f3849b23806dfa6a64f6d3a7b1e1e5f \
  --tokenizer-revision 8dc5f6055b90fe4b9422340810b270b9569f37f3
```

The command must print `recipes_emitted_pending_live_smoke` and produce the
three recipe files plus `development-panel-small.jsonl`,
`development-panel-audit.json`, `source-prefix-audit.json`, and
`preparation-result.json`. The result records one identity digest and all
root input identities. A normal run verifies data with
`load_training_records(..., admission_status="admitted")` and calls the
existing `verify_merged_parent_manifest`; it never imports a model framework.

Before launching, root should run CPU preflight on the uninterrupted and
split-first recipes:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" "$EXEC/experiments/training/campaign_rl_entry.py" \
  "$OUT/rl-two-update-uninterrupted.recipe.json" --preflight \
  --receipt "$OUT/uninterrupted-preflight.entry-result.json"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" "$EXEC/experiments/training/campaign_rl_entry.py" \
  "$OUT/rl-two-update-split-first.recipe.json" --preflight \
  --receipt "$OUT/split-first-preflight.entry-result.json"
```

These calls require the chosen run-root paths to be fresh native ext4 paths.
Preflight must report no framework import, 8,440 admitted rows, the G=4
geometry, the complete source schedule, the 4K evaluator wrapper, and the
accepted parent. After split-first completes, preflight the resume recipe
against the sealed checkpoint; a resume preflight before checkpoint-1 exists
must remain pending and must not be bypassed.

## Root-only live sequence

The separate theta0 gate must pass before any of these launches. The combined
three-attempt live smoke has a maximum wall budget of 1,800 seconds per
attempt, with a 60-second termination grace and a 120-second checkpoint
reserve. Root runs each attempt through the existing
`campaign_rl_launch.py`, which performs GNU `timeout` supervision and writes an
immutable supervision receipt. The entry receives that receipt through the
launcher environment and writes a distinct `*.entry-result.json`; a failed
entry must leave both receipts and its `output/failure.json` available.

```bash
LAUNCH="$EXEC/experiments/training/campaign_rl_launch.py"
"$PY" "$LAUNCH" "$OUT/rl-two-update-uninterrupted.recipe.json" \
  --receipt "$RUN_ROOT/uninterrupted-supervision.json"
"$PY" "$LAUNCH" "$OUT/rl-two-update-split-first.recipe.json" \
  --receipt "$RUN_ROOT/split-first-supervision.json"
"$PY" "$LAUNCH" "$OUT/rl-two-update-split-resume.recipe.json" \
  --receipt "$RUN_ROOT/split-resume-supervision.json"
```

Root should inspect the first two attempts before the resume launch. The
uninterrupted attempt must reach full checkpoints 1 and 2, perform evaluator
reads at both steps, record finite policy loss and nonzero finite adapter
gradients, and terminate `schedule_complete` at step 2. Split-first must stop
only at the step-1 control boundary with `lead_decision`, write a complete
full checkpoint, then the resume attempt must reach step 2 with
`schedule_complete`. A deadline stop, missing checkpoint, missing evaluator,
nonfinite gradient/reward, or altered tokenizer contract blocks the smoke.

Expected per-attempt files include:

```text
<run-root>/<attempt>/output/admitted-recipe.json
<run-root>/<attempt>/output/entry-preflight.json
<run-root>/<attempt>/output/parent-audit.json
<run-root>/<attempt>/output/cuda-occupancy.json
<run-root>/<attempt>/output/load-audit.json
<run-root>/<attempt>/output/post-trainer-contract.json
<run-root>/<attempt>/output/train-begin-tokenizer-contract.json
<run-root>/<attempt>/output/terminal-tokenizer-contract.json
<run-root>/<attempt>/output/terminal.json
<run-root>/<attempt>/output/generation-records.jsonl
<run-root>/<attempt>/output/gradient-records.jsonl
<run-root>/<attempt>/output/reward-records.jsonl
<run-root>/<attempt>/archive/full/checkpoint-1/
<run-root>/<attempt>/archive/full/checkpoint-2/
<run-root>/<attempt>/archive/adapters/checkpoint-1/
<run-root>/<attempt>/archive/adapters/checkpoint-2/
<run-root>/<attempt>/archive/evaluations/step-1.json
<run-root>/<attempt>/archive/evaluations/step-2.json
<run-root>/<attempt>-supervision.json
<run-root>/<attempt>-supervision.entry-result.json
```

## Resume comparison and acceptance

Compare the uninterrupted and split-resume attempts using their sealed
`campaign-manifest.json` identities and byte inventories. The following facts
must agree before the smoke is accepted:

- Parent, tokenizer, renderer, data, source, policy, and schedule identities,
  including the complete 24,000-draw schedule, selected-ID order, row
  identities, model-load capacity, and admission receipt hash.
- Generation record ordering and source groups. The first update consumes
  source-prefix positions 0–7 and the second 8–15; each has 32 completions.
  At the step-1 full checkpoint the sampler state must report
  `consumed_rows=128`, `consumed_prompt_copies=32`, and
  `source_draw_cursor=8`. At step 2 those values are 256, 64, and 16.
  `selected_id_index`, source schedule hash, and sampler geometry must match
  the source sequence rather than only the row count.
- Full checkpoint inventories contain adapter weights/config plus optimizer,
  scheduler, RNG, trainer state, and campaign state. Optimizer and scheduler
  steps, trainer `global_step`, update cursor, and complete optimizer-boundary
  metadata must agree. No resume may start from an adapter-only archive or a
  made-up parent/checkpoint.
- RNG and sampler reconstruction reproduce the next source groups. Compare
  the saved RNG file hashes and the resumed step-2 generation records with the
  uninterrupted step-2 records. Any difference is a resume parity failure
  even when both attempts reach step 2.
- Each step-1 and step-2 evaluation uses the exact small smoke denominator,
  package/family IDs, prompt/target NLL token denominators, protocol-valid and
  exact counts, strict no-op/false-suggestion/edit counts, cap and native-EOS
  counters. Evaluation must restore model mode, tokenizer fields, and RNG.
  The long main run separately uses the full 75-case DEV panel.
- Generation telemetry records native EOG and CONTROL handling, cap hits,
  prompt/completion lengths, and the exact stored prompt IDs. Gradient
  telemetry is finite and nonzero for the LoRA tensors only. Reward telemetry
  records family/length/cap/zero-variance groups and must preserve zero-
  variance groups rather than turning them into an apparent learning signal.
  Adapter attachment count is 294 and trainable parameter count is
  25,116,672.
- The supervisor receipt hash and bytes are unchanged after entry success or
  failure. The entry result is a distinct sibling artifact. This verifies the
  deadline/argv/PID evidence was not replaced by entry output.

The installed TRL source review admitted the response-logit direction
(`labels=None`, response-logit selection before FP32 conversion) as source
evidence only. The live smoke still must measure actual policy-log-probability
memory and dtype on the chosen parent; a full
`[8,~2240,130560]` FP32 conversion remains disallowed. The 4K DEV loader,
single CUDA occupancy, one-time generation/training transition, native EOS/PAD
alignment, finite gradients, and the interruption/resume comparison are all
root-owned live gates.
