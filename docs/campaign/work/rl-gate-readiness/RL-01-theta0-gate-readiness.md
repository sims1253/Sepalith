# RL-01 theta0 gate readiness

Packet status at 2026-09-12T15:07:51Z: `cpu_preparation_verified_live_gate_pending`.
This packet prepares the smallest deterministic theta0 diagnostic and records the
interfaces needed for the first real RL update. It did not load a model, import
CUDA, start a server, connect over SSH, or run a GRPO update. The live gate,
thresholds, data admission, parent choice, and resource lease remain lead-owned.

## Pinned inputs and contract

The fixture in
`docs/campaign/work/rl-gate-readiness/RL-01-theta0-gate-fixture.json` reads the
existing RL-02 candidate inputs without changing them:

| Input | Path | SHA256 | Count or role |
| --- | --- | --- | --- |
| train rows | `/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/eligible-train-rows.jsonl` | `e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602` | 8440 |
| context sidecar | `/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/context-sidecar.jsonl` | `6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f` | 8440 |
| selected train order | `/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/selected-train-ids-lead-order-v1.json` | `24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d` | ordered unique IDs |
| source draw schedule | `docs/campaign/work/rl-pool/source-row-draw-sequence-v4.json` | `892991f89d063544d85902eba04c7607f3867897749d3eee4f4e2ac6f407ee48` | candidate-only, 24000 draws |

The ordered selected-ID digest is
`7e994a3e74a642c149908b0737dfd2befed055e336f7b94400ee9773a6a88d2d`; the
selected row-identity digest is
`d3187195f0ebe401b69df244bbbff0f2a28fb0500185428701f933e31e8eab60`.
The source-draw sequence digest is
`d44a076d0b850fd607e7c613caa096d04ed352292ceeeb2e17f736dc20a089f2`.
The schedule is `frozen_candidate_only`, seed `3407`, G4, 8 prompt groups per
update, 32 completions per update, four-buffer reuse, and 24,000 source draws
over 8,365 distinct selected rows. It is a recipe binding only after the lead
admits RL-02.

The renderer and tokenization contract is PRM-03
`zeta2-prm03-v1`, using protocol source
`packages/sepalith/src/sepalith/campaign_protocol.py` at SHA256
`5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156` in the
execution checkout. The pinned tokenizer revision is
`8dc5f6055b90fe4b9422340810b270b9569f37f3`, tokenizer JSON SHA256
`3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, and
config SHA256
`e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`.
Prompt IDs are encoded without automatic special tokens, then receive one
manual BOS ID `0`; response termination accepts native EOG IDs `[1, 130073]`
but the canonical protocol response ends with EOS ID `1`. The RL geometry is
prompt `<=2048`, response cap `192`, and managed context `2240`. The final LF
is part of the prompt boundary.

The locked DEV evaluator input is separate and remains a DEV-only readout:
`/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl`,
75 cases, SHA256
`b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`. The
root-supplied DEV contract is all 75 predeclared IDs, cap `512`, and
`parameters.max_sequence_tokens=4096`; the observed maximum prompt is about
2619 tokens (maximum full case about 2631). The current live loader passes
`max_seq_length=2240`, so a real DEV readout needs the reviewed 4096 loaded
context option and an explicit runtime measurement. It must not silently shrink
the panel or alter the rollout limits of 2048 + 192.

## Deterministic theta0 fixture

Eight genuine TRAIN rows were selected from the accepted RL-02 input order for
coverage. Four have an observed history event and a structured edit, two are
strict no-op controls, and two are local-only diagnostics. The full source and
event identities, prompt/context hashes, and token-ID hashes are in the JSON
fixture.

| Row ID | Fixture class | Package / family | Path | Prompt IDs with BOS |
| --- | --- | --- | --- | ---: |
| `6e665944bef9a0f559f2b3e0` | history-supported structured edit | survminer / pipe_rewrite | `R/ggsurvplot_facet.R` | 1983 |
| `54932d7a5580bf14c1b25dd2` | history-supported structured edit | vcdExtra / format_propagation | `R/mosaic3d.R` | 1999 |
| `ac3d0958387057313cee640f` | history-supported structured edit | hutils / rename_propagation | `R/select_grep.R` | 1774 |
| `bcae0a35367d20da30a661c6` | history-supported structured edit | posterior / na_rm_propagation | `R/nested_rhat.R` | 1364 |
| `1337ddd1fa17179c7c2f228a` | strict no-op control | pslr / no_op | `R/cache.R` | 473 |
| `119118938fc7b2d234cd8f00` | strict no-op control | mermboost / no_op | `R/asFamily.R` | 127 |
| `00897dc804fa92c8cbe75998` | local-only diagnostic | ivreg2r / finish_block | `R/broom-methods.R` | 138 |
| `026b43a6a32414c9541589b8` | local-only diagnostic | tabxplor / roxygen_drafting | `R/fmt_class.R` | 477 |

The structured rows produce three context variants each:

* `evidence_supported` preserves the source-derived history.
* `missing_history` removes only `history`.
* `irrelevant_history` replaces `history` with one observed event from a
  deterministic donor cycle: `6e -> 549`, `549 -> ac3`, `ac3 -> bca`, and
  `bca -> 6e`.

The two no-op rows produce `strict_noop_control` variants. The two local-only
rows produce `local_only_control` variants with no history, references,
diagnostics, retrieval, or scope evidence. The result is 16 context variants
(12 structured-history variants plus two no-op and two local-only controls).
All are rendered from `PromptContext.from_mapping` and encoded with the pinned
tokenizer. The measured variant band is 127 to 2021 prompt IDs with BOS; every
variant satisfies `prompt <= 2048` and `prompt + 192 <= 2240`, and every prompt
ends in the required LF.

The variants are diagnostic contexts, not new RL training rows. No variant
payload contains target text, target body, target operation, replacement
region, or reward fields. The observed history is source-derived. A
missing-history or irrelevant-history result must not be converted into a
synthetic no-op label; counterfactuals remain diagnostic until the lead chooses
the scientific gate.

## Verification performed

From the PLAN checkout, the following command rechecked both input file hashes,
the selected-ID digest, all source identities, all 16 context derivations,
renderer hashes, tokenizer IDs, LF boundaries, and capacity. It imports only
the pinned protocol and tokenizer; it does not import a model or CUDA:

```bash
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHONPATH=packages/sepalith/src:experiments/training \
  /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python \
  docs/campaign/work/rl-gate-readiness/verify_theta0_fixture.py \
  --fixture docs/campaign/work/rl-gate-readiness/RL-01-theta0-gate-fixture.json \
  --execution-root /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912
```

Observed result:

```json
{"base_rows": 8, "model_or_cuda_loaded": false, "prompt_tokens_with_bos": {"max": 2021, "min": 127}, "selected_ids": 8440, "status": "verified", "target_or_reward_context_keys": false, "variants": 16}
```

The existing full-input loader was also called with
`admission_status="admitted"` as a validation argument. It returned 8,440
records with the digests above, first selected ID
`6e665944bef9a0f559f2b3e0`, and last selected ID
`01669b64a0334ac2978d52c2`. This call does not admit the candidate envelope;
the loader itself documents that admission is an explicit recipe assertion.

## Existing interfaces and missing gate plumbing

The pinned execution sources were inspected at these hashes:

| Source | SHA256 |
| --- | --- |
| `experiments/training/campaign_rl_entry.py` | `793a282570be4c90f1804ab43bd7320519fbc6f91f5eaaa4b87ce41bb99096ee` |
| `experiments/training/campaign_rl_launch.py` | `fa245326052bfae2997d91d80bbf8f45c055d35ffcd0b4b2b7d4440045ffb36a` |
| `experiments/training/campaign_rl_data.py` | `86ebc3fe80a976e3c35a29057234b341c9e06060135a87382bbf8c915b1db91b` |
| `experiments/training/campaign_rl_train.py` | `794348ccf5976133fec76ca8bf706fcabe0f7f4fde7fc2325bff226e748f87a0` |
| `experiments/training/campaign_eval.py` | `7064385d63e3900b87241da75525897acb6b55ffcd9e1a023689c9addfbaacfc9f` |
| `experiments/training/campaign_checkpoint.py` | `64202f4e5b88c94a74794442be3a4955f07d195079c9d64bbcff2850dbadd0ba` |

Its verified source behavior is clear: it accepts only
the predeclared DEV panel, builds target-bearing DEV rows at evaluation time,
generates deterministically with native EOG `[1,130073]`, writes per-case
records, and writes a terminal `evaluations/step-N.json`. It has no theta0
context-variant evaluator or cache/fresh ablation hook.

The existing RL entry path can run the admitted base dataset only after all of
these are supplied:

1. a lead-admitted RL-02 envelope with `admission_status="admitted"`;
2. a verified BF16 `merged_sft` parent manifest and aggregate weight identity;
3. a recipe carrying the seven identity fields `parent`, `tokenizer`,
   `renderer`, `data`, `source`, `policy`, and `schedule`;
4. fresh native-ext4 `output_dir`, `archive_root`, and `telemetry_path`;
5. a single-GPU lease and the live tokenizer/PEFT transition proof; and
6. a DEV evaluator factory or an explicit, reviewed pending-evaluation policy.

It cannot consume the 16 variants as RL rows: the data loader requires the
exact selected row order, exact sidecar artifact hash, row identities, source
identity, and the original row prompt. Replacing history changes the prompt
and sidecar identity, while the diagnostic variants intentionally carry no
target-bearing row. Keep the fixture outside the RL recipe identity.

The minimum root-owned theta0 evaluator must accept the fixture, a loaded
parent identity, and the pinned tokenizer/renderer, then for each variant run a
bounded direct generation in both fresh and cache-reuse arms when the runtime
can expose those controls. It must record the arm, prompt/context/weight
digests, returned token IDs, the first native terminal (canonical EOS or EOG),
inclusive cap status at 192, protocol/parser status, exact-region/no-op result
where the base source label is authoritative, policy log-probability status,
and synchronized wall time and peak resource readings. A cache arm that cannot
prove its state must be marked unsupported. The evaluator must write a
diagnostic receipt separate from RL `identity.data`, with complete/failed/
timeout terminal status and per-case partial evidence; it must not append to the
training rows or promote a counterfactual.

## Concrete root sequence

The following is the smallest real sequence. It is a recipe for root execution,
not a claim that this worker ran it.

### 1. Preflight the admitted parent and data

Root first writes a fresh recipe with the exact parent manifest wrapper and
executes the framework-free preflight:

```bash
EXEC=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
cd "$EXEC/experiments/training"
PYTHONPATH="$EXEC/packages/sepalith/src:$EXEC/experiments/training" \
  "$PYTHON" campaign_rl_entry.py /ABS/rl-theta0-smoke.recipe.json \
  --preflight --receipt /ABS/rl-theta0-smoke.preflight.entry-result.json
```

The parent wrapper is exactly `{ "path": "/ABS/merged-sft-parent.json",
"sha256": "<sha256-of-parent-manifest>" }`. The nested parent identity must
be `{ "kind": "merged_sft", "manifest_sha256": "...",
"merged_weights_sha256": "...", "base_model_revision": "...",
"sft_identity": { ... } }` and must match the verified manifest. Raw Midtrain,
Q8, or a serving checkpoint cannot satisfy this RL parent gate.

The `identity` object must contain these seven top-level objects exactly:
`parent`, `tokenizer`, `renderer`, `data`, `source`, `policy`, `schedule`.
The tokenizer object binds revision, tokenizer/config hashes, vocab `130560`,
BOS `0`, EOS/PAD `1`, and native EOG `[1,130073]`. The renderer binds
`zeta2-prm03-v1`, the tokenization policy, terminal `>>>>>>> UPDATED`, and
`[NO_EDIT]`. The data object binds rows/context/selected-ID hashes, the ordered
IDs, row identities, row count 8440, split `train`, sidecar artifact hash equal
to context hash, and explicit admission. The policy object binds LoRA r16/a16,
the seven target modules, 294 attachments, 25,116,672 trainable parameters,
prompt/completion caps 2048/192, G4 geometry, BNPO/group/beta0, and the sampled
generation policy. The schedule binds seed 3407, sampler
`campaign-repeat-manifest-order-v1`, one generation per update, and, once
admitted, the v4 source-draw hashes and geometry.

### 2. Run the theta0 diagnostic

Root must add or nominate a small evaluator module because no existing entry or
evaluator accepts context-only variants. The required command shape is:

```bash
PYTHONPATH="$EXEC/packages/sepalith/src:$EXEC/experiments/training" \
  "$PYTHON" /ABS/campaign_theta0_gate.py \
  --fixture "$PLAN/docs/campaign/work/rl-gate-readiness/RL-01-theta0-gate-fixture.json" \
  --parent-manifest /ABS/merged-sft-parent.json \
  --tokenizer /mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain \
  --context-length 4096 --prompt-cap 2048 --completion-cap 192 \
  --cache-arms fresh,reuse --deadline-seconds 1200 \
  --receipt /ABS/rl-theta0-gate.json
```

The module name is a required root implementation seam, not an existing file.
The live acceptance decision remains the lead's. The minimum terminal artifact
is a complete receipt with 16 variant records per supported arm, model/parent
identity, tokenizer/renderer hashes, actual returned IDs, EOS/EOG/cap and
protocol status, policy-logprob and memory fields, and a reason for any omitted
arm. Scientific thresholds are deliberately absent from this packet.

### 3. Run one actual G4 GRPO update

After the theta0 gate and the lead's data/parent admission, make a fresh
one-step recipe. Keep the training rollout caps at 2048/192, but configure the
loaded model context to 4096 so the same attempt can run the locked 75-case DEV
evaluator without shrinking it. The identity remains G4: candidate count 4,
32 rollout completions, microbatch 8, accumulation 4, eight prompt groups per
update, `steps_per_generation=4`, `num_iterations=1`, and v4 source-draw
`buffer_reuse=4`.

Use `max_steps=1`, `full_save_steps=1`, `light_save_steps=1`,
`evaluation_steps=[1]`, and a complete DEV evaluator binding. The recipe must
set `parameters.max_sequence_tokens=4096` for the evaluator and pass the
reviewed loader context-length option; it must not change the RL prompt or
completion caps. The sampled policy is the pinned
`do_sample=true, temperature=0.7, top_p=0.95, repetition_penalty=1.0`.

Launch only through the independent supervisor:

```bash
cd "$EXEC/experiments/training"
PYTHONPATH="$EXEC/packages/sepalith/src:$EXEC/experiments/training" \
  "$PYTHON" campaign_rl_launch.py /ABS/rl-theta0-smoke.recipe.json \
  --receipt /ABS/rl-theta0-smoke.supervision.json
```

The supervisor receipt, entry preflight, parent/load/tokenizer/adapter audits,
train-begin and terminal tokenizer contracts, reward records, telemetry, full
checkpoint manifest, DEV per-case/terminal evaluation, and entry-result receipt
are all required. A trainer exit or a finite loss alone is not an acceptance
signal. The one-step terminal must have a sealed full checkpoint with
`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`,
`campaign-state.json`, and deterministic sampler metadata.

### 4. Prove full-state interruption and resume

Use a second fresh recipe with the same identity, parent, data, policy, and
source-draw binding, `max_steps=2`, `full_save_steps=1`, `light_save_steps=1`,
and `evaluation_steps=[1,2]`. Stop the first supervised attempt only after
the full `checkpoint-1` has been sealed and its DEV step-1 readout is durable.
The second attempt sets `resume_from` to that full checkpoint and runs through
the same supervisor to step 2. Both attempts must use distinct fresh output,
archive, telemetry, supervision, and entry-result paths.

Compare the interrupted/resumed step-2 checkpoint and terminal evidence with an
uninterrupted step-2 control. The comparison must cover adapter bytes, optimizer,
scheduler, Python/NumPy/CUDA RNG state, trainer state, deterministic sampler
position, v4 source-draw position, tokenizer/renderer/parent/data identity,
actual update count, and DEV step-2 output. `verify_checkpoint` proves sealed
bytes and identity; it does not prove faithful resume by itself.

### 5. Review DEV and release the attempt

The evaluator must process exactly the 75 predeclared DEV IDs with cap 512 and
write `evaluations/cases-step-N.json` followed by
`evaluations/step-N.json`. The loaded context must be measured at 4096 because
the locked panel reaches approximately 2619 prompt tokens and approximately
2631 full tokens. No panel shrink, final-set read, or promotion follows from
the evaluator's process exit. Root records the lead's scientific decision and
stops the process at the campaign deadline or earlier if any identity,
protocol, memory, checkpoint, or resume condition fails.

## Time budget and unresolved gates

For the requested under-3600-second gate, a practical wall allocation is 0–300
seconds for parent/data preflight, 300–1500 for the bounded theta0 diagnostic,
1500–2400 for one G4 update and full save, 2400–3300 for interruption/resume
and DEV evidence, and 3300–3600 for receipt/hash review and shutdown reserve.
These are operational bounds, not quality thresholds. The later RL attempt uses
the campaign's 26-hour ceiling, light checkpoints every 100 steps, full
checkpoints every 500 steps, and the root-controlled hard stop; its actual rate
and memory must be measured from the one-step gate.

Remaining blockers are explicit:

* RL-02 remains candidate-only; this packet does not admit it.
* No accepted merged-SFT BF16 parent manifest is available in this worker.
* No live MiniCPM load, adapter attachment, GRPO generation, policy logprob,
  optimizer update, VRAM measurement, or full-resume comparison was run.
* The existing DEV evaluator cannot consume context-only theta0 variants and
  has no cache/fresh diagnostic interface.
* The current live loader uses context length 2240; root must use and measure
  the reviewed 4096 loaded-context option for the 75-case DEV readout.
* No scientific theta0 thresholds are asserted here. The 8K stress band is
  unsupported and pending; training admission remains 2048 prompt + 192
  response, with RL context 2240.

The immediate root action is therefore: admit RL-02 and bind the verified BF16
merged-SFT parent, implement the context-only theta0 evaluator at the command
seam above, run its complete receipt, then execute the one-step G4 and
interruption/resume recipes through `campaign_rl_launch.py`.
