# Corrective SFT feasibility packet

Status: preparation only. This packet does not create a training input, change
the original DEV panel, alter a checkpoint, or authorize a GPU launch.

The smallest safe experiment is a fresh 200-update SFT stage initialized from
the accepted merged SFT1000 theta0. It attaches a new LoRA and resets the
optimizer, scheduler, and training RNG. The corrected training input must be a
complete replacement `train-token-rows.jsonl` produced by the data worker. A
sparse six-row patch is not accepted by the current loader and would make the
input identity ambiguous.

## Pinned parent and source

The selected parent is:

* merged model directory: `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0`
* parent manifest SHA256: `1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12`
* merged `model.safetensors` SHA256: `499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d`
* config SHA256: `f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991`
* generation config SHA256: `7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269`
* tokenizer JSON SHA256: `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`
* tokenizer config SHA256: `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`

The reviewed immutable training source is snapshot
`26a07c58eebc1769a224a193a7b163e989125e4117236cc367b940e706d7af6f`, whose
manifest SHA256 is
`e50f9d5dd6ea52048af8abcfff5d0be29c5da8a38c0091a1aed93d82d1644a3d`. The
source files used in the feasibility review are pinned as follows:

| file | SHA256 |
| --- | --- |
| `campaign_sft.py` | `671a1da97939b21962cfaedda0dc7c2c6f2f01369cb2834ef351fd26c38541f9` |
| `campaign_sft_data.py` | `31e34c5a3a662724cf50dce1ea9e6a0d248c5f44bc0f5eb6b0a5b45894433104` |
| `campaign_sft_inputs.py` | `b89457804aa66c69214bdcb5dd31b6db6fe3877103990280126e169086451185` |
| `campaign_sampling.py` | `60e4e4d60b99a4b9851b3e35194f81fb2d0a2a3c47d0cac931c2091819c03dd0` |
| `campaign_checkpoint.py` | `64202f4e5b88c94a74794442be3a4955f07d195079c9d64bbcff2850dbadd0ba` |
| `campaign_control.py` | `9a5b0614f134ecf6b804d2f0d5e37a4fecf2b1dc53589b92f17580678f9dbdac` |
| `campaign_launch.py` | `043d8d5025b2fa1807972baca6e8eafe5540dcf9bb376e9404e6ecb1eb371e81` |
| `campaign_eval.py` | `7064385d63e3900b87241da75525897acb6b55d9ce1a023689c9addfbaacfc9f` |
| `campaign_protocol.py` | `5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156` |

The source review used the snapshot copy under
`/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/26a07c58eebc1769a224a193a7b163e989125e4117236cc367b940e706d7af6f/source`.

## Why a fresh theta0 stage is the valid route

`campaign_sft.py` loads one `model_path` and then calls
`FastLanguageModel.get_peft_model` with the fixed rank 32, alpha 64 target
module list. Pointing it at merged theta0 therefore gives a known parent and a
new LoRA with a fresh optimizer. The parent manifest records the merged model
and tokenizer identities, so the new recipe can bind those identities without
opening the old optimizer state.

The old SFT1000 full checkpoint is not a valid resume source after target
correction. Its checkpoint identity binds the old train-row and draw-schedule
hashes. More fundamentally, its optimizer moments and RNG state were produced
from the old labels. `campaign_checkpoint.verify_checkpoint(...,
require_full=True)` intentionally rejects a recipe with a different data or
schedule identity. The old adapter checkpoint remains provenance only.

Keeping the original base and loading the preserved SFT adapter separately
would require a new loading seam: the current source has no adapter-path field
and always creates a new adapter. It would also need a CPU fixture proving
that the separate base-plus-adapter forward matches the accepted merged theta0
and a fresh optimizer reset. That is a larger, unreviewed route, so it is not
the smallest safe experiment.

Resuming within the new corrective stage is valid. Keep `max_steps=200` in the
identity for all three bounded attempts, and change only `decision_steps` and
fresh output/archive paths between attempts:

1. Start from theta0 with `decision_steps=[50]`; save and evaluate full step 50.
2. Resume the full step-50 checkpoint with the same identity and
   `decision_steps=[100]`; save and evaluate full step 100.
3. Resume the full step-100 checkpoint with the same identity and
   `decision_steps=[200]`; save and evaluate full step 200.

Using separate recipes with `max_steps=50`, `100`, and `200` would change the
checkpoint identity and fail strict resume verification. The skeleton keeps
`full_every=50`, `light_every=50`, and evaluation steps `[50, 100, 200]` so
each decision boundary has a durable full state.

## Corrected input contract

The data worker must materialize a new complete training file from the exact
admitted source rows (`train_rows_sha256` currently
`7641bbdc8f609aca1e0ad72177561edddfdbdae1b49a8444c15470652bf5ebb6`). It must
stream all 11,764 rows, preserve row IDs, order, source provenance, family,
operation, and the 1,140 semantic no-op rows, and apply only the source-backed
correction to the affected TRAIN finish rows. The current family counts are

```
finish_block          4289
format_propagation    1719
na_rm_propagation      137
no_op                  1140
pipe_rewrite          2027
rename_propagation    1578
roxygen_drafting       874
```

The six finish DEV IDs found in the separate sensitivity audit are DEV rows;
they must not be copied into TRAIN or used as a substitute for identifying the
source-backed TRAIN corrections. The corrected TRAIN artifact must retain the
full mixture and no-op population. A finish-only six-row training run would be
an invalid overfit experiment.

### Overlay result and materialization boundary

The finish overlay worker has now produced a shadow artifact. Its receipt is
`docs/campaign/receipts/DAT-04-finish-target-overlay-v1-preparation.json`,
SHA256
`c88e79f55d0841da56a56d115a5e024756666ab9bd171ee73b2897b099d8c45f`. The
packet contains 5,000 `train_group` finish rows (packet SHA256
`42743e70dbde54dd0f4adab42ebd0c2b0a0e7d3a4c4596752ffca272b20adc25`). Of
those, 4,289 have exact lineage to both the DAT-05 registry (SHA256
`ef8ca082be4699d52dab67cb9c42628df4cc935479fb769d880961e7c3c2cf3d`) and
the accepted SFT train rows (SHA256
`7641bbdc8f609aca1e0ad72177561edddfdbdae1b49a8444c15470652bf5ebb6`). The
shadow overlay is
`docs/campaign/work/finish-target-overlay-v1/finish-target-overlay.jsonl`,
SHA256
`03cff340de0c2b0e641a0399ddc8de727c39c9aa5e51aeed984c6420117054cf`, 58,907,718
bytes and 5,000 records.

Strict v2 admission makes 4,051 of the lineage-matched rows eligible for
source-backed repair. The other 238 matched rows are rejected because their
target is not LF-terminated; 711 packet rows have no downstream registry or
SFT lineage. Thus 949 packet rows are currently excluded from a corrected
sampler. The 238 rows remain in the original train artifact as provenance, but
must not be drawn by a corrected-stage schedule until separately reviewed. The
711 absent rows cannot be reconstructed from this overlay. The eligible
corrected candidate population is therefore 4,051 finish rows plus the intact
7,475 non-finish SFT rows (11,526 schedulable records before any policy
quotas); the 1,140 semantic no-op rows remain in that non-finish population.

The overlay has not been tokenized (`tokenization_performed=false`), and it is
not a replacement train file. A data worker still must stream the original
11,764 rows, apply only the 4,051 admitted target replacements, leave the 238
rejected and all non-finish rows unchanged, and emit a new full row-file hash
plus correction manifest. Every changed row must retain its row ID, source and
geometry evidence, unchanged prompt token prefix, and pinned tokenizer and
protocol identity. The worker must then produce a 3,200-draw schedule that
excludes the 238 rejected IDs and records its resulting exposure; the old
first-3,200 schedule can be reused for isolation only if it contains no
excluded ID. It must not silently backfill or alter family/no-op quotas.

Each changed row must be rebuilt with the pinned tokenizer and protocol
builder, not by appending a token ID to an old row. The builder must recheck the
joint prompt/target boundary, `target_start`, `target_body_tokens`,
`target_terminal_tokens`, token counts, one manual BOS ID 0, one terminal EOS
ID 1, tokenizer identity, and the 4096-token ceiling. No truncation is
permitted. The prompt text and its token IDs remain unchanged; only the
source-derived target continuation and its dependent token arrays may change.

`full_text_collator` labels every non-padding position, including prompt and
terminal positions. It masks only right-padding with `-100`, so this stage
keeps the existing full-text objective. There is no separate target-token
cache to invalidate; the complete corrected row file and its hash are the
cache replacement. Changing to target-only masking would be a different SFT
objective and is outside this preparation.

The 200-update schedule needs exactly 3,200 row draws at effective batch 16.
For scientific isolation, the preferred schedule is a new manifest containing
the first 3,200 row IDs from the accepted 3,000-update schedule, with only its
`token_rows_sha256` and correction binding changed. This preserves the original
early exposure, subject to the explicit exclusion of the 238 rejected IDs. If
one of those IDs occurs in the prefix, the worker must create a deterministic
policy-approved backfill and report the changed exposure rather than claim
isolation. If the data worker rebuilds the finite schedule with
`campaign_sampling`, it must prove the same no-op/family policy and report any
changed row exposure; it must not silently change the mixture. The existing
`campaign_sft_inputs.py` only emits 50- and 3,000-update schedules, so it cannot
be called unchanged to create this 200-update artifact. A small standalone
builder or an explicitly reviewed extension is required; neither is made in
this packet.

The corrected DEV representation is a separate optional full 75-row diagnostic
panel. It must retain all original contexts and IDs, change only the six
source-backed finish targets, and carry a new SHA256 plus a manifest binding it
to the original panel SHA256
`b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`. The
original panel remains the historical comparator and is never overwritten. The
existing evaluator can consume a new panel path because it already reads
`recipe["development_panel"]`, but the new panel hash must enter the recipe
identity and all readouts must be labelled diagnostic. An original-panel run
and a corrected-panel run are separate measurements; the latter cannot replace
the former's score. A candidate corrected panel is now available at
`docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl`
(SHA256
`7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035`), with
75 records, six target changes, and 69 byte-identical records. Its independent
review remains a launch gate; the skeleton records it as a candidate rather
than silently treating it as the historical panel.

## Existing implementation and host constraints

No production source edit is required for the fresh theta0 route once the
complete corrected row file, 3,200-draw schedule, and chosen DEV panel are
sealed. `campaign_sft.preflight` already permits `max_steps=200`, insists on
effective batch 16, and retains the fixed LR, rank, alpha, 4096 context,
cosine schedule, and full-text labels. It verifies every declared file before
CUDA imports.

There are three concrete preparation dependencies:

* The input builder must create the 200-update schedule described above. The
  current builder hardcodes `(50, 3000)`.
* The target materializer must tokenize the complete corrected row file after
  joining the overlay. The current loader validates `target_start`, body and
  terminal tokens, BOS/EOS, and the 4096-token ceiling, but it cannot derive
  replacement target IDs from a text overlay. No target-only cache exists to
  refresh; the new complete row file is the cache replacement.
* The current trainer reads every corrected row into `rows`, then at
  `campaign_sft.py:334-335` constructs a Dataset from all rows and selects the
  draws. A later host-safe source snapshot may replace that with a draw-only
  Dataset construction, for example by gathering
  `[{"input_ids": rows[i]["input_ids"]} for i in draws]`, deleting `rows` and
  `draws`, and then calling `Dataset.from_list`. This preserves order and
  labels but needs a CPU fixture regression. It is an optional narrow memory
fix, not silently applied here.

No production source change is required for the fresh theta0 route itself:
the immutable trainer already supports the 200-step parameter and fresh LoRA
initialization. The concrete changes needed before a launch are new data
artifacts (full corrected rows, correction manifest, excluded-ID-aware draw
schedule, and optionally a separately hashed corrected DEV panel) and a new
recipe binding all of them. The optional memory fix above is a later reviewed
snapshot change only; it must preserve draw order and full-text labels. A
target-only masking change, adapter-resume change, or in-place mutation of the
accepted train/DEV artifacts is outside this preparation.

The current SFT source fsyncs per-step control telemetry and full checkpoint
files, but it does not measure Windows free memory or process RSS. The live
command therefore still requires the root-owned host supervisor and memory
guard. The source telemetry must show finite losses/gradients and each full
checkpoint must pass the existing manifest verifier; a host-guard stop or
driver event is a failed attempt requiring review. No quality or promotion
claim follows from command completion.

## Proposed recipe and launch shape

`corrective-sft-recipe-skeleton.json` is deliberately marked
`template_not_admitted`; its required overlay hashes and run paths are
placeholders. After the data worker seals those artifacts, the root can copy
the skeleton into a fresh recipe and replace every required value.

The launch shape for each attempt is:

```
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python \
  /home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/26a07c58eebc1769a224a193a7b163e989125e4117236cc367b940e706d7af6f/source/experiments/training/campaign_launch.py \
  /home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT-finish-correction-<attempt>/recipe.json \
  --receipt /home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT-finish-correction-<attempt>/supervision.json
```

Each attempt needs a new absent output and archive directory and a bounded
supervisor deadline. No command is authorized by this packet; root must first
bind the overlay, panel, source, storage, host guard, and available budget.

### First-stage time and compute estimate

The accepted SFT telemetry measured 4.5352441 seconds/update through step 1000
and 3.57099 seconds/update through step 2000. Those observations imply about
179--227 seconds of optimizer compute for the first 50-update decision, before
model load, tokenizer/data validation, DEV evaluation, checkpoint writes, and
host-guard reserve. The corresponding compute ranges are 357--454 seconds for
100 updates and 714--907 seconds for 200 updates. The skeleton therefore uses
an 1,800-second attempt bound as a planning value, not a runtime guarantee;
root must confirm budget and storage before admission. The three milestone
attempts are one fresh 50-update run followed by two strict same-identity
resumes, rather than three independent optimizer starts.

## Suggested acceptance gates

1. Verify the correction manifest against the original train-row hash, exact
   changed TRAIN IDs, all 11,764 row identities, family/no-op counts, and the
   complete corrected-file SHA256.
2. Verify every corrected row through the pinned protocol/tokenizer builder,
   including the joint encoding boundary and full sequence length. Reject any
   truncation, missing EOS, duplicate BOS/EOS, changed prompt, or unbound
   source geometry.
3. Verify the 3,200-draw manifest is complete, references only corrected
   TRAIN rows, has the new row-file hash, and preserves the declared exposure
   policy. Reject an infeasible or partial manifest.
4. Run `campaign_sft.py --preflight-only` through the immutable source. Confirm
   no CUDA started, all input hashes, corrected-panel denominator 75, and fresh
   output/archive paths.
5. At steps 50, 100, and 200, verify full checkpoint manifests, optimizer and
   scheduler state, RNG state, sampler draw count (`step * 16`), tokenizer
   contract, finite loss/gradient telemetry, and the corrected-panel readout.
6. Compare the corrected stage with theta0 on the original panel and the
   corrected diagnostic panel only after both artifacts are independently
   pinned. Preserve the original historical score and report all denominators.
   Stop on loss of editing/no-op coverage, protocol/cap regressions, host guard
   pressure, driver events, or any identity mismatch. The stage remains a
   diagnostic experiment and cannot open sealed final data.
