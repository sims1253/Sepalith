# C13: data selection, recipes and gates

Card [C13](../cards/C13-rft-dpo.md). Code: `packages/sepalith/src/sepalith/training/offline/`.
Commands: [build_c13.sh](build_c13.sh). Status: [status/C13.json](../status/C13.json).

The editing-SFT modules (`training/sft`, `optim`, `checkpoint`, `contracts`,
`guards`) are bound by hash into the arm A and arm B recipes, so C13 adds a new
package beside them and changes none of them. Arm B can keep resuming from
`main` after this merges.

## Inputs

- C12 census of arm A (checkpoint 536): `~/.local/state/sepalith/c12/census-arm-a/`,
  `census-summary.json` sha256 `eaeadf7b...9171`, every file hash-checked
  against it. Public copy: `campaign-20260915/C12/sepalith-2b-edit-sft-arm-a/`.
- TRAIN rows `DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl`
  (`ed1c7b4e...3aef`). Every census prompt hash equals its TRAIN row's prompt.
  No DEV row and nothing from the sealed set is read.

## RFT samples (user decision 2026-10-17: family-capped, 4,413)

1. **Pool.** Census samples with `exact` under reward v2.1, made distinct per
   prompt by generated ids, at most 2 per prompt: 7,528 samples on 7,382
   prompts (52,145 exact samples in total).
2. **Smallest edit first.** Within a prompt, samples are ranked by the
   line-level diff between the sample's body and the prompt's CURRENT region
   (the smallest edit wins). Ties go to the sample that is raw-identical to the
   gold target, then to fewer tokens, then to the lower sample index.
3. **Family caps.** Each family keeps at most its TRAIN share of the pool,
   `floor(train_rows_f / 15,006 x 7,528)`, and at most its supply. No family is
   scaled down to the scarcest one.

   | Family | TRAIN rows | Supply | Cap = selected |
   | --- | ---: | ---: | ---: |
   | finish_block | 7,785 | 1,181 | 1,181 |
   | pipe_rewrite | 1,983 | 1,995 | 994 |
   | format_propagation | 1,615 | 1,505 | 810 |
   | rename_propagation | 1,578 | 1,589 | 791 |
   | no_op | 1,094 | 1,100 | 548 |
   | na_rm_propagation | 136 | 137 | 68 |
   | roxygen_drafting | 815 | 21 | 21 |
   | **Total** | 15,006 | 7,528 | **4,413** |

4. **Order within a capped family.** Prompts are taken by census pass rate,
   lowest first, because mixed prompts carry the most signal. Ties are broken
   in a seeded order (3407). Each prompt's best sample comes first, then second
   samples, until the cap is reached. 4,324 rows are first samples and 89 are
   second samples. 1,952 rows come from mixed prompts (0 < pass < 1).
5. **Rows.** Each sample becomes a TRAIN-shaped row: the TRAIN row's prompt,
   the sample's body, the TRAIN terminal and EOS 1. This is the SFT render,
   masking and termination, and the rows pass `campaign_edit_data.validate_row`.
   The row id is `<train id>:rft:s<k>`, with provenance under the `rft` key.
   4,275 of the 4,413 rows are token-identical to their gold target. Most of
   the RFT signal is therefore re-weighting toward prompts the model already
   solves.

## RFT cohort and schedule

- **TRAIN mix (card: 30%).** The RFT rows plus 1,891 original TRAIN rows, so
  TRAIN rows are 30.0% of the 6,304-row cohort. The 1,891 are a uniform seeded
  sample of the 10,682 TRAIN rows whose prompt has no RFT row (4,324 prompts have one). That avoids
  duplicate sequences and puts the replay where RFT gives no signal (1,159
  finish_block, 148 roxygen_drafting, ...).
- **Schedule v2, unchanged.** 13 edits plus 3 no-ops per update gives 18.75%
  no-ops. No-op rows get at most 3 exposures (648 rows seen twice, 3 seen
  three times). The 435 updates use 6,960 draws and 7.09 M scheduled tokens.
  Gates fall at the halves: 218 and 435.

## DPO pairs

- All 18,268 census pair candidates on 9,497 prompts. Chosen is a distinct
  verified-correct sample (3,058 pairs) or the gold target (15,210). Rejected
  is a protocol-valid incorrect sample: 16,782 near misses, 1,155 wrong but
  parse-unverified, 164 missed edits, 157 syntax-invalid edits and 10 false
  suggestions on no-op prompts.
- Completions end in EOS 1, as SFT targets do. No end-of-generation id needed
  normalizing. The longest pair sequence is 2,987 tokens.
- **Pair schedule.** One seeded pass with 16 pairs per update, 1,142 updates.
  The last update is completed with the 4 earliest drawn pairs. Gates fall at
  571 and 1,142.

## Recipes (bound in `bound/`, templates in `templates/`)

| | RFT `rft-1-arm-a-536.json` | DPO `dpo-1-arm-a-536.json` |
| --- | --- | --- |
| Starting model | arm A 536 (`befe587a...`), reference = its gate 536 (score250 117, valid 74, cap 1, edit 27, nofp 8) | same, or the adopted RFT result after rebinding |
| LR (hidden / side) | SFT LR / 3 = 3.33333e-6 / 3.33333e-7 | SFT LR / 10 = 1e-6 / 1e-7 |
| Objective | target-only CE (SFT collator, masking gate) | sigmoid DPO, beta 0.1, summed completion log-probabilities, stored reference |
| Optimizer, schedule | aurora_mix with the CPT mechanics, bf16 stochastic rounding, 32 warmup updates, cosine to 10%, weight decay 0, batch 16 (micro-batch 1) | same; batch 16 pairs |
| Gates | 218, 435 | 571, 1,142 |
| Retention | retain_all | retain_all |

Both recipes also pin the DEV75 and DEV250 panels, batch-32 dynamic-cache
greedy generation (the same settings gate 536 used), the source manifest
`bound/source-manifest.json`, the template and the parent recipe.

**Switching the parent** is a rebind with `build_c13.sh bind`. A census on
another model (arm B, per R4) also needs `build_c13.sh data` and `templates`.

## Gates (pre-registered; `offline/rules.py`)

- **Every gate is R1-style.** It applies the R1 later-gate table. The first
  gate compares with the starting model's own gate result, later gates with
  the previous gate. R1's first-gate floor ("edit beats the parent") is not
  used: it is written for SFT from a CPT parent that cannot edit.
- **noedit_dodge**, stop-for-user. Edit cases answered `[NO_EDIT]` (DEV75 plus
  DEV250) rise by 3 or more while exact edits (DEV75 plus DEV250) do not rise.
- **dpo_chosen_collapse**, DPO only, stop-for-user. The 50-update means of
  log pi(chosen) - log pi_ref(chosen) fell three times in a row to below 0,
  while score250 is not above the starting model's. This is the card's stop
  condition.
- **R5** at the step's result: the final gate, or the last good gate after a
  rollback. The driver writes `r5-decision.json` and exits 0 on adopt or
  discard, or 3 in the stop-for-user band.
- **Before update 1.** RFT runs the SFT masking gate. DPO runs a pair-batch
  gate and a reference-alignment gate: the step-0 policy must reproduce the
  stored reference log-probabilities, within a mean of 0.5 nats and a maximum
  of 2.0 nats.

## Night cost (arm A rate: about 1,030 trained tokens/s, gates 10 to 13 min)

| Job | Work | Estimate |
| --- | --- | --- |
| `C13-rft-1-arm-a-536` | 7.09 M tokens, 2 gates | about 2.3 h: one night |
| `C13-dpo-1-<start>-reference` | 21.8 M tokens forward only | about 1.5 to 2.5 h |
| `C13-dpo-1-<start>` | 21.8 M tokens (prompt counted twice), 2 gates | about 6.3 h |

DPO therefore needs two nights. The jobs are resumable and the reference pass
keeps its verified prefix. Total: three nights, then the R4 step 2 re-census on
Kaggle (about 3.3 GPU-h).

## Iteration 2: census at temperature 1.2 (user decision 2026-10-22)

Iteration 1 adopted nothing. RFT-1 and DPO-1 both rolled back at their first
gate because `nofp` rose by 4 and 5, so R5 discarded both. R4 step 4 then
stopped for the user, with `M` at 12.5%. The user chose a second offline
iteration at temperature 1.2, again from arm A 536.

Only the census temperature changes. The prompts, k = 8, top_p 1.0, cap 1,024
and the seed scheme (3407 + census index, so the same seeds) stay as they were.
So do reward v2.1, the selection rules above, the TRAIN mix, schedule v2, the
recipes and the gates. The commands are the iteration-1 ones with
`ITER=2 MODEL_ID=sepalith-2b-edit-sft-arm-a-t12 TAG=arm-a-t12`, parent tag
`arm-a-536-t12` (see [build_c13.sh](build_c13.sh)).

| | Iteration 1 (T = 1.0) | Iteration 2 (T = 1.2) |
| --- | ---: | ---: |
| `M` | 12.5% | 14.7% |
| All-wrong share | 50.8% | 52.3% |
| Exact samples | 52,145 | 49,796 |
| Protocol-invalid samples | 438 | 18,771 (17,338 finish_block) |
| Distinct pool (at most 2 per prompt) | 7,528 | 7,340 |
| RFT rows (family-capped) | 4,413 | 4,113 (roxygen_drafting 7) |
| Cohort rows (30% TRAIN) | 6,304 | 5,876 |
| RFT updates, gates | 435; 218, 435 | 404; 202, 404 |
| DPO pairs (model-chosen) | 18,268 (3,058) | 18,888 (3,505) |
| DPO updates, gates | 1,142; 571, 1,142 | 1,181; 590, 1,181 |

The recipes are `bound/rft-2-arm-a-536-t12.json` and
`bound/dpo-2-arm-a-536-t12.json`. The jobs are `C13-rft-2-arm-a-536-t12`,
`C13-dpo-2-arm-a-536-t12-reference` and `C13-dpo-2-arm-a-536-t12`. As in
iteration 1, DPO from arm A 536 runs only if RFT-2 is discarded.
