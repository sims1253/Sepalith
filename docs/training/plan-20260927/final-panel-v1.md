# Final panel v1: a replacement constructor for the sealed final

**Status: design approved by the user on 2026-11-23 (A1). No final content
has been read.** Building the panel is a later step. It needs the user's
written weight and harness freeze and a separate written build authorization
(see [Access sequence](#access-sequence) and
[Build authorization draft](#build-authorization-draft)). Code:
[`docs/campaign/work/final-panel-v1/`](../../campaign/work/final-panel-v1/).

## User decisions (2026-11-23)

| Question | Answer | Effect |
| --- | --- | --- |
| Design and harness pin | "Approve; re-pin harness" | Design approved. The launcher's harness check moved to main `815967bb` ([sealed-final-v2.md](sealed-final-v2.md#harness-re-pin-2026-11-23)). A3 and A5 run from the main checkout. |
| Roxygen gap and code-cut-only finish | "Accept, report the mix" | The 250-case mix below stays. The final report states it and compares with [DEV250 reference rates at the same mix](#family-mix-and-dev250-reference). |
| Unread final reserve | "Hold back ~45%" | [Holdback split](#holdback-split-pre-registered): 105 build groups, 88 sealed |
| Rehearsal | "Yes, rehearse first" | [Dress rehearsal](#dress-rehearsal-2026-11-23) on dev groups before A2 |

## Why

On 2026-11-23 the sealed-final prep for
[C17 step 4](cards/C17-release-prep.md#4-recover-and-freeze-the-sealed-final-protocol)
found that `DAT-07-final-evaluator-cases.jsonl` is the DEV75 panel and not a
final panel. All 75 rows have `split: dev` and `source_ref.split: dev_group`.
The registry preflight reports `preflight_missing_approval` with 0 rows
admitted.

The real final split is the 193 `final_candidate_group` groups in
`DAT-02-global-split-v2.json` (sha256 `c4e1219a…8f09`). The v1 route from
those groups to cases is lost. `r2-final-assembly-v1/assemble_inputs.py`
survives in git. Its constructor modules (`final_constructor.py`,
`raw_source_case_builder.py`, `raw_source_tokenization.py`) and its selection
file (sha256 `f8037b71…1e1e`) were lost in the 2026-10-10 WSL reinstall.

On 2026-11-23 the user decided: "Build a new final panel."

This constructor turns those 193 never-used groups into cases. It uses the
same task families and the same PRM03 case format as DEV75 and DEV250, so
the [sealed-final v2 protocol](sealed-final-v2.md) scores them unchanged.

## What DEV75 and DEV250 are, and what survives

The DEV panels are `PromptContext` objects (`sepalith.prompt.prm03.v1`) plus a
target `{operation, region_new}`. Each panel row records two hashes:

- `prompt_sha256`, the SHA-256 of `render_prompt(context)`;
- `target_sha256`, the SHA-256 of `serialize_target(operation, region_new)`.

This holds for all 250 DEV250 rows and all 75 DEV75 rows, so both hashes can
be checked without a tokenizer.

The C05 DEV250 builder (`c05/build_dev250_packets.py`, `c05/build_dev250.py`)
built each family from the raw parent sources of the dev groups. Two of its
modules were lost with the `tuesday-execution` worktree:
`campaign_dev_propagation.py` and `campaign_dev_panel.py`. The other parts are
still in git. `recipe.json` pins every one of them by sha256.

| Part | In git |
| --- | --- |
| Rename, pipe, na_rm and format extractors | `experiments/synthetic-data/scenarios.py`, sha256 `cccf8ddf…250c`. This is the canonical-scenario pin in the DAT-08 review and the corrected-DEV75 manifest. |
| Code-cut finish rule | `cases/rules/rules_finish_block.py`, called through `c05/cut_dev_finish.py` |
| DAT-04B completion adapter `dat04b-completion-adapter-v3` | `lead/r2-task-materialization-a/reviewed-source/…/campaign_admission_completion.py`, sha256 `fafb852f…73cc` |
| DEV75 outer-brace correction | `finish-boundary-repair-v2/finish_boundary_repair_v2.py` |
| PRM05 source window | `sepalith.training.rl.campaign_token_audit.selected_context` |
| Allocation | C05's Edmonds-Karp max-flow. Its logic is copied unchanged into `FlowNetwork`. |

The constructor reimplements the two lost pieces:

- `convert_example` turns a scenario example into the post-event document,
  one history event, the region and the cursor;
- `noop_sites` is the no-op site rule.

### Reproduction on dev data (no final content)

`dev_equivalence.py` reads only dev material. It wrote
[`receipts/C17-final-panel-dev-equivalence.json`](receipts/C17-final-panel-dev-equivalence.json).

| Family | Check | Result |
| --- | --- | --- |
| rename, pipe, na_rm, format | Extractors run on the raw dev parent regenerate the DEV250 derived row | 23, 28, 3, 24 of 23, 28, 3, 24 |
| rename, pipe, na_rm, format | Converter plus PRM05 window gives the DEV250 context field for field, with identical prompt and target hashes | 78 / 78 |
| rename, pipe, na_rm, format | Same converter, against DEV75's recorded prompt and target hashes | 29 / 29 |
| no_op | Site rule plus C05 round-robin gives the same phase-1 candidate set per dev group | 25 / 25 groups |
| no_op | Context, prompt and target equal to DEV250 | 50 / 50 |
| finish_block | Code-cut harvest over the 105 eligible dev groups | 1,329 rows, byte-identical |
| finish_block | Adapter plus brace correction plus window gives the DEV250 context and target | 56 / 56 code-cut, 55 / 55 authored |

`event_id` and the range URIs never reach the prompt. The constructor derives
both from the row ID.

**What a rehearsal changes relative to C05.** Every rename seed is
`seed_for(group, path, family)`. C05 seeded by parent index, and that order
is lost. C05 also searched 40 na_rm seeds in DEV75's groups only; this
constructor does not. Both changes are deliberate final-specific choices.

`roxygen_drafting` cannot be built this way. DEV75 and DEV250 took their
roxygen cases from existing rows in the shared `scenarios_v1` batteries.
DAT-02 marks those batteries as final-ineligible ("until regenerated from raw
parents"). The v1 assembly also listed roxygen as a gap ("no admitted
constructor").

## Inputs

| Input | Identity | Use |
| --- | --- | --- |
| DAT-02 global split v2 | `/mnt/e/…/DAT-02-global-split-v2.json`, sha256 `c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09` | Group IDs, split labels, `identity_forms`, `flags` |
| Normalized CRAN corpus | `/mnt/nas/home/sepalith/normalized` (the real path behind `/mnt/h/sepalith/normalized`) | Parent sources `<Package>/<version>/<Package>/R/*.R`, `DESCRIPTION` |
| CRAN tarballs | `/mnt/nas/home/sepalith/tarballs/<Package>_<version>.tar.gz` | Raw pre-format sources for `format_propagation` |
| CPT document index | `CPT-prefix-extension-v1/combined-cache/index.sqlite3` | Skip any source whose sha256 is a CPT document (the C05 rule) |
| DEV75 | `DAT-07-final-dev-candidates.jsonl`, sha256 `bf278755c795b51ba58bc14c674edf76dcb7799475e5982adc7bd50f9c0d05df` (from the registry preflight). The user authorized reading DAT-07 registry metadata on 2026-11-23. Only `prompt_sha256` and identity fields are used. | Final prompts must not repeat it |
| DEV250 | `C05-dev250-v1r3/dev250-v1.jsonl`, sha256 `75f3a5ee…f487f22` | Same |
| Campaign tokenizer | `TOKENIZER_JSON_SHA256` and `TOKENIZER_CONFIG_SHA256` pins | Length filters through `build_training_row` |

The corrected DEV75 file (`7c144bcd…`) was lost with the planning worktree.
Its correction changed six finish targets and no prompts, so its prompt
hashes equal those of `bf278755…`.

**Parents.** Each `pkg:<name>` identity form of a group (sorted) must match
exactly one corpus directory, comparing `NFKC(name).lower()`. Zero matches or
several matches skip that form and are counted. The version directory is the
highest by `build_astfim.version_key`, with the directory name as tie-break.
A parent needs a `DESCRIPTION` with a nonempty `License:` field. A group
needs at least one parent. The C05 finish harvest normally re-picks the
version. Here it is pinned in memory to the checked version directory.

On the dev split, this rule reproduces all 165 parents of
`DAT-07-dev-parent-allowlist.json`. It leaves out the 4 dev groups that have
no CRAN parent. A final group is eligible when its split is
`final_candidate_group` and it has no `flags`, as in the v1 assembly.

Before any file of a package is read, `DESCRIPTION` included, `checked_tree`
refuses a version directory that holds any symlink or resolves outside the
corpus root.

## Case construction

Per parent, R files are visited in sorted order. A file is skipped when it:

- is empty or larger than `scenarios.MAX_FILE_BYTES`;
- is not UTF-8;
- has a lone CR or mixed LF and CRLF line endings;
- has a sha256 that is a CPT document.

C05 dropped mixed-EOL files later, in the window and the adapter. Here they
are skipped up front. The one exception is `finish_block`, which keeps C05's
own harvest filter (empty, size and CPT). Its EOL and encoding checks stay in
the DAT-04B adapter, so the finish route stays byte-identical to C05.

| Family | Constructor | Operation |
| --- | --- | --- |
| `rename_propagation` | `scenarios.extract_rename(cap=4)` with `Random(seed_for(group, path, family))`, then the converter | `replace` |
| `pipe_rewrite` | `scenarios.extract_pipe(cap=4)`, then the converter | `replace` |
| `na_rm_propagation` | `scenarios.extract_na_rm(cap=4)`, then the converter | `replace` |
| `format_propagation` | The tarball member `<Package>/R/<file>` is checked for an exact first path part, a regular file and the size caps. `scenarios.format_pairs_from_lines` compares it with the normalized file, then the converter runs. | `replace` |
| `no_op` | A non-comment line whose code (with quoted literals blanked and any inline `#` comment cut) matches `na.rm = TRUE` as a whole token (`already_applied_na_rm_true`, checked first) or ends in `\|>` (`already_applied_native_pipe_operator`). The cursor sits at the end of the line, there is no history, and `region_new = region_old`. | `no_op` |
| `finish_block` | C05's code-cut harvest: at most 6 bases per package, one per file, C05's body window and plain transforms. One transform per base by `Random(seed_for(group, base_id))`. Then the DAT-04B adapter, then DEV75's outer-brace correction with both of C05's parse gates. | `replace` |

`seed_for(*parts) = int(sha256_json([3407, *parts])[:16], 16)`.

**The converter** does the following:

1. Applies the example's single-hunk event to the before image. A hunk that
   does not apply drops the candidate.
2. Locates the region from the example's prefix and `region_old`. It needs
   exactly one match, then shifts the region by the event's line delta.
3. Builds the post-event document and one `observed_edit_diff` event:
   - `workspace_revision_before` and `range_utf16.content_sha256` are the
     before image's sha256;
   - the range is the event's full lines in the before image.
4. Builds the full-line replacement range and the cursor (`cursor_idx`, and
   its UTF-16 column).
5. Applies C05's two R parse gates: the post-event document must parse, and
   so must the post-edit document.
6. Passes the whole document to `selected_context`, the PRM05 window with a
   6,000 UTF-16 unit budget.

Every case then goes through `PromptContext.from_mapping` and
`build_training_row`.

**DAT-04B split gate.** The adapter accepts only `train_group` and
`dev_group`. A final build passes the true split, `final_candidate_group`,
and widens the adapter's `ALLOWED_SPLITS` in memory. It does so only inside
`build_panel`, after `authorize()` has accepted the user's receipt. No final
row is labelled dev.

**Order.**

1. Per (family, group), at most 3 candidates are generated, before any
   filter, as in C05. Parents are taken in `pkg:` order and families in a
   fixed order. No-op sites are taken round-robin across files, as in C05.
2. The filters run in generation order:
   - window and row validity;
   - the response with terminal and EOS is at most 1,024 tokens;
   - the sequence is at most 4,096 tokens;
   - no prompt equal to a DEV75 or DEV250 prompt;
   - one case per rendered prompt;
   - one case per row ID.
3. Allocation sorts the survivors by `sha256_json([3407, row_id])`.

Row IDs are `final-v1-` plus the first 24 hex characters of
`sha256_json([3407, group, package, path, family, site])`. The site is the
event line, cursor, prefix length, prefix hash and extractor note; the no-op
line and its rule; or the finish base, transform and cut.

## Sampling, size and caps (pre-registered)

All values were fixed before any final content was read. They copy DEV250's
sizing.

| Slot | Quota | Floor | DEV250 |
| --- | ---: | ---: | ---: |
| `finish_block` | 122 | 61 | 111 (55 authored, 56 code-cut) |
| `no_op` | 50 | 40 | 50 |
| `pipe_rewrite` | 28 | 14 | 28 |
| `format_propagation` | 24 | 12 | 24 |
| `rename_propagation` | 23 | 12 | 23 |
| `na_rm_propagation` | 3 | 0 | 3 |
| `roxygen_drafting` | 0 (gap) | 0 | 11 |
| **Total** | **250** | **200** | 250 |

Roxygen's 11 slots go to `finish_block`, TRAIN's largest family, following
the C05 2026-10-03 precedent. Every final finish case is code-cut. The
authored half of DEV250's finish cases was written by an LLM. It cannot be
reproduced, and authoring would send final text to an external model. As a
result, 122 of the 250 final cases are code-cut finish cases, against 56 of
250 in DEV250. Report this mix next to the final score.

**Group cap.** At most 3 cases per DAT-02 group, as in C05.

**Allocation.**

1. Pass 1 is C05's max-flow (source → slot → group → sink) with the quotas
   above.
2. Unfilled edit slots (pipe, format, rename, na_rm) move to `finish_block`.
   Pass 2 raises only finish_block's source edge and keeps augmenting the
   same flow. An augmenting path never sends flow back into the source, so
   no other slot can lose cases.
3. A `no_op` or `finish_block` shortfall is not moved. It makes the panel
   smaller.

**Floors.** The floors are construction stops, not pass criteria. They catch
a broken family, such as a missing tarball root, before anything is pinned.
The build stops if the panel has fewer than 200 cases or any family is below
its floor. The receipt then holds aggregate counts only.

Rehearsals of this recipe:

- **dev_group:** 250 cases from 105 groups. The 3 na_rm slots moved to
  finish_block, because DEV75 and DEV250 already use dev's na_rm supply.
- **193 train groups, CPT filter off** (train sources are CPT documents):
  250 cases from 91 groups, every quota filled.

In both, the launcher dry run constructed 250 of 250 cases with 0
mismatches.

## Holdback split (pre-registered)

On 2026-11-23 the user decided to hold back about 45% of the 193 final
groups for a later release. The split uses group IDs only. It reads each
DAT-02 group's `group_id`, `split` and `flags`, and no identity form, parent
or source.

- **Rule.** Take the eligible groups of the split (label
  `final_candidate_group`, no flags: 193). Order them by
  `sha256_json([3407, group_id])`, the order the rehearsal `--groups`
  option already uses. The first 105 groups build; the other 88 are sealed.
- **File.** [`holdback-split.json`](../../campaign/work/final-panel-v1/holdback-split.json),
  sha256 `2b284ba2f39260a2843634674faefd47191ed6a77160c47944cb0cb54ff6c30c`.
  It holds the DAT-02 pin, the rule, the counts and both sorted ID lists.
  The group IDs are opaque `g-` hashes. Commit `b06b7d52` pushed it before any
  final content was read; none has been read since.
- **Pin.** `recipe.json` carries `holdback.final_split_sha256`, so the
  authorization's `recipe_sha256` binds the split.
  `holdback.py check` recomputes the split from the pinned DAT-02 bytes and
  compares it with the file and the pin.
- **Enforcement.** For an authorized build, `_construct()` computes the
  split itself and refuses a final split that differs from the pin. It then
  keeps only the build groups, before any identity form, parent, package
  tree or tarball is looked at. An authorized build takes no group or quota
  override. `build_final()` checks the pin before it claims the attempt
  marker. The test `test_holdback_groups_are_never_opened` records every
  `open`, `listdir`, `scandir` and `stat` audit event of a synthetic
  authorized build and finds none under a holdback package's tree or
  tarball. With the filter removed, the same test fails.
- **Stop rule on the smaller pool.** The floors are unchanged. A3 stops for
  the user, with aggregate counts only, if the panel has fewer than 200 cases
  or any family is below its floor. That rule now applies to a 105-group
  pool. At 3 cases per group, 105 groups can supply at most 315 cases, so
  the group cap binds harder than with 193 groups. The dev rehearsal of the
  same size shows whether the quotas still fill.
- **Later use.** The 88 holdback groups are a second sealed pool. Using them
  needs a new user decision, and they must not be read by any C17 step.

## Family mix and DEV250 reference

The final mix (user: "Accept, report the mix"):

| Family | Final | DEV250 |
| --- | ---: | ---: |
| finish_block | 122, all code-cut | 111 (55 authored, 56 code-cut) |
| no_op | 50 | 50 |
| pipe_rewrite | 28 | 28 |
| format_propagation | 24 | 24 |
| rename_propagation | 23 | 23 |
| na_rm_propagation | 3 | 3 |
| roxygen_drafting | 0 | 11 |

The final rate is therefore not comparable with score250/250. The reference
is each model's DEV250 per-family rate weighted by the final mix, with
finish_block taken from the 56 code-cut DEV250 cases only.
[`dev250_reference.py`](../../campaign/work/final-panel-v1/dev250_reference.py)
reads only the DEV250 panel and the existing per-case DEV250 results. It
wrote [`receipts/C17-final-panel-dev250-reference.json`](receipts/C17-final-panel-dev250-reference.json).
Intervals use a stratified case bootstrap (10,000 replicates, seed 3407),
paired across models.

| Model | DEV250 source | score250 | Code-cut finish | Rate at final mix | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| arm-a-803 | SFT-v2-r4 gate, Transformers bf16, CUDA | 109 | 1 / 56 | 43.7% | 40.4 to 47.1 |
| c19-100-bf16 | C19 selection, Transformers bf16, CUDA | 112 | 2 / 56 | 45.7% | 42.0 to 49.6 |
| c19-100-q6_k | C17 quant gate, llama.cpp | 111 | 2 / 56 | 45.3% | 41.6 to 49.2 |

Paired differences at the final mix: C19 bf16 minus Arm A +2.1 pp (−0.8 to
+5.3); Q6_K minus C19 bf16 −0.4 pp (−1.2 to 0.0).

Every roster model scores near zero on code-cut finish (1 or 2 of 56), and
code-cut finish is 49% of the final panel. Expect the final rates near 45%
rather than near score250/250 (44 to 45%) for that reason, and expect the
finish family to give little separation between models. The final report
must show the mix, the reference rate at the actual built mix
(`dev250_reference.py --mix <build-receipt.json>`; a shortfall move changes
the mix) and the note that the references come from GPU or llama.cpp runs
while the final scoring runs on CPU.

## Dress rehearsal (2026-11-23)

The user asked for a full rehearsal of build, admission and scoring on dev
groups before the build authorization. It ran the real code paths from this
branch with
[`dress_rehearsal.py`](../../campaign/work/final-panel-v1/dress_rehearsal.py),
under `~/.local/state/sepalith/sealed-final-v2/rehearsals/` (mode 700). The
rehearsal authorization says `split: dev_group, rehearsal: true`. It binds
the real constructor, recipe, weights (`sha256_json` of the plan roster
`3ed055fd…e4fb`), harness (`874e0f0a…68f5`), memberships and a rehearsal
freeze statement, and it claims a one-shot attempt marker.

**Run 1: the recipe as registered (105 dev groups, real memberships).**
`build_final()` claimed its marker, read the holdback rule's 105 dev build
groups, and stopped with the pre-registered outcome `stopped_below_floor`.
It wrote only the aggregate build receipt and published no panel. 202 cases:
no_op 25 (floor 40) and pipe_rewrite 12 (floor 14). The DEV panels had
already used dev's no-op and pipe supply: 44 no-op and 14 pipe candidates
were dropped as `same_prompt_as_dev_panel`. The overlap audit also flagged
all 79 used groups, all of them through the `dev` inventory, as it must for
dev cases. Build time 90 s.

**Same-size supply check on train (105 train groups in the same seeded
order, CPT filter off, plain rehearsal).** `stopped_below_floor` again:
243 cases. Every family filled except pipe_rewrite, which reached 12 of 28
against its floor of 14. Only 12 pipe candidates exist in those 105 groups.
no_op filled 50 of 50.

**Finding for the user before A2.** At 105 groups the pipe floor of 14 is
likely to stop the final build: both same-size pools supply 12 pipe cases.
With 193 groups the train rehearsal filled all 28. The stop is
pre-registered, so a stop at A3 abandons the C17 sealed final. Any change
must be made now, before A3, and only by the user. Options:

- (a) Lower the pipe_rewrite floor to 6. A broken extractor gives 0, so the
  floor still catches it. Unfilled pipe slots move to code-cut finish, as
  already pre-registered.
- (b) Keep the floor and accept the stop risk.
- (c) Build from more groups. This would undo part of the 45% holdback.

**Run 2: admission and scoring (all 169 dev groups, dev-blind
memberships).** This run used the rehearsal-only field
`rehearsal_build_groups: 169`, which `authorize()` refuses on a final
receipt. Its memberships are the real eight inventories with the `dev` stage
emptied, so the launcher's audit can pass on dev cases. The build gave
`built`: 250 cases from 105 groups (finish 125, no_op 50, pipe 28,
format 24, rename 23, na_rm 0; the 3 na_rm slots moved to finish). The
overlap audit found 0 groups. The launcher dry run loaded and constructed
250 of 250 cases with 0 hash mismatches. Build time 90 s.

Scoring used the full launcher (`protocol.run()`): the freeze and plan pins,
the guard, `FrozenRuntime` with the pinned interpreter (`.venv-sft`,
CPython 3.12.15), distributions, tokenizer, weights, Q4_K_M-mixB-e4 file
pin and llama-server `3cb7ffb1a` (CPU build), then the seal, audit,
construct and scoring on CPU with 8 threads. The only in-memory change sets
the launcher's panel-split constant and the guard's semantic-receipt split
constant to `rehearsal_dev_group`.

CPU bf16 on this machine (AMD Ryzen 9 5900X, Zen 3, no AVX-512 or bf16
units, shared with other sessions at load 30 to 50) took 252 s for one
1,399-token prompt in a probe. The full roster, 250 cases and 3 models,
would not finish in a day, so scoring ran on a subset pre-registered in the
script before any rehearsal scoring: the first 30 row IDs in
`sha256_json([3407, "dress-rehearsal-subset", row_id])` order (finish 18,
pipe 6, no_op 4, format 1, rename 1). That subset is written as a panel of
its own, bound by its own cases hash and semantic receipt over the same
staged sources.

SCORING_RESULTS_PENDING

## Build authorization draft

A2 has two parts. The user writes and signs them; an agent must not issue
them. Fill in the bracketed values from the main checkout after this change
is fast-forwarded there.

**(a) Freeze statement** (plain text, its file sha256 goes into the
authorization):

> I freeze, for the C17 sealed final v2, the weights in the plan roster
> `pins/models.json` (arm-a-803, c19-100-bf16, c19-100-q6_k; `sha256_json`
> `3ed055fdfe80f45c3b002ba98d004d9bfb60875ec8d8b370aaba4f06c1c6e4fb`), the
> release artifact Q4_K_M-mixB-e4
> (`e24fbfbc5ea47c88647445ec682196fc67af9134d4fa36e826beeda045f1c71b`), and
> the harness `874e0f0a725a1db5d3245014c6cfda59d17f073092e3616f5fddf4b6a64968f5`
> (evaluator commit `815967bb`, package tree `a9510fb9`). No training,
> checkpoint, quantization or tuning decision may depend on the final
> results. Frozen at [UTC time].

**(b) Build authorization** (`build-authorization.json`; the user also gives
its sha256 in writing):

```json
{
  "schema": "sepalith.final-panel.v1.build-authorization",
  "status": "user-approved",
  "split": "final_candidate_group",
  "constructor_sha256": "[sha256 of final_panel.py on main]",
  "recipe_sha256": "[sha256 of recipe.json on main]",
  "weights_sha256": "3ed055fdfe80f45c3b002ba98d004d9bfb60875ec8d8b370aaba4f06c1c6e4fb",
  "harness_sha256": "874e0f0a725a1db5d3245014c6cfda59d17f073092e3616f5fddf4b6a64968f5",
  "memberships_sha256": "63b68fb718db9dfe20ed34bf1a520ed9b80c5b0b6799eaf9be81e4d119636736",
  "freeze_statement_sha256": "[sha256 of the freeze statement file]",
  "weights_and_harness_frozen": true,
  "out_dir": "/home/m0hawk/.local/state/sepalith/sealed-final-v2/panel-v1",
  "attempt_marker": "/home/m0hawk/.local/state/sepalith/sealed-final-v2/attempts/panel-v1-1.json"
}
```

The recipe hash binds the holdback split
(`2b284ba2f39260a2843634674faefd47191ed6a77160c47944cb0cb54ff6c30c`) and the
floors. If the user changes the pipe floor, the recipe hash changes and must
be re-read from main.

**What the freeze receipt (A4) will pin.** After A3 builds, the plan pins
the models, `release_artifacts`, the tokenizer tree, the runtime
(interpreter sha256, distributions, llama-server), and the A3 outputs
copied unchanged from `plan-fragment.json`: `cases` (file pin of
`cases.json`), `case_count`, `semantic_receipt`, `payloads_path`,
`source_roots` and `memberships` (`63b68fb7…`), plus the empty `state_dir`.
The freeze receipt (`sepalith.sealed-final.v2.freeze`) then pins
`plan_sha256` (the exact plan bytes), `weights_sha256` (`3ed055fd…e4fb`),
`harness_sha256` (`874e0f0a…68f5`), both freeze timestamps (taken from the
A2 statement), and `status: frozen` with all three unlock flags. The user
gives the receipt's own sha256 in writing, and the launcher checks it.

## Case fields and files

`cases.json` is `{"split": "final_locked_v1", "cases": [...]}`. Each case
carries exactly the `SourceCaseSpec` fields plus `function_ids`:

| Field | Value |
| --- | --- |
| `row_id` | See [Order](#case-construction) |
| `split_identity` | `<repository_id>::<package_id>::<group_id>::<row_id>`, the guard README's pattern |
| `package_id` | `pkg:<name>`, the DAT-02 identity form and the DEV panels' convention |
| `repository_id` | `pkg:<name>`. This is an explicit package identity, not an invented upstream URL (the v1 rule). |
| `group_id` | DAT-02 group ID |
| `family` | One of the six families |
| `source_path` | `<panel>/final-sources/<sha256>.R`, the staged copy of the case's raw input |
| `source_sha256` | sha256 of those bytes |
| `context_sha256` | `sha256_json(context)`, the guard's canonical JSON of the PRM03 context map after the window |
| `target_sha256` | `sha256_json({"operation": …, "region_new": […]})`. This is not the DEV panels' string hash. |
| `function_ids` | See below. Empty when no function applies. |

**Raw input per family.**

- Propagation and no-op: the normalized R file.
- `format_propagation`: the raw tarball member. Its normalized partner is
  staged too, and both are checked against CPT.
- `finish_block`: the original normalized parent file. Its sha256 is
  rechecked against the harvest's record. The simulated pre-edit document is
  bound by `context_sha256`.

**Function IDs.** Each top-level `name <- function` or `name = function`
whose span meets the region (in the post-event document) or the event (in
the before image) gets the ID `sha256("<Package>\0R/<file>\0<name>")[:16]`.
A finish case's document is a cut prefix, so its ID comes from the harvested
function's name, checked against the complete original source.
That is the exec pool's `function_id` scheme, with the corpus directory's
case.

**Other files**, all write-once and set to mode 0400 in a fresh mode-700
panel directory:

- `payloads.json` maps each `row_id` to `{"context", "target"}`.
- `semantic-receipt.json` has:
  - schema `dat08.semantic-source-case-receipt.v1`;
  - status `source_cases_verified`;
  - split `final_locked_v1`;
  - `source_lock_sha256` set to the sha256 of the exact `cases.json` bytes;
  - `content_access: post_freeze_source_bytes`;
  - `verified_case_ids`;
  - `constructor_sha256`;
  - four-field `case_bindings`;
  - `issuance_mismatches: []`. No registry holds earlier expected hashes for
    these cases.
- `provenance.json` holds the original paths and hashes. It stays private.
- `build-receipt.json` holds aggregate counts, exclusion reason counts, the
  overlap audit aggregate, the dry-run result, the four authorization hashes
  and file digests. It names no case, path or package.
- `plan-fragment.json` holds the plan keys that A3 fixes: `cases`,
  `case_count`, `semantic_receipt`, `payloads_path`, `source_roots` and
  `memberships`. A4 copies them unchanged.

**Staging.** `/mnt/nas/home` is CIFS without `serverino`, and the guard
compares inode identity. The protocol also asks for local inputs on a
filesystem with hard links and fsync. Each selected raw input is therefore
copied once, content-addressed, into `final-sources/`. This keeps package
and file names out of staged paths, so the guard's forbidden-component list
(`eval`, `heldout` and others) cannot be hit, and two inputs never share a
path. `plan.source_roots` is `[<panel>/final-sources]`.

The panel directory
(`~/.local/state/sepalith/sealed-final-v2/panel-v1/`) is separate from the
launcher's empty `state_dir` (`…/run-state/`). The finish intermediate rows
contain final text. They are written to `<panel>.work/` (mode 700) and
deleted when the build ends.

## Contamination

The constructor does not pre-filter groups by contamination. The v2 launcher
counts overlap and reports overlapping groups as a separate subset. The build
does two things:

1. It keeps C05's content guards. No source may be a CPT document; for format
   cases this covers both the raw member and the normalized file. No prompt
   may repeat a DEV prompt.
2. It runs the launcher's own `audit()` on the built cases against the pinned
   memberships, and records the aggregate in the build receipt. The user
   sees whether the 20% group ceiling would stop the run before any freeze
   receipt exists.

The 20% and 5% ceilings are unchanged. DAT-02 unions groups by package name,
URL and parent hash, so final groups are disjoint from TRAIN and DEV by
construction. The byte-level CPT check catches only identical documents, not
near copies.

## Access sequence

| Step | Who | Reads final content | Output |
| --- | --- | --- | --- |
| A0 | This PR | No. It reads dev and train groups only. DAT-02 is loaded as a whole, and only dev and train entries are selected. | Design, constructor, tests, DEV-equivalence receipt, rehearsals; inventories and pins in the state directory |
| A1 | Council, Sol, user | No | Approved design and code at an exact commit |
| A2 | User | No | (a) A written weight and harness freeze statement. (b) A separate build authorization, schema `sepalith.final-panel.v1.build-authorization`, which pins: `status: user-approved`, `split`, `constructor_sha256`, `recipe_sha256`, `weights_sha256`, `harness_sha256`, `memberships_sha256`, `freeze_statement_sha256`, `weights_and_harness_frozen: true`, the panel `out_dir` and an `attempt_marker` path. |
| A3 | Custodian agent, once | **Yes.** It reads the final groups' DAT-02 entries and those packages' sources and tarballs, under `checked_tree`. | The panel directory listed above |
| A4 | User | No | Plan finalized (`cases`, `case_count`, `semantic_receipt`, `memberships`, `source_roots`, `payloads_path`), then the freeze receipt |
| A5 | Launcher | Yes, as v2 specifies | `result.json` |

**A3 in order**, all in one process with fixed error text:

1. Verify the authorization file's exact sha256 and its constructor and
   recipe hashes.
2. Verify every module pin and the module closure: every imported repository
   module outside the package tree must be pinned. Also verify the
   memberships hash, the tokenizer file hashes, `protocol.harness_hash()`,
   `sha256_json` of the plan's models object and the freeze statement's file
   hash against the authorization. Compare them by value. Rehearsals check
   the DAT-02 pin too, because its split labels decide what may be read.
3. `build_final()` is the only entry point that reads final content. It
   re-verifies the receipt from disk and loads every input it consumes from
   the recipe's pins. It then creates the attempt marker exclusively, before
   any final metadata or source is read. The marker may not lie inside the
   output or work tree. `build_panel()` refuses the final split. The work
   directory is removed in a `finally` block.
4. Check the DAT-02 and DEV-panel pins.
5. Build.
6. Write the outputs.
7. Delete the work directory. If any step fails, the unpublished panel
   directory is deleted unread as well, and the attempt marker stays.
8. Run the **launcher dry run**: the v2 launcher's own `load_cases`, freeze
   and semantic-receipt validation, root and source-path checks,
   `O_NOFOLLOW` source reads and `construct()`, with the campaign tokenizer.
   It must give zero hash mismatches and construct every case. The real run
   takes its seal before these steps, so a defect found then would leave a
   dead run.

**Pre-registered outcomes of A3.** None of them may lead to a changed recipe
or a rerun that depends on what A3 showed.

- **Built:** go to A4.
- **Stopped below a floor** (`stopped_below_floor`): nothing is staged or
  written except the aggregate build receipt. The sealed final for C17 is
  abandoned unless the user decides otherwise. A new recipe is not an
  option.
- **Overlap above the 20% ceiling** (`stopped_over_overlap_ceiling`): the
  unchanged launcher would abort after taking its seal, so the sealed final
  for C17 is abandoned. A new recipe is not an option.
- **Crash from a code defect:** partial outputs are deleted unread. The
  defect is reproduced and fixed on dev data only. A rerun needs a new
  constructor hash, a review and a new authorization. The previous attempt
  marker stays as the record.
- **Environment failure** (mount, disk or permission) that is reproducible
  without any code change: partial outputs are deleted unread. The same
  constructor may run again under a new authorization and a new marker.
  Every content-free check (pins, tokenizer, CPT index, source roots, module
  closure) runs before the marker is claimed, so these failures normally
  stop the build before then. A refused package tree or an unreadable
  tarball is a counted skip, not a crash.

A3 replaces the lost selection file. The selection is a pure function of:

- the pinned constructor and recipe;
- DAT-02 v2;
- the corpus bytes;
- the CPT index;
- the DEV prompt hashes.

Determinism is shown on dev, not by a second final read. Two dev rehearsals
under different `PYTHONHASHSEED` values gave identical outputs.

## Inventories and pins

These are built in A0 by `final-panel-v1/inventories.py` with no final
content. They are written only to `~/.local/state/sepalith/sealed-final-v2/`
(mode 700). `memberships.json` has exactly the launcher's eight keys and five
lists. `coverage.json` records each stage's sources, rule and counts.

| Key | Source |
| --- | --- |
| `train` | Every DAT-02 group whose label is neither `dev_group` nor `final_candidate_group`: `train_group`, `quarantine_sft_v7`, `quarantine_historical_eval`, `quarantine_tu3`, `excluded_source_only`. The groups' `pkg:` forms go to `package_ids` and all identity forms to `split_identities`. Also every Arm A SFT row (DAT10 v4 braceindent) with its package, and the C18 SFT replay IDs. |
| `dev` | DAT-02 `dev_group` groups and forms, plus DEV75 and DEV250 row, package and group IDs |
| `cpt` | Every CPT-prefix-extension-v1 combined row (row, package, group, and document ID) and the C18 CPT replay |
| `c14` | Every C14 GRPO pool row, with packages joined from Arm A rows |
| `synth` | Every SYNTH v1, v2, v3 and v3.1 case row and seed package |
| `exec_train` | Train-role EXEC-ENV-v1 slices (function IDs, packages), plus the C18 train tasks and the C19 hard mix |
| `exec_validation` | The C18 validation tasks (C19 reuses them) |
| `exec_holdout` | Hold-out-role slice identities in `slices.json` only (function IDs and packages). No hold-out task, pool row or freeze file is opened. |

**Pins:**

- the bf16 checkpoint trees for `arm-a-803` and `c19-100-bf16`, every file;
- the Q6_K and mixB-e4 GGUFs;
- the campaign tokenizer tree;
- the interpreter's sha256 and the distribution versions;
- llama-server `3cb7ffb1a` (the CPU build).

## Pre-registered criteria

The v2 protocol's criteria and comparisons are unchanged:

- **Primary metric:** the fraction of admitted clean cases with an exact edit
  or a correct no-op.
- **Secondary counts:** unchanged.
- **Bootstrap:** a package bootstrap with 10,000 replicates and seed 3407.
- **Paired differences:**
  - C19 bf16 minus Arm A;
  - Q6_K minus C19 bf16;
  - adopted C20 minus C19 bf16. C20 was not adopted on 2026-11-23.
- **Ceilings:** the 20% overlap ceiling and the 5% hash-mismatch ceiling.

This design adds no criterion.

## Council review (2026-11-23)

Claude Fable 5.2, Gemini 4 Neon and GPT Astra 6.2 reviewed the design, all
at high effort.

| Delegate | Round 1 (design `98a777d0`) | Round 2 (design and code `4bcf7159`) |
| --- | --- | --- |
| Claude Fable 5.2 | 6.5/10 | 8/10, "approvable for the user's decision" |
| Gemini 4 Neon | 6.5/10 | 8/10, "approvable for the user's decision" |
| GPT Astra 6.2 | 6/10 | 7/10 |

**Adopted after round 1:**

- **A2 bindings:** the authorization binds the weights, harness, memberships
  and freeze statement, and claims a one-shot attempt marker.
- **A3 outcomes:** every outcome is pre-registered.
- **Dry run:** a pre-seal launcher dry run checks the outputs.
- **Explicit adapter widening:** the DAT-04B split gate is widened in memory
  only, under the authorization.
- **Staging:** sources are content-addressed, and the panel directory is
  separate from the launcher's state directory.
- **Parse gates:** C05's R parse gates are applied to propagation and finish.
- **Allocation:** pass 2 is augmenting only, no-op shortfalls are not moved,
  and per-family floors are added.
- **Determinism and order:** version picks are deterministic, mixed EOL is
  skipped up front, tarball members are checked, row IDs are deduplicated,
  and the order of steps is written down.
- **Coverage:** the train inventory covers every non-dev, non-final label,
  coverage is recorded per stage, and no-op equivalence is shown field for
  field.

**Adopted after round 2:**

- **One-shot claim:** `build_panel` itself re-verifies the authorization from
  disk and claims the marker. Every content-free check runs before the claim.
- **Values, not shapes:** the models and the freeze statement are compared by
  value.
- **Module closure:** the transitively imported `rewrite_*` modules are
  pinned, and the closure is checked.
- **Read order:** `checked_tree` runs before `DESCRIPTION` is read, and the
  finish harvest is pinned to the checked version.
- **Counted skips:** a refused package tree or an unreadable tarball is
  counted, not fatal.
- **Stops write nothing else:** below-floor and over-ceiling stops write only
  the aggregate receipt, and a failed build deletes its partial outputs
  unread.
- **Finish function IDs** come from the full source, and only `<-` and `=`
  count as assignments.
- **`plan-fragment.json`** is written at A3.
- **Normalized format files** get strict UTF-8 and EOL checks.
- **Inventory parsing:** the field parser cannot pick up nested keys, and
  missing identities are rejected.

**Declined or deferred, as follow-ups:**

- **Reserving unread final groups.** Decided by the user on 2026-11-23:
  [hold back 88 of 193](#holdback-split-pre-registered).
- **DEV250 reference rates at the final family mix.** Done:
  [Family mix and DEV250 reference](#family-mix-and-dev250-reference). No
  per-family final rate will be reported.
- **A full A3 dress rehearsal through a dev-split authorization.** Done
  before A2: [Dress rehearsal](#dress-rehearsal-2026-11-23).
- **Restoring the adapter's `ALLOWED_SPLITS`.** One build runs per process.
- **Sol S7, no-op sites inside multiline strings.** Declined. DEV250's own
  no-op cases include such lines, for example the `retentionmort_figure.R`
  lines inside a `cat("…")` R Markdown string. A syntax-aware exclusion would
  break fidelity to DEV. The target is the unchanged line either way.
- **A content-free `check_plan` at A4.** `plan-fragment.json` covers the
  A3-fixed keys.

## Open questions

All five were answered by the user on 2026-11-23 (see
[User decisions](#user-decisions-2026-11-23)):

1. Harness pin: re-pinned to main `815967bb` (option b).
2. Roxygen gap: accepted; the 11 slots go to `finish_block`.
3. Code-cut-only finish: accepted; the report states the mix.
4. Order of freeze and build: unchanged. A2 is a written freeze statement
   followed by a separate build authorization.
5. Unselected final groups: 88 of 193 are held back.
