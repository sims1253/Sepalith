# Final panel v1: a replacement constructor for the sealed final

**Status: design and code, awaiting user approval.** Neither reads final
content. Building the panel is a later step. It needs the user's written
weight and harness freeze and a separate written build authorization (see
[Access sequence](#access-sequence)). Code:
[`docs/campaign/work/final-panel-v1/`](../../campaign/work/final-panel-v1/).

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

- **Reserving unread final groups.** The user decides this (open question 5).
- **DEV250 reference rates at the final family mix.** These can be computed
  from existing DEV250 results before A3. No per-family final rate will be
  reported.
- **A full A3 dress rehearsal through a dev-split authorization.** It is
  recommended before A2. The final-only CLI branches are thin and are
  covered by unit tests of `authorize`, `claim_attempt` and the final
  `build_panel` path.
- **Restoring the adapter's `ALLOWED_SPLITS`.** One build runs per process.
- **Sol S7, no-op sites inside multiline strings.** Declined. DEV250's own
  no-op cases include such lines, for example the `retentionmort_figure.R`
  lines inside a `cat("…")` R Markdown string. A syntax-aware exclusion would
  break fidelity to DEV. The target is the unchanged line either way.
- **A content-free `check_plan` at A4.** `plan-fragment.json` covers the
  A3-fixed keys.

## Open questions

1. **Harness pin (blocker before A2).** `protocol.harness_hash()` requires
   `packages/sepalith/src` to equal `523efc07`. Main has since added 11 C20
   files (`training/sft/c20/*`, `training/c20_exception/*`), so it raises
   `evaluator_tree_differs_from_pin` on main and on this branch. The fix
   is either:
   - (a) run A3 and A5 from a checkout whose package tree equals the pin and
     which carries this constructor; or
   - (b) re-pin `EVALUATOR_COMMIT` under review. That gives a new harness
     hash. Alternatively, limit the check to the evaluator files the harness
     uses.

   This choice is the user's.
2. **Roxygen gap.** Accept a panel without `roxygen_drafting`, with its 11
   slots in `finish_block`?
3. **Code-cut-only finish.** Accept 122 code-cut finish cases in place of
   DEV250's authored and code-cut halves?
4. **Order of freeze and build.** This design keeps A2 as a separate written
   freeze statement before A3. A4 reuses its timestamps in the freeze
   receipt.
5. **Unselected final groups.** A3 reads all 193 groups' sources to find
   supply, and about 105 groups are expected to supply cases. Should a
   hash-ordered share of groups stay unread for a later release? That would
   need a smaller recipe, decided before A3.
