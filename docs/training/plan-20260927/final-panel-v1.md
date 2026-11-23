# Final panel v1: a replacement constructor for the sealed final

**Status: design, awaiting user approval.** This document and its code do not
read final content. Building the panel is a later step. It needs the user's
written weight and harness freeze and a separate written authorization (see
[Access sequence](#access-sequence)).

## Why

On 2026-11-23 the sealed-final prep for
[C17 step 4](cards/C17-release-prep.md#4-recover-and-freeze-the-sealed-final-protocol)
found that `DAT-07-final-evaluator-cases.jsonl` is the DEV75 panel and not a
final panel. All 75 rows have `split: dev` and `source_ref.split: dev_group`.
The registry preflight reports `preflight_missing_approval` with 0 rows
admitted. The real final split is the 193 `final_candidate_group` groups in
`DAT-02-global-split-v2.json` (sha256 `c4e1219a…8f09`). The v1 route from
those groups to cases is lost: `r2-final-assembly-v1/assemble_inputs.py`
survives in git, but its constructor modules (`final_constructor.py`,
`raw_source_case_builder.py`, `raw_source_tokenization.py`) and its selection
file (sha256 `f8037b71…1e1e`) went with the 2026-10-10 WSL reinstall. On
2026-11-23 the user decided: "Build a new final panel."

This constructor turns those 193 never-used groups into cases. It produces the
same task families and the same PRM03 case format as DEV75 and DEV250, so the
[sealed-final v2 protocol](sealed-final-v2.md) scores them unchanged.

## What DEV75 and DEV250 are, and what survives

The DEV panels are `PromptContext` objects (`sepalith.prompt.prm03.v1`) plus a
target `{operation, region_new}`. The prompt is `render_prompt(context)` and the
target is `serialize_target(operation, region_new)`. The recorded
`prompt_sha256` and `target_sha256` of all 250 DEV250 rows and all 75 DEV75
rows are the SHA-256 of those two strings, so they can be checked without a
tokenizer.

The C05 DEV250 builder (`c05/build_dev250_packets.py`, `c05/build_dev250.py`)
built each family from the raw parent sources of the dev groups. Some of its
modules were lost in the reinstall: `campaign_dev_propagation.py` and
`campaign_dev_panel.py` from the `tuesday-execution` worktree. The other parts
are still in git:

| Part | In git | Pin |
| --- | --- | --- |
| Rename, pipe, na_rm and format extractors | `experiments/synthetic-data/scenarios.py` | sha256 `cccf8ddf…250c` (the canonical-scenario pin in the DAT-08 review and the corrected-DEV75 manifest) |
| Code-cut finish rule | `experiments/synthetic-data/cases/rules/rules_finish_block.py` via `c05/cut_dev_finish.py` | file hashes recorded in the build receipt |
| DAT-04B completion adapter `dat04b-completion-adapter-v3` | `docs/campaign/work/lead/r2-task-materialization-a/reviewed-source/…/campaign_admission_completion.py` | sha256 `fafb852f…73cc` |
| DEV75 brace correction | `docs/campaign/work/finish-boundary-repair-v2/finish_boundary_repair_v2.py` | file hash in the build receipt |
| PRM05 source window | `sepalith.training.rl.campaign_token_audit.selected_context` | in the evaluator tree |

The constructor reimplements the two lost pieces:

- the converter that turns a scenario example into a post-event document,
  history event, region and cursor;
- the no-op site rule.

It reuses everything else unchanged.

### Reproduction on dev data (no final content)

These checks ran against dev-split data only, using DEV250 and its phase-1
artifacts on E:. The receipt is
`receipts/C17-final-panel-dev-equivalence.json`.

| Family | Check | Result |
| --- | --- | --- |
| rename, pipe, na_rm, format | Extractors run on the raw dev parent file regenerate the DEV250 derived row | 78 / 78 |
| rename, pipe, na_rm, format | Converter plus PRM05 window gives the DEV250 context field for field (`event_id` aside) | 78 / 78 |
| rename, pipe, na_rm, format | Same, against DEV75's recorded prompt and target hashes | see receipt |
| no_op | Site rule plus C05's round-robin gives the same phase-1 candidate set per dev group | 25 / 25 groups |
| finish_block | Code-cut harvest gives byte-identical derived rows | all eligible dev groups |
| finish_block | Adapter plus brace correction plus window gives the DEV250 context and target | 111 / 111 |

The rendered prompt never contains `event_id` or the range URIs. The
constructor derives both from the row ID, and they do not change any prompt.

`roxygen_drafting` cannot be built this way. DEV75 and DEV250 took their
roxygen cases from existing rows in the shared `scenarios_v1` batteries.
DAT-02 marks those batteries final-ineligible ("final-ineligible until
regenerated from raw parents"). The v1 assembly listed roxygen as a gap ("no
admitted constructor"). This design keeps it a gap (see
[Open questions](#open-questions)).

## Inputs

| Input | Identity | Use |
| --- | --- | --- |
| DAT-02 global split v2 | `/mnt/e/…/DAT-02-global-split-v2.json`, sha256 `c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09` | Group IDs, split labels, `identity_forms`, `flags` |
| Normalized CRAN corpus | `/mnt/nas/home/sepalith/normalized` (the real path behind `/mnt/h/sepalith/normalized`) | Parent sources `<Package>/<version>/<Package>/R/*.R`, `DESCRIPTION` |
| CRAN tarballs | `/mnt/nas/home/sepalith/tarballs/<Package>_<version>.tar.gz` | Raw pre-format sources for `format_propagation` |
| CPT document index | `/mnt/e/…/CPT-prefix-extension-v1/combined-cache/index.sqlite3` | Skip any source file whose sha256 is a CPT document (C05 rule) |
| DEV75 and DEV250 | DEV75: the `bf278755…` file recorded in the preflight. DEV250 `75f3a5ee…` | Prompt hashes that a final case must not repeat |
| Campaign tokenizer | `TOKENIZER_JSON_SHA256` and `TOKENIZER_CONFIG_SHA256` pins | Length filters through `build_training_row` |

**Parents.** A group's parents are its `pkg:<name>` identity forms. Each
must match exactly one directory of the normalized corpus, comparing
`NFKC(name).lower()`. The version directory is picked by
`build_astfim.pick_version_dir`, as in C05. On the dev split, this rule
reproduces all 165 parents of `DAT-07-dev-parent-allowlist.json` and leaves
out the 4 dev groups that have no CRAN parent. The rule is unchanged for
final groups. A final group is eligible when its split is
`final_candidate_group`, it has no `flags` (as in the v1 assembly), and it
has at least one parent with a `DESCRIPTION` `License:` field.

## Case construction

Per parent, files are visited in sorted order. A file is skipped if it is
empty, over `scenarios.MAX_FILE_BYTES`, not UTF-8 (lone CR included), or a CPT
document by sha256.

| Family | Constructor | Operation |
| --- | --- | --- |
| `rename_propagation` | `scenarios.extract_rename(cap=4)` with `Random(seed_for(group, path, family))`, then the converter | `replace` |
| `pipe_rewrite` | `scenarios.extract_pipe(cap=4)`, then the converter | `replace` |
| `na_rm_propagation` | `scenarios.extract_na_rm(cap=4)`, then the converter | `replace` |
| `format_propagation` | Tarball member `<Package>/R/<file>` against the normalized file, `scenarios.format_pairs_from_lines`, then the converter | `replace` |
| `no_op` | A non-comment line whose string-stripped text contains `na.rm = TRUE` (`already_applied_na_rm_true`) or ends in `\|>` (`already_applied_native_pipe_operator`). Cursor at end of line, empty history, `region_new = region_old`. | `no_op` |
| `finish_block` | C05 code-cut harvest: at most 6 bases per package, one per file, C05 body window and plain transforms. One transform per base by `Random(seed_for(group, base))`. Then the DAT-04B adapter and the DEV75 outer-brace correction. | `replace` |

`seed_for(*parts) = int(sha256_json([3407, *parts])[:16], 16)`.

The converter applies the example's one-hunk event to the before image. It
locates the region from the example's prefix and `region_old` (exactly one
match, or the candidate is dropped). It then builds the post-event document,
one `observed_edit_diff` history event, the full-line replacement range and
the cursor, and hands the whole document to `selected_context`, the PRM05
window with a 6,000 UTF-16 unit budget. Every case goes through
`PromptContext.from_mapping` and `build_training_row`.

Per (family, group), at most 3 candidates are kept, as in C05. No-op sites
are taken round-robin across files (first hit of each file, then the second,
and so on), as C05 did.

**Filters**, as in C05 phase 2:

- the response with terminal and EOS is at most 1,024 tokens;
- the sequence is at most 4,096 tokens;
- one case per rendered prompt;
- no prompt equal to any DEV75 or DEV250 prompt.

## Sampling, size and caps (pre-registered)

All values are fixed here, before any final content is read. They copy
DEV250's sizing.

| Slot | Quota | DEV250 |
| --- | ---: | ---: |
| `finish_block` | 122 | 111 (55 authored + 56 code-cut) |
| `no_op` | 50 | 50 |
| `pipe_rewrite` | 28 | 28 |
| `format_propagation` | 24 | 24 |
| `rename_propagation` | 23 | 23 |
| `na_rm_propagation` | 3 | 3 |
| `roxygen_drafting` | 0 (gap) | 11 |
| **Total** | **250** | 250 |

Roxygen's 11 slots move to `finish_block`, TRAIN's largest family, following
the C05 2026-10-03 precedent. The authored half of DEV250's finish cases was
written by an LLM and is not reproducible. Every final finish case is code-cut.

- **Group cap:** at most 3 cases per DAT-02 group, as in C05. 250 cases need
  at least 84 groups, and 193 groups are available.
- **Allocation:** C05's deterministic Edmonds-Karp max-flow
  (source → slot → group → sink), copied unchanged. Candidates enter in
  `sha256_json([3407, row_id])` order.
- **Shortfall:** if an edit family cannot fill its quota, its unfilled slots
  move to `finish_block` and the max-flow runs once more. If `finish_block` is
  still short, the panel is smaller than 250. The receipt records every
  shortfall.
- **Floor:** the build stops for the user, writing aggregate counts only, if
  the panel has fewer than 200 cases, fewer than 40 no-ops, or fewer than 84
  groups.

There is no tuning loop. The build runs once. A rerun must give the same bytes.

## Case fields

Each case in `cases.json` carries every `SourceCaseSpec` field plus
`function_ids`:

| Field | Value |
| --- | --- |
| `row_id` | `final-v1-` + the first 24 hex characters of `sha256_json([3407, group_id, package, relative_path, family, site])`. `site` is the region line and constructor, or the finish base and transform. It holds no source text. |
| `split_identity` | `<repository_id>::<package_id>::<group_id>::<row_id>`, the guard README's pattern |
| `package_id` | `pkg:<name>`, the DAT-02 identity form and the DEV panels' convention |
| `repository_id` | `pkg:<name>`. This is an explicit package identity, not an invented upstream URL (the v1 rule). |
| `group_id` | DAT-02 group ID |
| `family` | One of the six families above |
| `source_path` | Absolute path of the staged copy of the case's before image, under the build's `final-sources/` |
| `source_sha256` | sha256 of those bytes. For format cases this is the raw tarball member. |
| `context_sha256` | `sha256_json(context)`, the guard's canonical JSON of the PRM03 context map after the window |
| `target_sha256` | `sha256_json({"operation": …, "region_new": […]})` |
| `function_ids` | Each top-level `name <- function` whose span meets the region or the event: `sha256("<Package>\0R/<file>\0<name>")[:16]`, the exec pool's `function_id` scheme. Empty when no function applies. |

`payloads.json` maps each `row_id` to `{"context", "target"}`.

**Staging.** `/mnt/nas/home` is a CIFS mount without `serverino`, so its inode
numbers are generated by the client and need not survive a remount. The guard
compares inode identity and `O_NOFOLLOW` paths. The protocol also asks for
local inputs on a filesystem with hard links and fsync. The build therefore
copies each selected case's source bytes once into
`<state>/final-sources/<Package>/<version>/…`, read-only. For format cases it
also copies the normalized after file, and records that file's sha256 in the
build receipt. The plan's `source_roots` is that directory.

## Contamination

The constructor does not pre-filter groups by contamination. The v2 launcher
counts overlap and reports overlapping groups as a separate subset. The build
does two things:

1. It keeps C05's two content guards: no CPT source document, and no
   DEV-repeated prompt.
2. It runs the protocol's own `audit()` over the built cases and the eight
   membership inventories, and writes only the aggregate (`overlap_groups`,
   `overlap_groups_by_stage`). The user sees then whether the 20% group
   ceiling would stop the run, before any freeze receipt exists.

The 20% group ceiling and the 5% hash-mismatch ceiling stay as written in
the v2 protocol.

The eight inventories are built now (see
[Inventories](#inventories-and-pins)). Final groups are disjoint from TRAIN
and DEV by construction in DAT-02, because the same package name, URL or
parent hash unions groups. Overlap can still come from cross-package file
copies (caught by the CPT check) and from exec function IDs.

## Access sequence

| Step | Who | Reads final content | Output |
| --- | --- | --- | --- |
| A0 | This PR | No. It reads dev and train groups only. It also reads DAT-02 as a whole JSON object, but only dev and train entries are selected. | Design, constructor, tests, DEV-equivalence receipt; inventories and pins in the state directory |
| A1 | Council, Sol, user | No | Approved design |
| A2 | User | No | Written weight and harness freeze, and a separate written panel-build authorization naming the constructor file hash |
| A3 | Custodian agent, once | **Yes.** It reads the final groups' DAT-02 entries and those packages' source and tarball bytes once, under the allowlist and `O_NOFOLLOW`. | `final-sources/`, `cases.json`, `payloads.json`, a semantic receipt (`content_access: post_freeze_source_bytes`, `issuance_mismatches: []`), and a build receipt with aggregate counts only |
| A4 | User | No | Plan finalized (cases hash as `source_lock_sha256`), freeze receipt |
| A5 | Launcher | Yes, as v2 specifies | `result.json` |

A3 replaces the lost selection file. The selection is a pure function of:

- the pinned constructor;
- DAT-02 v2;
- the corpus bytes;
- the CPT index;
- the DEV prompt hashes;
- the seed and the quotas above.

The recipe of those pins goes into git now as `final-panel-v1/recipe.json`,
with no final identities. In A3, the `cases.json` hash becomes the source
lock. Construction and binding issuance are one step, because the constructor
computes the three hashes from the bytes it reads. No registry holds prior
expected hashes, so `issuance_mismatches` is empty.

At A3, the CLI refuses `final_candidate_group` without three flags:

- `--authorization` (a JSON file);
- `--authorization-sha256`;
- `--state` (a mode-700 directory).

The authorization must name `constructor_sha256`, `recipe_sha256` and
`status: "user-approved"`. Errors print a fixed message and no exception
text, as `protocol.main` does.

## Inventories and pins

These are built in A0 with no final content. They are written only to
`~/.local/state/sepalith/sealed-final-v2/` (mode 700).

| Key | Source |
| --- | --- |
| `train` | DAT-02 `train_group` and `quarantine_sft_v7` groups (IDs, `pkg:` names, all identity forms), plus every Arm A SFT row (DAT10 v4 braceindent) and its package |
| `dev` | DAT-02 `dev_group` groups, plus DEV75 (the `bf278755…` file from the preflight; identity keys only) and DEV250 row IDs, packages and groups |
| `cpt` | Every CPT-prefix-extension-v1 combined row: row, package, group, and document ID |
| `c14` | Every C14 GRPO pool row, with packages joined from Arm A rows |
| `synth` | Every SYNTH v1, v2, v3 and v3.1 case row and package |
| `exec_train` | Train-role EXEC-ENV-v1 slices (function IDs, packages), plus the C18 train tasks and the C19 hard mix |
| `exec_validation` | C18 validation tasks (C19 reuses them) |
| `exec_holdout` | Hold-out-role slice identities in `slices.json` only (function IDs and packages). No hold-out task file is opened. |

Pins: the bf16 checkpoint trees for `arm-a-803` and `c19-100-bf16`, the Q6_K
and mixB-e4 GGUFs, the campaign tokenizer tree, the interpreter hash, the
distribution versions, and llama-server `3cb7ffb1a` (CPU build).

## Pre-registered criteria

The v2 protocol's criteria and comparisons are unchanged:

- the primary metric is the fraction of admitted clean cases with an exact
  edit or a correct no-op;
- the secondary counts are unchanged;
- the bootstrap is a package bootstrap with 10,000 replicates and seed 3407;
- the paired differences are C19 bf16 − Arm A, Q6_K − C19 bf16, and adopted
  C20 − C19 bf16 (C20 was not adopted on 2026-11-23);
- the ceilings are the 20% overlap ceiling and the 5% hash-mismatch ceiling.

This design adds no new criterion. The build floor above is a construction
stop, not a pass criterion.

## Open questions

1. **Roxygen gap.** Accept a final panel without `roxygen_drafting`, with its
   11 slots moved to `finish_block`? The alternative is a separately reviewed
   raw-source roxygen constructor (`sepalith.data.synth.roxygen` exists, but
   its geometry is the SYNTH one, not DEV's).
2. **Code-cut-only finish cases.** DEV250's finish half was authored by an
   LLM. Final finish cases are code-cut only. Accept?
3. **Order of freeze and build.** The build reads final source bytes, so it
   comes after the weight and harness freeze (A2). The user then reviews the
   plan, which pins the built `cases.json`, and issues the freeze receipt
   (A4). Is A2 a separate written statement, or should the user defer the
   freeze statement until A4?
