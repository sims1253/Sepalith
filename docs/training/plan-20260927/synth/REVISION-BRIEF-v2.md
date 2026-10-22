# Seed revision brief (SYNTH-v2)

You revise ONE batch of existing seed files in place under `seeds-v2/<batch>/`.
The original (v1) brief is `AUTHORING-BRIEF.md`; its rules still hold
(original code, every file parses, ASCII only, no tabs, lines < 100 chars,
no `>>>>>>>`/`<<<<<<<`/`=======`/`[NO_EDIT]` strings) EXCEPT rule 5 (no blank
lines in bodies), which is replaced below. Do not touch `seeds/` (v1) or
any other batch. Do not use git. Use `nice -n 10` and at most 2 threads.

Why: the cut-off completions made from these files must look like real R
code. Reviewers found v1 bodies dense and comment-free (0.1% of targets had
a blank line vs 38.8% in the real training data), which teaches a house
style that conflicts with real golds.

## 1. Whitespace and comments (main task)

Edit every function body so its layout matches real package code. Measured
on the real training targets: about 7% of body lines are blank and about 5%
are full-line `#` comments; short bodies (under ~6 lines) rarely have
either; 13-20-line stretches have a blank line in ~46% and a comment in
~39% of cases; 30+-line stretches in ~65%.

- Put single blank lines between logical steps (validation / computation /
  assembly of the result, before a loop, before the final return value),
  where a careful developer would. Never two blank lines in a row, never a
  blank line directly after `{` or directly before `}`.
- Add full-line `# ...` comments where a real developer would explain intent
  (a step, a non-obvious choice, a TODO-free note). Vary density between
  files and authors: some files sparse, some well commented. Do not comment
  every line, and do not restate the code verbatim.
- Keep code semantics unchanged except for bug fixes (section 2).

## 2. Correctness review and fixes

Read every function critically and fix real bugs (NA handling, character vs
byte counts, NULL/empty inputs, wrong indexing, list vs data frame
confusion, off-by-one). Known findings to fix where they are in your batch:
- b18-web `relaydesk/R/webhooks.R` `safeEquals()`: compares by character
  count but iterates bytes; `safeEquals("é", "a")` returns NA. Make it
  byte-based (`charToRaw`) and return FALSE for unequal lengths.
- b03-datatable-io `logsift/R/csv_sniff.R` `looksNumeric()`:
  `looksNumeric(c("1", NA_character_))` returns FALSE; NA entries must be
  ignored (consistently with the sibling `looks*` functions).
- b23-simulation `herdwalk/R/agents.R` `summary.herd_population()`:
  `unname(counts) / nrow(object)` on a list-like object is dubious; use the
  population's actual size accessor.
- b13-geo `terraslab/R/leaflet.R` `quickRasterMap()`: `bounds` derived from
  `polygons`, which may be NULL; handle the NULL case.
- b05-oop `foliokit/tests/testthat/test-ledger.R`: tests must assert only
  behaviour the code actually has (e.g. currency defaults); fix any that
  don't.

## 3. Behavioural checks (new)

For each `R/*.R` (or script) file whose functions run in base R alone, add a
check file `checks/<same relative path>` (e.g. `seeds-v2/b02-base-stats/
checks/fitwright/R/bootstrap.R`). The harness sources the seed file into a
fresh environment, then evaluates the check file in that environment with
`Rscript --vanilla` (R 4.6.1, base and recommended packages only). Write tiny
example calls with `stopifnot()` assertions for the base-R functions (2-4
assertions per function is plenty). Skip functions that need add-on packages
(dplyr, data.table, shiny, R6, ggplot2, httr2, DBI, ...); list them in a
comment `# skipped: needs <pkg>`. A seed file that cannot be sourced without
add-on packages gets no check file. Run each check file yourself:

    nice -n 10 Rscript --vanilla -e 'e <- new.env(); sys.source("SEED.R", e); sys.source("CHECK.R", e)'

and make it pass (fix the seed if the check reveals a bug).

## 4. Roxygen conventions

Blocks in v1 were monotone. Revise them toward the real tag frequencies
(share of blocks with the tag): `@param` 93%, `@return` 60%, `@export` 44%,
`@keywords internal` 38%, `@description` 26%, `@noRd` 22%, `@rdname` 21%,
`@title` 10%, `@details` 6.5%, `@importFrom` 6%, `@inheritParams` 5.5%,
`@family` 4%. In practice: drop most `@family` tags; leave `@return` off
some internal helpers; mark internal helpers `@keywords internal` or `@noRd`
instead of `@export`; in one package per batch use explicit `@title` /
`@description` tags; group some siblings with `@rdname`; vary tag order
between packages (e.g. `@param ... @return ... @export` in one,
`@description ... @param ... @keywords internal` in another). `@param` names
must still match the signature in order.

When done, reply with: files changed, number of blank and comment lines
added (rough), bugs fixed (file: function: what), check files written and
whether all pass, and anything you could not do.
