# Seed bug sweep (SYNTH-v3.1)

Reviewers found two semantic bugs in the v3 seeds. Both were fixed by hand
(`b36-shiny/dashkit/R/mod_valueboxes.R` `kpi_targets`, and
`b45-sql/rowcast/R/validate.R` `check.params`); look at those diffs with
`git diff` (read-only) for reference. Your job is to sweep the batches
given in your task for the **same two bug classes**, fix every instance you
find, and add cheap counterexample checks.

## Bug class 1: one-sided range validation
The roxygen, a comment, the argument name or a sibling check says a value
must lie in a range (a share or proportion in [0, 1], a probability, a
percentage 0-100, a month 1-12, an hour 0-23, latitude -90..90, a
day-of-week 1-7, a quantile level in (0, 1), a positive bound below an
upper bound, ...), but the code only checks one side (e.g. `x < 0`). Fix the
check to enforce the documented range, with a message following the file's
existing message template. Do not invent new constraints that the docs or
names do not imply.

## Bug class 2: parsing raw text instead of the cleaned text
A function computes a cleaned/normalised/stripped version of its input (comments
or literals removed, whitespace squeezed, case folded, escapes removed,
quoted parts masked), but some branch or later step still reads the raw
input where the cleaned text was intended (or vice versa), so literals,
comments or escaped parts leak into the result. Fix the branch to use the
right variable.

## Checks
For every fix, add counterexamples to the file's check script
(`<batch>/checks/<same path>`): the boundary values (both ends valid, just
outside both ends an error) for class 1; an input whose literal/comment/escaped
part would have fooled the old code for class 2. Update any existing
assertion that encoded the buggy behaviour. Run every changed check with

    nice -n 10 Rscript --vanilla -e 'e <- new.env(); sys.source("SEED.R", e); sys.source("CHECK.R", e)'

and make it pass. Keep everything else unchanged: no refactors, no new
patterns, keep the file style, blank lines, comments and roxygen (update
roxygen only where the fix changes documented behaviour). Hard rules from
AUTHORING-BRIEF.md still apply (ASCII, no tabs, lines < 100, parse). No git
writes; `nice -n 10`, at most 2 threads; touch only your batches.

Reply with a table: file, function, class, what was wrong, what changed,
check lines added; plus the list of files you reviewed with no finding.
