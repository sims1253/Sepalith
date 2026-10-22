# Seed authoring brief (SYNTH-v1)

You are writing ORIGINAL R source files that a data pipeline will cut into
next-edit training cases for a small R code-completion model. The pipeline
cuts functions at statement boundaries ("finish this block"), removes roxygen
blocks ("draft the docs"), and reformats statements. Quality of the code is
the whole point: it must look like real code a competent R developer wrote.

## Output
- Write files under the directory given in your task, laid out like a real
  package or project: `<pkg>/R/<topic>.R`, `<pkg>/tests/testthat/test-<topic>.R`,
  `<app>/app.R`, `<project>/analysis/<step>.R`, etc. Invent fictional package
  names (lowercase, plausible, not an existing CRAN package name you know).
- Every file must parse: run `Rscript -e 'invisible(parse("FILE"))'` on each
  file you write and fix any error. (R 4.6 is installed; add-on packages are
  NOT installed, which is fine: we only parse, never run.)
- Do not run any formatter; the pipeline formats with `air`.

## Hard rules
1. Original code only. Do not reproduce functions from CRAN/GitHub packages
   from memory. Calling well-known package APIs (dplyr::mutate, ggplot2,
   data.table, shiny, R6, testthat, Rcpp wrappers, etc.) is expected; copying
   their implementations is not.
2. Each function must be complete, coherent and plausible: it uses its
   arguments, has realistic control flow, and returns something sensible.
   No placeholders, no `TODO: implement`, no lorem ipsum.
3. Functions should vary in length: roughly 30% short (3-10 body lines),
   50% medium (10-30), 20% long (30-60). Avoid >70-line functions.
4. Determinism matters. In every file, include at least two GROUPS of
   related functions that share structure (e.g. a family of validators, a
   getter/setter set, per-format readers, print/format/summary methods,
   parallel test blocks, symmetric x/y helpers, per-type S4 methods). Later
   members of a group should mirror earlier ones closely, the way real code
   does, so a reader could predict how a later one ends from the earlier ones.
   Also write functions whose endings are predictable from their own start:
   accumulate-then-return, validate-then-compute, loops that fill a
   preallocated result, `switch()` over `match.arg()` choices listed in the
   signature, `tryCatch` with symmetric handlers, early returns, etc.
5. Inside function bodies, use NO blank lines at all. (The pipeline adds
   blank lines by its own deterministic policy.) Blank lines between
   top-level definitions are fine. Use short `#` comments inside bodies
   where a real developer would (on their own line), not on every line.
6. Package files (`R/*.R`): every exported or internal function gets a
   roxygen block, written carefully:
   - a one-line title, then `#'` blank, then a 1-3 line description;
   - one `@param` per formal argument, in signature order, with the exact
     argument name (`@param ... Passed to [foo()].` for dots; you may use
     `@param x,y` for a pair documented together);
   - `@return` describing the value; then `@export`, or `@keywords internal`,
     or `@noRd` for helpers; optionally `@examples`, `@seealso`, `@family`,
     `@inheritParams other_fun` (then omit those params), `@rdname`, `@importFrom`.
   - Within one file keep the doc conventions consistent (same tag order,
     same phrasing for the same argument across sibling functions: if
     `data` means "A data frame of ..." in one function, reuse that exact
     sentence in siblings).
   Scripts, Shiny `app.R` and test files need no roxygen.
7. Mix styles across files (state the style in a first-line comment ONLY if
   natural; do not label it): snake_case vs camelCase vs dot.case names;
   `<-` (most files) vs `=` assignment (a few files); explicit `return()` vs
   implicit last value; magrittr `%>%` vs native `|>`; `pkg::fun()` vs
   bare calls; base R vs tidyverse idioms. Keep each file internally
   consistent.
8. No non-ASCII characters, no tabs, lines under 100 characters, no
   `\r`. Strings must not contain the text `>>>>>>>`, `<<<<<<<`, `=======`
   or `[NO_EDIT]`.

## Size
Write the number of files given in your task, each with 5-9 top-level
functions (test files: 5-9 `test_that()` blocks; app files: UI + server +
3-6 helper/module functions). When done, reply with a short list of files and
function counts. Do not write anything outside your directory.

## Wave 3 addendum
Wave 3 seeds feed mostly "finish the block" cases. In addition to the rules
above: favour functions whose bodies contain nested structure (if/else
chains, loops containing conditionals, tryCatch, switch, nested function
calls spanning several lines) and whose last 2-8 lines are predictable from
what came before (closing a loop and returning the accumulator, the else
branch mirroring the if branch, a final assembled list/data.frame whose
fields were computed above, symmetric cleanup). Keep each file's sibling
functions closely parallel. Aim for 20% long (30-60 line) functions.
