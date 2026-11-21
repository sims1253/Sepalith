# Seed authoring brief (SYNTH-v3)

You write ORIGINAL R source files under the directory given in your task.
A pipeline cuts each function at a line boundary and asks a small model to
complete the rest. A cut is only usable when the remainder is **determined by
what the model can see**: the roxygen block, the signature and the lines
above the cut, or one short sibling function directly above. Your job is to
write realistic code in which long remainders (10-60 lines) are determined
that way. All hard rules from `AUTHORING-BRIEF.md` still apply: original code
only, every file parses, ASCII only, no tabs, lines under 100 characters,
none of the strings `>>>>>>>`, `<<<<<<<`, `=======` or `[NO_EDIT]`, fictional
package names, no git, `nice -n 10`, at most 2 threads.

## Patterns (use several in every file)

1. **Parallel blocks inside one function.** 3-7 structurally parallel blocks
   (same calls, different names), each 3-8 lines, where every instance name is
   visible *before the first block*: in the signature, in a vector such as
   `fields <- c("start", "end", "duration")`, or in the roxygen `@param` /
   `@return` list. Examples: validate each argument in turn; summarise each
   column; build a settings list per axis; handle each event type; write one
   file per table.
2. **Dispatch tables.** `type <- match.arg(type)` with the choices in the
   signature, then `switch(type, a = {...}, b = {...}, ...)` whose arms are
   parallel (4-6 arms, 3-8 lines each).
3. **Mirrored branches.** `if/else` (or two loops) where the second half
   mirrors the first: min/max, x/y, left/right, head/tail, read/write,
   encode/decode, before/after, upper/lower bound.
4. **Spec-visible bodies.** The roxygen `@return` / `@details` or a short
   numbered comment plan at the top of the body names every element or step
   the body produces, with the same identifiers the code uses (e.g.
   "`@return` A list with elements `n`, `mean`, `sd`, `range`" and the body
   ends by building exactly that list).
5. **Compact method families.** 3-4 sibling functions (S3 methods, S4
   methods, per-format readers/writers, per-type handlers) that follow one
   template, **placed adjacent**, each 10-25 lines, so the previous sibling
   fits in a 40-line window above the current one.

## What makes a remainder undetermined (avoid in later instances)

- New free-text string literals. Error messages, labels and column names in
  later instances must follow a template established earlier (e.g.
  `sprintf("`%s` must be positive.", name)`, or a message reused verbatim) or
  be visible names.
- Calls to helper functions that are neither defined above in the same file
  nor already called earlier in the same function.
- Comments in later instances with new wording: reuse the wording of the
  corresponding earlier comment ("# Validate `end`" after "# Validate
  `start`"), or put the comment plan up front.
- Magic numbers that appear nowhere earlier.

## Style and mix

- Function lengths (body lines): 25% short (5-14), 40% medium (15-30), 35%
  long (31-60). The long ones should be long *because* they repeat a visible
  pattern, not because they contain new logic.
- Layout like real package code: single blank lines between logical steps
  (about 7% of body lines; short bodies usually none), full-line `#`
  comments where a developer would explain intent (about 5% of body lines).
  Never two blank lines in a row, no blank line right after `{` or right
  before `}`.
- Mostly **base R** (at least 60% of functions runnable with base R and the
  recommended packages only), so they can be executed by checks; the rest may
  use the add-on packages your domain needs.
- Roxygen on package files, with real-package tag frequencies: `@param` 93%,
  `@return` 60%, `@export` 44%, `@keywords internal` 38%, `@description`
  26%, `@noRd` 22%, `@rdname` 21%, `@title` 10%, `@details` 6.5%,
  `@importFrom` 6%, `@inheritParams` 5.5%, `@family` 4%. Vary tag order
  between packages; `@param` names must match the signature in order.
- Mix naming styles (snake_case, camelCase, dot.case), `<-` and `=`,
  explicit and implicit returns, between files; consistent within a file.

## Behavioural checks

For every file with base-R functions, write `checks/<same relative path>`
with tiny calls and `stopifnot()` assertions (2-4 per function; exercise
every pattern instance, e.g. every `switch` arm). The harness sources the
seed file into a fresh environment and evaluates the check file there:

    nice -n 10 Rscript --vanilla -e 'e <- new.env(); sys.source("SEED.R", e); sys.source("CHECK.R", e)'

Make every check pass. List functions you cannot check in a comment
`# skipped: needs <pkg>` (comments are not counted as calls).

## Size

Write the number of files given in your task, each with 5-8 top-level
functions. Reply with the file list, function counts, the share of
base-R-only functions, and which patterns each file uses.

## Wave B addendum (measured on wave A)

Wave A produced plenty of short determined cuts but too few medium/long ones:
most functions were short (S3 one-liners, small helpers), and later pattern
instances often introduced **new message text** (e.g. a new `stop()` sentence)
that the visible part never shows. For wave B:

- At least 70% of functions must have **20-60 body lines built from 4-8
  visible repetitions** of one pattern (parallel blocks, switch arms, per-field
  steps), with all instance names visible up front (signature `match.arg`
  choices, a names vector, roxygen `@param`/`@return` list).
- Every message, label or column-name string in a repeated block must be the
  **same template** as in the first instance with only the visible name
  changed (`sprintf("`%s` must be ...", "end")`, or build messages from the
  name with `paste0("`", name, "` must be ...")`), or a string already shown.
- Domain constants in later instances should be derivable from earlier ones
  or from visible names (e.g. a `factors <- c(day = 1, week = 7, ...)` table
  defined before the dispatch), not new magic numbers.
- Sibling families: 3-5 adjacent siblings of 12-30 lines each that differ only
  in visible names (per-format reader/writer, per-class method, per-statistic
  wrapper). Later siblings must not need helpers that are not visible above.
- Keep the TRAIN-like blank lines/comments, roxygen mix, base-R share and
  behavioural checks from the main brief.

## Wave C addendum (measured on waves A+B)

The usable cuts now come from patterns **inside one function**, because the
pipeline keeps prompts short (like real editing prompts: usually the
function's own roxygen, signature and first lines, 10-40 lines in total).
Cross-function sibling evidence often needs a longer window and is capped.
For wave C:

- Write 5-7 functions per file, almost all 25-55 body lines, each built from
  4-8 repetitions of one internal pattern whose instance names are visible in
  the signature, a names/table vector at the top of the body, or the first
  lines of roxygen. Put the repetitions early enough that a reader sees 1-2
  complete instances within the first ~25 lines of the function.
- Make about a third of the functions 35-55 body lines (long dispatch tables,
  per-field pipelines with 6-8 fields), still fully patterned.
- Keep roxygen blocks compact (title, one description line, `@param` lines,
  `@return`, one export/internal tag) so the prompt stays short.
- No small helpers, no one-line methods; helpers you need go inside the
  function or are called the same way in every instance (and appear before
  the first instance).
- Message/label strings: one template with only the visible name changed.
- Behavioural checks for every base-R function, exercising every instance.
