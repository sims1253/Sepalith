# Exec-env test authoring brief (convention exec-test-v1)

You write **behavioural tests for real R functions**. Sepalith (a ~2B R
code-editing model) is trained with a reward that splices the model's
completion back into the function and runs your tests. The reward accepts any
completion that behaves like the original, so it can finally reward correct
code that is not a character-for-character copy. Your tests decide what
"behaves like the original" means. Weak tests teach the model to write wrong
code; tests that pin incidental spelling teach it to guess. This brief is the
whole contract.

Read this file, then the code it points to only if you need to:
`packages/sepalith/src/sepalith/exec_env/` (`testlib.R`, `gate.py`).

## 1. Claim a slice

The slice manifest is
`/mnt/e/sepalith/campaign-20260915/data-work/EXEC-ENV-v1/slices.json`. Each
slice is a set of whole packages (about 100 functions), so no two authors ever
share a package. The materialised working tree is

```
/mnt/e/sepalith/campaign-20260915/data-work/EXEC-ENV-v1/authors/<slice>/<function id>/
    source.R    the function, its in-file helpers and roxygen (air-formatted). Read-only.
    meta.json   package, path, name, helpers, formals, shape. Read-only.
```

The manifest holds 472 training slices (`S000`-`S471`, 74-198 functions,
median 100; 47,607 functions from 6,590 packages) and 8 hold-out slices
(`H000`-`H007`, 800 functions from 206 packages). Take slices in id order:
the order is a hash shuffle, so the first slices are a representative mix.
In the pilot about 81% of functions gated and each gave about 11 tasks, so
roughly 17 to 42 slices give the 12k-30k tasks a ~1,500-update GRPO run
needs. Functions with a `PILOT` file already have tests from the pilot: re-run the gate and review them, rewrite
only if they fail.

Claim exactly one slice at a time, atomically:

```sh
ROOT=/mnt/e/sepalith/campaign-20260915/data-work/EXEC-ENV-v1/authors
mkdir "$ROOT/S017/.claim" && echo "<your agent name> $(date -Is)" > "$ROOT/S017/.claim/owner"
```

`mkdir` fails if someone else holds the slice; then pick another. Work only
inside your slice directory. When you are done, run the gate on the whole
slice and `touch "$ROOT/S017/.claim/done"`. Slices whose id starts with `H`
are the **frozen hold-out** (execution-eval set): same work, same bar, but
their tests are never used for training and are reviewed with extra care.
Don't work on an `H` slice and an `S` slice from the same package family in
one session (they never share packages; this is about keeping your notes
apart).

No git. Don't copy files elsewhere. Don't edit `source.R` or `meta.json`. If a
function is unusable, write `SKIP` in its directory with one line saying why
(`nondeterministic`, `needs a package that is not installed`, `function is
broken on every input`, `output only: behaviour is printed text`, `trivial: no
behaviour to test`), and move on. The gate ignores directories with `SKIP`.
A bug in one branch is not a reason to skip: test the working branches and
leave the quirk untested. Verify every "broken" or "missing helper" claim by
running the code first (two of the first skips were false). The sandbox runs
**R 4.6.1** with base and recommended packages only; `%||%` is in base.

## 2. What to write

One file per function: `test.R` in the function's directory. Optionally
`equivalent.txt` (section 5). The file is plain R evaluated top to bottom in an
environment where `source.R` has been sourced. Use only these helpers
(from `testlib.R`); each takes a **category** label first:

| Helper | Passes when |
| --- | --- |
| `check(cat, expr)` | `expr` is exactly `TRUE` |
| `check_equal(cat, actual, expected, tolerance = 1e-8)` | `all.equal` holds |
| `check_identical(cat, actual, expected)` | `identical()` holds |
| `check_error(cat, expr, class = NULL, pattern = NULL)` | `expr` signals an error of that class and/or matching message |
| `check_warning(cat, expr, pattern = NULL)` | `expr` signals a matching warning |

Every assertion is evaluated on its own; an error inside one counts as a
failure and the script continues. Only these helpers count. Printed output and
exit status count for nothing.

Categories group assertions by behaviour (`basic`, `edge`, `boundary`, `na`,
`empty`, `validation`, `types`, `options`, ...). The reward averages the pass
share **per category** first, so twenty easy assertions in one category cannot
outweigh one hard behaviour. The gate requires at least **12 hand-written
assertions** (frozen differential cases don't count) in at least **3
categories with at least 2 assertions each**. Use 3 to 5 categories that
match what the function actually does, never one category per assertion.

Example (`band_counts(scores, cut = 0.5)` returns named integer counts):

```r
# exec-test v1: band_counts
check_identical("counts", band_counts(c(0.1, 0.6, 0.9)), c(low = 1L, high = 2L, total = 3L))
check_identical("counts", band_counts(c(0.2, 0.3), cut = 0.25), c(low = 1L, high = 1L, total = 2L))
check_identical("counts", band_counts(c(0.9, 0.8, 0.7, 0.1), cut = 0.75), c(low = 2L, high = 2L, total = 4L))
check_identical("counts", band_counts(0.3), c(low = 1L, high = 0L, total = 1L))   # default cut pinned
check_identical("boundary", band_counts(c(0.5, 0.49999)), c(low = 1L, high = 1L, total = 2L))
check_identical("boundary", band_counts(c(0.25, 0.25), cut = 0.25), c(low = 0L, high = 2L, total = 2L))
check_identical("boundary", band_counts(c(-3, 7)), c(low = 1L, high = 1L, total = 2L))  # outside [0, 1]
check_identical("boundary", band_counts(c(1.5, 2), cut = 1), c(low = 0L, high = 2L, total = 2L))
check_identical("missing", band_counts(c(NA, 0.7, NA)), c(low = 0L, high = 1L, total = 1L))
check_identical("missing", band_counts(numeric()), c(low = 0L, high = 0L, total = 0L))
check_identical("missing", band_counts(c(NA_real_)), c(low = 0L, high = 0L, total = 0L))
check_error("missing", band_counts("a"), pattern = "scores must be numeric")
exec_cases <- function() {
  set.seed(42)  # every argument varies, including values outside [0, 1]
  lapply(1:10, function(i) list(scores = round(runif(sample(1:12, 1), -0.5, 1.5), 2), cut = round(runif(1), 2)))
}
```

### Frozen differential cases (recommended where inputs are easy to generate)

Define `exec_cases <- function() list(list(<args>), ...)` returning argument
lists (seeded RNG allowed: call `set.seed()` first). The gate runs the
original once on each case, in the sandbox, and writes `frozen.R` with one
`check_equal("differential", ...)` per case. These cover inputs you did not
think of and defeat lookup-table answers. Cases on which the original errors
are dropped. Keep each case small (the frozen file holds the outputs) and use
8 to 20 cases. **Vary every argument** across the cases (sizes, values,
options), not just the main input.

## 3. Run the gate (until it passes)

```sh
cd ~/sandbox/sep-exec-env   # or any checkout with the exec-env code
PYTHONPATH=packages/sepalith/src nice -n 10 \
  /home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python -m sepalith.exec_env.gate --workers 2 "$ROOT/S017/<function id>"
```

Pass a slice directory to gate all its functions. The gate writes
`gate.json` next to `test.R` and prints `pass`/`fail` with reasons. It
requires:

1. **Static**: the file parses; no `print`, `cat`, `message`,
   `capture.output`, `stopifnot`, `Sys.*`, timing, I/O, `source`, `library`,
   `assign`, `quit`; RNG only after `set.seed`; at most 30% *shape-only*
   assertions (only `is.*`, `inherits`, `class`, `length`, `nrow`, `dim`,
   `is.null`, or a `check_error` without `class`/`pattern`); token 12-gram
   containment of the function body in your test at most 0.25 (do not paste
   the implementation).
2. **The original passes every assertion in 3 fresh runs** with identical
   outcomes; at least 12 hand-written assertions in at least 3 categories of
   at least 2 each. A `check_error` pattern without `class =` must not match
   any of R's own messages (the gate checks it against R's message catalogs):
   `"length"`, `"minimum"` or `"numeric"` also match R's errors, so the test
   would still pass if the function's own check were deleted. Use the
   function's specific wording (`"scores must be numeric"`) or `class =`.
3. **The tests reach the function**: a stub whose body is
   `stop("exec-env stub")` may pass at most 20% of the assertions.
4. **Mutation sensitivity**: up to 60 mutants of the function (flipped
   comparisons and arithmetic, `&&`/`||`, dropped `!`, changed constants and
   strings, `TRUE`/`FALSE`, negated `is.na`, `if` conditions forced, deleted
   statements, duplicated self-updates, swapped arguments, `x[i + 1]`, early
   `return(NULL)` / `return(<first argument>)`). **Every surviving mutant is
   must-kill**: add an input that tells it apart from the original, or, only if
   none exists, declare it in `equivalent.txt` with a one-line reason
   (section 5). Survivors are listed in `gate.json` (`mutants.survivors`, with
   `original_lines` and `mutated_lines`).
5. **Self-probes** (optional, recommended): copy `source.R` to
   `probes/p1.R` (`mkdir -p probes && cp source.R probes/p1.R`) and make one
   plausible mistake a model could make: a wrong boundary, a dropped clamp or
   `...` forwarding, the wrong element, a missing branch. Write 2 or 3 per
   function. The gate runs every `probes/*.R` and fails if one passes all
   assertions (`gate.json` `probes`). This is how reviewers probe your suite:
   in the first review, 12% of their wrong versions passed every assertion.

The gate does not generate mutants that only change message wording (strings
and `call.` flags inside `stop()`, `warning()`, `message()`, `cat()`,
`print()`, ...), swaps of symmetric arguments (`max`, `sum`, `identical`,
...), or deletions of a trailing `x` / `return(x)` right after `x <- ...`.
Output-only methods (`print.*`, `plot.*`, `show`, ...) are skipped
automatically (`verdict: skip`). When the original fails, `gate.json` lists
the failing assertions (`original_failing_assertions`, with the statement text
when it can be mapped) and does not run mutants.

## 4. Quality bar

- **Assert values, not shapes.** Compute expected outputs by reading the code
  and reasoning about the input, then confirm by running the gate. Prefer
  `check_identical` for integers, characters, names and structure;
  `check_equal` for doubles.
- **Cover every branch** the function has: each `if`/`else`, each `switch`
  arm, loop bodies that run zero, one and many times, early returns.
- **Vary fixtures until every intermediate value matters.** Most reviewer
  false accepts came from fixtures that never reach a branch or make two
  computations coincide: a propensity of exactly 0.5, two folds where median
  equals mean, only a path graph, no ties, no blanks, nothing outside
  [0, 1]. For each intermediate value, include an input that changes it, and
  at least one input outside the comfortable range (overshooting, negative,
  tied, duplicated, empty strings).
- **Defaults and secondary arguments**: pin each default with an absolute
  expected value (a call that relies on the default), and give every other
  argument at least one non-default value, including `...` forwarding.
- **Assert what the function sets**: the returned S3 class
  (`check_identical(cat, class(res), c("foo", "list"))`), the names and values
  of returned fields and attributes, and stored side objects (a fitted tree
  kept in `res$cluster`). Three reviewer rejects were class-only gaps.
- **Edges**: empty input, length one, `NA`/`NaN` where the function handles or
  propagates them, boundaries of every comparison (both sides of `<` vs `<=`),
  zero and negative values where meaningful, factors vs characters where the
  function cares.
- **Errors**: test that invalid input errors, with `class =` where the function
  signals a classed condition, otherwise a `pattern =` specific to the
  function's own message (a distinctive phrase, not a generic word that R's
  own errors also contain).
- **Varied inputs**: never test with a single input family; mix sizes, orders
  and values so constants or lookup tables cannot pass.
- **Behaviour, not spelling.** Tests must accept any implementation that
  behaves like the original. Do not test the text of a message beyond a
  distinctive word, the order of attributes, or intermediate variables. Do
  test returned names and attributes when callers rely on them.
- **Inferability**: the task builder cuts each function at statement
  boundaries and shows the model the code above the cut. A cut is dropped
  automatically when your tests reference a string, `$` element or argument
  name that only appears in the hidden remainder. You do not need to avoid
  such assertions (they are often the behaviour that matters), but prefer
  assertions about values over assertions about exact message text, so more
  cuts survive.
- **The original is the specification, within reason.** Pin what the visible
  code clearly intends and callers rely on. Leave accidental quirks of the
  original untested (an `NA` column name past some size, an obviously wrong
  corner case): a correct rewrite should not be punished for fixing them.
- **Fake objects and S3**: build minimal lists/data frames with exactly the
  fields the function reads and set `class()` as needed. Prefer
  `strict(list(...), class = "foo")` (from `testlib.R`): its `$` and `[[`
  raise an error for any field the fake lacks. With a plain list, a completion
  that reads fields the real object would carry (and writes them into
  attributes, say) sees `NULL` and passes; in the pilot, arm A did exactly
  that. Use a plain list only when the function itself probes for optional
  fields (`is.null(x$opt)`). `strict()` appends `"exec_strict"` to the class
  (account for it in `class()`/`identical()` assertions) and cannot wrap a
  data frame (its own `$` method wins).
- **Dead code**: model completions sometimes add branches that never run on
  your inputs. Cover every argument the function reads with at least one
  non-default value, so code guarded by those arguments is exercised.
- **S3 methods** you define in `test.R` are not visible to dispatch from the
  function; register them with
  `registerS3method("generic", "class", function(...) ...)` or use real base
  or stats objects.
- **Messages and warnings**: the helpers suppress warnings (except
  `check_warning`), so `options(warn = ...)` effects are not observable; don't
  try. A `message()` can be made observable with
  `withCallingHandlers(expr, message = function(m) stop(conditionMessage(m)))`
  inside `check_error`.
- **Randomness**: call `set.seed()` in `test.R` before any random draw (also
  before calling a function that draws internally).
- **Visibility**: `withVisible(f(x))$visible` is allowed when callers depend
  on invisible returns; otherwise don't test it.
- **Names in tests**: don't give test variables the names of base functions
  (`single`, `c`, `t`, `df`): if an earlier statement fails, the base object
  is found instead and assertions can pass by accident.
- Keep the suite fast: well under 1 s for the whole file. No large
  simulations, no `Sys.sleep`.
- Tests are hidden from the model and never appear in prompts. Do not put
  hints in comments that could leak into data (`test.R` stays in the author
  tree).

## 5. Equivalent mutants

Some mutants cannot be told apart from the original by any input (for
example deleting a statement whose result is never used, or forcing an `if`
whose two arms compute the same value). List those in `equivalent.txt`, one
per line: `<mutant id> <one-line reason>`. They leave the kill-rate
denominator. Mutant ids are bound to the edit (span and replacement text);
`gate.json` lists `stale_equivalent_ids` (no longer generated) and
`declared_equivalent_but_killed` (your tests kill it, so it was not
equivalent): delete those lines. Reviewers check every entry; an entry that
is not genuinely equivalent fails the function. Do not use this file to get past the gate.

## 6. Anti-patterns (each fails review even if the gate passes)

- Tests that only check type, class, length or non-NULL-ness.
- Tests that pass on a mutant you could have distinguished with one more input.
- Tests that copy the implementation (recomputing the expected value with the
  same expression as the function) instead of independent expected values.
  Expected values may come from first principles, a different base-R route,
  or a hand-worked example.
- Tests that depend on printing, timing, locale, the working directory, the
  random seed of the session, or on each other's side effects.
- Tests wrapped in `tryCatch`/`try` that turn failures into passes.
- Asserting the full text of an error message or warning; or the opposite, a
  generic pattern (`"length"`) that R's own errors also match.
- Fixtures that leave a branch or an argument unexercised.
- One giant category, or a category per assertion.
- `exec_cases` that are trivial (all-zero, all-identical) or huge.

## 7. Review and audit

Every slice is reviewed by an agent of the **other** family (an Opus author is
reviewed by a GPT-6.1-Sol reviewer and vice versa). Reviewers read
`source.R`, `test.R`, `gate.json` and `equivalent.txt`, re-run the gate, and
reject functions that hit an anti-pattern. A uniform random 5% of passed
functions per slice goes to a human audit list
(`EXEC-ENV-v1/audit/human-sample.jsonl`, written by
`python -m sepalith.exec_env.slices --pool ... --out ... --audit-root EXEC-ENV-v1/authors --audit-out ...`). Hold-out (`H`) slices are reviewed
in full.

## 8. Rules

- CPU only, `nice -n 10`, at most 2 gate workers per author. No GPU, no network.
- Never open `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-*`,
  `~/.local/state/sepalith/campaign-20260915/sealed-final/`, DAT-08 final
  references, DEV panels, or any other slice's directory.
- Don't touch any git repository or worktree. Don't kill other processes.

## 9. Report

When your slice is done, reply with: slice id; functions passed / failed /
skipped (with the skip reasons); the median assertion count and median
kill rate from `gate.json`; the number of `equivalent.txt` entries; and any
function you think the pool should not contain (with the reason).
