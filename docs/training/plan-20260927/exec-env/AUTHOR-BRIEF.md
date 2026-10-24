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
broken on every input`, `trivial: no behaviour to test`), and move on.

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
outweigh one hard behaviour. Use 3 to 6 categories that match what the function
actually does.

Example (`band_counts(scores, cut = 0.5)` returns named integer counts):

```r
# exec-test v1: band_counts
check_identical("basic", band_counts(c(0.1, 0.6, 0.9)), c(low = 1L, high = 2L, total = 3L))
check_identical("basic", band_counts(c(0.2, 0.3), cut = 0.25), c(low = 1L, high = 1L, total = 2L))
check_identical("boundary", band_counts(c(0.5, 0.49999)), c(low = 1L, high = 1L, total = 2L))
check_identical("clamp", band_counts(c(-3, 7)), c(low = 1L, high = 1L, total = 2L))
check_identical("na", band_counts(c(NA, 0.7, NA)), c(low = 0L, high = 1L, total = 1L))
check_identical("empty", band_counts(numeric()), c(low = 0L, high = 0L, total = 0L))
check_error("validation", band_counts("a"), pattern = "numeric")
exec_cases <- function() {
  set.seed(42)
  lapply(1:8, function(i) list(scores = round(runif(8, -0.5, 1.5), 2), cut = round(runif(1), 2)))
}
```

### Frozen differential cases (recommended where inputs are easy to generate)

Define `exec_cases <- function() list(list(<args>), ...)` returning argument
lists (seeded RNG allowed: call `set.seed()` first). The gate runs the
original once on each case, in the sandbox, and writes `frozen.R` with one
`check_equal("differential", ...)` per case. These cover inputs you did not
think of and defeat lookup-table answers. Cases on which the original errors
are dropped. Keep each case small (the frozen file holds the outputs) and use
8 to 20 cases.

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
   outcomes, with at least 6 assertions in at least 2 categories (aim for 10
   to 25 assertions plus differential cases).
3. **The tests reach the function**: a stub whose body is
   `stop("exec-env stub")` may pass at most 20% of the assertions.
4. **Mutation sensitivity**: up to 60 mutants of the function (flipped
   comparisons and arithmetic, `&&`/`||`, dropped `!`, changed constants and
   strings, `TRUE`/`FALSE`, negated `is.na`, `if` conditions forced, deleted
   statements, duplicated self-updates, swapped arguments, `x[i + 1]`, early
   `return(NULL)` / `return(<first argument>)`). At least **70%** of the valid
   ones must fail at least one assertion. Survivors are listed in `gate.json`
   (`mutants.survivors`, with the mutated line): read them, then add an input
   that tells the mutant apart from the original.

## 4. Quality bar

- **Assert values, not shapes.** Compute expected outputs by reading the code
  and reasoning about the input, then confirm by running the gate. Prefer
  `check_identical` for integers, characters, names and structure;
  `check_equal` for doubles.
- **Cover every branch** the function has: each `if`/`else`, each `switch`
  arm, loop bodies that run zero, one and many times, early returns.
- **Edges**: empty input, length one, `NA`/`NaN` where the function handles or
  propagates them, boundaries of every comparison (both sides of `<` vs `<=`),
  zero and negative values where meaningful, factors vs characters where the
  function cares.
- **Errors**: test that invalid input errors, with `class =` where the function
  signals a classed condition, otherwise a `pattern =` of a word or two that
  identify the check (not the full sentence).
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
denominator. Reviewers check every entry; an entry that is not genuinely
equivalent fails the function. Do not use this file to get past the gate.

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
- Asserting the full text of an error message or warning.
- One giant category, or a category per assertion.
- `exec_cases` that are trivial (all-zero, all-identical) or huge.

## 7. Review and audit

Every slice is reviewed by an agent of the **other** family (an Opus author is
reviewed by a GPT-6.1-Sol reviewer and vice versa). Reviewers read
`source.R`, `test.R`, `gate.json` and `equivalent.txt`, re-run the gate, and
reject functions that hit an anti-pattern. A uniform random 5% of passed
functions per slice goes to a human audit list
(`EXEC-ENV-v1/audit/human-sample.jsonl`). Hold-out (`H`) slices are reviewed
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
