#!/usr/bin/env python3
"""Self-contained tests for context_builder.py, the shared scope-aware
context module (spec: docs/prompt-format.md, "Scope-aware context").

Run: nice -n 19 uv run python experiments/synthetic-data/test_context_builder.py

Part 1 — exactness checks on hand-written R fixtures: span/pin/outline
behavior for nested definitions, brace-less bodies, multi-line signatures,
= / <<- assignments, same-line one-liners, non-identifier assignment
targets, top-level cursors, out-of-range cursors, and the empty file.

Part 2 — corpus sweep over >= 30 real files from /mnt/h/sepalith/normalized
(packages strided across the sorted list for variety; rlang and dplyr forced
in for backtick-named signatures and file-size range). Per file, the test
walks the tree itself with the finish_block.py query pattern and asserts:

  - pin_split reassembles the file losslessly (prefix + pinned + rest)
  - a pin never truncates the function remainder: pinned runs from the
    cursor row through the enclosing function's end row, and the file
    continues only below that row
  - a cursor at top level returns no pin (enclosing is None)
  - every outline line is a single line matching the signature shape
    ^[A-Za-z._][\\w.]*\\s*(<-|=)   (backtick-quoted names, e.g. `%||%`,
    are counted separately: same shape, quoted symbol)
  - the outline equals the independently computed expectation, and the
    enclosing function's signature is absent from it while the cursor is
    inside that function

CPU-only, no network, no API keys, no LSP, no ry. Prints a summary and
exits nonzero on any failure.
"""
import re
import sys
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_builder as cb

ROOT = Path("/mnt/h/sepalith/normalized")
TARGET_FILES = 30
MAX_FILE_BYTES = 400_000          # finish_block.py convention
SIG_RE = re.compile(r"^[A-Za-z._][\w.]*\s*(<-|=)")   # required shape
BT_RE = re.compile(r"^`[^`]+`\s*(<-|=)")             # quoted-symbol variant

FAILS: list[str] = []
CHECKS = 0


def check(cond: bool, label: str) -> bool:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(label)
    return bool(cond)


# ---------------------------------------------------------------- part 1
FIXTURE = """#' docs
fit_model <- function(data,
                      weights,
                      method = "glm") {
  total <- sum(data * weights)
  inner <- function(y) {
    y + 1
  }
  inner(total)
}

sq = function(x) x^2

`%||%` <- function(x, y) if (is.null(x)) y else x

helpers <- list()
"""
L = FIXTURE.split("\n")[:-1]  # 16 lines, rows 0-15

# enclosing_function
enc = cb.enclosing_function(L, 4)          # mid fit_model body
check(enc == dict(name="fit_model",
                  signature='fit_model <- function(data, weights, method = "glm")',
                  start_line=1, end_line=9, start_byte=8, end_byte=193),
      f"fixture: span at row 4 == {enc}")
check(cb.enclosing_function(L, 6)["name"] == "fit_model",
      "fixture: nested inner def resolves to enclosing top-level fn")
check(cb.enclosing_function(L, 0) is None, "fixture: roxygen row is top level")
check(cb.enclosing_function(L, 10) is None, "fixture: blank row is top level")
check(cb.enclosing_function(L, 14) is None,
      "fixture: non-function assignment is top level")
sq = cb.enclosing_function(L, 11)
check(sq["name"] == "sq" and sq["start_line"] == sq["end_line"] == 11,
      "fixture: brace-less one-liner spans a single row")
check(cb.enclosing_function(L, -5) is None and cb.enclosing_function(L, 999) is None,
      "fixture: out-of-range cursor clamps to top level")
check(cb.enclosing_function([], 0) is None, "fixture: empty file -> None")

# outline
check(cb.outline(L, 0) == [
    'fit_model <- function(data, weights, method = "glm")',
    "sq <- function(x)",
    "`%||%` <- function(x, y)",
], f"fixture: outline at top level == {cb.outline(L, 0)}")
check(cb.outline(L, 4) == ["sq <- function(x)", "`%||%` <- function(x, y)"],
      "fixture: outline dedups the enclosing function's signature")
check(cb.outline(L, 11) == [
    'fit_model <- function(data, weights, method = "glm")',
    "`%||%` <- function(x, y)",
], "fixture: outline dedups a one-liner enclosing function")
check(cb.outline([], 0) == [], "fixture: empty file -> empty outline")

# pin_split
pre, pin, rest = cb.pin_split(L, 4)
check(pre == L[:4] and pin == L[4:10] and rest == L[10:],
      "fixture: pin at row 4 = prefix | fn remainder | file suffix")
check(pin[0] == "  total <- sum(data * weights)" and pin[-1] == "}",
      "fixture: pinned block starts at cursor row, ends at fn close")
check(cb.pin_split(L, 0) == ([], [], L), "fixture: top-level cursor -> no pin")
check(cb.pin_split(L, 14) == (L[:14], [], L[14:]),
      "fixture: non-function row -> no pin")
check(cb.pin_split(L, 15) == (L[:15], [], ["helpers <- list()"]),
      "fixture: last-row non-function -> no pin")
check(cb.pin_split(L, 999) == (L, [], []), "fixture: cursor past EOF clamps")
check(cb.pin_split([], 2) == ([], [], []), "fixture: empty file pin")

# same-line one-liners: first definition in document order wins
SAME = "a <- function(x) x; b <- function(y) y\nrest <- 1\n".split("\n")[:-1]
check(cb.enclosing_function(SAME, 0)["name"] == "a",
      "fixture: same-line tie resolves to first fn in document order")
check(cb.pin_split(SAME, 0) == ([], SAME[:1], SAME[1:]),
      "fixture: same-line tie pins through the first fn's end")

# <<- and = assignments; $-targets tracked as spans but not outlined
MISC = """set_val <<- function(v) {
  v
}
same = function(z) z
obj$method <- function(q) q
""".split("\n")[:-1]
check(cb.enclosing_function(MISC, 1)["name"] == "set_val",
      "fixture: <<- assignment recognized as top-level fn")
check(cb.enclosing_function(MISC, 2)["name"] == "set_val",
      "fixture: closing-brace row still inside the <<- fn")
check(cb.enclosing_function(MISC, 3)["name"] == "same",
      "fixture: = assignment recognized as top-level fn")
check(cb.enclosing_function(MISC, 4) is not None,
      "fixture: $-target function still resolvable as enclosing scope")
check(cb.outline(MISC, 1) == ["same <- function(z)"],
      f"fixture: $-target skipped, <<- fn deduped == {cb.outline(MISC, 1)}")
check(cb.outline(MISC, 3) == ["set_val <- function(v)"],
      f"fixture: = fn deduped, $-target skipped == {cb.outline(MISC, 3)}")
check(cb.pin_split(MISC, 1)[1] == MISC[1:3],
      "fixture: <<- fn pinned like any other")


# ---------------------------------------------------------------- part 2
def walk_functions(lines: list[str]) -> list[dict]:
    """Independent re-implementation of the finish_block.py query pattern
    (bytes + bisect only; Node.start_point is never read — see the module
    docstring)."""
    enc = [l.encode("utf-8") for l in lines]
    src = b"\n".join(enc)
    starts = list(accumulate((len(l) + 1 for l in enc), initial=0))

    def row(byte):
        return bisect_right(starts, byte) - 1

    out = []
    for node in cb._get_parser().parse(src).root_node.children:
        if node.type != "binary_operator":
            continue
        kids = node.children
        rhs = [c for c in kids if c.type == "function_definition"]
        if not rhs or not any(c.type in cb.ASSIGN_OPS for c in kids):
            continue
        fn = rhs[0]
        params = next((c for c in fn.children if c.type == "parameters"), None)
        name_node = kids[0]
        name = src[name_node.start_byte:name_node.end_byte].decode(
            "utf-8", "replace")
        ptxt = (src[params.start_byte:params.end_byte].decode("utf-8", "replace")
                if params is not None else "()")
        out.append(dict(
            name=name,
            signature=cb._one_line(f"{name} <- function{ptxt}"),
            plain=name_node.type == "identifier",
            start_line=row(node.start_byte),
            end_line=row(node.end_byte - 1),
        ))
    return out


def pick_files(pkgs: list[str]):
    """(package, path, lines) for 2 varied files per package until TARGET."""
    got = []
    for pkg in pkgs:
        try:
            ver_dirs = sorted(d for d in (ROOT / pkg).iterdir() if d.is_dir())
            if not ver_dirs:
                continue
            rdir = ver_dirs[0] / pkg / "R"
            if not rdir.is_dir():
                continue
            # dict-by-name: drvfs is case-insensitive (*.R vs *.r overlap)
            files = sorted({f.name: f for f in list(rdir.glob("*.R"))
                            + list(rdir.glob("*.r"))}.values(),
                           key=lambda f: f.name)
            sized = sorted((f for f in files
                            if f.stat().st_size <= MAX_FILE_BYTES),
                           key=lambda f: f.stat().st_size)
        except OSError:
            continue
        if not sized:
            continue
        picks = [sized[len(sized) // 2]]           # median size...
        if sized[-1] != picks[0]:
            picks.append(sized[-1])                # ...and the largest
        for f in picks:
            try:
                raw = f.read_bytes()
            except OSError:
                continue
            if not raw:
                continue
            lines = [l.decode("utf-8", "replace").rstrip("\r")
                     for l in raw.split(b"\n")]
            got.append((pkg, f.name, lines))
        if len(got) >= TARGET_FILES:
            break
    return got


def corpus_pkgs() -> list[str]:
    all_pkgs = sorted(p.name for p in ROOT.iterdir())
    stride = max(1, len(all_pkgs) // 40)
    spread = all_pkgs[::stride]
    forced = [p for p in ("rlang", "dplyr") if p in set(all_pkgs)]
    seen, ordered = set(), []
    for p in forced + [p for p in spread if p not in set(forced)]:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


stats = dict(packages_scanned=0, files=0, files_no_gap=0, fns=0, fn_cursors=0,
             top_cursors=0, outline_lines=0, outline_backtick=0, files_no_fn=0)

def forced_file(pkg: str, fname: str):
    """One hand-picked file: rlang/R/bytes.R has top-level backtick-named
    methods (`[.rlib_bytes` <- function...), exercising the quoted-symbol
    outline shape from the real corpus."""
    try:
        ver = sorted(d for d in (ROOT / pkg).iterdir() if d.is_dir())[0]
        raw = (ver / pkg / "R" / fname).read_bytes()
    except (OSError, IndexError):
        return None
    if not raw or len(raw) > MAX_FILE_BYTES:
        return None
    return (pkg, fname,
            [l.decode("utf-8", "replace").rstrip("\r") for l in raw.split(b"\n")])


files = pick_files(corpus_pkgs())
forced = forced_file("rlang", "bytes.R")
if forced is not None and not any(p == "rlang" and n == "bytes.R"
                                  for p, n, _ in files):
    files.insert(0, forced)
stats["packages_scanned"] = len({p for p, _, _ in files})
check(len(files) >= TARGET_FILES,
      f"corpus: only {len(files)} files selected (need {TARGET_FILES})")

for pkg, fname, lines in files:
    stats["files"] += 1
    where = f"{pkg}/R/{fname}"
    fns = walk_functions(lines)
    stats["fns"] += len(fns)
    if not fns:
        stats["files_no_fn"] += 1

    covered = set()
    for f in fns:
        covered.update(range(f["start_line"], f["end_line"] + 1))
    gap = next((r for r in range(len(lines)) if r not in covered), None)

    def expected_outline(exclude_idx: int) -> list[str]:
        """Mirror of the module's outline rules over the independent walk."""
        seen, out = set(), []
        for i, f in enumerate(fns):
            if i == exclude_idx or not f["plain"]:
                continue
            if f["signature"] not in seen:
                seen.add(f["signature"])
                out.append(f["signature"])
        return out

    def shape_check(outl: list[str], where_cur: int) -> None:
        for s in outl:
            stats["outline_lines"] += 1
            check("\n" not in s and s == s.strip(),
                  f"{where}: outline line not a single clean line: {s!r}")
            if not SIG_RE.match(s):
                if BT_RE.match(s):
                    stats["outline_backtick"] += 1
                else:
                    check(False,
                          f"{where}@{where_cur}: outline line fails shape "
                          f"regex: {s!r}")

    if gap is not None:
        stats["top_cursors"] += 1
        check(cb.enclosing_function(lines, gap) is None,
              f"{where}: cursor at top-level row {gap} must return None")
        check(cb.pin_split(lines, gap)[1] == [],
              f"{where}: cursor at top-level row {gap} must pin nothing")
        outl = cb.outline(lines, gap)
        shape_check(outl, gap)
        check(outl == expected_outline(-1),
              f"{where}: outline mismatch vs independent walk")
    else:
        stats["files_no_gap"] += 1

    sample = fns if len(fns) <= 6 else [fns[i] for i in sorted(
        {0, len(fns) - 1, len(fns) // 5, 2 * len(fns) // 5,
         3 * len(fns) // 5, 4 * len(fns) // 5})]
    for f in sample:
        for cur in {f["start_line"], (f["start_line"] + f["end_line"]) // 2,
                    f["end_line"]}:
            stats["fn_cursors"] += 1
            enc = cb.enclosing_function(lines, cur)
            if not check(enc is not None,
                         f"{where}: cursor {cur} inside fn got None"):
                continue
            # the module's span must equal the independent walk's pick for
            # this row (first covering definition, document order)
            enc_idx = next(i for i, g in enumerate(fns)
                           if g["start_line"] <= cur <= g["end_line"])
            g = fns[enc_idx]
            check((enc["name"], enc["signature"], enc["start_line"],
                   enc["end_line"]) == (g["name"], g["signature"],
                                        g["start_line"], g["end_line"]),
                  f"{where}: span differs from walk at cursor {cur}: "
                  f"{enc} vs {g}")
            check(enc["start_line"] <= cur <= enc["end_line"],
                  f"{where}: span does not cover cursor {cur}")
            pre, pin, rest = cb.pin_split(lines, cur)
            check(pre + pin + rest == lines,
                  f"{where}: pin_split not lossless at cursor {cur}")
            check(pin == lines[cur:enc["end_line"] + 1] and bool(pin),
                  f"{where}: pin truncates fn remainder at cursor {cur}")
            check(rest == lines[enc["end_line"] + 1:],
                  f"{where}: rest does not start below the fn at cursor {cur}")
            check(len(pre) == cur,
                  f"{where}: prefix not strictly above cursor {cur}")
            # outline dedup against the pin
            outl = cb.outline(lines, cur)
            shape_check(outl, cur)
            check(outl == expected_outline(enc_idx),
                  f"{where}: outline mismatch at cursor {cur}")

print("=" * 62)
print("test_context_builder summary")
print("=" * 62)
print(f"checks run:        {CHECKS}")
print(f"failures:          {len(FAILS)}")
for f in FAILS[:20]:
    print(f"  FAIL {f}")
if len(FAILS) > 20:
    print(f"  ... and {len(FAILS) - 20} more")
for k, v in stats.items():
    print(f"{k:>18}: {v}")
print(f"result: {'PASS' if not FAILS and CHECKS else 'FAIL'}")
sys.exit(1 if FAILS or not CHECKS else 0)
