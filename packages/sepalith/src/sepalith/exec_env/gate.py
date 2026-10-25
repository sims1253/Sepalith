"""Quality gate for exec-env test files (convention exec-test-v1).

    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.exec_env.gate DIR [DIR ...]

``DIR`` is a function directory (``source.R`` + ``test.R``) or a directory of
them (a slice). For each function the gate writes ``gate.json`` (and
``frozen.R`` when the test defines ``exec_cases``) and prints one line.
``--summary`` adds a per-function table of failing rules and a rule tally;
``--report-only`` prints that table from existing ``gate.json`` files without
re-running (to migrate a finished slice: run with ``--summary``, fix, repeat).

A test file passes when all of these hold:

1. **Static**: it parses; it uses the assertion helpers only (no printing,
   timing, I/O, sourcing or package loading, RNG only after ``set.seed``); at
   least ``MIN_HAND_CHECKS`` hand-written assertions in at least
   ``MIN_CATEGORIES`` categories of at least ``MIN_PER_CATEGORY`` each;
   ``check_error`` patterns without ``class`` must not match R's own messages
   (the msgids of R's message catalogs), so deleting the function's own check
   cannot pass on a generic R error;
   at most ``MAX_SHAPE_SHARE`` of the assertions only check type, class,
   length or NULL-ness, or are ``check_error`` without ``class``/``pattern``;
   it does not copy the implementation (token 12-gram containment of the
   function body in the test at most ``MAX_COPY_CONTAINMENT``).
2. **Original passes, deterministically**: ``RUNS_ORIGINAL`` runs in fresh
   sandboxed workers, every assertion passes and the outcomes are identical.
3. **Observability**: a stub whose body is ``stop("exec-env stub")`` passes at
   most ``MAX_STUB_PASS_SHARE`` of the assertions (the tests reach the function).
4. **Mutation sensitivity**: up to ``MUTANT_LIMIT`` class-stratified mutants of
   the target function (``mutants.py``); invalid mutants (do not parse or do
   not source) and mutants the author declared equivalent in
   ``equivalent.txt`` (``<mutant id> <reason>`` per line) leave the
   denominator; every other valid mutant must fail at least one assertion (or
   time out / crash). Survivors are listed in ``gate.json``.
5. **Self-probes** (optional): every ``probes/*.R`` in the function directory
   (a full plausible-but-wrong copy of ``source.R``) must fail at least one
   assertion.
6. **Differential checks** (v3, ``r_diff.R``): every call the suite makes to
   the function is recorded and perturbed. Each declared equivalent must match
   the original on all of these inputs (value, class, attributes, NA/NaN,
   error vs value, warning or not; inputs the original rejects and inputs
   inside a ``SCOPED[cond]``/``BUGFIX[cond]``/``QUIRK[cond]`` condition are excluded; PRINT
   entries are replayed too). Every
   argument the body reads takes 2+ distinct supplied values (a logical flag:
   its non-default value once), no two arguments are always equal, and
   one-nonzero-per-row/column matrix fixtures draw a warning. A generic
   condition class (``simpleError``, ...) does not exempt a ``check_error``
   pattern from the R-message check; the function's own message text (2+ words
   matching a literal in the unit) and anchored full messages (3+ words) do.
   Returned closures are compared by behaviour on probe inputs; ``call``,
   ``terms`` and ``heading`` values are ignored unless the suite reads them.
   ``gate.json`` records ``gate_version`` (name, commit, source fingerprint).

Optional frozen differential cases: if ``test.R`` defines
``exec_cases <- function() list(list(<args>), ...)`` (seeded RNG allowed), the
gate evaluates the original on each case once, inside the sandbox, and writes
``frozen.R`` with one ``check_equal("differential", ...)`` per case. The suite
used for gating and rewards is ``test.R`` followed by ``frozen.R``; the
original is never reachable at reward time.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from sepalith.exec_env import mutants as mutants_module
from sepalith.exec_env.harness import SANDBOX, TESTLIB, ExecHarness, ExecJob, ExecResult, namespaces_available

GATE_SCHEMA = "sepalith.exec-env.gate.v1"
GATE_VERSION_NAME = "v3"
_GATE_VERSION: dict | None = None


def gate_version() -> dict:
    """The gate's identity for freezes: name, git commit (and whether exec_env files differ from it) and a
    sha256 fingerprint of the exec_env sources (gate, mutants, harness, R scripts, testlib, sandbox)."""
    global _GATE_VERSION
    if _GATE_VERSION is None:
        import hashlib
        here = Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for path in sorted(here.iterdir()):
            if path.suffix in (".py", ".R", ".sh"):
                digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=False).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, check=False).stdout.strip()
        _GATE_VERSION = {"name": GATE_VERSION_NAME, "commit": commit, "exec_env_modified": bool(dirty),
                         "fingerprint": digest.hexdigest()[:16]}
    return _GATE_VERSION
MIN_HAND_CHECKS = 12  # hand-written assertions; frozen differential cases do not count
MIN_CATEGORIES = 3  # hand-written categories ("differential" does not count)
MIN_PER_CATEGORY = 2
MAX_SHAPE_SHARE = 0.3
MAX_COPY_CONTAINMENT = 0.25
RUNS_ORIGINAL = 3
MAX_STUB_PASS_SHARE = 0.2
MUTANT_LIMIT = 60
MIN_VALID_MUTANTS = 5
MIN_KILL_RATE = 1.0  # every valid survivor must be killed or declared equivalent (with a reason)
TIMEOUT = 5.0

FORBIDDEN_TEST_CALLS = re.compile(
    r"(?<![\w.])(capture\.output|print|cat|message|Sys\.[A-Za-z]+|system\.time|proc\.time|sink|"
    r"source|sys\.source|library|require|requireNamespace|loadNamespace|file|readLines|writeLines|readRDS|"
    r"saveRDS|load|save|download\.file|url|tempfile|unlink|setTimeLimit|quit|q|trace|assign|"
    r"stopifnot|testthat::\w+|expect_\w+)\s*\(")
RNG_CALLS = re.compile(r"(?<![\w.])(sample|sample\.int|runif|rnorm|rbinom|rpois|rexp|rgamma|rbeta|rt|rchisq)\s*\(")
CHECK_LINE = re.compile(r"^\s*(check|check_true|check_equal|check_identical|check_error|check_warning)\s*\(")
SHAPE_EXPR = re.compile(r"^\s*!?\s*(is\.\w+|inherits|length|lengths|class|typeof|mode|NROW|NCOL|nrow|ncol|dim|is\.null|"
                        r"identical\(\s*class|identical\(\s*length|all\(\s*is\.\w+)\s*\(")


def strip_strings_comments(text: str) -> str:
    from sepalith.data.synth.common import STRING_RE, strip_comment
    return "\n".join(STRING_RE.sub('""', strip_comment(line)) for line in text.split("\n"))


def check_statements(test: str) -> list[tuple[str, str]]:
    """(helper, joined text) for each top-level assertion call, multi-line calls joined."""
    out = []
    lines = test.split("\n")
    index = 0
    while index < len(lines):
        match = CHECK_LINE.match(lines[index])
        if not match:
            index += 1
            continue
        buf, depth, j = [], 0, index
        while j < len(lines):
            code = strip_strings_comments(lines[j])
            depth += code.count("(") - code.count(")")
            buf.append(lines[j].strip())
            j += 1
            if depth <= 0:
                break
        out.append((match.group(1), " ".join(buf)))
        index = j
    return out


def shape_only(helper: str, text: str) -> bool:
    if helper == "check_error":
        return "class" not in text and "pattern" not in text and not re.search(r",\s*\"[^\"]*\"\s*\)\s*$", text)
    inner = re.sub(r"^\s*check\w*\s*\(\s*\"[^\"]*\"\s*,", "", text)
    return bool(SHAPE_EXPR.match(inner))


_R_MESSAGES: list[str] | None = None


def r_messages() -> list[str]:
    """English msgids of R's message catalogs (base, stats, utils, methods, ...), read from the .mo files."""
    global _R_MESSAGES
    if _R_MESSAGES is None:
        import struct
        out: set[str] = set()
        proc = subprocess.run(["Rscript", "--vanilla", "-e", "cat(R.home())"], capture_output=True, text=True, check=False)
        root = Path(proc.stdout.strip() or "/usr/lib/R") / "library" / "translations" / "en@quot" / "LC_MESSAGES"
        for mo in sorted(root.glob("*.mo")):
            data = mo.read_bytes()
            magic = struct.unpack("<I", data[:4])[0]
            order = "<" if magic == 0x950412DE else ">"
            n, offset = struct.unpack(order + "II", data[8:16])
            for i in range(n):
                length, start = struct.unpack(order + "II", data[offset + 8 * i:offset + 8 * i + 8])
                for msg in data[start:start + length].split(b"\0"):
                    if msg:
                        out.add(msg.decode("utf-8", "replace"))
        _R_MESSAGES = sorted(out)
    return _R_MESSAGES


ERROR_PATTERN = re.compile(r"pattern\s*=\s*\"((?:[^\"\\]|\\.)*)\"")


def _pattern_words(pattern: str) -> int:
    return len(re.sub(r"\\.|[\^$.*+?()\[\]{}|]", " ", pattern).split())


def _own_literals(unit: str) -> list[str]:
    """String literals in the unit's code (comments removed): the messages the function itself raises."""
    from sepalith.data.synth.common import STRING_RE, strip_comment
    return [m[1:-1] for line in unit.split("\n") for m in STRING_RE.findall(strip_comment(line))]


def loose_error_patterns(statements: list[tuple[str, str]], unit: str = "") -> list[str]:
    """``check_error`` patterns (without a specific ``class``) that also match one of R's own messages.

    Allowed despite a catalog match: a pattern of 2+ words that matches a string literal of the
    function's own file (its own ``stop()``/``stopifnot``/``match.arg`` text), and an anchored
    pattern (``^...$``) of 3+ words that spells out a full message (a deliberately delegated error).
    Bare generic words are always rejected.
    """
    messages = r_messages()
    literals = _own_literals(unit)
    out = []
    for helper, text in statements:
        if helper != "check_error":
            continue
        classes = re.findall(r"\bclass\s*=\s*(?:c\()?\s*\"([^\"]+)\"", text)
        if classes and not set(classes) <= GENERIC_CONDITION_CLASSES:
            continue  # a specific condition class identifies the function's own error
        patterns = ERROR_PATTERN.findall(text)
        if classes and not patterns:
            out.append(f"class={classes[0]} without a pattern")
            continue
        for raw in patterns:
            pattern = raw.encode().decode("unicode_escape")
            try:
                compiled = re.compile(pattern)
            except re.error:
                continue
            words = _pattern_words(pattern)
            if words >= 2 and any(compiled.search(literal) for literal in literals):
                continue  # the function's own message
            if words >= 3 and pattern.startswith("^") and pattern.endswith("$"):
                continue  # an anchored full message (a deliberately delegated error)
            if any(compiled.search(message) for message in messages):
                out.append(raw)
    return sorted(set(out))


def static_checks(test: str, unit: str, target_text: str) -> tuple[list[str], dict]:
    from sepalith.data.synth import dedup
    problems = []
    code = strip_strings_comments(test)
    bad = sorted({m.group(1) for m in FORBIDDEN_TEST_CALLS.finditer(code)})
    if bad:
        problems.append("forbidden_calls:" + ",".join(bad))
    rng = RNG_CALLS.search(code)
    if rng and ("set.seed(" not in code or code.index("set.seed(") > rng.start()):
        problems.append("rng_without_seed")
    statements = check_statements(test)
    shape = sum(1 for helper, text in statements if shape_only(helper, text))
    share = shape / len(statements) if statements else 1.0
    if statements and share > MAX_SHAPE_SHARE:
        problems.append(f"shape_only_share:{share:.2f}")
    loose = loose_error_patterns(statements, unit)
    if loose:
        problems.append("error_pattern_matches_r_message:" + "|".join(loose)[:200])
    body_tokens = dedup.text_tokens([line for line in target_text.split("\n") if not line.lstrip().startswith("#")])
    hashes = dedup.kgram_hashes(dedup.token_ids(body_tokens))
    index = dedup.TokenIndex("test")
    index.add(test)
    index.freeze()
    copy, run = dedup.overlap(index.hits(hashes))
    if copy > MAX_COPY_CONTAINMENT:
        problems.append(f"copies_implementation:{copy:.2f}")
    return problems, {"assertion_calls_static": len(statements), "shape_only_share": round(share, 3),
                      "implementation_containment": round(copy, 3), "implementation_longest_run": run}


def target_span(unit: str) -> tuple[int, int, int]:
    """(first line, body brace line, last line) of the unit's last top-level function (1-based)."""
    tokens = mutants_module.parse_tokens([unit])[0]
    if tokens is None:
        raise ValueError("unit does not parse")
    tops = [t for t in tokens if t.parent == 0 and not t.terminal]
    target = max(tops, key=lambda t: (t.line1, t.col1))
    fun = min((t for t in tokens if t.token == "FUNCTION" and t.line1 >= target.line1), key=lambda t: (t.line1, t.col1))
    brace = min((t for t in tokens if t.token == "'{'" and (t.line1, t.col1) > (fun.line1, fun.col1)),
                key=lambda t: (t.line1, t.col1))
    return target.line1, brace.line1, target.line2


def stub_unit(unit: str) -> str:
    """The unit with the target body replaced by a stop() (header and closing line kept)."""
    lines = unit.rstrip("\n").split("\n")
    _first, brace, last = target_span(unit)
    head = lines[brace - 1]
    position = head.rindex("{")
    return "\n".join([*lines[:brace - 1], head[:position + 1], '  stop("exec-env stub")', "}",
                      *lines[last:]]) + "\n"


def top_names(unit: str, names: Sequence[str]) -> tuple[str, ...]:
    return tuple(names)


R_DIFF = Path(__file__).with_name("r_diff.R")
DIFF_MAX_INPUTS = 150
GENERIC_CONDITION_CLASSES = {"simpleError", "error", "condition", "simpleCondition", "simpleWarning", "warning"}


NUMTOL_MAX = 1e-2


RELAX_TAGS = re.compile(r"^((?:\s*(?:NUMTOL\[[^\]]*\]|ATTR\[[^\]]*\]|NAMES)\s*,?)+)\s*:")
NAMES_ATTRS = ("names", "dimnames", "row.names")


def relaxation_of(reason: str) -> tuple[str, list[str]] | None:
    """``(tolerance, attributes)`` from leading ``NUMTOL[tol]`` / ``ATTR[a,b]`` / ``NAMES`` tags (combinable,
    e.g. ``ATTR[iter] NUMTOL[1e-6]:``), or None. ``NAMES`` is ``ATTR[names,dimnames,row.names]``."""
    match = RELAX_TAGS.match(reason.strip())
    if not match:
        return None
    tags = match.group(1)
    tol = re.findall(r"NUMTOL\[\s*([^\]]*?)\s*\]", tags)
    attrs = [a.strip() for group in re.findall(r"ATTR\[([^\]]*)\]", tags) for a in group.split(",") if a.strip()]
    if re.search(r"\bNAMES\b", tags):
        attrs += list(NAMES_ATTRS)
    return (tol[-1] if tol else ""), sorted(set(attrs))


def declared_conditions(equivalents: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Per declared mutant: ``("cond", <R condition>)`` for ``SCOPED/BUGFIX/QUIRK[cond]:``;
    ``("relax", "<tol>|<attr,attr>")`` for ``NUMTOL[tol]`` / ``ATTR[attrs]`` / ``NAMES`` tags (numbers compared
    with the tolerance, the listed attributes ignored, everything else exact); else ``("", "")``. Every
    declaration without a condition must hold on every replayed input."""
    out = {}
    for mid, reason in equivalents.items():
        relax = relaxation_of(reason)
        if relax is not None:
            out[mid] = ("relax", f"{relax[0]}|{','.join(relax[1])}")
        else:
            cond = condition_of(reason)
            out[mid] = ("cond", cond) if cond else ("", "")
    return out


def condition_of(reason: str) -> str:
    """The ``<cond>`` of ``SCOPED[<cond>]:`` / ``BUGFIX[...]`` / ``QUIRK[...]``, brackets balanced (so
    ``x[1] > 0`` and ``nrow(m[, 1, drop = FALSE]) == 0`` work), quotes respected; "" if none."""
    text = reason.strip()
    head = re.match(r"^(SCOPED|BUGFIX|QUIRK)\[", text)
    if not head:
        return ""
    depth, quote, escaped = 1, "", False
    for index in range(head.end(), len(text)):
        ch = text[index]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            continue
        if ch in "\"'`":
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                rest = text[index + 1:].lstrip()
                return text[head.end():index].strip() if rest.startswith(":") else ""
    return ""


def differential_checks(unit: str, suite: str, target: str, mutants: Sequence[mutants_module.Mutant],
                        conditions: dict[str, tuple[str, str]], *, timeout: float = 240.0) -> dict:
    """Run r_diff.R in the sandbox: recorded test calls, perturbed inputs, declared-mutant comparisons."""
    with tempfile.TemporaryDirectory(prefix="sepalith-diff-") as tmp:
        root = Path(tmp)
        (root / "work").mkdir()
        (root / "mutants").mkdir()
        (root / "unit.R").write_text(unit, encoding="utf-8")
        (root / "suite.R").write_text(suite, encoding="utf-8")
        for mutant in mutants:
            (root / "mutants" / f"{mutant.id}.R").write_text(mutant.text, encoding="utf-8")
        (root / "mutants" / "conditions.tsv").write_text(
            "".join(f"{mid}\t{kind}\t{value}\n" for mid, (kind, value) in conditions.items()), encoding="utf-8")
        cmd = ["Rscript", "--vanilla", str(R_DIFF), str(root / "unit.R"), str(root / "suite.R"), str(TESTLIB), target,
               str(root / "mutants"), str(DIFF_MAX_INPUTS)]
        if namespaces_available():
            cmd = ["unshare", "-rnm", "--propagation", "private", "sh", str(SANDBOX), str(root / "work"), "256", "--", *cmd]
        if shutil.which("prlimit"):
            cmd = ["prlimit", "--as=4294967296", *cmd]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"error": "differential check timed out"}
    out: dict = {"calls": None, "inputs": None, "args": {}, "always_equal": [], "same": {}, "diff": {},
                 "excluded": {}, "bad_conditions": {}}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        kind = parts[0]
        if kind == "CALLS":
            out["calls"] = int(parts[1])
        elif kind == "INPUTS":
            out["inputs"] = int(parts[1])
        elif kind == "ARG":
            out["args"][parts[1]] = {"distinct_supplied": int(parts[2]), "default_used": parts[3] == "TRUE",
                                     "logical_flag": parts[4] == "TRUE", "sparse": parts[5] == "TRUE"}
        elif kind == "EQUAL":
            out["always_equal"].append([parts[1], parts[2]])
        elif kind == "SAME":
            out["same"][parts[1]] = int(parts[2])
        elif kind == "DIFF":
            out["diff"][parts[1]] = {"input": parts[2], "original": parts[3], "mutant": parts[4]}
        elif kind == "EXCL":
            out["excluded"][parts[1]] = {"excluded": int(parts[2]), "eligible": int(parts[3]),
                                         "share": round(int(parts[2]) / int(parts[3]), 3) if int(parts[3]) else None}
        elif kind == "BADCOND":
            out["bad_conditions"][parts[1]] = parts[2] if len(parts) > 2 else "invalid"
        elif kind == "FAIL":
            out["error"] = parts[1]
        elif kind == "WARN":
            out.setdefault("warnings", []).append(parts[1])
    if out["calls"] is None and "error" not in out:
        out["error"] = "differential check produced no output: " + proc.stderr.strip()[-200:]
    return out


def frozen_cases(unit: str, test: str, target: str, *, timeout: float = 30.0) -> tuple[str | None, str | None]:
    """frozen.R text from ``exec_cases`` (original evaluated once in the sandbox), or (None, error)."""
    if not re.search(r"^exec_cases\s*(<-|=)\s*function", test, flags=re.M):
        return None, None
    script = r"""
args <- commandArgs(trailingOnly = TRUE)
e <- new.env(parent = globalenv())
sys.source(args[[1]], envir = e)
lib <- new.env(parent = globalenv())
sys.source(args[[4]], envir = lib)
t <- new.env(parent = lib)
for (n in ls(e, all.names = TRUE)) assign(n, get(n, envir = e), envir = t)
for (h in c("check", "check_true", "check_equal", "check_identical", "check_error", "check_warning"))
  assign(h, function(...) invisible(NULL), envir = t)
sys.source(args[[2]], envir = t)
cases <- get("exec_cases", envir = t)()
ctl <- c("keepNA", "keepInteger", "niceNames", "showAttributes", "digits17")
fn <- args[[3]]
out <- character()
for (case in cases) {
  value <- tryCatch(do.call(get(fn, envir = e), case), error = function(err) err)
  if (inherits(value, "error")) next
  a <- paste(deparse(case, control = ctl), collapse = "\n")
  v <- paste(deparse(value, control = ctl), collapse = "\n")
  out <- c(out, sprintf('check_equal("differential", do.call(%s, %s), %s)', fn, a, v))
}
writeLines(out)
"""
    with tempfile.TemporaryDirectory(prefix="sepalith-freeze-") as tmp:
        work = Path(tmp) / "work"
        work.mkdir()
        for name, text in (("unit.R", unit), ("test.R", test), ("freeze.R", script)):
            (Path(tmp) / name).write_text(text, encoding="utf-8")
        cmd = ["Rscript", "--vanilla", str(Path(tmp) / "freeze.R"), str(Path(tmp) / "unit.R"), str(Path(tmp) / "test.R"),
               target, str(TESTLIB)]
        if namespaces_available():
            cmd = ["unshare", "-rnm", "--propagation", "private", "sh", str(SANDBOX), str(work), "64", "--", *cmd]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return None, "exec_cases_timeout"
    if proc.returncode != 0:
        return None, "exec_cases_failed:" + proc.stderr.strip()[-200:]
    text = proc.stdout
    if len(text) > 200_000:
        return None, "exec_cases_output_too_large"
    return text, None


def suite_text(directory: Path) -> str:
    test = (directory / "test.R").read_text(encoding="utf-8")
    frozen = directory / "frozen.R"
    return test + ("\n" + frozen.read_text(encoding="utf-8") if frozen.is_file() else "")


def killed(result: ExecResult, expected: int) -> bool:
    return result.status != "ok" or result.error_class in ("timeout", "candidate_exit") or result.total != expected \
        or result.passed < expected


def declared_equivalents(directory: Path) -> dict[str, str]:
    path = directory / "equivalent.txt"
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").split("\n"):
            parts = line.strip().split(None, 1)
            if parts and not parts[0].startswith("#"):
                out[parts[0]] = parts[1] if len(parts) > 1 else ""
    return out


OUTPUT_ONLY = re.compile(r"^(print|plot|autoplot|show|lines|points|image|persp|text|cat)\.|^(show|display)$")


def gate_function(directory: Path, harness: ExecHarness, *, mutant_limit: int = MUTANT_LIMIT) -> dict:
    meta = json.loads((directory / "meta.json").read_text())
    if OUTPUT_ONLY.search(meta["name"]):
        report = {"schema": GATE_SCHEMA, "gate_version": gate_version(), "id": meta["id"], "name": meta["name"], "verdict": "skip",
                  "problems": ["output_only_method"]}
        (directory / "gate.json").write_text(json.dumps(report, indent=1) + "\n")
        return report
    unit = (directory / "source.R").read_text(encoding="utf-8")
    test = (directory / "test.R").read_text(encoding="utf-8")
    expect = (*meta["helpers"], meta["name"])
    first, _brace, last = target_span(unit)
    target_text = "\n".join(unit.split("\n")[first - 1:last])
    report: dict = {"schema": GATE_SCHEMA, "gate_version": gate_version(), "id": meta["id"], "name": meta["name"],
                    "problems": []}
    problems, stats = static_checks(test, unit, target_text)
    report["static"] = stats
    report["problems"] += problems
    frozen, error = frozen_cases(unit, test, meta["name"])
    if error:
        report["problems"].append(error)
    if frozen is not None:
        (directory / "frozen.R").write_text(frozen, encoding="utf-8")
        report["frozen_cases"] = frozen.count('check_equal("differential"')
    elif (directory / "frozen.R").exists():
        (directory / "frozen.R").unlink()
    suite = test + ("\n" + frozen if frozen else "")
    originals = harness.run_many([ExecJob(unit, suite, expect, TIMEOUT, f"orig-{i}") for i in range(RUNS_ORIGINAL)])
    first_run = originals[0]
    report["original"] = first_run.to_dict()
    outcomes = {tuple(r.checks) for r in originals}
    report["checks"] = first_run.total
    report["categories"] = dict(Counter(c for c, _ in first_run.checks))
    original_ok = first_run.status == "ok" and not first_run.error_class and first_run.passed == first_run.total
    if not original_ok:
        report["problems"].append(f"original_fails:{first_run.error_class or ''}:{first_run.passed}/{first_run.total}")
        statements = check_statements(test)
        failing = [i for i, (_c, ok) in enumerate(first_run.checks) if not ok]
        # assertion i maps to the i-th assertion statement when every statement ran exactly once
        aligned = len(statements) + (frozen or "").count('check_equal("differential"') == first_run.total
        report["original_failing_assertions"] = [
            {"index": i + 1, "category": first_run.checks[i][0],
             **({"statement": statements[i][1][:300]} if aligned and i < len(statements) else {})}
            for i in failing[:20]]
        if first_run.error:
            report["original_error"] = first_run.error
    if len(outcomes) != 1:
        report["problems"].append("nondeterministic_original")
    hand = Counter(c for c, _ in first_run.checks if c != "differential")
    report["hand_written_checks"] = sum(hand.values())
    if sum(hand.values()) < MIN_HAND_CHECKS:
        report["problems"].append(f"too_few_hand_written_assertions:{sum(hand.values())}")
    if len(hand) < MIN_CATEGORIES:
        report["problems"].append(f"too_few_categories:{len(hand)}")
    thin = sorted(c for c, n in hand.items() if n < MIN_PER_CATEGORY)
    if thin:
        report["problems"].append("categories_with_one_assertion:" + ",".join(thin)[:200])
    expected = first_run.total
    if not original_ok:
        report["verdict"] = "fail"
        report["mutants"] = {"note": "not run: the original fails"}
        (directory / "gate.json").write_text(json.dumps(report, indent=1) + "\n")
        return report
    stub = harness.run(ExecJob(stub_unit(unit), suite, expect, TIMEOUT, "stub"))
    stub_share = stub.passed / expected if expected else 1.0
    report["stub_pass_share"] = round(stub_share, 3)
    if stub_share > MAX_STUB_PASS_SHARE:
        report["problems"].append(f"tests_do_not_reach_function:{stub_share:.2f}")
    tokens = mutants_module.parse_tokens([unit])[0]
    assert tokens is not None
    all_mutants = mutants_module.generate(unit, tokens)
    chosen = mutants_module.sample(all_mutants, mutant_limit, meta["id"])
    results = harness.run_many([ExecJob(m.text, suite, expect, TIMEOUT, m.id) for m in chosen])
    equivalents = declared_equivalents(directory)
    by_op: dict[str, list[int]] = {}
    survivors, invalid, valid, kills, declared = [], 0, 0, 0, 0
    declared_killed: list[str] = []
    records = []
    for mutant, result in zip(chosen, results, strict=True):
        if result.parse_ok is False or result.structure_ok is False or (result.sourced_ok is False
                                                                       and result.error_class == "source_error"):
            invalid += 1
            continue
        dead = killed(result, expected)
        records.append({**mutant.to_dict(), "killed": dead})
        if mutant.id in equivalents:
            if not dead:
                declared += 1
                continue
            declared_killed.append(mutant.id)
        valid += 1
        cell = by_op.setdefault(mutant.op, [0, 0])
        cell[1] += 1
        if dead:
            kills += 1
            cell[0] += 1
        else:
            survivors.append({**mutant.to_dict(), "original_lines": list(mutant.original),
                              "mutated_lines": list(mutant.replacement)})
    generated_ids = {m.id for m in all_mutants}
    stale = sorted(set(equivalents) - generated_ids)
    rate = kills / valid if valid else 0.0
    report["mutants"] = {"generated": len(all_mutants), "sampled": len(chosen), "invalid": invalid,
                         "declared_equivalent": declared, "valid": valid, "killed": kills,
                         "kill_rate": round(rate, 3), "by_op": {k: {"killed": v[0], "valid": v[1]} for k, v in sorted(by_op.items())},
                         "survivors": survivors, "stale_equivalent_ids": stale,
                         "declared_equivalent_but_killed": declared_killed}
    report["mutant_outcomes"] = records
    if valid < MIN_VALID_MUTANTS:
        report["problems"].append(f"too_few_valid_mutants:{valid}")
    elif rate < MIN_KILL_RATE:
        report["problems"].append(f"unkilled_survivors:{valid - kills} (kill or declare each; rate {rate:.2f})")
    # v3: differential verification of every declared equivalent, argument diversity, fixture density
    declared_mutants = [m for m in all_mutants if m.id in equivalents]
    conditions = declared_conditions(equivalents)
    for mid, (kind, value) in sorted(conditions.items()):
        if kind == "relax":
            tol_text, attrs = value.split("|", 1)
            if tol_text:
                try:
                    tolerance = float(tol_text)
                except ValueError:
                    tolerance = float("nan")
                if not 0 < tolerance <= NUMTOL_MAX:
                    report["problems"].append(f"numtol_invalid:{mid}={tol_text} (0 < tol <= {NUMTOL_MAX})")
                    conditions[mid] = ("relax", f"|{attrs}")
            if not tol_text and not attrs:
                report["problems"].append(f"relaxation_empty:{mid}")
    diff = differential_checks(unit, suite, meta["name"], declared_mutants, conditions)
    report["differential"] = diff
    report.setdefault("warnings", [])
    if "error" in diff:
        if declared_mutants:
            # unverified declarations must not pass silently
            report["problems"].append("equivalence_unverified: " + str(diff["error"])[:150])
        else:
            report["warnings"].append("differential_check_unavailable: " + str(diff["error"])[:150])
    report["warnings"] += diff.get("warnings", [])
    for mid in sorted(diff.get("diff", {})):
        report["problems"].append(f"equivalence_refuted:{mid}")
    for mid, why in sorted(diff.get("bad_conditions", {}).items()):
        report["problems"].append(f"condition_invalid:{mid} ({why[:120]})")
    for mid, cell in sorted(diff.get("excluded", {}).items()):
        if conditions.get(mid, ("", ""))[0] == "cond" and cell["eligible"] and cell["excluded"] == cell["eligible"]:
            report["problems"].append(f"blanket_condition:{mid} (excludes all {cell['eligible']} replayed inputs)")
    if diff.get("calls"):
        # each argument the body reads takes 2+ distinct supplied values (a logical flag: its
        # non-default value at least once)
        narrow = sorted(a for a, v in diff["args"].items()
                        if (v["distinct_supplied"] < 1 if v["logical_flag"] else v["distinct_supplied"] < 2))
        if narrow:
            report["problems"].append("argument_not_varied:" + ",".join(narrow)[:200])
        for a, b in diff["always_equal"]:
            report["problems"].append(f"arguments_always_equal:{a}={b}")
        sparse = sorted(a for a, v in diff["args"].items() if v["sparse"])
        if sparse:
            report["warnings"].append("sparse_matrix_fixtures:" + ",".join(sparse))
    elif "error" not in diff:
        report["warnings"].append("no_recorded_calls: the suite never calls the function by name")
    probes = sorted((directory / "probes").glob("*.R")) if (directory / "probes").is_dir() else []
    if probes:
        outcomes = harness.run_many([ExecJob(p.read_text(encoding="utf-8"), suite, expect, TIMEOUT, p.name) for p in probes])
        report["probes"] = {}
        for probe, result in zip(probes, outcomes, strict=True):
            if result.parse_ok is False or result.structure_ok is False:
                verdict = "invalid"
            else:
                verdict = "rejected" if killed(result, expected) else "passes"
            report["probes"][probe.name] = verdict
            if verdict == "passes":
                report["problems"].append(f"probe_passes:{probe.name}")
    report["verdict"] = "pass" if not report["problems"] else "fail"
    (directory / "gate.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def function_dirs(paths: Sequence[Path]) -> list[Path]:
    out = []
    for path in paths:
        if (path / "source.R").is_file() and (path / "meta.json").is_file():
            if (path / "test.R").is_file() and not (path / "SKIP").exists():
                out.append(path)
        else:
            out += sorted(p.parent for p in path.glob("*/test.R")
                          if (p.parent / "source.R").is_file() and not (p.parent / "SKIP").exists())
    return out


def rule_of(problem: str) -> str:
    """The rule name of a gate problem (``kill_rate:0.5`` -> ``kill_rate``)."""
    return problem.split(":", 1)[0].split(" ", 1)[0]


def summarize_slice(dirs: Sequence[Path]) -> dict:
    """Per-function verdicts and failing rules from existing ``gate.json`` files (no re-run)."""
    rows, rules, verdicts = [], Counter(), Counter()
    for directory in dirs:
        path = directory / "gate.json"
        report = json.loads(path.read_text()) if path.is_file() else {"verdict": "not_gated", "problems": []}
        failing = sorted({rule_of(p) for p in report.get("problems", [])})
        verdicts[report["verdict"]] += 1
        rules.update(failing if report["verdict"] not in ("pass", "skip") else [])
        name = report.get("name") or (json.loads((directory / "meta.json").read_text()).get("name")
                                      if (directory / "meta.json").is_file() else "")
        rows.append({"function": directory.name, "name": name, "verdict": report["verdict"],
                     "hand_written": report.get("hand_written_checks"), "rules": failing})
    return {"verdicts": dict(verdicts), "failing_rules": dict(rules.most_common()), "functions": rows}


def print_summary(summary: dict) -> None:
    print("\n== summary ==")
    for row in summary["functions"]:
        rules = ",".join(row["rules"]) if row["verdict"] not in ("pass", "skip") else ""
        print(f"{row['verdict']:9} {row['function']}  {row['name'][:40]:40}  {rules}")
    print("verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["verdicts"].items())))
    if summary["failing_rules"]:
        print("failing rules: " + ", ".join(f"{k} {v}" for k, v in summary["failing_rules"].items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mutants", type=int, default=MUTANT_LIMIT)
    parser.add_argument("--summary", action="store_true",
                        help="after gating, print one line per function with its failing rules and a rule tally")
    parser.add_argument("--report-only", action="store_true",
                        help="do not run anything: summarise the existing gate.json files")
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    args = parser.parse_args(argv)
    dirs = function_dirs(args.dirs)
    if not args.report_only:
        with ExecHarness(workers=min(args.workers, 8)) as harness:
            for directory in dirs:
                try:
                    report = gate_function(directory, harness, mutant_limit=args.mutants)
                except Exception as error:  # noqa: BLE001 - one broken directory must not stop a slice
                    report = {"verdict": "error", "problems": [f"{type(error).__name__}: {error}"]}
                    (directory / "gate.json").write_text(json.dumps(report, indent=1) + "\n")
                mutants = report.get("mutants", {})
                print(f"{report['verdict']:5} {directory.name} checks={report.get('checks')} "
                      f"kill={mutants.get('killed')}/{mutants.get('valid')} stub={report.get('stub_pass_share')} "
                      f"{';'.join(report['problems'])}", flush=True)
    summary = summarize_slice(dirs)
    if args.summary or args.report_only:
        print_summary(summary)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    ok = sum(v for k, v in summary["verdicts"].items() if k in ("pass", "skip"))
    print(json.dumps({"functions": len(dirs), "passed": ok}))
    return 0 if ok == len(dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
