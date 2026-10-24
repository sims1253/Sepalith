"""Quality gate for exec-env test files (convention exec-test-v1).

    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.exec_env.gate DIR [DIR ...]

``DIR`` is a function directory (``source.R`` + ``test.R``) or a directory of
them (a slice). For each function the gate writes ``gate.json`` (and
``frozen.R`` when the test defines ``exec_cases``) and prints one line.

A test file passes when all of these hold:

1. **Static**: it parses; it uses the assertion helpers only (no printing,
   timing, I/O, sourcing or package loading, RNG only after ``set.seed``); at
   least ``MIN_CHECKS`` assertions in at least ``MIN_CATEGORIES`` categories;
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
   denominator; at least ``MIN_KILL_RATE`` of the rest must fail at least one
   assertion (or time out / crash). Survivors are listed in ``gate.json``.

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
import subprocess
import tempfile

from sepalith.exec_env import mutants as mutants_module
from sepalith.exec_env.harness import SANDBOX, ExecHarness, ExecJob, ExecResult, namespaces_available

GATE_SCHEMA = "sepalith.exec-env.gate.v1"
MIN_CHECKS = 6
MIN_CATEGORIES = 2
MAX_SHAPE_SHARE = 0.3
MAX_COPY_CONTAINMENT = 0.25
RUNS_ORIGINAL = 3
MAX_STUB_PASS_SHARE = 0.2
MUTANT_LIMIT = 60
MIN_VALID_MUTANTS = 5
MIN_KILL_RATE = 0.7
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


def frozen_cases(unit: str, test: str, target: str, *, timeout: float = 30.0) -> tuple[str | None, str | None]:
    """frozen.R text from ``exec_cases`` (original evaluated once in the sandbox), or (None, error)."""
    if not re.search(r"^exec_cases\s*(<-|=)\s*function", test, flags=re.M):
        return None, None
    script = r"""
args <- commandArgs(trailingOnly = TRUE)
e <- new.env(parent = globalenv())
sys.source(args[[1]], envir = e)
t <- new.env(parent = e)
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
               target]
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


def gate_function(directory: Path, harness: ExecHarness, *, mutant_limit: int = MUTANT_LIMIT) -> dict:
    meta = json.loads((directory / "meta.json").read_text())
    unit = (directory / "source.R").read_text(encoding="utf-8")
    test = (directory / "test.R").read_text(encoding="utf-8")
    expect = (*meta["helpers"], meta["name"])
    first, _brace, last = target_span(unit)
    target_text = "\n".join(unit.split("\n")[first - 1:last])
    report: dict = {"schema": GATE_SCHEMA, "id": meta["id"], "name": meta["name"], "problems": []}
    problems, stats = static_checks(test, unit, target_text)
    report["static"] = stats
    report["problems"] += problems
    frozen, error = frozen_cases(unit, test, meta["name"])
    if error:
        report["problems"].append(error)
    if frozen is not None:
        (directory / "frozen.R").write_text(frozen, encoding="utf-8")
        report["frozen_cases"] = frozen.count("\n")
    elif (directory / "frozen.R").exists():
        (directory / "frozen.R").unlink()
    suite = test + ("\n" + frozen if frozen else "")
    originals = harness.run_many([ExecJob(unit, suite, expect, TIMEOUT, f"orig-{i}") for i in range(RUNS_ORIGINAL)])
    first_run = originals[0]
    report["original"] = first_run.to_dict()
    outcomes = {tuple(r.checks) for r in originals}
    categories = sorted({c for c, _ in first_run.checks})
    report["checks"] = first_run.total
    report["categories"] = dict(Counter(c for c, _ in first_run.checks))
    if first_run.status != "ok" or first_run.error_class or first_run.passed != first_run.total:
        report["problems"].append(f"original_fails:{first_run.error_class or ''}:{first_run.passed}/{first_run.total}")
    if len(outcomes) != 1:
        report["problems"].append("nondeterministic_original")
    if first_run.total < MIN_CHECKS:
        report["problems"].append(f"too_few_assertions:{first_run.total}")
    if len(categories) < MIN_CATEGORIES:
        report["problems"].append(f"too_few_categories:{len(categories)}")
    expected = first_run.total
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
    records = []
    for mutant, result in zip(chosen, results, strict=True):
        if result.parse_ok is False or result.structure_ok is False or (result.sourced_ok is False
                                                                       and result.error_class == "source_error"):
            invalid += 1
            continue
        dead = killed(result, expected)
        records.append({**mutant.to_dict(), "killed": dead})
        if mutant.id in equivalents and not dead:
            declared += 1
            continue
        valid += 1
        cell = by_op.setdefault(mutant.op, [0, 0])
        cell[1] += 1
        if dead:
            kills += 1
            cell[0] += 1
        else:
            changed = [line for line in mutant.text.split("\n")[mutant.line1 - 1:mutant.line1 + 1]][:2]
            survivors.append({**mutant.to_dict(), "mutated_lines": changed})
    rate = kills / valid if valid else 0.0
    report["mutants"] = {"generated": len(all_mutants), "sampled": len(chosen), "invalid": invalid,
                         "declared_equivalent": declared, "valid": valid, "killed": kills,
                         "kill_rate": round(rate, 3), "by_op": {k: {"killed": v[0], "valid": v[1]} for k, v in sorted(by_op.items())},
                         "survivors": survivors}
    report["mutant_outcomes"] = records
    if valid < MIN_VALID_MUTANTS:
        report["problems"].append(f"too_few_valid_mutants:{valid}")
    elif rate < MIN_KILL_RATE:
        report["problems"].append(f"kill_rate:{rate:.2f}")
    report["verdict"] = "pass" if not report["problems"] else "fail"
    (directory / "gate.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def function_dirs(paths: Sequence[Path]) -> list[Path]:
    out = []
    for path in paths:
        if (path / "source.R").is_file() and (path / "meta.json").is_file():
            if (path / "test.R").is_file():
                out.append(path)
        else:
            out += sorted(p.parent for p in path.glob("*/test.R") if (p.parent / "source.R").is_file())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mutants", type=int, default=MUTANT_LIMIT)
    args = parser.parse_args(argv)
    dirs = function_dirs(args.dirs)
    passed = 0
    with ExecHarness(workers=min(args.workers, 8)) as harness:
        for directory in dirs:
            try:
                report = gate_function(directory, harness, mutant_limit=args.mutants)
            except Exception as error:  # noqa: BLE001 - one broken directory must not stop a slice
                report = {"verdict": "error", "problems": [f"{type(error).__name__}: {error}"]}
                (directory / "gate.json").write_text(json.dumps(report, indent=1) + "\n")
            passed += report["verdict"] == "pass"
            mutants = report.get("mutants", {})
            print(f"{report['verdict']:5} {directory.name} checks={report.get('checks')} "
                  f"kill={mutants.get('killed')}/{mutants.get('valid')} stub={report.get('stub_pass_share')} "
                  f"{';'.join(report['problems'])}", flush=True)
    print(json.dumps({"functions": len(dirs), "passed": passed}))
    return 0 if passed == len(dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
