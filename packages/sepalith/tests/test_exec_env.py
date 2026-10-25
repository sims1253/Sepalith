"""Exec-env: sandboxed harness, mutants, gate, per-cut tasks and the execution reward."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from sepalith.campaign_protocol import PromptContext, serialize_target
from sepalith.data.synth.cases import Case

HAS_R = shutil.which("Rscript") is not None
HAS_AIR = shutil.which("air") is not None

UNIT = '''clamp_unit <- function(x) {
  pmin(pmax(x, 0), 1)
}

#' Summarise scores into bands
#'
#' @param scores Numeric vector of scores.
#' @param cut Threshold between low and high.
band_counts <- function(scores, cut = 0.5) {
  if (!is.numeric(scores)) {
    stop("scores must be numeric")
  }
  scores <- clamp_unit(scores[!is.na(scores)])
  low <- sum(scores < cut)
  high <- sum(scores >= cut)
  total <- length(scores)
  if (total == 0) {
    return(c(low = 0L, high = 0L, total = 0L))
  }
  out <- c(low = low, high = high, total = total)
  storage.mode(out) <- "integer"
  out
}
'''
TEST = '''check_identical("counts", band_counts(c(0.1, 0.6, 0.9)), c(low = 1L, high = 2L, total = 3L))
check_identical("counts", band_counts(c(0.2, 0.3), cut = 0.25), c(low = 1L, high = 1L, total = 2L))
check_identical("counts", band_counts(c(0.9, 0.8, 0.7, 0.1), cut = 0.75), c(low = 2L, high = 2L, total = 4L))
check_identical("counts", band_counts(0.3), c(low = 1L, high = 0L, total = 1L))
check_identical("boundary", band_counts(c(0.5, 0.49999)), c(low = 1L, high = 1L, total = 2L))
check_identical("boundary", band_counts(c(0.25, 0.25), cut = 0.25), c(low = 0L, high = 2L, total = 2L))
check_identical("boundary", band_counts(c(-3, 7)), c(low = 1L, high = 1L, total = 2L))
check_identical("boundary", band_counts(c(1.5, 2), cut = 1), c(low = 0L, high = 2L, total = 2L))
check_identical("missing", band_counts(c(NA, 0.7, NA)), c(low = 0L, high = 1L, total = 1L))
check_identical("missing", band_counts(numeric()), c(low = 0L, high = 0L, total = 0L))
check_identical("missing", band_counts(c(NA_real_)), c(low = 0L, high = 0L, total = 0L))
check_error("missing", band_counts("a"), pattern = "scores must be numeric")
exec_cases <- function() {
  set.seed(42)
  lapply(1:6, function(i) list(scores = round(runif(8, -0.5, 1.5), 2), cut = round(runif(1), 2)))
}
'''
EXPECT = ("clamp_unit", "band_counts")
EQUIVALENT = """b9bb95e29d clamping twice is idempotent
6fc928b60f sum() of an empty logical and length() are already 0L, so the general path returns the same vector
de02ee2c26 same as deleting the empty-input branch
80ac0b2942 same as deleting the empty-input branch
973faa667a sum() of logicals and length() are already integer
"""


def replace_body(unit: str, old: str, new: str) -> str:
    assert old in unit
    return unit.replace(old, new)


@unittest.skipUnless(HAS_R, "needs Rscript")
class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from sepalith.exec_env.harness import ExecHarness
        cls.harness = ExecHarness(workers=2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.harness.close()

    def run_one(self, source: str, timeout: float = 3.0):
        from sepalith.exec_env.harness import ExecJob
        return self.harness.run(ExecJob(source, TEST, EXPECT, timeout, "t"))

    def test_gold_passes_every_assertion(self):
        result = self.run_one(UNIT)
        self.assertEqual((result.passed, result.total), (12, 12))
        self.assertEqual(result.by_category()["counts"], (4, 4))

    def test_wrong_code_fails_some(self):
        result = self.run_one(replace_body(UNIT, "scores < cut", "scores <= cut"))
        self.assertLess(result.passed, result.total)

    def test_quit_ends_only_the_child(self):
        result = self.run_one(replace_body(UNIT, "  out\n}", '  quit("no")\n}'))
        self.assertEqual(result.status, "candidate_exit")
        self.assertEqual(self.run_one(UNIT).passed, 12)

    def test_hang_times_out(self):
        result = self.run_one(replace_body(UNIT, "  out\n}", "  repeat {}\n}"), timeout=1.0)
        self.assertEqual(result.error_class, "timeout")
        self.assertEqual(result.passed, 0)

    def test_escaping_the_function_breaks_structure(self):
        result = self.run_one(replace_body(UNIT, "  out\n}", "  out\n}\nidentical <- function(...) TRUE"))
        self.assertFalse(result.structure_ok)

    def test_global_planting_cannot_hijack_assertions(self):
        planted = replace_body(UNIT, "  out\n}", "  identical <<- function(...) TRUE\n  NULL\n}")
        # the locked global environment makes `<<-` an error: only the validation assertion and the
        # empty-input early returns (which never reach the plant) still pass
        self.assertEqual(self.run_one(planted).passed, 3)

    def test_seeded_rng_works_in_tests(self):
        from sepalith.exec_env.harness import ExecJob
        test = 'set.seed(1)\nx <- runif(3)\nset.seed(1)\ncheck("rng", identical(runif(3), x))\n'
        result = self.harness.run(ExecJob(UNIT, test, EXPECT, 3.0, "rng"))
        self.assertEqual((result.passed, result.total), (1, 1))

    def test_strict_fakes_reject_unknown_fields(self):
        from sepalith.exec_env.harness import ExecJob
        src = "f <- function(o) {\n  o$a + 1\n}\ng <- function(o) {\n  o$a + length(o$extra)\n}\n"
        test = ('o <- strict(list(a = 1), class = "thing")\ncheck_equal("s", f(o), 2)\ncheck_error("s", g(o))\n'
                'check_equal("s", g(list(a = 1)), 1)\ncheck("s", inherits(o, "thing"))\n')
        result = self.harness.run(ExecJob(src, test, ("f", "g"), 3.0, "strict"))
        self.assertEqual((result.passed, result.total), (4, 4))

    def test_filesystem_and_network_are_closed(self):
        if not self.harness.sandbox:
            self.skipTest("no user namespaces")
        probe = Path(tempfile.gettempdir()) / "sepalith-exec-probe.txt"
        writes = replace_body(UNIT, "  out\n}", f'  writeLines("x", "{probe}")\n  out\n}}')
        self.assertLess(self.run_one(writes).passed, 12)
        self.assertFalse(probe.exists())
        net = replace_body(UNIT, "  out\n}", '  readLines(url("http://example.com"))\n  out\n}')
        self.assertLess(self.run_one(net).passed, 12)


@unittest.skipUnless(HAS_R, "needs Rscript")
class ConditionParsingTests(unittest.TestCase):
    def test_nested_brackets_and_quotes(self):
        from sepalith.exec_env.gate import condition_of
        self.assertEqual(condition_of("SCOPED[x[1] > 0]: only positive starts"), "x[1] > 0")
        self.assertEqual(condition_of("QUIRK[nrow(m[, 1, drop = FALSE]) == 0]: empty"), "nrow(m[, 1, drop = FALSE]) == 0")
        self.assertEqual(condition_of('BUGFIX[any(x == "]")]: literal bracket'), 'any(x == "]")')
        self.assertEqual(condition_of("BUGFIX[l[[\"a\"]][2] < 0] : spaced colon"), 'l[["a"]][2] < 0')
        self.assertEqual(condition_of("PRINT: no condition"), "")
        self.assertEqual(condition_of("SCOPED[x[1] > 0 no colon"), "")


@unittest.skipUnless(HAS_R and HAS_AIR, "needs Rscript and air")
class NumericToleranceTests(unittest.TestCase):
    def test_numtol_declarations(self):
        from sepalith.exec_env import mutants
        from sepalith.exec_env.gate import gate_function
        from sepalith.exec_env.harness import ExecHarness
        unit = "fit <- function(x, k = 2) {\n  y <- x * k\n  z <- y + 0.00001\n  z\n}\n"
        bump = next(m for m in mutants.generate(unit, mutants.parse_tokens([unit])[0])
                    if m.op == "constant" and "0.00001 * 2" in m.text)
        test = "".join(f'check_equal("{c}", fit({x}, k = {k}), {x} * {k}, tolerance = 1e-3)\n'
                       for c, x, k in [("a", 1, 2), ("a", 2, 3), ("b", 3, 1), ("b", 4, 5), ("c", -1, 2), ("c", 0.5, 4)])
        with tempfile.TemporaryDirectory() as tmp, ExecHarness(workers=2) as harness:
            d = Path(tmp) / "f"
            d.mkdir()
            (d / "source.R").write_text(unit)
            (d / "test.R").write_text(test)
            (d / "meta.json").write_text(json.dumps({"id": "f", "package": "p", "path": "R/f.R", "name": "fit",
                                                      "helpers": [], "formals": ["x", "k"]}))
            outcomes = {}
            for tol in ("1e-4", "1e-7", "0.5"):
                (d / "equivalent.txt").write_text(f"{bump.id} NUMTOL[{tol}]: offset noise\n")
                outcomes[tol] = gate_function(d, harness)["problems"]
        self.assertFalse(any(bump.id in p for p in outcomes["1e-4"]), outcomes["1e-4"])
        self.assertIn(f"equivalence_refuted:{bump.id}", outcomes["1e-7"])
        self.assertTrue(any(p.startswith(f"numtol_invalid:{bump.id}") for p in outcomes["0.5"]))


@unittest.skipUnless(HAS_R and HAS_AIR, "needs Rscript and air")
class DeclarationKindTests(unittest.TestCase):
    """NUMTOL through S4 slots, NAMES, and invalid conditions."""

    def run_gate(self, unit: str, name: str, formals: list[str], test: str, declarations: list[str]) -> list[list[str]]:
        from sepalith.exec_env.gate import gate_function
        from sepalith.exec_env.harness import ExecHarness
        out = []
        with tempfile.TemporaryDirectory() as tmp, ExecHarness(workers=2) as harness:
            d = Path(tmp) / "f"
            d.mkdir()
            (d / "source.R").write_text(unit)
            (d / "test.R").write_text(test)
            (d / "meta.json").write_text(json.dumps({"id": "f", "package": "p", "path": "R/f.R", "name": name,
                                                      "helpers": [], "formals": formals}))
            for line in declarations:
                (d / "equivalent.txt").write_text(line + "\n")
                out.append(gate_function(d, harness)["problems"])
        return out

    def test_numtol_reaches_s4_slots(self):
        from sepalith.exec_env import mutants
        unit = "fit <- function(x, k = 2) {\n  y <- x * k\n  z <- y + 0.00001\n  Matrix::Matrix(z, nrow = 1)\n}\n"
        bump = next(m for m in mutants.generate(unit, mutants.parse_tokens([unit])[0])
                    if m.op == "constant" and "0.00001 * 2" in m.text)
        test = "".join(f'check_equal("{c}", as.numeric(fit(c({x}, 1), k = {k})), c({x}, 1) * {k}, tolerance = 1e-3)\n'
                       for c, x, k in [("a", 1, 2), ("a", 2, 3), ("b", 3, 1), ("b", 4, 5), ("c", -1, 2), ("c", 0.5, 4)])
        ok, tight = self.run_gate(unit, "fit", ["x", "k"], test,
                                  [f"{bump.id} NUMTOL[1e-4]: offset noise", f"{bump.id} NUMTOL[1e-7]: offset noise"])
        self.assertFalse(any(bump.id in p for p in ok), ok)
        self.assertIn(f"equivalence_refuted:{bump.id}", tight)

    def test_names_declaration(self):
        from sepalith.exec_env import mutants
        unit = 'lab <- function(x, k = 2) {\n  out <- x * k\n  names(out) <- paste0("v", seq_along(x))\n  out\n}\n'
        rename = next(m for m in mutants.generate(unit, mutants.parse_tokens([unit])[0]) if m.description == "string altered")
        test = "".join(f'check_identical("{c}", unname(lab({x}, k = {k})), unname({x} * {k}))\n'
                       for c, x, k in [("a", "1:2", 2), ("a", "3", 3), ("b", "c(2, 5)", 1), ("b", "4", 5), ("c", "-1", 2),
                                       ("c", "c(0.5, 1)", 4)])
        names_ok, plain, bad = self.run_gate(unit, "lab", ["x", "k"], test,
                                             [f"{rename.id} NAMES: label prefix only",
                                              f"{rename.id} labels only",
                                              f"{rename.id} QUIRK[k \\&\\& TRUE]: escaped operator"])
        self.assertFalse(any(rename.id in p for p in names_ok), names_ok)
        self.assertIn(f"equivalence_refuted:{rename.id}", plain)
        self.assertTrue(any(p.startswith(f"condition_invalid:{rename.id}") for p in bad), bad)


class MutantTests(unittest.TestCase):
    def test_classes_and_spans(self):
        from sepalith.exec_env import mutants
        tokens = mutants.parse_tokens([UNIT])[0]
        found = mutants.generate(UNIT, tokens)
        ops = {m.op for m in found}
        self.assertTrue({"relational", "constant", "statement_delete", "early_return", "branch", "logical"}
                        <= ops)
        self.assertTrue(all(m.line1 >= 9 for m in found))  # never the helper
        self.assertFalse(any("!!is.na" in m.text for m in found))  # no double negation duplicate
        picked = mutants.sample(found, 10, "seed")
        self.assertEqual(len(picked), 10)
        self.assertEqual([m.id for m in picked], [m.id for m in mutants.sample(found, 10, "seed")])


@unittest.skipUnless(HAS_R and HAS_AIR, "needs Rscript and air")
class GateTaskRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from sepalith.exec_env.harness import ExecHarness
        cls.harness = ExecHarness(workers=2)
        cls.tmp = tempfile.mkdtemp(prefix="sepalith-exec-test-")
        cls.fdir = Path(cls.tmp) / "S000" / "fx"
        cls.fdir.mkdir(parents=True)
        (cls.fdir / "source.R").write_text(UNIT)
        (cls.fdir / "meta.json").write_text(json.dumps({"id": "fx", "package": "fixture", "path": "R/bands.R",
                                                          "name": "band_counts", "helpers": ["clamp_unit"],
                                                          "formals": ["scores", "cut"]}))
        (cls.fdir / "equivalent.txt").write_text(EQUIVALENT)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.harness.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def gate(self, test: str) -> dict:
        from sepalith.exec_env.gate import gate_function
        (self.fdir / "test.R").write_text(test)
        return gate_function(self.fdir, self.harness)

    def test_good_suite_passes_and_freezes_cases(self):
        report = self.gate(TEST)
        self.assertEqual(report["verdict"], "pass", report["problems"])
        self.assertEqual(report["categories"]["differential"], 6)
        self.assertGreaterEqual(report["mutants"]["kill_rate"], 0.7)

    def test_self_probes_must_be_rejected(self):
        probes = self.fdir / "probes"
        probes.mkdir(exist_ok=True)
        try:
            (probes / "caught.R").write_text(UNIT.replace("scores < cut", "scores <= cut"))
            (probes / "missed.R").write_text(UNIT.replace("  out\n}", "  out[] <- out\n  out\n}"))
            report = self.gate(TEST)
            self.assertEqual(report["probes"], {"caught.R": "rejected", "missed.R": "passes"})
            self.assertIn("probe_passes:missed.R", report["problems"])
        finally:
            shutil.rmtree(probes)

    def test_false_equivalence_is_refuted(self):
        from sepalith.exec_env import mutants
        flip = next(m for m in mutants.generate(UNIT, mutants.parse_tokens([UNIT])[0]) if m.description == "< -> >=")
        (self.fdir / "equivalent.txt").write_text(EQUIVALENT + f"{flip.id} PRINT: only changes output\n")
        try:
            report = self.gate(TEST)
        finally:
            (self.fdir / "equivalent.txt").write_text(EQUIVALENT)
        self.assertIn(f"equivalence_refuted:{flip.id}", report["problems"])
        self.assertIn("input", report["differential"]["diff"][flip.id])
        self.assertEqual(len(report["differential"]["same"]), 5)  # the genuine equivalents survive the check

    def test_conditions_see_defaults_and_blankets_fail(self):
        first = EQUIVALENT.splitlines()[0].split()[0]
        rest = "\n".join(EQUIVALENT.splitlines()[1:]) + "\n"
        try:
            # a condition must give one TRUE/FALSE on every replayed input (here also the empty `cut`)
            (self.fdir / "equivalent.txt").write_text(f"{first} QUIRK[cut == 0.5]: default cut only\n" + rest)
            report = self.gate(TEST)
            self.assertTrue(any(p.startswith(f"condition_invalid:{first}") for p in report["problems"]))
            (self.fdir / "equivalent.txt").write_text(f"{first} QUIRK[isTRUE(cut == 0.5)]: default cut only\n" + rest)
            report = self.gate(TEST)
            share = report["differential"]["excluded"][first]["share"]
            self.assertTrue(0 < share < 1, share)  # `cut` is unbound in calls relying on its default
            self.assertFalse(any(p.startswith("blanket_condition") for p in report["problems"]))
            (self.fdir / "equivalent.txt").write_text(f"{first} QUIRK[TRUE]: anything goes\n" + rest)
            report = self.gate(TEST)
            self.assertTrue(any(p.startswith(f"blanket_condition:{first}") for p in report["problems"]))
        finally:
            (self.fdir / "equivalent.txt").write_text(EQUIVALENT)

    def test_unvaried_argument_flagged(self):
        fixed = "\n".join(line for line in TEST.splitlines() if "cut = " not in line or "exec_cases" in line)
        report = self.gate(fixed.replace("lapply(1:6, function(i) list(scores = round(runif(8, -0.5, 1.5), 2), "
                                         "cut = round(runif(1), 2)))", "lapply(1:6, function(i) list(scores = runif(8)))"))
        self.assertTrue(any(p.startswith("argument_not_varied:cut") for p in report["problems"]), report["problems"])

    def test_generic_class_does_not_bypass_pattern_check(self):
        from sepalith.exec_env.gate import loose_error_patterns
        found = loose_error_patterns([("check_error", 'check_error("v", f(1), class = "simpleError", pattern = "length")'),
                                      ("check_error", 'check_error("v", f(1), class = "simpleError")'),
                                      ("check_error", 'check_error("v", f(1), class = "myPkgError")')])
        self.assertEqual(found, ["class=simpleError without a pattern", "length"])

    def test_own_and_anchored_messages_allowed(self):
        from sepalith.exec_env.gate import loose_error_patterns
        unit = 'f <- function(x) {\n  if (all(is.na(x))) stop("all times contain an NA")\n  x\n}\n'
        statements = [("check_error", 'check_error("v", f(NA), pattern = "all times contain an NA")'),
                      ("check_error", 'check_error("v", f(NA), pattern = "^missing values and NaN\'s not allowed if \'na.rm\' is FALSE$")'),
                      ("check_error", 'check_error("v", f(NA), pattern = "NA")')]
        self.assertEqual(loose_error_patterns(statements, unit), ["NA"])

    def test_loose_error_pattern_flagged(self):
        report = self.gate(TEST.replace('pattern = "scores must be numeric"', 'pattern = "numeric"'))
        self.assertTrue(any(p.startswith("error_pattern_matches_r_message") for p in report["problems"]))

    def test_shape_only_suite_fails(self):
        weak = "\n".join(f'check("{c}", is.integer(band_counts(c(0.{i}, 0.9))))' for i, c in
                         enumerate(["a", "a", "b", "b", "b", "a"], 1)) + "\n"
        report = self.gate(weak)
        self.assertEqual(report["verdict"], "fail")
        self.assertTrue(any(p.startswith("shape_only_share") for p in report["problems"]))
        self.assertTrue(any(p.startswith(("unkilled_survivors", "tests_do_not_reach")) for p in report["problems"]))

    def test_forbidden_test_calls(self):
        from sepalith.exec_env.gate import static_checks
        problems, _ = static_checks('print(band_counts(1))\nstopifnot(TRUE)\n', UNIT, UNIT)
        self.assertTrue(any(p.startswith("forbidden_calls") for p in problems))

    def test_reward_outcomes(self):
        from sepalith.exec_env.gate import suite_text
        from sepalith.exec_env.reward import ExecReward, ExecTask
        self.assertEqual(self.gate(TEST)["verdict"], "pass")
        lines = UNIT.rstrip("\n").split("\n")
        cut = lines.index("  total <- length(scores)")
        doc = tuple(lines[:cut])
        target = tuple(lines[cut:])
        case = Case("finish_block", "exec:fixture", "R/bands.R", doc, len(doc), len(doc), target)
        report = json.loads((self.fdir / "gate.json").read_text())
        task = ExecTask("t", "finish_block", PromptContext.from_dict(case.to_context()), "\n".join(target),
                        "\n".join(doc), "",
                        EXPECT, 3.0, suite_text(self.fdir), report["categories"], UNIT)
        reward = ExecReward({"t": task}, self.harness, None)
        alt = list(target)
        alt[0] = "  n_all <- length(scores)"
        alt[1] = "  if (n_all == 0) {"
        alt = [line.replace("total = total", "total = n_all") for line in alt]
        texts = {
            "gold": serialize_target("replace", target),
            "alt": serialize_target("replace", alt),
            "wrong": serialize_target("replace", [t.replace("0L, high = 0L", "1L, high = 0L") for t in target]),
            "quit": serialize_target("replace", ['  quit("no")', "}"]),
            "escape": serialize_target("replace", [*target, "identical <- function(...) TRUE"]),
            "syntax": serialize_target("replace", [*target[:-1], "  out(", "}"]),
            "padded": serialize_target("replace", [target[0], "  if (FALSE) {",
                                                   *[f"    unused_{i} <- {i}" for i in range(10)], "  }", *target[1:]]),
            "noop": serialize_target("no_op", []),
            "invalid": "  total <- length(scores)\n",
        }
        records = dict(zip(texts, reward.score_texts(["t"] * len(texts), list(texts.values())), strict=True))
        self.assertEqual(records["gold"]["reward"], 1.0)
        self.assertTrue(records["gold"]["exact_match"])
        self.assertEqual(records["alt"]["reward"], 1.0)
        self.assertFalse(records["alt"]["exact_match"])
        self.assertTrue(0 < records["wrong"]["reward"] < 1)
        self.assertEqual(records["quit"]["outcome"], "forbidden_construct")
        self.assertEqual(records["escape"]["outcome"], "structure_violation")
        self.assertEqual(records["syntax"]["outcome"], "applied_syntax_invalid")
        self.assertEqual(records["noop"]["outcome"], "missed_edit")
        self.assertEqual(records["padded"]["reward"], 0.5)  # all assertions pass, overlong guard halves it
        self.assertIn("overlong", records["padded"])
        self.assertEqual(records["invalid"]["reward"], -0.5)


if __name__ == "__main__":
    unittest.main()
