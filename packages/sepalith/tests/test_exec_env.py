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
TEST = '''check_identical("basic", band_counts(c(0.1, 0.6, 0.9)), c(low = 1L, high = 2L, total = 3L))
check_identical("basic", band_counts(c(0.2, 0.3), cut = 0.25), c(low = 1L, high = 1L, total = 2L))
check_identical("boundary", band_counts(c(0.5, 0.49999)), c(low = 1L, high = 1L, total = 2L))
check_identical("clamp", band_counts(c(-3, 7)), c(low = 1L, high = 1L, total = 2L))
check_identical("na", band_counts(c(NA, 0.7, NA)), c(low = 0L, high = 1L, total = 1L))
check_identical("empty", band_counts(numeric()), c(low = 0L, high = 0L, total = 0L))
check_error("validation", band_counts("a"), pattern = "numeric")
exec_cases <- function() {
  set.seed(42)
  lapply(1:6, function(i) list(scores = round(runif(8, -0.5, 1.5), 2), cut = round(runif(1), 2)))
}
'''
EXPECT = ("clamp_unit", "band_counts")


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
        self.assertEqual((result.passed, result.total), (7, 7))
        self.assertEqual(result.by_category()["basic"], (2, 2))

    def test_wrong_code_fails_some(self):
        result = self.run_one(replace_body(UNIT, "scores < cut", "scores <= cut"))
        self.assertLess(result.passed, result.total)

    def test_quit_ends_only_the_child(self):
        result = self.run_one(replace_body(UNIT, "  out\n}", '  quit("no")\n}'))
        self.assertEqual(result.status, "candidate_exit")
        self.assertEqual(self.run_one(UNIT).passed, 7)

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
        # empty-input early return (which never reaches the plant) still pass
        self.assertEqual(self.run_one(planted).passed, 2)

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
        self.assertLess(self.run_one(writes).passed, 7)
        self.assertFalse(probe.exists())
        net = replace_body(UNIT, "  out\n}", '  readLines(url("http://example.com"))\n  out\n}')
        self.assertLess(self.run_one(net).passed, 7)


@unittest.skipUnless(HAS_R, "needs Rscript")
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

    def test_shape_only_suite_fails(self):
        weak = "\n".join(f'check("{c}", is.integer(band_counts(c(0.{i}, 0.9))))' for i, c in
                         enumerate(["a", "a", "b", "b", "b", "a"], 1)) + "\n"
        report = self.gate(weak)
        self.assertEqual(report["verdict"], "fail")
        self.assertTrue(any(p.startswith("shape_only_share") for p in report["problems"]))
        self.assertTrue(any(p.startswith(("kill_rate", "tests_do_not_reach")) for p in report["problems"]))

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
