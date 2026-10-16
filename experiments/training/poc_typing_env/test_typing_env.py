#!/usr/bin/env python3
"""Typing-RL PoC tests (stdlib-only; Rscript integration skipped if R
is absent). Run: python3 -m unittest test_typing_env -v"""
from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d3_client import PROCESS_QUESTIONS, StubJudge            # noqa: E402
from env import (EnvConfig, EpisodeRow, TypingRLEnv,          # noqa: E402
                 bracket_ok, extract_code, glm_style_score, judge_parse,
                 parse_pass_fail, perf_score, repeated_line)
from policy import MockTypist                                 # noqa: E402
from tasks import Task, TASKS                                 # noqa: E402

HAVE_R = shutil.which("Rscript") is not None


class RewardMath(unittest.TestCase):
    def test_perf_score(self):
        self.assertAlmostEqual(perf_score(1.0, 1.0), 0.0)
        self.assertAlmostEqual(perf_score(16.0, 1.0), 1.0)
        self.assertAlmostEqual(perf_score(1.0, 16.0), -1.0)
        self.assertAlmostEqual(perf_score(4.0, 1.0), 0.5)
        self.assertEqual(perf_score(1.0, 0.0), -1.0)
        self.assertEqual(perf_score(1000.0, 1.0), 1.0)   # clamped

    def test_scalar_gating(self):
        env = TypingRLEnv(TASKS["rolling_mean"])
        good = EpisodeRow(task_id="t", code="f", correctness=1.0, perf=1.0,
                          style=1.0, guard=1.0)
        # full visible pass earns the seta bonus: 0.55*(1+0.2)+0.45
        self.assertAlmostEqual(env._scalar(good), 0.55 * 1.2 + 0.45)
        broken = EpisodeRow(task_id="t", code="f", correctness=0.0, perf=1.0,
                            style=1.0, guard=1.0)
        self.assertAlmostEqual(env._scalar(broken),
                               (0.20 + 0.15 + 0.10) * 0.3)

    def test_style_mixes_tiers(self):
        env = TypingRLEnv(TASKS["rolling_mean"])
        row = EpisodeRow(task_id="t", code="f", glm_judge={
            "elegance": 5, "readability": 5, "r_idiom": 5,
            "reward_hacking": 5, "parse_error": False})
        row.d3_terminal = {"quality": {"label": "elegant", "probs": {}}}
        self.assertAlmostEqual(env._style(row), 1.0)
        row.d3_terminal = {"quality": {"label": "unusable", "probs": {}}}
        self.assertAlmostEqual(env._style(row), 0.5)   # (1.0+0.0)/2


class Parsing(unittest.TestCase):
    def test_extract_code(self):
        self.assertEqual(extract_code("noise\n```r\nf <- 1\n```"),
                         "f <- 1")
        self.assertEqual(extract_code("```R\nf <- 2\n```"), "f <- 2")
        self.assertEqual(extract_code("f <- 3"), "f <- 3")

    def test_judge_parse(self):
        ok = judge_parse('{"elegance": 4, "notes": "fine"}')
        self.assertFalse(ok["parse_error"])
        self.assertEqual(ok["elegance"], 4)
        self.assertTrue(judge_parse("no json")["parse_error"])
        fenced = judge_parse("```json\n{\"elegance\": 3}\n```")
        self.assertEqual(fenced["elegance"], 3)
        self.assertIsNone(glm_style_score({"parse_error": True}))

    def test_pass_fail_protocol(self):
        frac, rows, err = parse_pass_fail(
            "PASS basic\nFAIL na_window - not equal\n", "")
        self.assertAlmostEqual(frac, 0.5)
        self.assertEqual(rows[1]["name"], "na_window")
        frac0, rows0, err0 = parse_pass_fail("", "Error: unexpected ')'")
        self.assertEqual((frac0, rows0), (0.0, []))
        self.assertIn("unexpected", err0)


class PrefixChecks(unittest.TestCase):
    def test_bracket_and_repeat(self):
        self.assertTrue(bracket_ok("f <- function(x {"))
        self.assertFalse(bracket_ok("f <- function(x)]"))
        self.assertTrue(repeated_line("a <- 1\nb <- 2\na <- 1"))
        self.assertFalse(repeated_line("a <- 1\nb <- 2\nc <- 3"))
        self.assertTrue(repeated_line("x\nx"))


class CascadeRouting(unittest.TestCase):
    """wiki model-routing directive: cheap tier triggers the big judge
    only on ambiguity."""

    def d3(self, quality, hacky="no", looks="yes"):
        return {"quality": {"label": quality, "probs": {}},
                "hacky": {"label": hacky, "probs": {}},
                "looks_correct": {"label": looks, "probs": {}}}

    def test_routes(self):
        from env import cascade_decision
        self.assertEqual(cascade_decision(0.0, self.d3("unusable")),
                         "dead")
        self.assertEqual(cascade_decision(0.0, self.d3("poor")), "dead")
        self.assertEqual(cascade_decision(1.0, self.d3("elegant")), "clean")
        self.assertEqual(cascade_decision(1.0, self.d3("good")), "clean")
        self.assertEqual(cascade_decision(1.0, self.d3("good", hacky="yes")),
                         "run")
        self.assertEqual(cascade_decision(1.0, self.d3("good", looks="no")),
                         "run")
        self.assertEqual(cascade_decision(0.5, self.d3("acceptable")), "run")
        self.assertEqual(cascade_decision(1.0, self.d3("acceptable")), "run")
        self.assertEqual(cascade_decision(
            1.0, self.d3("elegant"), route_skip_clean=False), "run")


class StubDecisionTier(unittest.TestCase):
    def test_deterministic_normalized(self):
        j = StubJudge()
        a1 = j.ask("state text", PROCESS_QUESTIONS)
        a2 = StubJudge().ask("state text", PROCESS_QUESTIONS)
        self.assertEqual(a1, a2)
        for q, ans in a1.items():
            self.assertIn(ans["label"], ans["probs"] or
                          [ans["label"]], msg=q)

    def test_env_process_checks_bounded(self):
        cfg = EnvConfig(check_every=1, max_checks=3, max_chars=4000)
        env = TypingRLEnv(TASKS["rolling_mean"], judge=StubJudge(),
                          cfg=cfg)
        env.reset()
        text = "```r\nf <- function(x, w) {\n  a <- 1\n  a <- 1\n  b <- 2\n}\n```\n"
        done = False
        for chunk in text.splitlines(keepends=True):
            if done:
                break
            _, _r, done, _i = env.step(chunk)
        self.assertLessEqual(len(env.process_checks), 3)
        self.assertTrue(done)


TINY_TASK = Task(
    task_id="tiny_mean",
    prompt="Write `midpoint(x)` returning (min(x)+max(x))/2.",
    tests_r='''
.t("basic", stopifnot(isTRUE(all.equal(midpoint(c(1, 2, 3)), 2))))
.t("neg", stopifnot(isTRUE(all.equal(midpoint(c(-4, 0, 4)), 0))))
''',
    bench_r='''
.data <- c(rnorm(200000), NA)
invisible(midpoint(.data))
.bench(naive_impl(.data), tag = "BASE")
.bench(midpoint(.data))
''',
    naive_r='''
naive_impl <- function(x) {
  lo <- x[1]; hi <- x[1]
  for (v in x) { if (is.na(v)) next; if (v < lo) lo <- v; if (v > hi) hi <- v }
  (lo + hi) / 2
}
''',
)


@unittest.skipUnless(HAVE_R, "Rscript not installed")
class EndToEndR(unittest.TestCase):
    def test_full_episode(self):
        good = ("```r\nmidpoint <- function(x) {\n"
                "  (min(x, na.rm = TRUE) + max(x, na.rm = TRUE)) / 2\n"
                "}\n```\n")
        env = TypingRLEnv(TINY_TASK, judge=StubJudge(),
                          cfg=EnvConfig(check_every=2, max_chars=800))
        env.reset()
        gen = MockTypist(good).type_solution(TINY_TASK.prompt)
        done = False
        while not done:
            try:
                chunk = next(gen)
            except StopIteration:
                break
            _, _r, done, _i = env.step(chunk)
        row = env.finalize()
        self.assertEqual(row.correctness, 1.0)
        self.assertIsNotNone(row.perf)
        self.assertGreater(row.perf, 0.0)      # vectorized beats the loop
        self.assertGreater(row.scalar, 0.4)

    def test_broken_code_gated(self):
        bad = "```r\nmidpoint <- function(x) { x[1] }\n```\n"
        env = TypingRLEnv(TINY_TASK, judge=StubJudge(),
                          cfg=EnvConfig(max_chars=800))
        env.reset()
        gen = MockTypist(bad).type_solution(TINY_TASK.prompt)
        done = False
        while not done:
            try:
                chunk = next(gen)
            except StopIteration:
                break
            _, _r, done, _i = env.step(chunk)
        row = env.finalize()
        self.assertLess(row.correctness, 1.0)
        self.assertLess(row.scalar, 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
