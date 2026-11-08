"""SYNTH-v3.1: determinacy rules for constants, regex, schemas, element names,
defaults and string digits; call. = FALSE thinning; held-out selection; dose."""
from __future__ import annotations

import json
import unittest

from sepalith.data.synth.build import select_heldout, seed_wave
from sepalith.data.synth.cases import Case
from sepalith.data.synth.cohort import base_id, dose, repeat_rows
from sepalith.data.synth.common import assess
from sepalith.data.synth.normalize import CALL_FALSE_KEEP_SHARE, keeps_call_false, thin_call_false

PROMPT = ["#' @param x A numeric vector.", "summarise_x <- function(x, tol = 0.5) {",
          "  out <- list(mean = mean(x))", "  out$mean <- round(out$mean, 3)"]


def hard_kinds(target: list[str], prompt: list[str] = PROMPT) -> set[str]:
    return {h.split(":", 1)[0] for h in assess(target, prompt)[1]}


class AssessV31Tests(unittest.TestCase):
    def test_constant_absent_from_the_prompt_is_hard(self):
        self.assertIn("unspecified_constant", hard_kinds(["  if (out$mean > 0.01) out$mean <- 0", "  out", "}"]))
        self.assertNotIn("unspecified_constant", hard_kinds(["  out$mean <- out$mean * tol + 3", "  out", "}"],
                                                            PROMPT + ["  # three: 3"]))

    def test_regex_policy_needs_visible_evidence(self):
        self.assertIn("unspecified_regex", hard_kinds(['  stopifnot(grepl("^[A-Z]{2}$", x))', "}"]))
        shown = PROMPT + ['  ok <- grepl("^[A-Z]{2}$", names(x))']
        self.assertNotIn("unspecified_regex", hard_kinds(['  stopifnot(grepl("^[A-Z]{2}$", x))', "}"], shown))

    def test_output_schema_names_must_be_visible(self):
        self.assertIn("unspecified_schema", hard_kinds(["  list(scaled = x / tol, norms = out)", "}"]))
        self.assertNotIn("unspecified_schema", hard_kinds(["  list(mean = out$mean, x = x)", "}"]))

    def test_element_names_and_invented_defaults_are_novel(self):
        novel, _hard = assess(["  out$median <- t.test(x, mu = 0)$estimate", "  out", "}"],
                              PROMPT)
        self.assertIn("elt:median", novel)
        self.assertIn("default:mu", novel)
        novel, _hard = assess(["  out$mean <- round(out$mean, 3)", "  out", "}"], PROMPT)
        self.assertFalse([n for n in novel if n.startswith(("elt:", "default:"))])

    def test_digits_in_strings_must_be_visible(self):
        self.assertIn("unspecified_literal_number", hard_kinds(['  stamp <- "2024-01-31"', "}"]))

    def test_sentence_final_dot_is_not_a_compound_name(self):
        prompt = ['  if (!inherits(start, "Date")) stop("`start` must be a Date.")', "f <- function(start, end) {"]
        self.assertEqual(assess(['  if (!inherits(end, "Date")) stop("`end` must be a Date.")', "}"], prompt)[1], [])
        self.assertIn("unspecified_literal_identifier",
                      hard_kinds(['  names(out) <- "x_scaled"', "}"]))


class CallFalseThinningTests(unittest.TestCase):
    def test_thinning_is_per_file_and_deterministic(self):
        text = 'f <- function(x) {\n  stop(\n    "bad",\n    call. = FALSE\n  )\n  warning("w", call. = FALSE)\n}\n'
        paths = [f"b01-x/p/R/file{i}.R" for i in range(400)]
        kept = [p for p in paths if keeps_call_false(p)]
        self.assertAlmostEqual(len(kept) / len(paths), CALL_FALSE_KEEP_SHARE, delta=0.05)
        thinned_path = next(p for p in paths if not keeps_call_false(p))
        thinned, status = thin_call_false(text, thinned_path)
        self.assertEqual(status, "thinned")
        self.assertNotIn("call.", thinned)
        self.assertIn('stop(\n    "bad"\n  )', thinned)
        self.assertEqual(thin_call_false(text, kept[0]), (text, "kept"))
        self.assertEqual(thin_call_false("x <- 1\n", thinned_path)[1], "absent")


class SeedWaveTests(unittest.TestCase):
    def test_waves(self):
        self.assertEqual([seed_wave(b) for b in ("b01-a", "b24-x", "b25-v", "b33-n", "b48-p")],
                         ["v2", "v2", "v3A", "v3B", "v3C"])


def fb_case(unit: str, cut: int, target_lines: int, alt: int = 0, body: str = "x") -> tuple[Case, tuple]:
    target = tuple(f"  v{cut}_{i} <- {body}{i}_{unit}" for i in range(target_lines - 1)) + ("}",)
    case = Case("finish_block", "synth:p", "R/a.R", ("f <- function(x) {",), 1, 1, target,
                meta={"unit_key": unit, "cut_key": f"{unit}:{cut}", "alt_rank": alt, "variant": "plain"})
    return case, (None, None, None)


class HeldoutTests(unittest.TestCase):
    def test_heldout_uses_unused_units_only_one_cut_each(self):
        final = [fb_case("u1", 2, 4)]
        kept = final + [fb_case("u1", 3, 12), fb_case("u2", 2, 4), fb_case("u2", 3, 12), fb_case("u3", 5, 25),
                        fb_case("u3", 5, 25, alt=1)]
        chosen, report = select_heldout(kept, final)
        units = [c.meta["unit_key"] for c, _ in chosen]
        self.assertNotIn("u1", units)
        self.assertEqual(sorted(units), ["u2", "u3"])
        self.assertTrue(all(c.meta["alt_rank"] == 0 for c, _ in chosen))
        self.assertEqual(report["units_shared_with_training"], 0)

    def test_heldout_refuses_a_target_shown_in_a_training_window(self):
        shared = fb_case("u2", 2, 7, body="shared_long_value_")
        training = Case("finish_block", "synth:p", "R/b.R", ("g <- function(x) {", *shared[0].target[:-1]), 7, 7,
                        ("}",), meta={"unit_key": "u9"})
        chosen, report = select_heldout([shared], [(training, (None, None, None))])
        self.assertEqual(chosen, [])
        self.assertEqual(report["refused"], {"target_line_4gram_in_training_window": 1})


class DoseTests(unittest.TestCase):
    def test_repeat_rows_gives_distinct_ids(self):
        lines = [json.dumps({"id": "a"}) + "\n", json.dumps({"id": "b"}) + "\n"]
        ids = [json.loads(line)["id"] for line in repeat_rows(lines, 2)]
        self.assertEqual(ids, ["a", "b", "a~x2", "b~x2"])
        self.assertEqual(base_id("a~x2"), "a")

    def test_token_shares(self):
        def draw(rid: str, family: str = "finish_block", target: int = 10, prompt: int = 100) -> dict:
            return {"row_id": rid, "family": family, "supervised_target_tokens": target, "prompt_tokens": prompt}
        draws = [draw("t1", target=30), draw("t2"), draw("s1", target=20, prompt=50), draw("s1~x2", target=20, prompt=50),
                 draw("n1", family="roxygen_drafting", target=20)]
        result = dose(draws, {"s1"}, {"t1": "medium", "t2": "short", "s1": "long"})
        self.assertEqual(result["draws"]["synth_finish"], 2)
        self.assertEqual(result["draws"]["synth_share_of_finish_draws"], 0.5)
        tokens = result["tokens"]
        self.assertEqual(tokens["synth_share_of_finish_target_tokens"], round(40 / 80, 4))
        self.assertEqual(tokens["synth_share_of_all_supervised_tokens"], round(40 / 100, 4))
        self.assertEqual(tokens["synth_share_of_finish_prompt_tokens"], round(100 / 300, 4))
        self.assertEqual(tokens["medium_long_share_of_finish_target_tokens"],
                         {"cohort": round(70 / 80, 4), "train": 0.75, "synth": 1.0})


if __name__ == "__main__":
    unittest.main()
