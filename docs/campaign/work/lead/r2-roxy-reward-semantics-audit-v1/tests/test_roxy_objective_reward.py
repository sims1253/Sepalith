import pathlib, subprocess, sys, tempfile, unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from roxy_objective_reward import extract_visible_signature, check_roxygen_candidate


class RoxygenObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.signature = extract_visible_signature([
            "helper <- function(x, y = list(a = 1, b = 2), ...) {", "  x", "}",
        ])

    def check(self, text): return check_roxygen_candidate(text, self.signature)

    def test_source_signature_and_supported_alternative(self):
        self.assertEqual(self.signature.parameters, ("x", "y", "..."))
        result = self.check("#' Compute a documented value\n#' @param x first input\n#' @param y optional settings\n#' @param ... forwarded arguments\n#' @return a value")
        self.assertTrue(result["objective_pass"])
        self.assertEqual(result["proposed_reward_effect"], "no_positive_credit")
        self.assertEqual(result["semantic_correctness"], "not_measured")

    def test_invented_parameter_fails(self):
        result = self.check("#' Text\n#' @param x x\n#' @param y y\n#' @param ... dots\n#' @param invented unsupported")
        self.assertIn("invented_param:invented", result["errors"])

    def test_empty_documentation_fails(self):
        self.assertIn("empty_documentation", self.check("")["errors"])

    def test_duplicate_and_missing_tags_fail(self):
        result = self.check("#' Text\n#' @param x first\n#' @param x duplicate")
        self.assertIn("duplicate_param:x", result["errors"])
        self.assertIn("missing_param:y", result["errors"])

    def test_non_roxygen_code_and_unresolved_inheritance_fail(self):
        result = self.check("#' Text\n#' @inheritParams other\nx <- system('never')")
        self.assertIn("non_roxygen_line", result["errors"])
        self.assertIn("unsupported_tags_require_source_resolution", result["errors"])

    def test_fixed_roxygen_harness_parses_without_executing_candidate(self):
        candidate = "#' Compute value\n#' @param x input\n#' @param y setting\n#' @param ... dots\n#' @return a value\n"
        with tempfile.TemporaryDirectory() as root:
            doc = pathlib.Path(root) / "doc.R"; params = pathlib.Path(root) / "params.txt"
            doc.write_text(candidate); params.write_text("x\ny\n...\n")
            run = subprocess.run(["Rscript", "--vanilla", str(HERE / "roxygen_parse_only.R"), str(doc), str(params)])
            self.assertEqual(run.returncode, 0)
            doc.write_text(candidate + "#' @eval system('never')\n")
            run = subprocess.run(["Rscript", "--vanilla", str(HERE / "roxygen_parse_only.R"), str(doc), str(params)])
            self.assertEqual(run.returncode, 2)


if __name__ == "__main__": unittest.main()
