"""Compare against actual legacy renderer source without importing its module."""
import ast
from pathlib import Path
import unittest

from sepalith.protocol import EditContext, render_context


def legacy_renderer():
    root = Path(__file__).resolve().parents[3]
    path = root / "experiments/eval/run_eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"with_cursor", "render_zeta2"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names or len(functions) != len(names):
        raise AssertionError("Legacy renderer functions changed; review parity extraction")
    constants = [node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "CURSOR2"
                         for target in node.targets)]
    if len(constants) != 1:
        raise AssertionError("Legacy cursor marker changed; review parity extraction")
    namespace = {"CURSOR2": ast.literal_eval(constants[0])}
    # Execute only these two function definitions. Module imports, CLI, server
    # code and model evaluation do not enter this namespace or run.
    module = ast.Module(body=list(functions), type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["render_zeta2"]


class LegacyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy = staticmethod(legacy_renderer())

    def test_baseline_fixtures_match_actual_legacy_bytes(self):
        fixtures = {
            "cursor_on_interior_line": {
                "path": "R/fit.R", "prefix": ["fit <- function(x) {"],
                "region_old": ["  x <- na.omit(x)", "  mean(x)", "}"],
                "suffix": ["", "fit(values)"], "cursor_idx": 1,
            },
            "empty_edit": {
                "path": "empty.R", "prefix": [], "region_old": [],
                "suffix": [], "cursor_idx": -1,
            },
            "suffix_whitespace_unicode_and_history": {
                "path": "R/café.R", "prefix": ["# café  ", ""],
                "region_old": ["x <- 1  ", ""], "suffix": ["", "# end  ", ""],
                "cursor_idx": 1,
                "event_diff": "```diff\nUser edited café.R\n\n-x <- 0\n+x <- 1\n```\n",
            },
            "partial_line_at_cursor": {
                "path": "R/mean.R", "prefix": [], "region_old": ["mean(x, na.rm = "],
                "suffix": [")", ""], "cursor_idx": 0, "event_diff": None,
            },
            "out_of_range_cursor_and_literal_markers": {
                "path": "R/markers.R", "prefix": ["# <filename>literal"],
                "region_old": ['x <- "<<<<<<< CURRENT"'], "suffix": ["# <[fim-middle]>"],
                "cursor_idx": 20, "event_diff": "\n\n-a\r\n+b\r\n",
                "future_metadata": {"message": "must not enter prompt"},
            },
        }
        for name, example in fixtures.items():
            with self.subTest(name=name):
                actual = render_context(EditContext.from_legacy(example)).encode("utf-8")
                expected = self.legacy(example).encode("utf-8")
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
