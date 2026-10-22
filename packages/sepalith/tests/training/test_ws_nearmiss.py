"""Unit checks for the decode-time whitespace analysis (docs/training/plan-20260927/decode-time)."""
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "docs/training/plan-20260927/decode-time/ws_nearmiss.py"
spec = importlib.util.spec_from_file_location("ws_nearmiss", SCRIPT)
assert spec is not None and spec.loader is not None
ws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ws)


def test_ws_types():
    assert ws.ws_types(("a",), ("a",)) is None
    assert ws.ws_types(("a",), ("b",)) is None
    assert ws.ws_types(("x <- 1 ",), ("x <- 1",)) == {"trailing_whitespace"}
    assert ws.ws_types(("", "  x"), ("  x",)) == {"leading_blank_lines"}
    assert ws.ws_types(("x", "", "y"), ("x", "y")) == {"interior_blank_lines"}
    assert ws.ws_types(("f(a,b)",), ("f(a, b)",)) == {"intra_line_whitespace"}
    assert ws.ws_types(("f(a,", "b)"), ("f(a, b)",)) == {"line_wrapping"}
    types = ws.ws_types(("    x", "      y"), ("  x", "    y"))
    assert "indentation" in types and "indent:uniform_offset" in types
    types = ws.ws_types(("\tx",), ("  x",))
    assert "indent:tabs_vs_spaces" in types


def test_body_unit_ignores_base_indent():
    assert ws.body_unit(["    a <- 1"]) == (None, 0)
    assert ws.body_unit(["  f <- function() {", "      x", "          y"]) == (4, 2)


def test_trailing_blank_trim_keeps_leading_blank():
    assert ws.drop_trailing_blank(["", "  x", "", "  "]) == ["", "  x"]
    assert ws.strip_outer(["", "  x", ""]) == ["  x"]


class WsNearmissTests(unittest.TestCase):
    """unittest entry point (the repo runs unittest; the functions above also work under pytest)."""

    def test_ws_types(self):
        test_ws_types()

    def test_body_unit_ignores_base_indent(self):
        test_body_unit_ignores_base_indent()

    def test_trailing_blank_trim_keeps_leading_blank(self):
        test_trailing_blank_trim_keeps_leading_blank()


if __name__ == "__main__":
    unittest.main(verbosity=2)
