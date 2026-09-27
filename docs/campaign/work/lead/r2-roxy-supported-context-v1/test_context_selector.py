#!/usr/bin/env python3
"""Synthetic checks for the DAT-10 source-span policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from tree_sitter import Language, Parser
import tree_sitter_r


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dat10_selector_test_module", HERE / "context_selector.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
parser = Parser(Language(tree_sitter_r.language()))


def test_reference_closure_and_unrelated_omission() -> None:
    source = """\nmain <- function(x, verbose = FALSE) {\n  y <- helper(x)\n  if (verbose) print(y)\n  y\n}\n\nhelper <- function(z) {\n  z + 1\n}\n\nunrelated <- function(q) q - 1\n"""
    lines = source.splitlines()
    _prefix, suffix, evidence = module.select_source_spans(
        source=source.encode(), source_lines=lines, anchor_line=0,
        target_lines=["#' @param x value", "#' @return result"],
        suffix_lines=lines[1:6], parser=parser,
    )
    assert evidence["status"].startswith("source_spans_selected")
    assert any("main <-" in line for line in suffix)
    assert any("helper <-" in line for line in suffix)
    assert not any("unrelated <-" in line for line in suffix)
    assert evidence["resolved_reference_names"] == ["helper"]
    assert evidence["docs_evidence"]["documented_params_missing_from_formals"] == []


def test_formal_mismatch_is_queued_and_target_is_retained() -> None:
    source = """\nmain <- function(x) {\n  x\n}\n"""
    lines = source.splitlines()
    _prefix, suffix, evidence = module.select_source_spans(
        source=source.encode(), source_lines=lines, anchor_line=0,
        target_lines=["#' @param absent text"], suffix_lines=lines[1:], parser=parser,
    )
    assert evidence["status"] == "source_spans_selected_docs_formal_revalidation_required"
    assert any("main <-" in line for line in suffix)
    assert evidence["semantic_support_claim"] is False


def test_target_body_span_is_not_length_cropped() -> None:
    body = "\n".join("  value <- value + 1" for _ in range(180))
    source = "\nmain <- function(value) {\n" + body + "\n}\n"
    lines = source.splitlines()
    _prefix, suffix, evidence = module.select_source_spans(
        source=source.encode(), source_lines=lines, anchor_line=0,
        target_lines=["#' @param value a value"], suffix_lines=lines[1:4], parser=parser,
    )
    assert len(suffix) >= 180
    assert evidence["target_definition_span"][1] >= 180


if __name__ == "__main__":
    for name in sorted(globals()):
        if name.startswith("test_"):
            globals()[name]()
    print("context selector synthetic tests passed")

