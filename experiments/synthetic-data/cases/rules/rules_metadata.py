"""rules_metadata.py — code-smell METADATA rules (no rewrite rows).

The compounding design throws cyclomatic complexity and friends at the
registry in TWO roles: as rewrite TRIGGERS (catalog: complexity-reduction,
nesting-flattening, function-splitting — all D3/D4 author-LLM rewrites) and
as METADATA: difficulty knobs and restraint selectors stamped on the base
sample. This module implements the metadata role: annotate() computes the
smell vector once per base sample; the runner carries it on every derived
row (`smells` key) and reports the corpus distribution.

Cyclomatic complexity convention (lintr/cyclocomp-compatible): 1 + count of
if / for / while / repeat / ifelse() / && / || decision points in the
function body.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

import scenarios as S                       # noqa: E402
from scenarios import node_text             # noqa: E402
import cases.validators as V                # noqa: E402
from cases.rules import rule                 # noqa: E402

_DECISION_TYPES = {"if_statement", "for_statement", "while_statement",
                   "repeat_statement"}


def _cyclo(src, body) -> int:
    c = 1
    for n in V._walk(body):
        if n.type in _DECISION_TYPES:
            c += 1
        elif n.type == "call" and S.callee_name(src, n) == "ifelse":
            c += 1
        elif n.type == "binary_operator" and len(n.children) >= 2 and \
                node_text(src, n.children[1]) in (b"&&", b"||"):
            c += 1
    return c


def _max_nesting(body) -> int:
    """Max control-structure nesting of the body (1 = flat body). Depth of a
    control node = 1 + number of enclosing control ancestors; the metric
    counts STRUCTURES, so `if () { for () { if ... } }` scores 4 (body,
    if, for, inner if). Ancestor walks stop at the body by byte-span
    comparison (tree-sitter hands out fresh Node wrappers — Python `is` is
    unreliable, the compound.py lesson)."""
    def _is_body(n) -> bool:
        return n.type == body.type and n.start_byte == body.start_byte \
            and n.end_byte == body.end_byte

    best = 1
    for n in V._walk(body):
        if n.type not in _DECISION_TYPES:
            continue
        d = 1
        p = n.parent
        while p is not None and not _is_body(p):
            if p.type in _DECISION_TYPES:
                d += 1
            p = p.parent
        best = max(best, d + 1)          # +1: statements inside the control
    return best


@rule(id="smell_metadata", family="smell_metadata", determinism="D1",
      kind="metadata", requires=["fn_body"],
      signal="AST metrics: cyclomatic complexity (1 + decision points), "
             "max control-nesting depth, non-blank body lines",
      restraint="high-complexity samples are the RESTRAINT pool for rewrite "
                "rules (a smell present but NOT rewritten without an "
                "explicit user intent signal) and the difficulty knob for "
                "curriculum sweeps",
      status="new")
class SmellMetadata:
    SELFTEST = [
        (b'f <- function(x) {\n  if (x > 1) {\n    for (i in 1:10) {\n'
         b'      if (i && x) x <- x + 1\n    }\n  }\n  x\n}\n',
         dict(expect_annotations={"cyclo": 5, "nesting": 4})),
        (b'f <- function(x) {\n  x + 1\n}\n',
         dict(expect_annotations={"cyclo": 1, "nesting": 1})),
    ]

    def annotate(self, bs):
        src = bs.b.src
        cyclo = _cyclo(src, bs.body)
        nesting = _max_nesting(bs.body)
        flags = []
        if cyclo >= 15:
            flags.append("cyclo_lintr")        # lintr cyclocomp default flag
        elif cyclo >= 8:
            flags.append("cyclo_elevated")
        if nesting >= 5:
            flags.append("nesting_deep")
        if bs.nbody > 40:
            flags.append("fn_long")
        return dict(cyclo=cyclo, nesting=nesting, body_lines=bs.nbody,
                    flags=flags,
                    thresholds=dict(cyclo_elevated=8, cyclo_lintr=15,
                                    nesting_deep=5, fn_long=40))
