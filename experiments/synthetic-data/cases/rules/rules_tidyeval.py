"""rules_tidyeval.py — tidy-eval rewrite rules (ideation-tournament winner X-006,
rule family C_DATA_PRONOUN). Proof-of-concept wave: one deterministic rule
implemented through the existing registry gates (selftest + base-samples).

C1: `.data$col` -> `col` INSIDE a data-masked verb call (filter/mutate/
summarise/... including namespaced dplyr::/stats:: forms). Deterministic
ground truth: the rewrite is pure token surgery on a tree-sitter site, and
`verify` checks exactly one `.data$` prefix disappears with the column
symbol intact. Restraint: `.data` used OUTSIDE a masked verb (plain list
access `l$x` semantics differ; also `.data[[`-indexed form) is left alone.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

import scenarios as S                       # noqa: E402
from scenarios import node_text             # noqa: E402
import cases.validators as V                # noqa: E402
from cases.rules import Rewrite, Site, rule  # noqa: E402
from cases.rules.rules_lint import _geom_ok, _one_row, _txt  # noqa: E402

_MASKED_VERBS = {
    "filter", "mutate", "summarise", "summarize", "select", "group_by",
    "arrange", "transmute", "rename", "slice", "distinct", "count",
    "reframe", "left_join", "inner_join", "right_join", "full_join",
}
_DOTDATA_RE = re.compile(rb"(?<![\w.$])\.data\$")

_CALL_RE = re.compile(rb"([A-Za-z.][\w.]*)(::)?([A-Za-z.][\w.]*)?\s*\(")


def _callee_name(src, call) -> str | None:
    """Final name of a call's callee, `dplyr::filter(` -> filter."""
    kids = [k for k in call.children if k.type != "("]
    if not kids:
        return None
    name = _txt(src, kids[0])
    return name.rsplit("::", 1)[-1].split("$")[-1]


def _in_masked_call(src, n) -> bool:
    """True when node n has a call ancestor whose callee is a masked verb.
    The .data pronoun is only guaranteed equivalent to the bare symbol
    inside a data-masked context — everywhere else the rewrite is WRONG
    (the restraint curriculum)."""
    p = n.parent
    while p is not None:
        if p.type == "call":
            if _callee_name(src, p) in _MASKED_VERBS:
                return True
            return False          # innermost enclosing call decides
        p = p.parent
    return False


@rule(id="tidyeval_data_pronoun", family="tidyeval_rewrite", determinism="D2",
      kind="rewrite", requires=["fn_body"],
      signal="`.data$col` pronoun inside a data-masked verb call "
             "(filter/mutate/summarise/...); tree-sitter ancestor walk",
      restraint="suppressed OUTSIDE masked verb calls (plain list `$` "
                "semantics) and for `.data[[`-bracket forms — dropping "
                "the pronoun there changes program meaning",
      status="new", extends="tidyeval family (ideation X-006)")
class TidyevalDataPronoun:
    PRESCREEN = [rb"\.data\$"]
    SELFTEST = [
        (b'f <- function(df) {\n'
         b'  dplyr::filter(df, .data$value > 0)\n'
         b'  1\n'
         b'}\n',
         dict(expect_sites=1)),
        (b'f <- function(df) {\n'
         b'  mutate(df, y = .data$x + .data$z)\n'
         b'  1\n'
         b'}\n',
         dict(expect_sites=2)),
        (b'f <- function(df, l) {\n'
         b'  a <- l$.data$x\n'          # no pronoun on a plain list
         b'  b <- df[[".data$x"]]\n'
         b'  a && b\n'
         b'}\n',
         dict(expect_sites=0, why="no .data$ pronoun site")),
        (b'f <- function(df) {\n'
         b'  x <- .data$value\n'        # outside any masked verb
         b'  filter(df, value > 0)\n'
         b'}\n',
         dict(expect_sites=0,
              why="restraint: .data$ outside a masked call is not "
                  "pronoun-guaranteed")),
    ]

    def detector(self, bs):
        src = bs.b.src
        out = []
        for n in V._walk(bs.body):
            if n.type != "identifier" or _txt(src, n) != ".data":
                continue
            p = n.parent
            if p is None or p.type != "extract_operator":
                continue                    # `.data[[` or bare use: skip
            if _txt(src, p.children[0]) != ".data":
                continue                    # `x$.data`-shaped: skip
            if not _in_masked_call(src, n):
                continue                    # restraint gate
            if not _one_row(p):
                continue
            row = n.start_point[0]
            if not _geom_ok(bs, row):
                continue
            right = p.children[-1]           # [left, '$', right]
            col_tok = _txt(src, right)
            out.append(Site(row=row, sb=n.start_byte, eb=right.end_byte,
                            payload=dict(old=f".data${col_tok}", new=col_tok),
                            note=f".data${col_tok} -> {col_tok} "
                                 f"(data-masked pronoun elision)"))
        return out

    def rewrite(self, bs, site):
        b = bs.b
        lb = b.line_bytes(site.row)
        col = site.sb - b.starts[site.row]
        old_b = site.payload["old"].encode()
        new_lb = lb[:col] + site.payload["new"].encode() \
            + lb[col + len(old_b):]
        new_line = new_lb.decode("utf-8", "replace").rstrip("\r")
        if not new_line.strip():
            return None
        return Rewrite(lines=[new_line],
                       meta=dict(old_tok=site.payload["old"],
                                 new_tok=site.payload["new"]),
                       span_text=site.payload["new"])

    def verify(self, old_text, new_text):
        # multi-site safe: exactly ONE `.data$` prefix elided, the column
        # symbol preserved, nothing else moved
        old_n = len(_DOTDATA_RE.findall(old_text.encode()))
        new_n = len(_DOTDATA_RE.findall(new_text.encode()))
        if new_n != old_n - 1:
            return False, f"expected exactly one .data$ elided " \
                          f"({old_n} -> {new_n})"
        if ".data" in new_text and _DOTDATA_RE.search(new_text.encode()) \
                and not _DOTDATA_RE.search(old_text.encode()):
            return False, "new .data$ introduced"
        return True, ""
