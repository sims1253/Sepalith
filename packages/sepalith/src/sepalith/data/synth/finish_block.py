"""finish_block cases: cut a braced top-level expression at a line boundary.

Geometry follows the TRAIN family (DAT10 v4): the prompt holds the function's
head (optionally its roxygen block, optionally a deterministic outline or
static-diagnostics header above it, or the preceding file content), the region
is empty at the cut (or, for full-body cuts, the signature line with the
cursor at its end), the suffix is empty, and the target is the remainder of
the expression through its closing line. Gold text is air-formatted source,
so the final brace sits at the header column (the v4-braceindent convention).

Cut boundaries: statement starts at any depth, the top of a full-line comment
run above a statement, and nested closing lines (target starts with ``}``).
Short remainders (1-8 lines) are preferred; see ``select_cuts``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import random
import re

from sepalith.data.synth.cases import Case, window_start_for
from sepalith.data.synth.common import novelty, rng_for, strip_comment, word_in
from sepalith.data.synth.structure import FileStructure, Unit

CLOSE_RE = re.compile(r"^\}[)\]]*$")
WINDOW_BUDGET = 4500
NOVELTY_LIMITS = {"short": 4, "medium": 8, "long": 14}
VARIANT_WEIGHTS = {"plain": 0.35, "file": 0.25, "strip": 0.15, "outline": 0.125, "diag": 0.125}


@dataclass(frozen=True)
class Cut:
    line: int  # 1-based first target line
    kind: str  # statement | comment_lead | nested_close | final_close
    length: int

    @property
    def bucket(self) -> str:
        return "short" if self.length <= 8 else "medium" if self.length <= 20 else "long"


def eligible(unit: Unit, lines: Sequence[str]) -> bool:
    outer = unit.outer_block()
    if outer is None or unit.end - unit.start > 90:
        return False
    if not CLOSE_RE.match(lines[unit.end - 1].strip()):
        return False
    if not lines[outer.open_line - 1].rstrip().endswith("{"):
        return False
    return bool(outer.stmt_lines)


def cuts_for(unit: Unit, lines: Sequence[str], structure: FileStructure) -> list[Cut]:
    outer = unit.outer_block()
    assert outer is not None
    header_end = outer.open_line
    comment_lines = structure.comments
    protected = structure.string_lines()
    inner = [b for b in unit.blocks if outer.open_line <= b.open_line and b.close_line <= outer.close_line]
    found: dict[int, str] = {}
    for block in inner:
        for line, col in zip(block.stmt_lines, block.stmt_cols, strict=True):
            if line <= header_end or line in protected or lines[line - 1][:col - 1].strip():
                continue
            found.setdefault(line, "statement")
            top = line
            while (top - 1 > header_end and (top - 1) in comment_lines
                   and lines[top - 2].lstrip().startswith("#")):
                top -= 1
            if top != line:
                found.setdefault(top, "comment_lead")
        if block is not outer and block.close_line > block.open_line:
            if lines[block.close_line - 1].lstrip().startswith("}"):
                found.setdefault(block.close_line, "nested_close")
    found[outer.close_line] = "final_close"
    return sorted((Cut(line, kind, unit.end - line + 1) for line, kind in found.items()), key=lambda c: c.line)


def header_lines(kind: str, unit: Unit, lines: Sequence[str], cut: Cut) -> list[str]:
    outer = unit.outer_block()
    fun = unit.main_function()
    assert outer is not None and fun is not None
    n_stmts = len(outer.stmt_lines)
    typed = sum(1 for end in outer.stmt_end_lines if end < cut.line)
    params = ", ".join(fun.formals[:8]) or "none"
    if kind == "outline":
        return ["# ---- outline " + "-" * 32,
                f"# fn: {unit.name or '<anon>'}({params})",
                f"# plan: {n_stmts} top-level statements; {typed} typed above the cursor",
                f"# docs: {'yes' if unit.roxygen else 'no'}",
                "# " + "-" * 46]
    body = "\n".join(strip_comment(line) for line in lines[outer.open_line:cut.line - 1])
    unused = [p for p in fun.formals if p != "..." and not word_in(p, body)]
    open_braces = sum(1 for b in unit.blocks
                      if outer.open_line <= b.open_line < cut.line <= b.close_line and b.close_line <= outer.close_line)
    return ["# ---- diagnostics (static) " + "-" * 20,
            f"# typed statements: {typed} of {n_stmts}",
            f"# args not yet referenced: {', '.join(unused[:8]) or 'none'}",
            f"# unclosed braces in typed code: {open_braces}",
            "# " + "-" * 46]


def safe_lines(structure: FileStructure) -> set[int]:
    """0-based lines outside every top-level expression."""
    inside: set[int] = set()
    for unit in structure.units:
        inside.update(range(unit.start - 1, unit.end))
    total = max((u.end for u in structure.units), default=0) + 1
    return {i for i in range(total + 5000) if i not in inside}


def build_case(unit: Unit, cut: Cut, variant: str, lines: Sequence[str], structure: FileStructure,
               package_id: str, path: str, *, signature_region: bool = False) -> Case:
    outer = unit.outer_block()
    assert outer is not None
    head = list(lines[unit.start - 1:cut.line - 1])
    target = tuple(lines[cut.line - 1:unit.end])
    roxy = list(lines[unit.roxygen[0] - 1:unit.roxygen[1]]) if unit.roxygen else []
    meta = {"rule": f"fb_{cut.kind}", "variant": variant, "cut_line": cut.line, "unit": unit.name or unit.kind,
            "unit_kind": unit.kind, "unit_lines": [unit.start, unit.end], "bucket": cut.bucket,
            "target_lines": len(target), "signature_region": signature_region}
    if variant == "file":
        doc = tuple(lines[:cut.line - 1])
        first = (unit.roxygen[0] if unit.roxygen else unit.start) - 1
        start = window_start_for(doc, len(doc), WINDOW_BUDGET, safe_lines(structure))
        start = min(start, first)
    else:
        prefix: list[str] = []
        if variant in ("outline", "diag"):
            prefix += header_lines(variant, unit, lines, cut)
        if variant != "strip":
            prefix += roxy
        doc = tuple(prefix + head)
        start = 0
    case = Case("finish_block", package_id, path, doc, len(doc), len(doc), target, window_start=start, meta=meta)
    if signature_region:
        # Region = the signature line ending in "{", cursor at its end; target restates it.
        sig = doc[-1]
        case = Case("finish_block", package_id, path, doc, len(doc) - 1, len(doc), (sig, *target),
                    window_start=start, cursor=(0, len(sig)), meta=meta)
    prompt = "\n".join(doc[start:])
    novel = novelty(target, prompt)
    case.meta.update(novel_tokens=novel[:20], novelty=len(novel))
    return case


def select_cuts(unit: Unit, cuts: list[Cut], rng: random.Random) -> list[Cut]:
    outer = unit.outer_block()
    assert outer is not None
    shorts = [c for c in cuts if 2 <= c.length <= 8]
    mediums = [c for c in cuts if c.bucket == "medium"]
    longs = [c for c in cuts if c.bucket == "long"]
    chosen: list[Cut] = []
    nested = [c for c in shorts if c.kind == "nested_close"]
    if nested and rng.random() < 0.5:
        chosen.append(rng.choice(nested))
    pool = [c for c in shorts if c not in chosen]
    rng.shuffle(pool)
    chosen += pool[:max(0, 2 - len(chosen))]
    if mediums:
        chosen.append(rng.choice(mediums))
    first = min(cuts, key=lambda c: c.line)
    if first.length > 20 and first.length <= 45 and rng.random() < 0.6:
        chosen.append(first)
    elif longs and rng.random() < 0.3:
        chosen.append(rng.choice(longs))
    finals = [c for c in cuts if c.kind == "final_close"]
    if finals and rng.random() < 0.04:
        chosen.append(finals[0])
    if first.length <= 20 and first not in chosen and rng.random() < 0.5:
        chosen.append(first)  # full body of a short function
    return sorted(set(chosen), key=lambda c: c.line)


def generate(lines: Sequence[str], structure: FileStructure, package_id: str, path: str) -> list[Case]:
    out: list[Case] = []
    for unit in structure.units:
        if not eligible(unit, lines):
            continue
        cuts = cuts_for(unit, lines, structure)
        rng = rng_for("fb", package_id, path, unit.start)
        outer = unit.outer_block()
        is_fun = unit.main_function() is not None and outer is not None and outer.open_line >= unit.start
        first_line = min(c.line for c in cuts)
        for cut in select_cuts(unit, cuts, rng):
            weights = {k: w for k, w in VARIANT_WEIGHTS.items()
                       if (k != "strip" or unit.roxygen) and (k not in ("outline", "diag") or is_fun)}
            variants = [rng.choices(list(weights), weights=list(weights.values()))[0]]
            if rng.random() < 0.2:
                rest = {k: w for k, w in weights.items() if k not in variants}
                if rest:
                    variants.append(rng.choices(list(rest), weights=list(rest.values()))[0])
            for variant in variants:
                signature = cut.line == first_line and variant != "file" and rng.random() < 0.12
                case = build_case(unit, cut, variant, lines, structure, package_id, path,
                                  signature_region=signature)
                if case.meta["novelty"] > NOVELTY_LIMITS[cut.bucket]:
                    case.meta["rejected"] = "novelty"
                out.append(case)
    return out
