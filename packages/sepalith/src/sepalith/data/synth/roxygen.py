"""roxygen_drafting cases: draft the roxygen block above a definition.

Geometry follows TRAIN: up to 30 prefix lines of preceding file content, an
empty region where the block starts, the documented definition as suffix, and
the block as target (``full``). A second geometry (``tags``) keeps the typed
title and description in the prefix and asks for the tag section only.

Exact prose is the bottleneck for this family (C10's 3,861 extra rows did not
move DEV), so cases are admitted only when the target is mostly determined by
the prompt: every line is either skeleton (``#'``, ``@export``,
``@keywords internal``, ``@noRd``, a bare tag) or appears verbatim in the
prompt (sibling blocks reuse parameter, return and tag lines). The share of
determined lines is recorded per case as ``determined_share``.
"""
from __future__ import annotations

from collections.abc import Sequence
import re

from sepalith.data.synth.cases import Case, skip_blank
from sepalith.data.synth.common import rng_for
from sepalith.data.synth.structure import FileStructure, Unit

PREFIX_MAX_LINES = 30
VALID_TAGS = frozenset("""param return returns export keywords noRd examples examplesIf seealso family
inheritParams rdname importFrom import title description details name aliases describeIn method field slot
format source references note section template md concept usage evalRd include exportMethod exportClass
exportS3Method docType encoding author inherit inheritDotParams eval order useDynLib returns""".split())
SKELETON = re.compile(r"^#'\s*(@export|@keywords internal|@noRd|@md|@examples|@return|@returns|)\s*$")
TAG_RE = re.compile(r"^#'\s*@(\w+)(.*)$")
MIN_SHARE = {"full": 0.65, "tags": 0.7}


def tags_of(block: Sequence[str]) -> list[tuple[str, str]]:
    out = []
    for line in block:
        match = TAG_RE.match(line.strip())
        if match:
            out.append((match.group(1), match.group(2).strip()))
    return out


def check_block(block: Sequence[str], unit: Unit) -> list[str]:
    """Problems with the gold block: unknown tags and @param/signature mismatches."""
    problems = []
    tags = tags_of(block)
    for tag, _ in tags:
        if tag not in VALID_TAGS:
            problems.append(f"unknown_tag:{tag}")
    if not block or not block[0].startswith("#'") or block[0].strip() in ("#'",):
        problems.append("missing_title")
    documented: list[str] = []
    for tag, rest in tags:
        if tag == "param":
            if not rest:
                problems.append("param_without_name")
                continue
            documented += [name for name in rest.split()[0].split(",") if name]
    if len(set(documented)) != len(documented):
        problems.append("duplicate_param")
    fun = unit.main_function()
    if fun is not None:
        formals = list(fun.formals)
        if any(name not in formals for name in documented):
            problems.append("param_not_in_signature")
        order = [formals.index(name) for name in documented if name in formals]
        if order != sorted(order):
            problems.append("param_order_differs_from_signature")
        inherits = any(tag in ("inheritParams", "inheritDotParams", "inherit", "rdname", "describeIn")
                       for tag, _ in tags)
        if not inherits and "noRd" not in [t for t, _ in tags]:
            missing = [name for name in formals if name not in documented]
            if missing:
                problems.append("formal_not_documented")
    else:
        names = set(unit.symbols)
        for f in unit.funs:
            names.update(f.formals)
        if any(name not in names for name in documented):
            problems.append("param_not_in_definition")
    return problems


def determined_share(target: Sequence[str], prompt_lines: Sequence[str]) -> float:
    seen = {line.strip() for line in prompt_lines}
    good = sum(1 for line in target if SKELETON.match(line.strip()) or line.strip() in seen)
    return good / len(target) if target else 0.0


def generate(lines: Sequence[str], structure: FileStructure, package_id: str, path: str) -> list[Case]:
    out: list[Case] = []
    inside: set[int] = set()
    for unit in structure.units:
        inside.update(range(unit.start - 1, unit.end))
    for unit in structure.units:
        if unit.roxygen is None:
            continue
        r0, r1 = unit.roxygen[0] - 1, unit.roxygen[1]  # 0-based [r0, r1)
        block = tuple(lines[r0:r1])
        problems = check_block(block, unit)
        start = r0
        for index in range(r0 - 1, max(-1, r0 - PREFIX_MAX_LINES - 1), -1):
            if index not in inside:
                start = index
        start = skip_blank(lines, start, r0)
        rng = rng_for("roxy", package_id, path, unit.start)
        first_tag = next((i for i, line in enumerate(block) if line.lstrip().startswith("#' @")), None)
        geometries = ["full"]
        if first_tag is not None and first_tag >= 2 and rng.random() < 0.5:
            geometries.append("tags")
        for geometry in geometries:
            cut = r0 if geometry == "full" or first_tag is None else r0 + first_tag
            # In the tags geometry, a blank `#'` separator before the tags stays typed.
            target = tuple(lines[cut:r1])
            # Pre-edit document: the file through the definition, without the drafted lines.
            doc = (*lines[:cut], *lines[r1:unit.end])
            case = Case("roxygen_drafting", package_id, path, doc, cut, cut, target,
                        window_start=start, window_end=len(doc),
                        meta={"rule": f"roxy_{geometry}", "unit": unit.name or unit.kind, "unit_kind": unit.kind,
                              "unit_lines": [unit.start, unit.end], "target_lines": len(target)})
            prompt = list(doc[start:])
            share = determined_share(target, prompt)
            case.meta.update(determined_share=round(share, 3), roxygen_problems=problems)
            if problems:
                case.meta["rejected"] = "roxygen_check"
            elif share < MIN_SHARE[geometry]:
                case.meta["rejected"] = "determinacy"
            out.append(case)
    return out
