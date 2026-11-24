"""Seed normalisation: optional blank-line policy, then `air format`.

v2 (default ``authored``): blank lines and comments are kept as the seed
authors wrote them (the v2 seeds were revised to TRAIN-like density), and
air collapses runs of blank lines to one. The C12 census records 309
finish_block near misses (156 prompts) whose canonical region equals gold
but whose file differs, e.g. by an extra blank line
(receipts/C12-census.json, observations.near_miss_with_canonical_region_equal),
so blank-line placement in golds matters; v2 matches TRAIN's distribution
instead of v1's dense house style.

v1 policies (kept for reproducing SYNTH-v1): ``dense`` removes blank lines
inside top-level expressions; ``sectioned`` puts one blank line before each
full-line comment. Blank lines inside multi-line strings are never touched.

v3.1: the seeds put ``call. = FALSE`` on almost every ``stop()``/``warning()``
(99.6% of calls; TRAIN 15%), so 36.8% of v3 finish rows show it against 4.3%
in TRAIN. ``thin_call_false`` removes the argument from a deterministic 88% of
seed files (whole files, so a prompt and its target always agree).
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
import re

from sepalith.data.synth.structure import FileStructure
from sepalith.training.rl.air_canonical import AirFormatter

STYLES = ("dense", "sectioned")
CALL_FALSE_KEEP_SHARE = 0.12  # files that keep `call. = FALSE` (TRAIN: 15% of stop/warning calls carry it)
CALL_FALSE_RE = re.compile(r",\s*call\.\s*=\s*FALSE\b")


def keeps_call_false(relative_path: str) -> bool:
    digest = int(hashlib.sha256(("call-false:" + relative_path).encode()).hexdigest(), 16)
    return digest % 10_000 < CALL_FALSE_KEEP_SHARE * 10_000


def thin_call_false(text: str, relative_path: str) -> tuple[str, str]:
    """(text, status): status is absent, kept or thinned."""
    if not CALL_FALSE_RE.search(text):
        return text, "absent"
    if keeps_call_false(relative_path):
        return text, "kept"
    return CALL_FALSE_RE.sub("", text), "thinned"


def style_for(relative_path: str) -> str:
    digest = int(hashlib.sha256(("style:" + relative_path).encode()).hexdigest(), 16)
    return "sectioned" if digest % 10 < 3 else "dense"


def apply_blank_policy(lines: Sequence[str], structure: FileStructure, style: str) -> list[str]:
    if style not in STYLES:
        raise ValueError(style)
    protected = structure.string_lines()
    inside: set[int] = set()
    for unit in structure.units:
        inside.update(range(unit.start + 1, unit.end))  # 1-based, excluding first/last line
    out: list[str] = []
    for number, line in enumerate(lines, start=1):
        if number in inside and number not in protected:
            if not line.strip():
                continue
            stripped = line.strip()
            if (style == "sectioned" and stripped.startswith("#") and not stripped.startswith("#'")
                    and out and out[-1].strip() and not out[-1].rstrip().endswith(("{", "(", "["))
                    and not out[-1].strip().startswith("#")):
                out.append("")
        out.append(line)
    return out


def format_text(formatter: AirFormatter, text: str) -> str | None:
    formatted = formatter.format(text)
    if formatted is None:
        return None
    if not formatted.endswith("\n"):
        formatted += "\n"
    return formatted


def normalise_file(source: Path, relative: str, structure: FileStructure, formatter: AirFormatter, *,
                   blank_policy: str = "authored", call_false: bool = False) -> tuple[str | None, dict]:
    text = source.read_text(encoding="utf-8")
    info: dict = {"seed": relative, "seed_sha256": hashlib.sha256(text.encode()).hexdigest()}
    problems = []
    if "\t" in text or "\r" in text:
        problems.append("tab_or_cr")
    if any(ord(ch) > 127 for ch in text):
        problems.append("non_ascii")
    for marker in (">>>>>>>", "<<<<<<<", "=======", "[NO_EDIT]", "<|user_cursor|>", "<[fim-"):
        if marker in text:
            problems.append("reserved_marker")
    if not structure.ok:
        problems.append("seed_does_not_parse")
    if problems:
        info["rejected"] = sorted(set(problems))
        return None, info
    lines = text.rstrip("\n").split("\n")
    if blank_policy == "authored":
        style = "authored"
        styled = "\n".join(lines) + "\n"
    else:
        style = style_for(relative)
        styled = "\n".join(apply_blank_policy(lines, structure, style)) + "\n"
    if call_false:
        styled, info["call_false"] = thin_call_false(styled, relative)
    formatted = format_text(formatter, styled)
    if formatted is None:
        info["rejected"] = ["air_failed"]
        return None, info
    again = format_text(formatter, formatted)
    if again != formatted:
        info["rejected"] = ["air_not_idempotent"]
        return None, info
    info.update(style=style, normalised_sha256=hashlib.sha256(formatted.encode()).hexdigest())
    return formatted, info
