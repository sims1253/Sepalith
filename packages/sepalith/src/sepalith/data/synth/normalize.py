"""Seed normalisation: deterministic blank-line policy, then `air format`.

The C12 census showed that 434 of 477 whitespace-only finish_block near
misses differ from gold only in blank-line placement (air keeps single blank
lines, so they survive canonical comparison). SYNTH-v1 therefore gives every
seed file one of two predictable body styles before formatting:

- ``dense``: no blank lines inside top-level expressions;
- ``sectioned``: inside top-level expressions, a blank line before every
  full-line ``#`` comment whose previous line neither opens a block nor is
  itself a comment, and nowhere else.

Blank lines inside multi-line strings are never touched. The style is a pure
function of the seed path, so it is reproducible.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path

from sepalith.data.synth.structure import FileStructure
from sepalith.training.rl.air_canonical import AirFormatter

STYLES = ("dense", "sectioned")


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


def normalise_file(source: Path, relative: str, structure: FileStructure, formatter: AirFormatter) -> tuple[str | None, dict]:
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
    style = style_for(relative)
    lines = text.rstrip("\n").split("\n")
    styled = "\n".join(apply_blank_policy(lines, structure, style)) + "\n"
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
