"""Parse structure of R files via base R's getParseData (never evaluates code)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).with_name("r_structure.R")


@dataclass(frozen=True)
class Block:
    open_line: int
    open_col: int
    close_line: int
    close_col: int
    stmt_lines: tuple[int, ...]
    stmt_cols: tuple[int, ...]
    stmt_end_lines: tuple[int, ...]


@dataclass(frozen=True)
class Function:
    line: int
    end_line: int
    formals: tuple[str, ...]
    body_line: int
    body_braced: bool


@dataclass(frozen=True)
class Unit:
    """A top-level expression; lines are 1-based and inclusive."""

    start: int
    end: int
    end_col: int
    kind: str
    name: str
    blocks: tuple[Block, ...]
    funs: tuple[Function, ...]
    symbols: frozenset[str]
    roxygen: tuple[int, int] | None = None  # 1-based inclusive line span

    def outer_block(self) -> Block | None:
        """The braced block that closes on the unit's last line."""
        closing = [b for b in self.blocks if b.close_line == self.end]
        return min(closing, key=lambda b: (b.open_line, b.open_col)) if closing else None

    def main_function(self) -> Function | None:
        if self.kind == "assign_function" and self.funs:
            return min(self.funs, key=lambda f: f.line)
        return None


@dataclass
class FileStructure:
    path: str
    ok: bool
    error: str | None
    strings: list[tuple[int, int]] = field(default_factory=list)
    comments: frozenset[int] = frozenset()
    units: list[Unit] = field(default_factory=list)

    def string_lines(self) -> set[int]:
        """Lines strictly inside or ending a multi-line string (not the opening line)."""
        out: set[int] = set()
        for start, end in self.strings:
            out.update(range(start + 1, end + 1))
        return out


def _ints(value: object) -> tuple[int, ...]:
    if isinstance(value, list):
        return tuple(int(str(v)) for v in value)
    if isinstance(value, (int, float, str)):
        return (int(value),)
    return ()


def _unit(raw: dict) -> Unit:
    blocks = tuple(Block(int(b["open_line"]), int(b["open_col"]), int(b["close_line"]), int(b["close_col"]),
                         _ints(b["stmt_lines"]), _ints(b["stmt_cols"]), _ints(b["stmt_end_lines"]))
                   for b in raw.get("blocks") or [])
    funs = tuple(Function(int(f["line"]), int(f["end_line"]),
                          tuple(f["formals"]) if isinstance(f["formals"], list) else (f["formals"],),
                          int(f["body_line"]), bool(f["body_braced"]))
                 for f in raw.get("funs") or [])
    symbols = raw.get("symbols") or []
    return Unit(int(raw["start"]), int(raw["end"]), int(raw["end_col"]), str(raw["kind"]), str(raw["name"]),
                blocks, funs, frozenset(symbols if isinstance(symbols, list) else [symbols]))


def attach_roxygen(structure: FileStructure, lines: Sequence[str]) -> None:
    """Mark the contiguous ``#'`` block directly above each unit."""
    units = []
    for unit in structure.units:
        end = unit.start - 1  # 1-based line above the unit
        start = end
        while start >= 1 and lines[start - 1].lstrip().startswith("#'"):
            start -= 1
        span = (start + 1, end) if start < end else None
        units.append(Unit(unit.start, unit.end, unit.end_col, unit.kind, unit.name, unit.blocks, unit.funs,
                          unit.symbols, span))
    structure.units = units


def describe(paths: Sequence[Path], *, batch: int = 40) -> dict[str, FileStructure]:
    out: dict[str, FileStructure] = {}
    for index in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[index:index + batch]]
        proc = subprocess.run(["Rscript", "--vanilla", str(SCRIPT), *chunk], capture_output=True, text=True,
                              check=True, timeout=600)
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            structure = FileStructure(raw["file"], bool(raw["ok"]), raw.get("error"))
            if structure.ok:
                structure.strings = [tuple(s) for s in raw.get("strings") or []]  # type: ignore[misc]
                structure.comments = frozenset(_ints(raw.get("comments")))
                structure.units = [_unit(u) for u in raw.get("units") or []]
                text = Path(raw["file"]).read_text(encoding="utf-8")
                attach_roxygen(structure, text.split("\n"))
            out[raw["file"]] = structure
    return out
