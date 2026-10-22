"""format_propagation cases: the user is reformatting a file statement by
statement; the history shows one statement rewritten into air layout, the
region is the next statement still in the old layout, the target is its air
layout.

Sources are air-formatted, so the old layout is synthesised by collapsing a
multi-line statement (``one_line``: joined on one line; ``aligned``: arguments
continued under the opening parenthesis, closing parenthesis attached). A
collapse is used only if air turns the collapsed file back into the original
file byte for byte, so every target is exactly air's output in context.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sepalith.campaign_protocol import utf16_length
from sepalith.data.synth.cases import Case, Event, skip_blank, window_start_for
from sepalith.data.synth.structure import FileStructure
from sepalith.training.rl.air_canonical import AirFormatter

PREFIX_BUDGET = 3500
SUFFIX_BUDGET = 2000
MAX_ONE_LINE = 110
MAX_GAP = 60


@dataclass(frozen=True)
class Stmt:
    start: int  # 0-based first line
    end: int  # 0-based exclusive
    unit_index: int


def one_line(lines: Sequence[str]) -> list[str] | None:
    out = lines[0].rstrip()
    for line in lines[1:]:
        part = line.strip()
        if out.endswith(("(", "[")) or part.startswith((")", "]")):
            out += part
        else:
            out += " " + part
    return [out] if len(out) <= MAX_ONE_LINE else None


def aligned(lines: Sequence[str]) -> list[str] | None:
    head = lines[0].rstrip()
    if not head.endswith("(") or len(lines) < 4:
        return None
    last = lines[-1].strip()
    if not last.startswith(")"):
        return None
    body = list(lines[1:-1])
    base = len(body[0]) - len(body[0].lstrip())
    column = len(head)
    out = [head + body[0].strip()]
    for line in body[1:]:
        indent = len(line) - len(line.lstrip())
        if indent < base:
            return None
        out.append(" " * (column + indent - base) + line.strip())
    out[-1] += last
    return out


def candidates(lines: Sequence[str], structure: FileStructure) -> list[Stmt]:
    protected = structure.string_lines()
    found = []
    for index, unit in enumerate(structure.units):
        spans = []
        if not unit.blocks and not unit.funs:
            spans.append((unit.start, unit.end))
        for block in unit.blocks:
            for line, end in zip(block.stmt_lines, block.stmt_end_lines, strict=True):
                spans.append((line, end))
        for start, end in spans:
            if end - start < 1 or end - start > 14:
                continue
            chunk = lines[start - 1:end]
            if lines[start - 1][:len(lines[start - 1]) - len(lines[start - 1].lstrip())].strip():
                continue
            if any("{" in line or "}" in line or "#" in line for line in chunk):
                continue
            if any(n in protected for n in range(start, end + 1)):
                continue
            found.append(Stmt(start - 1, end, index))
    return sorted(set(found), key=lambda s: s.start)


def generate(lines: Sequence[str], structure: FileStructure, package_id: str, path: str,
             formatter: AirFormatter) -> list[Case]:
    original = "\n".join(lines) + "\n"
    stmts = candidates(lines, structure)
    collapsed: dict[tuple[int, str], list[str]] = {}
    for stmt in stmts:
        chunk = lines[stmt.start:stmt.end]
        for style, fn in (("one_line", one_line), ("aligned", aligned)):
            new = fn(chunk)
            if new is None:
                continue
            test = [*lines[:stmt.start], *new, *lines[stmt.end:]]
            if formatter.format("\n".join(test) + "\n") == original:
                collapsed[(stmt.start, style)] = new
    unit_end = {i: u.end for i, u in enumerate(structure.units)}
    unit_start = {i: u.start - 1 for i, u in enumerate(structure.units)}
    inside = set()
    for unit in structure.units:
        inside.update(range(unit.start - 1, unit.end))
    safe = {i for i in range(len(lines) + 1) if i not in inside}
    out: list[Case] = []
    for style in ("one_line", "aligned"):
        usable = [s for s in stmts if (s.start, style) in collapsed]
        for previous, region in zip(usable, usable[1:], strict=False):
            if region.start - previous.end > MAX_GAP or region.start < previous.end:
                continue
            # Current document: everything from the region down in the old layout.
            following = [s for s in usable if s.start >= region.start]
            doc: list[str] = []
            cursor = 0
            for stmt in following:
                doc += lines[cursor:stmt.start]
                doc += collapsed[(stmt.start, style)]
                cursor = stmt.end
            doc += lines[cursor:]

            def new_pos(old: int, following: list[Stmt] = following, style: str = style) -> int:
                """Map an old line boundary (not inside a collapsed statement) to the new document."""
                return old - sum((s.end - s.start) - len(collapsed[(s.start, style)])
                                 for s in following if s.end <= old)

            region_lines = collapsed[(region.start, style)]
            pre = list(doc)
            pre[previous.start:previous.end] = collapsed[(previous.start, style)]
            event = Event(path, tuple(pre), previous.start, tuple(collapsed[(previous.start, style)]),
                          tuple(lines[previous.start:previous.end]))
            need = min(previous.start, unit_start[region.unit_index], unit_start[previous.unit_index])
            start = window_start_for(doc, region.start, PREFIX_BUDGET, safe)
            if start > need:
                start = max(i for i in safe if i <= need) if any(i <= need for i in safe) else 0
                start = skip_blank(doc, start, need)
            r_end = region.start + len(region_lines)
            # Smallest safe end at or after the containing unit, then grow within budget.
            needed_end = new_pos(unit_end[region.unit_index])
            grown = needed_end
            used = 0
            for index in range(needed_end, len(doc)):
                used += utf16_length(doc[index]) + 1
                if used > SUFFIX_BUDGET:
                    break
                grown = index + 1
            # Back off to a top-level boundary so the window parses.
            safe_ends = {new_pos(u.end) for u in structure.units} | {new_pos(i + 1) for i in safe if i < len(lines)}
            window_end = max(e for e in safe_ends if needed_end <= e <= grown) if any(
                needed_end <= e <= grown for e in safe_ends) else needed_end
            case = Case("format_propagation", package_id, path, tuple(doc), region.start, r_end,
                        tuple(lines[region.start:region.end]), window_start=start, window_end=window_end,
                        cursor=(0, len(region_lines[0]) - len(region_lines[0].lstrip())), events=(event,),
                        meta={"rule": f"fmt_{style}", "event_line": previous.start + 1,
                              "region_line": region.start + 1, "target_lines": region.end - region.start,
                              "prefix_chars": sum(len(x) + 1 for x in doc[start:region.start]),
                              # air turns the collapsed file back into the source byte for byte (checked above)
                              "air_roundtrip": True})
            if case.meta["prefix_chars"] > 8000:
                case.meta["rejected"] = "window_too_large"
            out.append(case)
    return out
