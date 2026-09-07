"""parsep.py — config-aware prediction parse + the extension's two
post-heuristics, applied offline.

Parse = extension.ts parsePrediction semantics with the two searchable
toggles (marker-line drop, degenerate-repetition cut); the cut at the first
">>>>>>>", the <|user_cursor|> removal and the blank-strip are always on.
The scorer's fixed tail applies run_eval.norm (per-line rstrip + trailing
blank pop) identically to every candidate so validator comparisons match
the eval_scenarios convention.

Post-heuristics (offline mapping, documented in H1_RESULTS.md):
  comment_heuristic ON (extension default): cursor line is a non-empty
    comment and the first predicted line is code-looking -> prepend "".
  full_region_replace OFF (non-default): when the first predicted line
    starts with the typed partial (>= 4 chars trimmed), the extension would
    APPEND after the partial instead of replacing the line -> the scored
    first line becomes typed_partial + pred[0]. ON (default) scores the
    prediction as-is (whole-line replace).
"""
from __future__ import annotations

import re

MARKER_LINE = re.compile(
    r"^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|<\[fim-(middle|prefix|"
    r"suffix)\]>|<\|user_cursor\|>|<\|outline\|>)\s*$")
CODE_FIRST = re.compile(r"^[A-Za-z.][\w.$]*\s*(<-|=|\()")


def _norm(lines: list[str]) -> list[str]:
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def parse_prediction(text: str, cfg: dict) -> list[str]:
    """extension.parsePrediction with the config's parse toggles."""
    if type(text) is str:
        end = text.find(">>>>>>>")
        if end != -1:
            text = text[:end]
    else:
        if ">>>>>>>" in text:
            text = text.split(">>>>>>>")[0]
    text = text.replace("<|user_cursor|>", "")
    lines = [l[:-1] if l.endswith("\r") else l for l in text.split("\n")]
    if cfg["parse_marker_drop"]:
        lines = [l for l in lines if not MARKER_LINE.match(l)]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    if cfg["parse_rep_cut"]:
        for i in range(2, len(lines)):
            if lines[i] == lines[i - 1] == lines[i - 2]:
                lines = lines[:i]
                break
    return _norm(lines)


def cursor_position(region_old: list[str], cursor_idx: int):
    """(current_line, typed_partial) for the cursor's character offset in
    the joined region string (validate_example's coordinate)."""
    acc = 0
    for l in region_old:
        if acc <= cursor_idx <= acc + len(l):
            return l, l[:cursor_idx - acc]
        acc += len(l) + 1
    return region_old[-1], region_old[-1]


def apply_post_heuristics(pred_lines: list[str], row: dict, cfg: dict) -> list[str]:
    if not pred_lines:
        return pred_lines
    current_line, typed_partial = cursor_position(row["region_old"],
                                                  row["cursor_idx"])
    lines = list(pred_lines)
    if cfg["comment_heuristic"]:
        first = lines[0].strip()
        cl = current_line.strip()
        if (cl.startswith("#") and cl != "#" and CODE_FIRST.match(first)):
            lines.insert(0, "")
    if not cfg["full_region_replace"]:
        tp = typed_partial.strip()
        if len(tp) >= 4 and lines and lines[0].strip().startswith(tp):
            lines = [typed_partial + lines[0].lstrip()] + lines[1:]
    return lines
