"""render.py — config-aware prompt builders for the H1 harness search.

Two builders, both markers-FROZEN to the zeta2 canonical set:

  render_scenario(row, cfg, texts)      D_harness/held-out scenario rows:
      the run_eval.render_zeta2 part order (edit_history + filename +
      prefix + CURRENT/region/=======/fim-middle), with the config's
      prefix-cap/scope knobs and the GEPA text slots applied. At
      DEFAULT_CONFIG/DEFAULT_TEXTS this is byte-identical to
      assemble_sft_v2.edit_row's prompt (unit-pinned).

  build_scoped_prompt(lines, cl, cc, ...)  noopFP rows: the
      eval_noop_fp.build_prompt port (extension v0.0.6 suffix-completion
      render, <|user_cursor|> in the region), with the config's cap /
      pin / outline / suffix-truncation-direction knobs. At DEFAULT_CONFIG
      byte-identical to eval_noop_fp.build_prompt (unit-pinned).

Knob semantics (offline mapping of the extension knobs, documented in
H1_RESULTS.md): prefix+suffix share MAX_PREFIX_SUFFIX_CHARS; the prefix is
truncated from its START (keep tail); under "protect-pin" an active pin's
span is never cut (the pin falls back outline-only when it cannot fit,
mirroring buildScope); under "hard-cut" the cut is purely by budget. The
suffix in the scoped builder follows buildScopedPrompt: pin active → pinned
remainder leads + older lines cut from their END under the cap
(protect-pin), or the flat suffix cut from its TOP at the cap (hard-cut);
pin inactive → suffix unbounded (current extension behavior) under
protect-pin, capped under hard-cut.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))            # run_eval

from rscan import build_scope, line_span_chars           # noqa: E402
from harness_config import DEFAULT_TEXTS                 # noqa: E402

CURSOR2 = "<|user_cursor|>"


def _ev_lines(event_diff: str) -> list[str]:
    ev = event_diff or ""
    for tag in ("```diff\n", "```"):
        ev = ev.replace(tag, "")
    lines = ev.splitlines()
    if lines and lines[0].startswith("User edited"):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines


def _cut_prefix_keep_tail(prefix: list[str], budget: int) -> tuple[list[str], int]:
    """Truncate the prefix from its START keeping the tail (v0.0.6 rule)."""
    keep = used = 0
    for i in range(len(prefix) - 1, -1, -1):
        if used + len(prefix[i]) + 1 > budget:
            break
        used += len(prefix[i]) + 1
        keep += 1
    truncated = len(prefix) - keep
    return prefix[truncated:], truncated


def _scope_sections(lines, cursor_line, cfg, texts):
    """(pin, outline_lines) under the config's scope knobs."""
    if cfg["pin"] == "off" and cfg["outline"] == "off":
        return None, []
    pin_cap = 0 if cfg["pin"] == "off" else cfg["pin"]
    scope = build_scope(lines, cursor_line, pin_cap, cfg["outline"])
    pin = scope["pin"] if cfg["pin"] != "off" else None
    outline = []
    if scope["outline"]:
        outline = [texts.get("outline_header") or "<|outline|>"] + scope["outline"]
    return pin, outline


def _prefix_under_cap(prefix: list[str], suffix_chars: int, cfg, pin,
                      cursor_line, prefix_offset: int):
    """Apply MAX_PREFIX_SUFFIX_CHARS to the prefix, honoring pin protection.

    pin is (start, end, name) in the FULL line window; prefix_offset is the
    index of prefix[0] in that window. Returns (kept_prefix, truncated)."""
    cap = cfg["prefix_suffix_cap"]
    budget = max(0, cap - suffix_chars)
    if pin is not None and cfg["suffix_truncation"] == "protect-pin":
        prot_start = max(pin[0] - prefix_offset, 0)  # protected prefix span
        prot = prefix[prot_start:]
        prot_chars = sum(len(l) + 1 for l in prot)
        if prot_chars <= budget:
            above, rem = prefix[:prot_start], budget - prot_chars
            keep = used = 0
            for i in range(len(above) - 1, -1, -1):
                if used + len(above[i]) + 1 > rem:
                    break
                used += len(above[i]) + 1
                keep += 1
            trunc = len(above) - keep
            return above[trunc:] + prot, trunc
        # pin cannot fit -> falls back outline-only (buildScope rule)
    return _cut_prefix_keep_tail(prefix, budget)


def render_scenario(row: dict, cfg: dict, texts: dict | None = None) -> tuple[str, dict]:
    texts = texts or DEFAULT_TEXTS
    row = dict(row)
    row.setdefault("suffix", [])
    prefix = list(row["prefix"])
    suffix = list(row["suffix"])
    region = list(row["region_old"])
    ev = _ev_lines(row.get("event_diff"))

    lines = prefix + region
    cursor_line = len(prefix)  # the region begins right after the prefix
    pin, outline = _scope_sections(lines, cursor_line, cfg, texts)

    suffix_chars = sum(len(l) + 1 for l in suffix)
    kept, truncated = _prefix_under_cap(prefix, suffix_chars, cfg, pin,
                                        cursor_line, 0)

    # cursor marker exactly as run_eval.with_cursor (line-index semantics)
    region_render = list(region)
    if 0 <= row["cursor_idx"] < len(region_render):
        region_render[row["cursor_idx"]] += CURSOR2

    parts = ["<[fim-suffix]>"] + suffix + ["<[fim-prefix]><filename>edit_history"]
    if ev:
        parts += ev + [""]
    parts += [f"<filename>{row['path']}"] + kept + outline
    if texts.get("checklist_line"):
        parts.append(texts["checklist_line"])
    if texts.get("instruction_line"):
        parts.append(texts["instruction_line"])
    parts += ["<<<<<<< CURRENT"] + region_render + ["=======", "<[fim-middle]>"]
    meta = dict(truncated_prefix=truncated, pin=bool(pin), outline_lines=len(outline),
                scope_mode=("pin+outline" if pin and outline else "pin" if pin
                            else "outline" if outline else "off"))
    return "\n".join(parts), meta


def build_scoped_prompt(lines: list[str], cursor_line: int, cursor_char: int,
                        rel_path: str, cfg: dict, texts: dict | None = None):
    """noopFP builder: the extension v0.0.6 render under the config knobs."""
    texts = texts or DEFAULT_TEXTS
    line = lines[cursor_line] if cursor_line < len(lines) else ""
    before, after = line[:cursor_char], line[cursor_char:]
    region_old = [before + CURSOR2]
    cap = cfg["prefix_suffix_cap"]

    pin, outline = _scope_sections(lines, cursor_line, cfg, texts)

    if pin is not None and pin[1] >= cursor_line:
        pinned = lines[cursor_line + 1: pin[1] + 1]
        older = lines[pin[1] + 1:]
        if cfg["suffix_truncation"] == "protect-pin":
            used = len(after) + 1 + sum(len(l) + 1 for l in pinned)
            older_keep = 0
            for i in range(len(older)):
                if used + len(older[i]) + 1 > cap:
                    break
                used += len(older[i]) + 1
                older_keep += 1
            suffix = [after] + pinned + older[:older_keep]
            suffix_truncated = len(older) - older_keep
        else:  # hard-cut: flat suffix cut from its TOP at the cap
            flat = [after] + lines[cursor_line + 1:]
            used, keep = 0, 0
            for i in range(len(flat)):
                if used + len(flat[i]) + 1 > cap:
                    break
                used += len(flat[i]) + 1
                keep += 1
            suffix = flat[:keep]
            suffix_truncated = len(flat) - keep
    else:
        suffix = [after] + lines[cursor_line + 1:]
        if cfg["suffix_truncation"] == "hard-cut":
            used, keep = 0, 0
            for i in range(len(suffix)):
                if used + len(suffix[i]) + 1 > cap:
                    break
                used += len(suffix[i]) + 1
                keep += 1
            suffix_truncated = len(suffix) - keep
            suffix = suffix[:keep]
        else:
            suffix_truncated = 0  # protect-pin, no pin: suffix unbounded (current)

    suffix_chars = sum(len(l) + 1 for l in suffix)
    prefix = lines[:cursor_line]
    kept, truncated = _prefix_under_cap(prefix, suffix_chars, cfg, pin,
                                        cursor_line, 0)

    parts = (["<[fim-suffix]>"] + suffix +
             [f"<[fim-prefix]><filename>{rel_path}"] + kept + outline)
    if texts.get("checklist_line"):
        parts.append(texts["checklist_line"])
    if texts.get("instruction_line"):
        parts.append(texts["instruction_line"])
    parts += ["<<<<<<< CURRENT"] + region_old + ["=======", "<[fim-middle]>"]
    meta = dict(truncated_prefix=truncated, truncated_suffix=suffix_truncated,
                pin=bool(pin), outline_lines=len(outline))
    return "\n".join(parts), meta
