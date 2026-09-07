"""rscan.py — line-for-line Python port of the scope functions in
extensions/vscode-sepalith/src/context_build.ts (cleanRLine, the signature
regexes, findEnclosingFunctionByScan, outlineFromScan, formatOutline,
buildScope). Everything is PURE; unit tests pin parity against the TS
semantics on hand-checked cases (see test_harness_search.py).
"""
from __future__ import annotations

import re

# JS \w == [A-Za-z0-9_]; keep the classes explicit so parity is exact.
NAMED_SIG = re.compile(r"^\s*([A-Za-z._][A-Za-z0-9._]*)\s*(?:<-|=)\s*function\s*\(")
ANON_SIG = re.compile(r"(?:^|[(,=\s])function\s*\(")


def clean_r_line(line: str) -> str:
    """Blank string bodies first (a '#' inside a string is not a comment),
    then drop the comment — braces inside either never count."""
    s = line if type(line) is str and "'" not in line and '"' not in line else re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", '""', line)
    return re.sub(r"#.*", "", s) if "#" in s else s


def match_signature(cleaned: str):
    """(is_signature, name|None). TS: null = not a signature line;
    {name: null} = an anonymous function(...) signature."""
    m = NAMED_SIG.match(cleaned)
    if m:
        return True, m.group(1)
    if ANON_SIG.search(cleaned):
        return True, None
    return False, None


def net_braces(cleaned: str) -> int:
    n = 0
    for ch in cleaned:
        if ch == "{":
            n += 1
        elif ch == "}":
            n -= 1
    return n


def find_enclosing_function_by_scan(lines: list[str], cursor_line: int):
    """(start_line, end_line, name|None) of the innermost enclosing function
    or None. Exact port of findEnclosingFunctionByScan (brace depth tracked
    from the top of the SIGNATURE line)."""
    cleaned = [clean_r_line(l) for l in lines]
    for c in range(cursor_line, -1, -1):
        is_sig, name = match_signature(cleaned[c] if c < len(cleaned) else "")
        if not is_sig:
            continue
        depth = 0
        saw_open = False
        inside = False
        for i in range(c, cursor_line + 1):
            text = cleaned[i] if i < len(cleaned) else ""
            if "{" in text:
                saw_open = True
            depth += net_braces(text)
            inside = saw_open and depth > 0  # set BEFORE the breaks
            if depth < 0:
                break
            if saw_open and depth <= 0:
                break
        if not inside:
            continue
        for i in range(cursor_line + 1, len(lines)):
            depth += net_braces(cleaned[i])
            if depth <= 0:
                return (c, i, name)
        return (c, len(lines) - 1, name)  # unclosed — approximate
    return None


def outline_from_scan(lines: list[str]):
    """[(1-based line, name)] top-level named signatures of the window."""
    entries = []
    depth = 0
    for i, line in enumerate(lines):
        cleaned = clean_r_line(line)
        m = NAMED_SIG.match(cleaned)
        if depth == 0 and m and m.group(1) is not None:
            entries.append((i + 1, m.group(1)))
        depth = max(0, depth + net_braces(cleaned))  # clamp: survive garbage
    return entries


def format_outline(entries, max_entries: int, max_chars: int) -> list[str]:
    """<= max_entries entries and <= max_chars chars, then an ellipsis line."""
    out, used, dropped = [], 0, 0
    for e in entries:
        text = f"{e[0]} {e[1]}"
        if len(out) >= max_entries or used + len(text) + 1 > max_chars:
            dropped = len(entries) - len(out)
            break
        out.append(text)
        used += len(text) + 1
    if dropped > 0:
        out.append(f"... ({dropped} more)")
    return out


def line_span_chars(lines: list[str], start: int, end: int) -> int:
    n = 0
    for i in range(start, min(end, len(lines) - 1) + 1):
        n += len(lines[i]) + 1
    return n


def build_scope(lines: list[str], cursor_line: int, pin_chars_cap: int,
                outline_cfg) -> dict:
    """Port of buildScope: pin active only if the enclosing function fits the
    cap; the pinned function's outline entry is dropped (doc rule 1)."""
    enclosing = find_enclosing_function_by_scan(lines, cursor_line)
    pin = None
    if enclosing and enclosing[0] <= cursor_line <= enclosing[1]:
        if line_span_chars(lines, enclosing[0], enclosing[1]) <= pin_chars_cap:
            pin = enclosing
    entries = outline_from_scan(lines)
    if pin:
        entries = [e for e in entries if e[0] - 1 != pin[0]]
    outline = []
    if outline_cfg != "off":
        max_entries, max_chars = outline_cfg
        outline = format_outline(entries, max_entries, max_chars)
    return {"pin": pin, "outline": outline, "mode":
            ("pin+outline" if pin and outline else "pin" if pin
             else "outline" if outline else "off")}
