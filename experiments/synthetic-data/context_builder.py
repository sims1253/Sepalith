#!/usr/bin/env python3
"""Scope-aware context builder, shared by dataset renderers and the extension.

Implements the "Scope-aware context" section of docs/prompt-format.md so
training data and inference prompts cannot drift apart: one module computes
the enclosing-function pin and the file outline for both sides.

Public API (lines: list[str] without trailing newlines; cursor_line: 0-based
row index, the same convention as VS Code Position.line):

  enclosing_function(lines, cursor_line) -> span dict | None
      {name, signature, start_line, end_line, start_byte, end_byte} for the
      top-level function definition containing the row, else None at top
      level. "Top-level function definition" is the finish_block.py query
      pattern: a root-level binary_operator with a <- / = / <<- operator and
      a function_definition RHS. Scope resolution is syntax-only; it never
      requires LSP or ry (spec rule 3). A cursor inside a nested definition
      resolves to its enclosing top-level function; same-line one-liners
      resolve to the first in document order. Bytes are offsets into
      "\\n".join(lines), the same source the spans are parsed from.

  outline(lines, cursor_line) -> list[str]
      One line per top-level signature, rendered `name <- function(<args>)`
      with internal whitespace collapsed (multi-line signatures become one
      line). Signatures only — no bodies, no nested definitions (spec rule
      2). The enclosing function's signature is dropped while the cursor is
      inside it (dedup against the pinned scope, spec rule 1); exact
      duplicate signature lines collapse to one. Non-identifier assignment
      targets (x$y <- function, names(x) <- function) are skipped: they have
      no clean one-line signature.

  pin_split(lines, cursor_line) -> (prefix, pinned, rest)
      The scope-pin split: prefix = lines strictly above the cursor; when
      the cursor is inside a function, pinned = the function remainder from
      the cursor row through its end row (the suffix head that truncation
      must never eat); rest = the file below the function. At top level
      nothing is pinned (spec rules 1-2). prefix + pinned + rest always
      reassembles the file exactly; the renderer splits pinned[0] at the
      cursor column for the <|cursor|> partial-line zone.

CLI: python context_builder.py <file.R> <cursor_line> [--brief]
      Prints {enclosing, outline, pin} as JSON (cursor_line 0-based) for
      eyeballing and for the extension to shell out to during development.
      --brief replaces the pin line lists with counts + boundary lines.

Tree-sitter discipline (inherited from finish_block.py / roxygen_drafting.py):
rows are derived from byte offsets via bisect over per-line start offsets —
Node.start_point / end_point are NEVER touched. In this tree_sitter build
point access segfaults the interpreter; the byte attributes (start_byte,
end_byte, type, children) are the stable surface both sibling scripts use.

CPU-only, no LSP, no ry, no network.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from bisect import bisect_right
from itertools import accumulate

try:  # no fallback by design: scope resolution must be tree-sitter (spec)
    import tree_sitter_r
    from tree_sitter import Language, Parser
except ImportError as e:
    sys.exit(f"tree-sitter-r unavailable ({e}); refusing to run without it")

ASSIGN_OPS = ("<-", "=", "<<-")  # roxygen_drafting.py convention

_LANGUAGE = Language(tree_sitter_r.language())  # keep alive: the C parser
_parser: Parser | None = None                   # points into it (GC -> segfault)


def _get_parser() -> Parser:
    global _parser
    if _parser is None:
        _parser = Parser(_LANGUAGE)
    return _parser


def _one_line(s: str) -> str:
    """Collapse whitespace runs (incl. newlines) to single spaces, then trim
    the spaces a wrapped signature leaves after '(' and before ')'."""
    s = " ".join(s.split())
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", s))


def _functions(src: bytes, starts: list[int]) -> list[dict]:
    """Top-level function definitions via the finish_block.py query pattern.

    starts = byte offset of each line start (row 0 -> 0); rows come from
    bisect, never from Node.start_point (see module docstring).
    """

    def row(byte: int) -> int:
        return bisect_right(starts, byte) - 1

    out = []
    tree = _get_parser().parse(src)  # local ref: nodes are read while it lives
    for node in tree.root_node.children:
        if node.type != "binary_operator":
            continue
        kids = node.children
        rhs = [c for c in kids if c.type == "function_definition"]
        if not rhs or not any(c.type in ASSIGN_OPS for c in kids):
            continue
        fn = rhs[0]
        params = next((c for c in fn.children if c.type == "parameters"), None)
        name_node = kids[0]
        name = src[name_node.start_byte:name_node.end_byte].decode(
            "utf-8", "replace")
        ptxt = (src[params.start_byte:params.end_byte].decode("utf-8", "replace")
                if params is not None else "()")
        out.append(dict(
            name=name,
            signature=_one_line(f"{name} <- function{ptxt}"),
            plain_name=name_node.type == "identifier",
            start_line=row(node.start_byte),
            end_line=row(node.end_byte - 1),  # roxygen_drafting.py convention
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        ))
    return out


def _analyze(lines: list[str]) -> tuple[bytes, list[int], list[dict]]:
    enc = [l.encode("utf-8") for l in lines]
    src = b"\n".join(enc)
    starts = list(accumulate((len(l) + 1 for l in enc), initial=0))
    return src, starts, _functions(src, starts)


def _clamp(cursor_line: int, n_lines: int) -> int:
    return max(0, min(cursor_line, n_lines))


def _enclosing(fns: list[dict], line: int) -> dict | None:
    for f in fns:
        if f["start_line"] <= line <= f["end_line"]:
            return f
    return None


def enclosing_function(lines: list[str], cursor_line: int) -> dict | None:
    """Span of the top-level function containing cursor_line, else None."""
    _, _, fns = _analyze(lines)
    f = _enclosing(fns, _clamp(cursor_line, len(lines)))
    if f is None:
        return None
    return {k: f[k] for k in ("name", "signature", "start_line", "end_line",
                              "start_byte", "end_byte")}


def outline(lines: list[str], cursor_line: int) -> list[str]:
    """One-line-per-top-level-signature outline, deduped against the pin."""
    _, _, fns = _analyze(lines)
    enc = _enclosing(fns, _clamp(cursor_line, len(lines)))
    seen, out = set(), []
    for f in fns:
        if f is enc:
            continue  # fully present in the prompt; drop from the index
        if not f["plain_name"]:
            continue  # no clean one-line signature (x$y <- function, ...)
        if f["signature"] in seen:
            continue
        seen.add(f["signature"])
        out.append(f["signature"])
    return out


def pin_split(lines: list[str], cursor_line: int
              ) -> tuple[list[str], list[str], list[str]]:
    """(prefix, pinned function remainder, rest-of-file) line lists."""
    _, _, fns = _analyze(lines)
    cur = _clamp(cursor_line, len(lines))
    enc = _enclosing(fns, cur)
    if enc is None:
        return lines[:cur], [], lines[cur:]
    return (lines[:cur],
            lines[cur:enc["end_line"] + 1],
            lines[enc["end_line"] + 1:])


def _cli() -> int:
    ap = argparse.ArgumentParser(
        description="Scope-aware context for one R file at one cursor row.")
    ap.add_argument("file", help="R source file")
    ap.add_argument("cursor_line", type=int,
                    help="cursor row, 0-based (tree-sitter / VS Code convention)")
    ap.add_argument("--brief", action="store_true",
                    help="replace pin line lists with counts + boundary lines")
    args = ap.parse_args()
    try:
        raw = open(args.file, "rb").read()
    except OSError as e:
        print(f"cannot read {args.file}: {e}", file=sys.stderr)
        return 2
    lines = [l.decode("utf-8", "replace").rstrip("\r") for l in raw.split(b"\n")]
    enc = enclosing_function(lines, args.cursor_line)
    outl = outline(lines, args.cursor_line)
    prefix, pinned, rest = pin_split(lines, args.cursor_line)
    if args.brief:
        pin = dict(prefix_n=len(prefix), pinned_n=len(pinned), rest_n=len(rest),
                   pinned_first=pinned[0] if pinned else None,
                   pinned_last=pinned[-1] if pinned else None)
    else:
        pin = dict(prefix=prefix, pinned=pinned, rest=rest)
    print(json.dumps(dict(file=args.file, cursor_line=args.cursor_line,
                          enclosing=enc, outline=outl, pin=pin),
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
