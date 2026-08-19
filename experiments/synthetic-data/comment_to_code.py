#!/usr/bin/env python3
"""comment_to_code scenarios (block level): user writes a comment, cursor at
its end, the model proposes the code below it.

Two variants built from the normalized CRAN corpus
(/mnt/h/sepalith/normalized/<pkg>/<ver>/<pkg>/R/*.R) via tree-sitter-r:

  comment_to_code_real       REAL pairs (no LLM): intra-body `#` comments
                             (NOT roxygen `#'`) immediately followed by 2-10
                             code statements with no blank line and no other
                             comment before the next blank line.
  comment_to_code_synthetic  REVERSE-synthetic: comment-FREE statement blocks
                             (2-8 lines, >= 2 function calls or a pipe chain);
                             a free LLM writes ONE comment line, the real code
                             becomes the ground truth.

Example JSON shape (block-level; cursor_idx is a character offset into
"\n".join(region_old) exactly like scenarios.py, so the cursor at the end of
the comment line is the empty region line right below it):

  {"family": "comment_to_code_real", "package": ..., "path": ...,
   "prefix": [lines ending with the comment line],
   "region_old": [""], "region_new": [statement lines], "cursor_idx": 0,
   "event_diff": "", "note": ..., "generator": ...}  # generator: synthetic only

Validation mirrors scenarios.py rigor: every constructed example must pass
validate_example() (structure, cursor convention, comment/roxygen rules,
region_new re-parses as clean statements, block references are not
undefined-heavy), and calibrate() asserts the no-op baseline (predict
nothing) scores exactly 0.

Usage:
  uv run python experiments/synthetic/comment_to_code.py --calibrate
  uv run python experiments/synthetic/comment_to_code.py --packages 150
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import scenarios as S
from scenarios import (Bundle, IDENT_RE, RESERVED, exact_reward, iter_bundles,
                       list_packages, node_text, noop_baseline_score, parser,
                       traverse)

ROOT = S.ROOT
OUT_DIR = S.OUT_DIR
FAMILIES = ("comment_to_code_real", "comment_to_code_synthetic")
TARGET_A = 3000
TARGET_B = 4000
MAX_PER_FILE_A = 12          # spread examples across files/packages
MAX_PER_FILE_B = 6
MAX_PREFIX_LINES = 30
MAX_BLOCK_LINES = 25
MIN_API_KEYS = ("family", "package", "path", "prefix", "region_old",
                "region_new", "cursor_idx", "event_diff", "note")

# best-effort "common globals" for the undefined-name heuristic
BASE_FNS = {
    "abs", "all", "any", "apply", "array", "as.character", "as.complex",
    "as.data.frame", "as.Date", "as.factor", "as.integer", "as.list",
    "as.logical", "as.matrix", "as.numeric", "as.POSIXct", "as.vector",
    "assign", "attr", "attributes", "basename", "c", "cat", "cbind",
    "ceiling", "class", "colMeans", "colnames", "colSums", "cummax", "cummin",
    "cumprod", "cumsum", "cut", "data.frame", "date", "diag", "diff", "dim",
    "dimnames", "dir", "dir.create", "dirname", "do.call", "double",
    "droplevels", "eval", "exists", "exp", "expression", "factor", "file",
    "file.copy", "file.exists", "file.path", "file.remove", "Filter",
    "findInterval", "floor", "format", "formatC", "get", "gsub", "head",
    "ifelse", "integer", "interaction", "invisible", "is.character",
    "is.data.frame", "is.element", "is.factor", "is.finite", "is.function",
    "is.infinite", "is.list", "is.logical", "is.na", "is.nan", "is.null",
    "is.numeric", "is.object", "is.vector", "lapply", "length", "levels",
    "list", "list.files", "log", "log10", "log2", "logical", "lower.tri",
    "make.names", "mapply", "Map", "match", "matrix", "max", "mean",
    "merge", "message", "min", "missing", "mode", "nchar", "ncol",
    "nlevels", "normalizePath", "nrow", "numeric", "order", "paste",
    "paste0", "pmax", "pmin", "print", "prod", "quote", "range", "rank",
    "rbind", "read.csv", "readLines", "read.table", "regexpr", "rep",
    "require", "requireNamespace", "return", "rev", "round", "rowMeans",
    "rownames", "rowSums", "sample", "sapply", "scale", "seq", "seq_along",
    "seq_len", "setdiff", "setequal", "setNames", "sign", "sin", "sort",
    "split", "sprintf", "sqrt", "stop", "stopifnot", "sub", "subset",
    "substr", "substring", "sum", "summary", "suppressMessages",
    "suppressWarnings", "switch", "system.file", "system.time", "t", "table",
    "tabulate", "tail", "tapply", "tempdir", "tempfile", "tolower",
    "toupper", "transform", "trimws", "try", "tryCatch", "typeof", "unlist",
    "unique", "unname", "upper.tri", "vapply", "vector", "warning", "which",
    "which.max", "which.min", "with", "within", "write.csv", "write.table",
    "writeLines", "xtfrm",
}
BASE_CONSTS = {
    "LETTERS", "letters", "month.abb", "month.name", "pi", "R.version",
    "R.version.string", "version", ".Machine", ".Platform", ".Random.seed",
    ".Options", "commandArgs", "formals", "body", "environment", "T", "F",
}
COMMON_GLOBALS = BASE_FNS | BASE_CONSTS | RESERVED

HASH_IN_LINE = re.compile(rb"#")
LHS_RE = re.compile(r"^\s*([A-Za-z.][A-Za-z0-9._]*)\s*(?:<-|=|<<-)")
FOR_VAR_RE = re.compile(r"^\s*for\s*\(\s*([A-Za-z.][A-Za-z0-9._]*)\s+in\b")


def is_comment_line(line: str) -> bool:
    return line.lstrip().startswith("#")


def is_roxygen(line: str) -> bool:
    return line.lstrip().startswith("#'")


def line_has_inline_comment(line_bytes: bytes) -> bool:
    return HASH_IN_LINE.search(S.strip_strings(line_bytes)) is not None


# ---------------------------------------------------------------------------
# tree-sitter fragment helpers
# ---------------------------------------------------------------------------

def parse_fragment(text: str):
    return parser.parse(text.encode("utf-8", "surrogateescape"))


def fragment_statements(text: str) -> list | None:
    """Top-level statements of a code fragment, or None if it does not parse
    cleanly (any ERROR/missing node -> None)."""
    tree = parse_fragment(text)
    if tree.root_node.has_error:
        return None
    for n in traverse(tree.root_node):
        if n.type == "ERROR" or n.is_missing:
            return None
    return [c for c in tree.root_node.children if c.is_named]


def fragment_calls_and_pipes(text: str) -> tuple[int, bool]:
    tree = parse_fragment(text)
    calls = sum(1 for n in traverse(tree.root_node) if n.type == "call")
    stripped = S.strip_strings(text.encode())
    has_pipe = b"%>%" in stripped or b"|>" in stripped
    return calls, has_pipe


# ---------------------------------------------------------------------------
# definedness heuristic (best-effort, tree-sitter identifiers)
# ---------------------------------------------------------------------------

def _skip_identifier(n) -> bool:
    """Identifiers that are structurally qualified/positional (df$col,
    pkg::fun, formula terms) must not count as unbound references.
    Caller-position identifiers are reported separately (allowed set is
    wider: same-package helper functions are invisible to us)."""
    p = n.parent
    if p is None:
        return True
    if p.type in ("extract_operator", "namespace_operator", "slot_operator"):
        named = [c for c in p.children if c.is_named]
        if named and n is not named[0]:
            return True  # RHS of $ / :: / @
    if p.type == "call" and p.children and p.children[0] is n:
        return False  # caller position: handled separately
    anc = p
    while anc is not None:
        if anc.type == "formula":
            return True  # data-frame columns in formulas -> allowed
        anc = anc.parent
    return False


def unbound_refs(block_text: str, bound: set[str]) -> tuple[list[str], list[str]]:
    """(unbound plain identifiers, unbound caller names) in block_text."""
    raw = block_text.encode("utf-8", "surrogateescape")
    tree = parse_fragment(block_text)
    refs, callers = [], []
    for n in traverse(tree.root_node):
        if n.type != "identifier":
            continue
        name = node_text(raw, n).decode("utf-8", "replace")
        if not IDENT_RE.match(name) or _skip_identifier(n):
            continue
        if S.parent_is_caller(n):
            callers.append(name)
        else:
            refs.append(name)
    inner = block_bindings(block_text)  # names bound within the block itself
    known = bound | inner | COMMON_GLOBALS
    return (sorted({r for r in refs if r not in known}),
            sorted({c for c in callers if c not in known}))


def block_bindings(text: str) -> set[str]:
    raw = text.encode("utf-8", "surrogateescape")
    out: set[str] = set()
    for n in traverse(parse_fragment(text).root_node):
        if n.type == "binary_operator" and n.children and \
                n.children[0].type == "identifier":
            out.add(node_text(raw, n.children[0]).decode("utf-8", "replace"))
        elif n.type == "for_statement":
            for c in n.children:
                if c.type == "identifier":
                    out.add(node_text(raw, c).decode("utf-8", "replace"))
    return out


def identifier_count(block_text: str) -> int:
    return sum(1 for n in traverse(parse_fragment(block_text).root_node)
               if n.type == "identifier" and not _skip_identifier(n))


def undefined_heavy(block_text: str, bound: set[str],
                    max_unbound: int = 3, max_ratio: float = 0.25) -> bool:
    """Extraction-time gate: skip blocks that reference clearly-undefined
    names (neither bound earlier, nor common globals)."""
    plain, calls = unbound_refs(block_text, bound)
    unbound = len(plain) + len(calls)
    total = identifier_count(block_text)
    return unbound > max_unbound or (total > 0 and unbound / total > max_ratio)


# ---------------------------------------------------------------------------
# function-body geometry
# ---------------------------------------------------------------------------

def function_bodies(b: Bundle) -> list[dict]:
    """All braced function bodies with their innermost-enclosure metadata."""
    out = []
    src = b.src
    for fn in traverse(b.tree.root_node):
        if fn.type != "function_definition":
            continue
        body = next((c for c in fn.children if c.type == "braced_expression"), None)
        if body is None:
            continue
        params = next((c for c in fn.children if c.type == "parameters"), None)
        name, assign_start = "", fn.start_byte
        parent = fn.parent
        if parent is not None and parent.type == "binary_operator":
            assign_start = parent.start_byte
            name = node_text(src, parent.children[0]).decode("utf-8", "replace")
        pnames: set[str] = set()
        if params is not None:
            for p in traverse(params):
                if p.type == "identifier":
                    pnames.add(node_text(src, p).decode("utf-8", "replace"))
        out.append(dict(fn=fn, body=body, name=name,
                        assign_start=assign_start, params=pnames))
    return out


def innermost_body(bodies: list[dict], byte: int) -> dict | None:
    best = None
    for t in bodies:
        if t["body"].start_byte < byte < t["body"].end_byte:
            if best is None or (t["body"].end_byte - t["body"].start_byte) < \
                    (best["body"].end_byte - best["body"].start_byte):
                best = t
    return best


def enclosing_bindings(b: Bundle, bodies: list[dict], inner: dict,
                       upto_row: int) -> set[str]:
    """Names plausibly bound before row `upto_row`: parameters of the
    innermost + enclosing functions, assignment/for-loop LHS in the prefix
    lines, and file-scope function names (same-file helpers)."""
    bound: set[str] = set(inner["params"])
    for t in bodies:  # enclosing function parameters
        if t is not inner and t["body"].start_byte < inner["body"].start_byte \
                and inner["body"].end_byte < t["body"].end_byte:
            bound |= t["params"]
    start_row, _ = b.rowcol(inner["assign_start"])
    for r in range(start_row, upto_row):
        if r >= b.nlines():
            break
        m = LHS_RE.match(b.line_str(r)) or FOR_VAR_RE.match(b.line_str(r))
        if m:
            bound.add(m.group(1))
    for n in b.tree.root_node.children:  # file-scope function names
        if n.type == "binary_operator" and n.children and \
                n.children[0].type == "identifier":
            if any(c.type == "function_definition" for c in n.children):
                bound.add(node_text(b.src, n.children[0]).decode("utf-8", "replace"))
    return bound


# ---------------------------------------------------------------------------
# variant A: real comment -> code pairs
# ---------------------------------------------------------------------------

def block_rows_after(b: Bundle, comment_row: int, body_end_row: int,
                     max_lines: int = MAX_BLOCK_LINES) -> list[int] | None:
    """Rows of code statements immediately after a comment: contiguous,
    non-blank, comment-free (full-line AND inline), inside the braces.
    Returns None if the run is empty (a blank line or another comment comes
    first)."""
    rows = []
    r = comment_row + 1
    while r < body_end_row and r < b.nlines() and len(rows) < max_lines:
        line = b.line_str(r)
        if not line.strip() or is_comment_line(line):
            break
        if line.lstrip().startswith("}"):
            break  # closing brace of the enclosing block ends the region
        if line_has_inline_comment(b.line_bytes(r)):
            break  # a trailing # comment also counts as 'another comment'
        rows.append(r)
        r += 1
    return rows or None


def extract_comment_pairs(b: Bundle, cap: int = MAX_PER_FILE_A) -> list[dict]:
    out = []
    src = b.src
    bodies = function_bodies(b)
    comments = [n for n in traverse(b.tree.root_node)
                if n.type == "comment" and n.start_byte < len(src)]
    for cn in comments:
        if len(out) >= cap:
            break
        txt = node_text(src, cn).decode("utf-8", "replace").strip()
        if not txt.startswith("#") or txt.startswith("#'"):
            continue  # intra-body plain comments only (never roxygen)
        row, _ = b.rowcol(cn.start_byte)
        if not is_comment_line(b.line_str(row)):
            continue  # trailing comment after code -> cursor semantics unclear
        inner = innermost_body(bodies, cn.start_byte)
        if inner is None:
            continue  # top-level comment, not inside any function body
        body_end_row, _ = b.rowcol(inner["body"].end_byte - 1)
        rows = block_rows_after(b, row, body_end_row)
        if rows is None:
            continue
        block = [b.line_str(r) for r in rows]
        stmts = fragment_statements("\n".join(block))
        if stmts is None or not (2 <= len(stmts) <= 10):
            continue
        start_row, _ = b.rowcol(inner["assign_start"])
        if row - start_row + 1 > MAX_PREFIX_LINES:
            continue
        prefix = [b.line_str(r) for r in range(start_row, row + 1)]
        bound = enclosing_bindings(b, bodies, inner, row)
        if undefined_heavy("\n".join(block), bound):
            continue
        note = (f"real comment -> following {len(stmts)} statement(s) "
                f"(fn {inner['name'] or '<anon>'})")
        ex = make_example("comment_to_code_real", b, prefix, block, note)
        try:
            validate_example(ex)
        except AssertionError:
            continue
        out.append(ex)
    return out


# ---------------------------------------------------------------------------
# variant B: comment-free blocks -> LLM comment
# ---------------------------------------------------------------------------

def candidate_blocks(b: Bundle, cap: int = MAX_PER_FILE_B) -> list[dict]:
    """Comment-free statement blocks (2-8 lines, >= 2 calls or a pipe chain)
    with the prefix lines up to the block start."""
    out = []
    bodies = function_bodies(b)
    for inner in bodies:
        if len(out) >= cap:
            break
        body_start_row, _ = b.rowcol(inner["body"].start_byte)
        body_end_row, _ = b.rowcol(inner["body"].end_byte - 1)
        r = body_start_row + 1
        while r < body_end_row and len(out) < cap:
            line = b.line_str(r)
            if (not line.strip() or is_comment_line(line)
                    or line.lstrip().startswith("}")
                    or line_has_inline_comment(b.line_bytes(r))):
                r += 1
                continue
            if r > body_start_row + 1 and is_comment_line(b.line_str(r - 1)):
                r += 1
                continue  # a real comment sits right above -> skip the run
            run = []  # maximal run of usable lines
            while r < body_end_row:
                line = b.line_str(r)
                if (not line.strip() or is_comment_line(line)
                        or line.lstrip().startswith("}")
                        or line_has_inline_comment(b.line_bytes(r))):
                    break
                run.append(r)
                r += 1
            if len(run) >= 2:
                out.extend(_windows(b, inner, bodies, run))
            r += 1
    return out


def _windows(b: Bundle, inner: dict, bodies: list[dict], run: list[int]) -> list[dict]:
    """Cut a clean-parsing line run into non-overlapping 2-8 line blocks of
    complete statements, keeping those with >= 2 calls or a pipe chain."""
    text = "\n".join(b.line_str(r) for r in run)
    stmts = fragment_statements(text)
    if stmts is None:
        return []
    spans = []
    for st in stmts:  # rows are fragment-relative; run[0] is its file row
        spans.append((st.start_point[0], st.end_point[0]))
    out = []
    i = 0
    while i < len(spans):
        chosen = None
        for j in range(i, len(spans)):
            lines = spans[j][1] - spans[i][0] + 1
            if lines > 8:
                break
            if lines >= 2:
                chosen = j
                if lines >= 3:  # prefer compact blocks: shorter prompts =
                    break        # less LLM reasoning, fewer null contents
        if chosen is None:
            i += 1
            continue
        rows = list(range(run[0] + spans[i][0], run[0] + spans[chosen][1] + 1))
        block = [b.line_str(r) for r in rows]
        block_text = "\n".join(block)
        calls, has_pipe = fragment_calls_and_pipes(block_text)
        if calls >= 2 or has_pipe:
            start_row, _ = b.rowcol(inner["assign_start"])
            prefix = [b.line_str(r) for r in range(start_row, rows[0])]
            if len(prefix) + 1 <= MAX_PREFIX_LINES and prefix:
                bound = enclosing_bindings(b, bodies, inner, rows[0])
                out.append(dict(prefix_lines=prefix, block=block, bound=bound,
                                calls=calls, pipe=has_pipe,
                                fn=inner["name"] or "<anon>"))
        i = chosen + 1
    return out


# ---------------------------------------------------------------------------
# record construction + validation
# ---------------------------------------------------------------------------

def make_example(family: str, b: Bundle, prefix: list[str],
                 block: list[str], note: str, generator: str | None = None) -> dict:
    ex = {
        "family": family,
        "package": b.package,
        "path": b.rel,
        "prefix": prefix,
        "region_old": [""],          # empty line at the cursor, right below
        "region_new": list(block),   # the proposed code block
        "cursor_idx": 0,             # cursor at start of that empty line
        "event_diff": "",            # no event in this family
        "note": note,
    }
    if generator is not None:
        ex["generator"] = generator
    return ex


def _gate_ok(comment: str) -> bool:
    return bool(comment) and len(comment) <= 90 and "\n" not in comment \
        and ";" not in comment and "<-" not in comment and "(" not in comment


def _split_top_commas(s: str) -> list[str]:
    parts, depth, start, q = [], 0, 0, None
    for i, ch in enumerate(s):
        if q:
            if ch == "\\":
                continue
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def prefix_bindings(prefix: list[str]) -> set[str]:
    """Names plausibly visible at the cursor, from the prefix text alone:
    assignment/for LHS plus every function-header parameter name."""
    bound: set[str] = set()
    for s in prefix:
        m = LHS_RE.match(s) or FOR_VAR_RE.match(s)
        if m:
            bound.add(m.group(1))
    text = "\n".join(prefix)
    for m in re.finditer(r"function\s*\(", text):
        depth, i = 1, m.end()
        start = i
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        for part in _split_top_commas(text[start:i - 1]):
            tok = part.split("=")[0].strip()
            if IDENT_RE.match(tok or ""):
                bound.add(tok)
    return bound


def validate_example(ex: dict) -> None:
    """Assert the example is internally consistent (mirrors scenarios.py
    rigor, adapted to the block-level no-event shape)."""
    for k in MIN_API_KEYS:
        assert k in ex and ex[k] is not None, f"missing field {k}"
    assert ex["family"] in FAMILIES
    assert ex["event_diff"] == "", "comment_to_code has no triggering event"
    for f in ("prefix", "region_old", "region_new"):
        assert isinstance(ex[f], list) and ex[f], f"{f} must be non-empty list"
        assert all(isinstance(l, str) and "\n" not in l for l in ex[f]), \
            f"{f} must be single-line strings"
    assert ex["region_old"] == [""], "region_old must be the empty cursor line"
    assert ex["cursor_idx"] == 0, "cursor must sit at the start of the empty line"
    assert ex["region_old"] != ex["region_new"], "GT must change the region"
    assert ex["region_new"] != [""]
    block = "\n".join(ex["region_new"])
    for l in ex["region_new"]:
        assert l.strip(), "no blank lines inside region_new"
        assert not is_comment_line(l), "no comment lines inside region_new"
        assert not line_has_inline_comment(l.encode()), \
            "no inline comments inside region_new"
    fam = ex["family"]
    stmts = fragment_statements(block)
    assert stmts is not None, "region_new must re-parse as valid statements"
    if fam == "comment_to_code_real":
        assert 2 <= len(stmts) <= 10, f"expected 2-10 statements, got {len(stmts)}"
        assert "generator" not in ex
    else:
        assert 1 <= len(stmts) <= 10, f"expected 1-10 statements, got {len(stmts)}"
        assert isinstance(ex.get("generator"), str) and ex["generator"]
    last = ex["prefix"][-1]
    assert is_comment_line(last), "prefix must end with the comment line"
    assert not is_roxygen(last), "roxygen comments never trigger this family"
    if fam == "comment_to_code_synthetic":
        content = last.lstrip()[1:].strip()
        assert _gate_ok(content), f"synthetic comment failed the gate: {content!r}"
    # definedness re-check (softer than the extraction gate: callers may be
    # same-file/same-package helpers invisible from prefix + block alone)
    plain, _calls = unbound_refs(block, prefix_bindings(ex["prefix"]))
    total = identifier_count(block)
    assert len(plain) <= 4, f"plain unbound references: {plain}"
    assert not total or len(plain) / total <= 0.5, \
        f"plain unbound references: {plain}"


def normalize_block(block: list[str]) -> str:
    return "\n".join(l.strip() for l in block)


# ---------------------------------------------------------------------------
# free-LLM comment generation (opencode primary, openrouter fallback)
# ---------------------------------------------------------------------------

OPENCODE_URL = "https://opencode.ai/zen/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ZAI_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
OPENCODE_MODEL = "deepseek-v4-flash-free"
OPENROUTER_MODEL = "dots-studio/dots-3-note-preview:free"
PROMPT = ('Write ONE concise R comment (max 80 chars, no code, describes what '
          'this block does). Code:\n\n{code}\n\nJSON only: {{"comment": string}}')
OPENCODE_COOLDOWN_S = 300.0  # free-tier quota resets slowly; re-probe later

API_STATS = {
    "opencode": dict(attempts=0, ok=0, err_429=0, err_provider=0, err_other=0,
                     err_timeout=0, err_json=0, lat_s=0.0),
    "zai": dict(attempts=0, ok=0, err_429=0, err_provider=0, err_other=0,
                err_timeout=0, err_json=0, lat_s=0.0),
    "openrouter": dict(attempts=0, ok=0, err_429=0, err_provider=0, err_other=0,
                       err_timeout=0, err_json=0, lat_s=0.0),
}
_tls = threading.local()
_opencode_until = 0.0
_stats_lock = threading.Lock()
_pace_lock = threading.Lock()
_pace_next = {"opencode": 0.0, "openrouter": 0.0, "zai": 0.0}
_PACE_GAP_S = {"opencode": 0.0, "openrouter": 6.5, "zai": 0.3}
# opencode: quota-based 429s, fast responses -> no artificial gap needed;
# openrouter dots-3:free: shared 1000/day free-model cap + provider
# congestion -> patient pacing converts retries into steady completions.


def _paced_wait(source: str):
    """Per-source pacer shared by all workers (concurrency is still <= 3;
    this only spaces request STARTS)."""
    with _pace_lock:
        now = time.time()
        start = max(now, _pace_next[source])
        _pace_next[source] = start + _PACE_GAP_S[source]
    delay = start - now
    if delay > 0:
        time.sleep(delay)


class _Retryable(Exception):
    """kind: 'rate' (429/5xx/provider) -> real backoff; 'json'/'net' -> 1s."""

    def __init__(self, msg: str, kind: str = "rate"):
        super().__init__(msg)
        self.kind = kind


def _session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "curl/8.5.0"  # plain urllib UA gets 403
        _tls.session = s
    return s


def _bump(source: str, key: str, dt: float = 0.0):
    with _stats_lock:
        d = API_STATS[source]
        d["attempts"] = d.get("attempts", 0) + 1
        d[key] = d.get(key, 0) + 1
        d["lat_s"] = d.get("lat_s", 0.0) + dt
        if key == "ok":
            global _last_ok_ts
            _last_ok_ts = time.time()


_last_ok_ts = time.time()
_outage_lock = threading.Lock()
_next_probe = 0.0


def _outage_gate():
    """During a total outage (no 2xx for 5 min) idle the workers, letting a
    single real probe request through every minute until a provider answers;
    then traffic resumes at full (paced) rate."""
    global _next_probe
    while time.time() - _last_ok_ts > 300:
        with _outage_lock:
            now = time.time()
            if now >= _next_probe:
                _next_probe = now + 60.0
                return  # this worker's next request is the probe
        time.sleep(15)


def _post(url: str, api_key: str, payload: dict, timeout: float,
          source: str) -> str:
    t0 = time.time()
    try:
        _paced_wait("opencode" if "opencode" in url else "openrouter")
        with _session().post(url, json=payload, timeout=timeout,
                             headers={"Authorization": f"Bearer {api_key}"}) as r:
            dt = time.time() - t0
            if r.status_code != 200:
                body = r.text[:200]
                if r.status_code == 429:
                    _bump(source, "err_429", dt)
                elif "provider" in body.lower() or r.status_code >= 500:
                    _bump(source, "err_provider", dt)
                else:
                    _bump(source, "err_other", dt)
                raise _Retryable(f"http {r.status_code}: {body}")
            _bump(source, "ok", dt)
            return r.json()["choices"][0]["message"]["content"]
    except _Retryable:
        raise
    except requests.RequestException as e:
        _bump(source, "err_timeout", time.time() - t0)
        raise _Retryable(f"{type(e).__name__}: {e}", kind="net")
    except (KeyError, IndexError, ValueError) as e:
        _bump(source, "err_json", time.time() - t0)
        raise _Retryable(f"{type(e).__name__}: {e}", kind="json")


def _extract_comment(content: str | None, source: str) -> str | None:
    if not isinstance(content, str) or not content.strip():
        _bump(source, "err_json")  # e.g. dots-3 reasoning-only null content
        raise _Retryable(f"non-string content: {str(content)[:80]!r}", kind="json")
    txt = content.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-z]*\n?|```$", "", txt).strip()
    try:
        val = json.loads(txt).get("comment")
    except ValueError:
        _bump(source, "err_json")
        return None
    return val if isinstance(val, str) else None


def _opencode_available() -> bool:
    return time.time() >= _opencode_until


def _set_opencode_cooldown(seconds: float = OPENCODE_COOLDOWN_S) -> None:
    global _opencode_until
    _opencode_until = time.time() + seconds


def _plain_code(code) -> str:
    """Code embedded in the comment-generation PROMPT is ALWAYS plain text:
    a list of block lines is newline-joined. A Python list repr (`['a', 'b']`)
    must never leak into a prompt (contaminated the first 82 synthetic rows)."""
    if isinstance(code, (list, tuple)):
        return "\n".join(str(l) for l in code)
    return code


def call_opencode(code: str, api_key: str) -> str | None:
    payload = {"model": OPENCODE_MODEL, "max_tokens": 300,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "user",
                             "content": PROMPT.format(code=_plain_code(code))}]}
    content = _post(OPENCODE_URL, api_key, payload, timeout=60, source="opencode")
    return _extract_comment(content, "opencode")


def call_openrouter(code: str, api_key: str) -> str | None:
    payload = {"model": OPENROUTER_MODEL, "max_tokens": 3000,
               "reasoning": {"effort": "low"},
               "messages": [{"role": "user",
                             "content": PROMPT.format(code=_plain_code(code))}]}
    content = _post(OPENROUTER_URL, api_key, payload, timeout=75,
                    source="openrouter")
    # dots-3 prefixes reasoning whitespace; strip before json.loads
    return _extract_comment(content, "openrouter")


def normalize_comment(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    c = raw.strip()
    c = c.lstrip("#").strip()          # models sometimes include the '#'
    c = re.sub(r"\s+", " ", c)
    if c.startswith('"') and c.endswith('"') and len(c) > 1:
        c = c[1:-1].strip()
    return c or None


def _backoff(kind: str, i: int) -> float:
    return (i + 1) * 5.0 if kind == "rate" else 1.0


def call_zai(code, api_key: str) -> str | None:
    payload = {
        "model": "glm-5.3",
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": PROMPT.format(code=_plain_code(code))}],
        "response_format": {"type": "json_object"},
        "max_tokens": 1500, "temperature": 0.7,
    }
    txt = _post(ZAI_URL, api_key, payload, timeout=60, source="zai")  # returns content
    if not txt.strip():
        raise _Retryable("zai empty content", kind="json")
    import json as _json
    try:
        obj = _json.loads(txt)
        c = obj.get("comment") if isinstance(obj, dict) else None
        if c and str(c).strip():
            return str(c)
    except Exception:
        pass
    return _extract_comment(txt, "zai")


def generate_comment(code, opencode_key: str,
                     openrouter_key: str, zai_key: str = "") -> tuple[str | None, str]:
    """One comment attempt: opencode (2 tries, backoff) then openrouter
    (3 tries, backoff on 429 / intermittent 'Provider returned error').
    `code` is the candidate block (list of lines) or plain text; either way
    the prompt embeds it newline-joined via _plain_code — never a list repr.
    Returns (comment, model_tag); comment is None if every try failed."""
    if zai_key:
        for i in range(2):
            try:
                c = call_zai(code, zai_key)
            except _Retryable as e:
                c = None
                if e.kind == "json" and i >= 1:
                    break
                time.sleep(_backoff(e.kind, i))
            if c is not None:
                return normalize_comment(c), "zai/glm-5.3"
    if _opencode_available():
        for i in range(2):
            try:
                c = call_opencode(code, opencode_key)
            except _Retryable as e:
                c = None
                time.sleep(_backoff(e.kind, i))
            if c is not None:
                return normalize_comment(c), f"opencode/{OPENCODE_MODEL}"
        _set_opencode_cooldown()
    for i in range(3):  # fallback tier
        try:
            c = call_openrouter(code, openrouter_key)
        except _Retryable as e:
            c = None
            if e.kind == "json" and i >= 1:
                break  # persistent null/unparsable content: stop burning quota
            time.sleep(_backoff(e.kind, i))
        if c is not None:
            return normalize_comment(c), f"openrouter/{OPENROUTER_MODEL}"
    return None, ""


def make_synthetic_example(cand: dict, comment: str, generator: str) -> dict:
    """Assemble a variant-B record from a candidate dict + gated comment."""
    prefix = list(cand["prefix_lines"]) + [f"  # {comment}"]
    note = (f"synthetic comment ({generator}) -> real {len(cand['block'])}-line "
            f"block ({cand['calls']} calls{'/pipe' if cand['pipe'] else ''}; "
            f"fn {cand['fn']})")
    ex = make_example("comment_to_code_synthetic",
                      _cand_bundle(cand["package"], cand["path"]), prefix,
                      cand["block"], note, generator)
    return ex


class _cand_bundle:
    """Duck-typed Bundle stand-in for candidates (package/path only)."""

    def __init__(self, package: str, rel: str):
        self.package = package
        self.rel = rel


# ---------------------------------------------------------------------------
# corpus scan (variant A + variant B candidates + comment density)
# ---------------------------------------------------------------------------

def scan_corpus(package_names, seed: int, target_a: int, n_candidates: int,
                time_budget_s: int = 900, verbose: bool = False):
    """Single pass over the sampled packages: real pairs, comment-free block
    candidates, and per-package comment-density stats. Each file is read and
    parsed exactly once (shared by both variants)."""
    rng = random.Random(seed)
    real, cands = [], []
    seen_blocks: set[str] = set()
    density: dict[str, dict] = {}
    t0 = time.time()
    files = 0
    for b in iter_bundles(package_names, rng):
        files += 1
        bodies = function_bodies(b)
        intra = [n for n in traverse(b.tree.root_node)
                 if n.type == "comment"
                 and node_text(b.src, n).lstrip().startswith(b"#")
                 and not node_text(b.src, n).lstrip().startswith(b"#'")]
        d = density.setdefault(b.package, dict(bodies=0, with_comment=0))
        d["bodies"] += len(bodies)
        d["with_comment"] += sum(
            1 for t in bodies
            if any(t["body"].start_byte < n.start_byte < t["body"].end_byte
                   for n in intra))
        if len(real) < target_a:
            for ex in extract_comment_pairs(b):
                key = normalize_block(ex["region_new"])
                if key in seen_blocks:
                    continue
                seen_blocks.add(key)
                real.append(ex)
        if len(cands) < n_candidates:
            for cand in candidate_blocks(b):
                key = normalize_block(cand["block"])
                if key in seen_blocks:
                    continue
                seen_blocks.add(key)
                cand["package"], cand["path"] = b.package, b.rel
                cands.append(cand)
        if verbose and files % 250 == 0:
            print(f"  files={files} elapsed={time.time()-t0:.0f}s "
                  f"real={len(real)} candidates={len(cands)}", flush=True)
        if time.time() - t0 > time_budget_s:
            break
        if len(real) >= target_a and len(cands) >= n_candidates:
            break
    rng.shuffle(cands)
    stats = dict(files=files, elapsed_s=round(time.time() - t0, 1),
                 real_pairs=len(real), candidates=len(cands),
                 density=density)
    return real, cands, stats


def density_summary(density: dict) -> dict:
    """% of function bodies containing >= 1 intra-body comment, overall and
    per-package variation."""
    pcts = [(pkg, d["with_comment"] / d["bodies"])
            for pkg, d in density.items() if d["bodies"]]
    vals = sorted(p for _, p in pcts)
    tot_b = sum(d["bodies"] for d in density.values())
    tot_c = sum(d["with_comment"] for d in density.values())
    q = lambda x: vals[min(len(vals) - 1, int(x * len(vals)))] if vals else 0.0
    return {
        "bodies_total": tot_b,
        "bodies_with_intra_comment": tot_c,
        "pct_with_intra_comment": round(100 * tot_c / tot_b, 2) if tot_b else 0.0,
        "per_package_pct": {
            "n_packages": len(vals),
            "mean": round(100 * sum(vals) / len(vals), 2) if vals else 0.0,
            "min": round(100 * vals[0], 2) if vals else 0.0,
            "p25": round(100 * q(0.25), 2),
            "median": round(100 * q(0.50), 2),
            "p75": round(100 * q(0.75), 2),
            "max": round(100 * vals[-1], 2) if vals else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# variant B runner (LLM comments, concurrency <= 3)
# ---------------------------------------------------------------------------

def _nas_write_lines(path, lines, tries: int = 20, wait_s: float = 30.0):
    """Write JSONL to the NAS store, riding out drvfs ENOMEM flaps.

    The WSL mount intermittently refuses new opens under heavy churn
    (OSError errno 12). Skipping a PARTIAL checkpoint only loses progress
    since the last good write; raising would kill a multi-hour API run.
    """
    for attempt in range(tries):
        try:
            with open(path, "w") as fh:
                for ex in lines:
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            return
        except OSError as e:
            if attempt == tries - 1:
                print(f"  [nas-write] giving up on {path}: {e}", flush=True)
                return
            print(f"  [nas-write] {e}; retry {attempt + 1}/{tries - 1} in {wait_s:.0f}s",
                  flush=True)
            time.sleep(wait_s)


def build_synthetic(cands: list[dict], target: int, opencode_key: str,
                    openrouter_key: str, verbose: bool = True,
                    zai_key: str = "",
                    partial_path: Path | None = None,
                    resume: list[dict] | None = None,
                    deadline_s: float = 46800.0) -> tuple[list[dict], dict]:
    """Generate + gate LLM comments for candidate blocks (concurrency 3).
    Outage-resilient: blocks whose every API try failed go back on the queue
    (max 2 rounds), and when no request has succeeded for 5+ minutes workers
    pause instead of burning candidates, so provider 429 storms are waited out."""
    from collections import deque

    out: list[dict] = list(resume or [])
    done_blocks = {normalize_block(e["region_new"]) for e in out}
    cands = [c for c in cands if normalize_block(c["block"]) not in done_blocks]
    todo = deque(range(len(cands[: int(target * 1.6) + 50])))
    attempts: dict[int, int] = {}
    lock = threading.Lock()
    gate_first_reject = {"n": 0}
    dropped = {"api": 0, "gate": 0, "validate": 0, "precheck": 0, "exhausted": 0}
    t0 = time.time()

    def work(idx):
        if len(out) >= target or time.time() - t0 > deadline_s:
            return
        if time.time() - t0 <= deadline_s:
            _outage_gate()  # circuit breaker (see docstring)
        cand = cands[idx]
        # pre-filter BEFORE spending API calls (mirrors the validator's
        # plain-reference check so post-hoc validate drops stay ~0)
        block_text = "\n".join(cand["block"])
        plain, _ = unbound_refs(block_text, prefix_bindings(cand["prefix_lines"]))
        total = identifier_count(block_text)
        if len(plain) > 4 or (total and len(plain) / total > 0.5):
            with lock:
                dropped["precheck"] += 1
            return
        comment, gen = generate_comment(cand["block"], opencode_key,
                                        openrouter_key, zai_key)
        regenerated = False
        if comment is not None and not _gate_ok(comment):
            gate_first_reject["n"] += 1
            regenerated = True
            comment2, gen2 = generate_comment(cand["block"], opencode_key,
                                              openrouter_key, zai_key)
            if comment2 is not None and _gate_ok(comment2):
                comment, gen = comment2, gen2  # regenerate once, as specified
            else:
                comment = None
        if comment is None:
            n = attempts.get(idx, 0) + 1
            attempts[idx] = n
            with lock:
                if n >= 2:
                    dropped["exhausted" if regenerated else "api"] += 1
                else:
                    todo.append(idx)  # requeue: outage or transient failure
            return
        ex = make_synthetic_example(cand, comment, gen)
        try:
            validate_example(ex)
        except AssertionError:
            with lock:
                dropped["validate"] += 1
            return
        with lock:
            if len(out) < target:
                out.append(ex)

    last_print = 0.0
    stop_heartbeat = threading.Event()

    def heartbeat():
        while not stop_heartbeat.wait(300):
            oc, orr = API_STATS["opencode"], API_STATS["openrouter"]
            outage = max(0.0, time.time() - _last_ok_ts)
            print(f"  [hb] synth={len(out)}/{target} queue={len(todo)} "
                  f"oc[ok={oc['ok']} e429={oc['err_429']}] "
                  f"or[ok={orr['ok']} e429={orr['err_429']} "
                  f"ejson={orr['err_json']}] "
                  f"no_ok_for={outage:.0f}s elapsed={time.time()-t0:.0f}s",
                  flush=True)
            if partial_path is not None:
                _nas_write_lines(partial_path, out)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    with ThreadPoolExecutor(max_workers=3) as pool:  # concurrency <= 3 total
        while todo and len(out) < target and time.time() - t0 <= deadline_s:
            batch = []
            while todo and len(batch) < 90:
                batch.append(todo.popleft())
            futures = [pool.submit(work, i) for i in batch]
            for f in futures:
                f.result()
            if verbose and time.time() - last_print > 60:
                last_print = time.time()
                oc, orr = API_STATS["opencode"], API_STATS["openrouter"]
                outage = max(0.0, time.time() - _last_ok_ts)
                print(f"  synth={len(out)}/{target} queue={len(todo)} "
                      f"oc[ok={oc['ok']} e429={oc['err_429']}] "
                      f"or[ok={orr['ok']} e429={orr['err_429']} "
                      f"epr={orr['err_provider']} ejson={orr['err_json']} "
                      f"etmo={orr['err_timeout']}] dropped={dropped} "
                      f"no_ok_for={outage:.0f}s "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            if partial_path is not None:  # crash-safe partial progress
                _nas_write_lines(partial_path, out)
    stop_heartbeat.set()
    api = {
        src: {k: (round(v, 1) if k == "lat_s" else v) for k, v in d.items()}
        for src, d in API_STATS.items()
    }
    for src, d in api.items():
        d["mean_latency_s"] = round(d.pop("lat_s") / d["ok"], 2) if d["ok"] else None
    api["gate_first_rejects"] = gate_first_reject["n"]
    api["dropped"] = dropped
    api["wall_s"] = round(time.time() - t0, 1)
    api["n_this_run"] = len(out) - len(resume or [])
    api["throughput_per_min"] = round(
        api["n_this_run"] / max(0.1, api["wall_s"]) * 60, 1)
    return out, api


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def calibrate(n_real: int = 30, n_synth: int = 5, n_pkgs: int = 25,
              seed: int = 7) -> dict:
    """Small-scale run: construct + validate examples of both variants and
    assert the no-op baseline (predict nothing) scores exactly 0."""
    all_pkgs = list_packages()
    rng = random.Random(seed)
    sample = rng.sample(all_pkgs, min(n_pkgs, len(all_pkgs)))
    real, cands, scan = scan_corpus(sample, seed=seed, target_a=n_real,
                                    n_candidates=max(30, n_synth * 6),
                                    time_budget_s=300)
    assert len(real) >= 5, f"only {len(real)} real pairs constructed"
    for ex in real:
        validate_example(ex)
        assert noop_baseline_score(ex) == 0.0
        assert exact_reward(ex["region_new"], ex["region_new"]) == 1.0
    synth, api = build_synthetic(cands, n_synth,
                                 os.environ.get("OPENCODE_API_KEY", ""),
                                 os.environ.get("OPENROUTER_API_KEY", ""),
                                 verbose=False, deadline_s=600,
                                 zai_key=os.environ.get("ZAI_API_KEY", ""))
    assert len(synth) >= 3, f"only {len(synth)} synthetic examples built"
    for ex in synth:
        validate_example(ex)
        assert noop_baseline_score(ex) == 0.0
    return {
        "packages_sampled": len(sample),
        "files": scan["files"],
        "elapsed_s": scan["elapsed_s"],
        "families": {
            "comment_to_code_real": {
                "n_constructed": len(real), "all_valid": True,
                "noop_baseline_mean": 0.0, "noop_baseline_max": 0.0,
            },
            "comment_to_code_synthetic": {
                "n_constructed": len(synth), "all_valid": True,
                "noop_baseline_mean": 0.0, "noop_baseline_max": 0.0,
                "generators": sorted({e["generator"] for e in synth}),
            },
        },
        "api": api,
        "comment_density": density_summary(scan["density"]),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def print_spot(exs: list[dict], n: int = 5, seed: int = 1):
    rng = random.Random(seed)
    for i, ex in enumerate(rng.sample(exs, min(n, len(exs))), 1):
        print(f"--- spot {ex['family']} #{i} [{ex['package']} {ex['path']}]")
        for l in ex["prefix"][-4:]:
            print(f"  P| {l}")
        print(f"  >> cursor at end of comment (region_old={ex['region_old']!r}, "
              f"cursor_idx={ex['cursor_idx']})")
        for l in ex["region_new"][:8]:
            print(f"  N| {l}")
        if len(ex["region_new"]) > 8:
            print(f"  N| ... (+{len(ex['region_new']) - 8} more lines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-total", type=int, default=900,
                    help="hard cap on packages scanned (sample + extras)")
    ap.add_argument("--target-a", type=int, default=TARGET_A)
    ap.add_argument("--target-b", type=int, default=TARGET_B)
    ap.add_argument("--time-budget", type=int, default=1500)
    ap.add_argument("--api-deadline-s", type=int, default=46800,
                    help="wall-clock cap for the LLM phase (waits out 429 "
                         "storms instead of burning candidates)")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.calibrate:
        rep = calibrate()
        print(json.dumps(rep, indent=1))
        return

    all_pkgs = list_packages()
    rng = random.Random(args.seed)
    sample = rng.sample(all_pkgs, min(args.packages, len(all_pkgs)))
    # if the ~150-package sample cannot reach the targets after dedup, keep
    # scanning fresh packages (same seed stream) instead of falling short
    extra = [p for p in all_pkgs if p not in set(sample)]
    rng.shuffle(extra)
    package_list = sample + extra[: max(0, args.max_total - len(sample))]

    n_candidates = min(int(args.target_b * 1.8) + 100, 12000)
    cache = Path(__file__).resolve().parent / ".c2c_cache.json"
    if cache.exists():  # restart support: skip the corpus rescan
        blob = json.loads(cache.read_text())
        real, cands, scan = blob["real"], blob["cands"], blob["scan"]
        scan["density"] = blob["density"]
        print(f"loaded cache: {len(real)} real pairs, {len(cands)} candidates")
    else:
        print(f"scanning ~{args.packages} packages (up to {len(package_list)} "
              f"if targets unmet): real pairs (target {args.target_a}) + "
              f"{n_candidates} comment-free candidates ...")
        real, cands, scan = scan_corpus(
            package_list, seed=args.seed, target_a=args.target_a,
            n_candidates=n_candidates, time_budget_s=args.time_budget,
            verbose=True)
        try:
            slim = {k: v for k, v in scan.items() if k != "density"}
            cache.write_text(json.dumps(dict(real=real, cands=[
                {k: v for k, v in c.items() if k != "bound"} for c in cands],
                scan=slim, density=scan["density"])))
        except OSError:
            pass
    dens = density_summary(scan["density"])

    args.out.mkdir(parents=True, exist_ok=True)
    _nas_write_lines(args.out / "comment_to_code_real.jsonl", real)

    print(f"generating synthetic comments for up to {args.target_b} blocks "
          f"(concurrency 3) ...")
    partial = args.out / "comment_to_code_synthetic.partial.jsonl"
    resume = []
    if partial.exists():  # continue a previous interrupted API phase
        for line in partial.read_text().splitlines():
            try:
                ex = json.loads(line)
                validate_example(ex)
                resume.append(ex)
            except (ValueError, AssertionError):
                pass
        print(f"resuming from partial: {len(resume)} accepted examples")
    synth, api = build_synthetic(cands, args.target_b,
                                 os.environ.get("OPENCODE_API_KEY", ""),
                                 os.environ.get("OPENROUTER_API_KEY", ""),
                                 partial_path=partial, resume=resume,
                                 deadline_s=args.api_deadline_s,
                                 zai_key=os.environ.get("ZAI_API_KEY", ""))
    _nas_write_lines(args.out / "comment_to_code_synthetic.jsonl", synth)
    (args.out / "comment_to_code_synthetic.partial.jsonl").unlink(missing_ok=True)

    # merge with existing stats.json (scenarios.py families stay untouched)
    stats_path = args.out / "stats.json"
    stats = {}
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except (ValueError, OSError):
            stats = {}
    stats.setdefault("counts", {})
    stats["counts"]["comment_to_code_real"] = len(real)
    stats["counts"]["comment_to_code_synthetic"] = len(synth)
    for fam, exs in (("comment_to_code_real", real),
                     ("comment_to_code_synthetic", synth)):
        rng2 = random.Random(1)
        samp = rng2.sample(exs, min(200, len(exs))) if exs else []
        stats[f"{fam}_noop_baseline_mean(sampled)"] = \
            round(sum(noop_baseline_score(e) for e in samp) / len(samp), 4) \
            if samp else None
        stats[f"{fam}_written"] = len(exs)
    stats["comment_to_code"] = {
        "packages_requested": args.packages,
        "packages_with_files_scanned": len(scan["density"]),
        "files": scan["files"],
        "scan_elapsed_s": scan["elapsed_s"],
        "seed": args.seed,
        "comment_density": dens,
        "api": api,
    }
    stats_path.write_text(json.dumps(stats, indent=1))

    print(json.dumps(stats["counts"], indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "counts"}, indent=1))
    print("\n===== spot examples =====")
    print_spot(real)
    print()
    print_spot(synth)


if __name__ == "__main__":
    main()
