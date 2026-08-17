#!/usr/bin/env python3
"""Programmatic edit-scenario training examples with exact ground truth.

Three scenario families built from the normalized CRAN corpus
(/mnt/h/sepalith/normalized/<pkg>/<ver>/<pkg>/R/*.R) via tree-sitter-r:

  rename_propagation  an identifier (function argument / local variable /
                       column-name string literal) occurring >= 3 times in one
                       function body; the event renames ONE occurrence, the
                       target edit renames the NEXT occurrence.
  pipe_rewrite         a simple direct magrittr chain `lhs %>% rhs(args)`
                       (no "." placeholder, RHS a parenthesized call); the
                       event rewrites one %>% to the native |> , the target
                       rewrites the next %>% on another line.
  na_rm_propagation    inside a dplyr summarise/mutate-style call, 2+
                       mean(/sd(/var( calls lacking na.rm; the event adds
                       `na.rm = TRUE` to one, the target adds it to the next.

Example JSON shape (all regions are single lines, so edits are exactly
verifiable; cursor_idx is a character offset into "\n".join(region_old)):

  {"family": ..., "package": ..., "path": ..., "prefix": [lines],
   "region_old": [lines], "region_new": [lines], "cursor_idx": int,
   "event_diff": "User edited \"path\":\n\n```diff\n@@ .. @@\n-old\n+new\n```",
   "note": ...}

Also provides exact_reward(pred, gold) and calibrate() which validates every
constructed example (region_old->region_new must be EXACTLY the intended
transformation) and scores a no-op baseline (predict region_old unchanged,
reward ~0).

Usage:
  uv run python experiments/synthetic/scenarios.py --calibrate
  uv run python experiments/synthetic/scenarios.py --packages 200
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import time
from bisect import bisect_right
from pathlib import Path

import tree_sitter_r
from tree_sitter import Language, Parser

ROOT = Path("/mnt/h/sepalith/normalized")
OUT_DIR = Path("/mnt/h/sepalith/datasets/scenarios_v1")
STATS_PATH = OUT_DIR / "stats.json"
FAMILIES = ("rename_propagation", "pipe_rewrite", "na_rm_propagation")
MAX_PER_FAMILY = 5000
MAX_FILE_BYTES = 300_000
MAX_FILES_PER_PKG = 40

parser = Parser(Language(tree_sitter_r.language()))

# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

RESERVED = {
    "TRUE", "FALSE", "NULL", "NA", "NA_integer_", "NA_real_", "NA_character_",
    "NA_complex_", "NaN", "Inf", "T", "F", "if", "else", "for", "while",
    "repeat", "function", "in", "next", "break", "library", "require",
    "return", "self", "super",
}
IDENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._]*$")


def node_text(src: bytes, n) -> bytes:
    return src[n.start_byte:n.end_byte]


def traverse(n):
    stack = [n]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def strip_strings(b: bytes) -> bytes:
    """Blank out string-literal contents so regexes never match inside them."""
    out = bytearray(b)
    i, q = 0, None
    while i < len(out):
        c = out[i]
        if q is None and c in (0x22, 0x27):  # " or '
            q = c
        elif q is not None:
            if c == 0x5C:  # backslash escape
                out[i] = 0x78
                if i + 1 < len(out):
                    out[i + 1] = 0x78
                i += 2
                continue
            if c == q:
                q = None
            elif c != 0x0A:  # keep newlines (multi-line strings)
                out[i] = 0x78
        i += 1
    return bytes(out)


def derive_new_name(old: str) -> str | None:
    """Deterministic rename: '_tmp' suffix stripped, '.' -> '_', else append 2."""
    if old in RESERVED or not IDENT_RE.match(old or ""):
        return None
    if old.endswith("_tmp"):
        new = old[: -len("_tmp")]
    elif "." in old:
        new = old.replace(".", "_")
    else:
        new = old + "2"
    if new == old or not IDENT_RE.match(new) or new in RESERVED:
        return None
    return new


def event_diff_for(path: str, lineno: int, old_line: str, new_line: str) -> str:
    return (f'User edited "{path}":\n\n'
            f"```diff\n@@ -{lineno} +{lineno} @@\n-{old_line}\n+{new_line}\n```")


EVENT_DIFF_RE = re.compile(
    r'^User edited "(?P<path>.+)":\n\n```diff\n'
    r"@@ [^@]*@@\n-(?P<old>.*)\n\+(?P<new>.*)\n```$", re.DOTALL)


def exact_reward(pred_lines, region_new_lines) -> float:
    """1.0 on exact match after rstrip-normalisation, else line-F1 (difflib)."""
    p = [l.rstrip() for l in (pred_lines or [])]
    g = [l.rstrip() for l in (region_new_lines or [])]
    while p and p[-1] == "":
        p.pop()
    while g and g[-1] == "":
        g.pop()
    if p == g:
        return 1.0
    if not p or not g:
        return 0.0
    sm = difflib.SequenceMatcher(a=p, b=g, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    prec = matched / len(p)
    rec = matched / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# corpus loading (each file is read exactly once, parsed once, shared)
# ---------------------------------------------------------------------------

class Bundle:
    __slots__ = ("package", "rel", "src", "tree", "lines", "starts",
                 "id_names", "str_contents")

    def __init__(self, package: str, rel: str, src: bytes):
        self.package = package
        self.rel = rel
        self.src = src
        self.tree = parser.parse(src)
        self.lines = src.split(b"\n")
        starts, off = [], 0
        for ln in self.lines:
            starts.append(off)
            off += len(ln) + 1
        self.starts = starts
        self.id_names: set[str] = set()
        self.str_contents: set[str] = set()
        for n in traverse(self.tree.root_node):
            if n.type == "identifier":
                self.id_names.add(node_text(src, n).decode("utf-8", "replace"))
            elif n.type == "string":
                self.str_contents.add(
                    node_text(src, n).decode("utf-8", "replace"))

    def rowcol(self, byte: int) -> tuple[int, int]:
        row = bisect_right(self.starts, byte) - 1
        return row, byte - self.starts[row]

    def line_bytes(self, row: int) -> bytes:
        return self.lines[row]

    def line_str(self, row: int) -> str:
        return self.lines[row].decode("utf-8", "replace").rstrip("\r")

    def nlines(self) -> int:
        return len(self.lines)


def list_packages(root: Path = ROOT) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


TIDY_CACHE = Path(__file__).resolve().parent / ".tidy_pkgs_cache.json"


def tidy_packages(cache: Path = TIDY_CACHE, timeout: int = 600) -> list[str]:
    """Packages that mention dplyr in DESCRIPTION (cached pre-scan).

    na_rm_propagation (and to a lesser degree pipe_rewrite) need
    tidyverse-style code, which is rare in a uniform random package sample;
    scanning DESCRIPTION files (~1.5 min once, SMB) is far cheaper than
    grepping all sources (~10 min).
    """
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except (ValueError, OSError):
            pass
    t0 = time.time()
    pkgs = []
    for p in list_packages():
        if time.time() - t0 > timeout:
            break
        try:
            ver = next((ROOT / p).iterdir())
            text = (ver / p / "DESCRIPTION").read_text(errors="replace")
        except (StopIteration, OSError, NotADirectoryError):
            continue
        if re.search(r"\bdplyr\b|\btidyverse\b", text):
            pkgs.append(p)
    try:
        cache.write_text(json.dumps(pkgs))
    except OSError:
        pass
    return pkgs


def iter_bundles(package_names, rng: random.Random, max_files=MAX_FILES_PER_PKG):
    """Yield one Bundle per source file (each file read exactly once)."""
    for pkg in package_names:
        try:
            ver_dir = next((ROOT / pkg).iterdir())
            rdir = ver_dir / pkg / "R"
        except (StopIteration, FileNotFoundError, NotADirectoryError):
            continue
        if not rdir.is_dir():
            continue
        try:
            files = sorted(list(rdir.glob("*.R")) + list(rdir.glob("*.r")))
        except OSError:
            continue
        if len(files) > max_files:
            files = rng.sample(files, max_files)
        for f in files:
            try:
                src = f.read_bytes()
            except OSError:
                continue
            if not src or len(src) > MAX_FILE_BYTES:
                continue
            yield Bundle(pkg, f"R/{f.name}", src)


def make_example(family: str, b: Bundle, region_row: int, region_new: str,
                 cursor_col_bytes: int, event_row: int, event_old: str,
                 event_new: str, note: str) -> dict:
    old_line = b.line_str(region_row)
    prefix = [b.line_str(r) for r in range(max(0, region_row - 10), region_row)]
    lb = b.line_bytes(region_row)
    cursor_idx = len(lb[:cursor_col_bytes].decode("utf-8", "replace"))
    return {
        "family": family,
        "package": b.package,
        "path": b.rel,
        "prefix": prefix,
        "region_old": [old_line],
        "region_new": [region_new],
        "cursor_idx": cursor_idx,
        "event_diff": event_diff_for(b.rel, event_row + 1, event_old, event_new),
        "note": note,
    }


# ---------------------------------------------------------------------------
# family 1: rename_propagation
# ---------------------------------------------------------------------------

def callee_name(src: bytes, call_node) -> str | None:
    """Last component of a call's function expression (foo, ns::foo, x$foo)."""
    if not call_node.children:
        return None
    head = call_node.children[0]
    last = head
    while last.children and last.type in ("namespace_operator", "extract_operator"):
        named = [c for c in last.children if c.is_named]
        if not named:
            break
        last = named[-1]
    return node_text(src, last).decode("utf-8", "replace") if last.type == "identifier" else None


def extract_rename(b: Bundle, rng: random.Random, cap: int = 6) -> list[dict]:
    out = []
    src = b.src
    fn_nodes = [n for n in traverse(b.tree.root_node)
                if n.type == "function_definition"]

    for fn in fn_nodes:
        body = next((c for c in fn.children if c.type == "braced_expression"), None)
        params = next((c for c in fn.children if c.type == "parameters"), None)
        if body is None:
            continue
        fn_name = ""
        parent = fn.parent
        if parent is not None and parent.type == "binary_operator":
            fn_name = node_text(src, parent.children[0]).decode("utf-8", "replace")

        # all identifier + string occurrences inside this body (skip callers)
        id_occ: dict[str, list[tuple[int, int]]] = {}
        callers: set[str] = set()
        for n in traverse(body):
            if n.type == "call":
                cn = callee_name(src, n)
                if cn:
                    callers.add(cn)
            elif n.type == "identifier":
                name = node_text(src, n).decode("utf-8", "replace")
                if parent_is_caller(n):
                    callers.add(name)
                else:
                    id_occ.setdefault(name, []).append(
                        (n.start_byte, n.end_byte))
        str_occ: dict[str, list[tuple[int, int]]] = {}
        for n in traverse(body):
            if n.type == "string":
                raw = node_text(src, n).decode("utf-8", "replace")
                inner = raw[1:-1] if len(raw) >= 2 else ""
                if (raw[:1] in ('"', "'") and IDENT_RE.match(inner or "")
                        and inner not in RESERVED):
                    str_occ.setdefault(raw, []).append(
                        (n.start_byte, n.end_byte))

        # candidate declared names: parameters + simple LHS assignments
        declared: set[str] = set()
        if params is not None:
            for p in traverse(params):
                if p.type == "identifier":
                    declared.add(node_text(src, p).decode("utf-8", "replace"))
        for n in traverse(body):
            if n.type == "binary_operator" and n.children and \
                    n.children[0].type == "identifier":
                declared.add(node_text(src, n.children[0]).decode("utf-8", "replace"))

        for kind, occ_map in (("variable", id_occ), ("string", str_occ)):
            cands = []
            for tok, occs in occ_map.items():
                if kind == "variable":
                    inner = tok
                    if (len(inner) < 3 or inner in RESERVED or inner.startswith(".")
                            or not IDENT_RE.match(inner) or inner in callers
                            or inner not in declared):
                        continue
                else:
                    inner = tok[1:-1]
                    if len(inner) < 3:
                        continue
                if len(occs) < 3:
                    continue
                new_inner = derive_new_name(inner)
                if new_inner is None:
                    continue
                if kind == "string":
                    q = tok[0]  # preserve the literal's quote style
                    new_tok = f"{q}{new_inner}{q}"
                else:
                    new_tok = new_inner
                if new_tok == tok:
                    continue
                if new_inner in b.id_names or new_inner in b.str_contents:
                    continue  # collision anywhere in file -> ambiguous, skip
                cands.append((tok, new_tok, occs))
            if not cands:
                continue
            rng.shuffle(cands)
            emitted = 0
            for tok, new_tok, occs in cands:
                if emitted >= cap:
                    break
                occs = sorted(occs)
                # event = first occurrence; target = one of the later ones
                ev_start, _ = occs[0]
                ev_row, ev_col = b.rowcol(ev_start)
                targets = occs[1:]
                if not targets:
                    continue
                # usually the immediately-next occurrence, sometimes a later one
                t_start, _t_end = targets[0] if rng.random() < 0.7 \
                    else rng.choice(targets)
                t_row, t_col = b.rowcol(t_start)
                if t_row == ev_row or t_row >= b.nlines():
                    continue  # keep event/target on distinct lines
                ev_line = b.line_bytes(ev_row)
                ev_new = (ev_line[:ev_col] + new_tok.encode()
                          + ev_line[ev_col + len(tok.encode()):])
                ev_new_s = ev_new.decode("utf-8", "replace").rstrip("\r")
                ev_old_s = b.line_str(ev_row)
                if ev_new_s == ev_old_s:
                    continue
                lb = b.line_bytes(t_row)
                t_new = (lb[:t_col] + new_tok.encode()
                         + lb[t_col + len(tok.encode()):])
                t_new_s = t_new.decode("utf-8", "replace").rstrip("\r")
                t_old_s = b.line_str(t_row)
                if t_new_s == t_old_s:
                    continue
                note = (f"rename {tok} -> {new_tok} ({kind}, occurrence "
                        f"after the edited one; fn {fn_name or '<anon>'})")
                out.append(make_example(
                    "rename_propagation", b, t_row, t_new_s, t_col,
                    ev_row, ev_old_s, ev_new_s, note))
                emitted += 1
    return out


def parent_is_caller(n) -> bool:
    p = n.parent
    return (p is not None and p.type == "call"
            and p.children and p.children[0] is n)


# ---------------------------------------------------------------------------
# family 2: pipe_rewrite
# ---------------------------------------------------------------------------

DOT_TOKEN_RE = re.compile(rb"(?<![\w.])\.(?![\w.])")
OTHER_SPECIAL = re.compile(rb"%<>%|%T>%|%\$%|\|%>")


def _pipe_ok(b: Bundle, op_node, lhs, rhs) -> bool:
    src = b.src
    if rhs.type != "call":
        return False  # bare fn / { } / lambda / ( ...) can't map 1:1 to |>
    lhs_txt = strip_strings(node_text(src, lhs))
    rhs_txt = strip_strings(node_text(src, rhs))
    if DOT_TOKEN_RE.search(lhs_txt) or DOT_TOKEN_RE.search(rhs_txt):
        return False  # magrittr "." placeholder / lambda -> ambiguous
    row, _ = b.rowcol(op_node.start_byte)
    line = b.line_bytes(row)
    if OTHER_SPECIAL.search(strip_strings(line)):
        return False
    return True


def extract_pipe(b: Bundle, rng: random.Random, cap: int = 4) -> list[dict]:
    src = b.src
    occs = []
    for n in traverse(b.tree.root_node):
        if n.type != "binary_operator":
            continue
        kids = n.children
        if len(kids) < 3 or kids[1].type != "special":
            continue
        if node_text(src, kids[1]) != b"%>%":
            continue
        if _pipe_ok(b, kids[1], kids[0], kids[2]):
            row, col = b.rowcol(kids[1].start_byte)
            occs.append((kids[1].start_byte, row, col))
    out = []
    occs.sort()
    for (ev_b, ev_row, ev_col), (t_b, t_row, t_col) in zip(occs, occs[1:]):
        if ev_row == t_row:
            continue  # event and target must be distinct lines
        ev_line = b.line_bytes(ev_row)
        ev_new = (ev_line[:ev_col] + b"|>"
                  + ev_line[ev_col + 3:]).decode("utf-8", "replace").rstrip("\r")
        ev_old = b.line_str(ev_row)
        t_line = b.line_bytes(t_row)
        t_new = (t_line[:t_col] + b"|>"
                 + t_line[t_col + 3:]).decode("utf-8", "replace").rstrip("\r")
        t_old = b.line_str(t_row)
        if t_new == t_old or ev_new == ev_old:
            continue
        note = "rewrite magrittr pipe %>% to native pipe |> (next occurrence after the edited one)"
        out.append(make_example("pipe_rewrite", b, t_row, t_new, t_col,
                                ev_row, ev_old, ev_new, note))
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# family 3: na_rm_propagation
# ---------------------------------------------------------------------------

STAT_FNS = ("mean", "sd", "var")
DPLYR_VERBS = {"summarise", "summarize", "mutate", "transmute", "reframe",
               "summarise_at", "summarize_at", "mutate_at",
               "summarise_if", "summarize_if", "mutate_if",
               "summarise_all", "summarize_all", "mutate_all"}
NA_RM_EQ = re.compile(rb"(?<![\w.])na\.rm\s*=")
TIDY_MARKER = re.compile(rb"dplyr|tidyverse|summaris|summariz|%>%")


def extract_na_rm(b: Bundle, rng: random.Random, cap: int = 4) -> list[dict]:
    src = b.src
    if not TIDY_MARKER.search(src):
        return []
    if re.search(rb"(?<![\w.])(?:mean|sd|var)\s*<-", src):
        return []  # stat fn shadowed locally -> adding na.rm could be wrong

    groups: dict[int, list] = {}
    all_calls: list[tuple] = []
    for n in traverse(b.tree.root_node):
        if n.type != "call":
            continue
        fn = callee_name(src, n)
        if fn not in STAT_FNS:
            continue
        call_txt = strip_strings(node_text(src, n))
        if NA_RM_EQ.search(call_txt):
            continue  # already has na.rm
        if n.start_point[0] != n.end_point[0]:
            continue  # single-line calls only (exact line-region GT)
        body = call_txt[call_txt.find(b"(") + 1:-1].strip()
        if not body or body.endswith(b","):
            continue  # zero-arg call or trailing comma -> ambiguous insert
        row, col_end = b.rowcol(n.end_byte - 1)  # the closing ')'
        lb = b.line_bytes(row)
        ins = col_end
        while ins > 0 and lb[ins - 1:ins] in (b" ", b"\t"):
            ins -= 1
        rec = (n.start_byte, row, ins, fn)
        all_calls.append(rec)
        # grouping by enclosing dplyr summarise/mutate-style call (primary)
        anc, verb = n.parent, None
        while anc is not None:
            if anc.type == "call":
                cn = callee_name(src, anc)
                if cn in DPLYR_VERBS:
                    verb = anc
                    break
            anc = anc.parent
        if verb is not None:
            groups.setdefault(verb.id, []).append(rec)

    out = []

    def emit(ev, t):
        (ev_b, ev_row, ev_ins, ev_fn) = ev
        (t_b, t_row, t_ins, t_fn) = t
        if ev_row == t_row:
            return False
        ev_chars = len(b.line_bytes(ev_row)[:ev_ins].decode("utf-8", "replace"))
        ev_old = b.line_str(ev_row)
        ev_new = ev_old[:ev_chars] + ", na.rm = TRUE" + ev_old[ev_chars:]
        t_chars = len(b.line_bytes(t_row)[:t_ins].decode("utf-8", "replace"))
        t_old = b.line_str(t_row)
        t_new = t_old[:t_chars] + ", na.rm = TRUE" + t_old[t_chars:]
        if t_new == t_old or ev_new == ev_old:
            return False
        note = f"add na.rm = TRUE to {t_fn}( lacking it (next call after the edited one)"
        out.append(make_example("na_rm_propagation", b, t_row, t_new, t_ins,
                                ev_row, ev_old, ev_new, note))
        return True

    # primary: two+ stat calls inside the SAME summarise/mutate call
    pairs = []
    for _, calls in sorted(groups.items()):
        calls.sort()
        pairs.extend(zip(calls, calls[1:]))
    # fallback: consecutive qualifying calls anywhere in the same tidyverse file
    all_calls.sort()
    pairs.extend(zip(all_calls, all_calls[1:]))
    seen_pairs = set()
    for ev, t in pairs:
        if len(out) >= cap:
            break
        key = (ev[0], t[0])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        emit(ev, t)
    return out


# ---------------------------------------------------------------------------
# validation + calibration
# ---------------------------------------------------------------------------

_BOUNDARY = re.compile(r"[\w.]")
_TOKEN_PATS = (r'"[^"\n]*"', r"'[^'\n]*'", r"[A-Za-z][A-Za-z0-9._]*")


def _single_token_edit(old: str, new: str, pat: str):
    """If `new` == `old` with exactly ONE occurrence of pattern `pat` replaced
    (token-boundary safe), return (old_tok, new_tok); else None."""
    for m in re.finditer(pat, old):
        tail = old[m.end():]
        if not new.startswith(old[:m.start()]) or not new.endswith(tail):
            continue
        n = new[m.start(): len(new) - len(tail)]
        if not n or len(new) - len(tail) < m.start():
            continue
        b = old[m.start() - 1] if m.start() else ""
        a = old[m.end()] if m.end() < len(old) else ""
        na = new[m.start() + len(n)] if m.start() + len(n) < len(new) else ""
        if (b and _BOUNDARY.match(b)) or (a and _BOUNDARY.match(a)) \
                or (na and _BOUNDARY.match(na)):
            continue  # would split a longer token -> not a whole-token edit
        if new[:m.start()] + n + tail == new:
            return m.group(0), n
    return None


def _single_insert_before_close(old: str, new: str, ins: str) -> bool:
    """True if `new` == `old` with exactly `ins` inserted once at a position
    whose remainder (after optional spaces) starts with ')'. This guarantees a
    trailing-argument insertion like `, na.rm = TRUE`."""
    if new.count(ins) != old.count(ins) + 1:
        return False
    for i in range(len(old) + 1):
        if old[:i] + ins + old[i:] == new \
                and old[i:].lstrip(" \t").startswith(")"):
            return True
    return False


def validate_example(ex: dict) -> None:
    """Assert the example is internally consistent and region_old->region_new
    is EXACTLY the intended transformation (family-specific)."""
    for k in ("family", "package", "path", "prefix", "region_old",
              "region_new", "cursor_idx", "event_diff", "note"):
        assert k in ex and ex[k] is not None, f"missing field {k}"
    assert ex["family"] in FAMILIES
    for f in ("prefix", "region_old", "region_new"):
        assert isinstance(ex[f], list) and ex[f], f"{f} must be non-empty list"
        assert all(isinstance(l, str) and "\n" not in l for l in ex[f]), \
            f"{f} must be single-line strings"
    assert ex["region_old"] != ex["region_new"], "GT must change the region"
    joined = "\n".join(ex["region_old"])
    assert isinstance(ex["cursor_idx"], int) and 0 <= ex["cursor_idx"] <= len(joined)
    m = EVENT_DIFF_RE.match(ex["event_diff"])
    assert m, "event_diff malformed"
    assert m.group("old") != m.group("new"), "event must be a real edit"
    assert m.group("path") == ex["path"]

    old, new = ex["region_old"][0], ex["region_new"][0]
    fam = ex["family"]
    _assert_transformation(fam, old, new)
    _assert_transformation(fam, m.group("old"), m.group("new"))


def _assert_transformation(fam: str, old: str, new: str) -> None:
    """region/event old->new lines must be EXACTLY the family's edit."""
    assert old != new, (fam, old)
    if fam == "pipe_rewrite":
        edit = _single_token_edit(old, new, r"%>%")
        assert edit == ("%>%", "|>"), (old, new, edit)
    elif fam == "na_rm_propagation":
        assert _single_insert_before_close(old, new, ", na.rm = TRUE"), (old, new)
    elif fam == "rename_propagation":
        # try every token pattern; any one producing a valid identifier(or
        # quoted identifier) rename pair is sufficient (string spans can
        # align misleadingly when the real edit is nearby)
        cands = [_single_token_edit(old, new, p) for p in _TOKEN_PATS]
        assert any(c and _valid_rename_pair(*c) for c in cands), (old, new, cands)


def _valid_rename_pair(o: str, n: str) -> bool:
    if o == n:
        return False
    if o[:1] in ('"', "'"):
        return (n[:1] == o[:1] and n[-1:] == o[-1] and len(o) >= 3
                and bool(IDENT_RE.match(o[1:-1] or ""))
                and bool(IDENT_RE.match(n[1:-1] or ""))
                and o[1:-1] not in RESERVED and n[1:-1] not in RESERVED)
    return (bool(IDENT_RE.match(o or "")) and bool(IDENT_RE.match(n or ""))
            and o not in RESERVED and n not in RESERVED)


def noop_baseline_score(ex: dict) -> float:
    """Score of the 'do nothing' policy: predict region_old unchanged."""
    return exact_reward(ex["region_old"], ex["region_new"])


def build_examples(package_names, seed=13, per_family_cap=MAX_PER_FAMILY,
                   time_budget_s=780, verbose=False):
    rng = random.Random(seed)
    buckets = {f: [] for f in FAMILIES}
    t0 = time.time()
    files = 0
    seen_pkgs = set()
    for b in iter_bundles(package_names, rng):
        files += 1
        seen_pkgs.add(b.package)
        for fam, fn in (("rename_propagation", extract_rename),
                        ("pipe_rewrite", extract_pipe),
                        ("na_rm_propagation", extract_na_rm)):
            if len(buckets[fam]) >= per_family_cap:
                continue
            for ex in fn(b, rng):
                validate_example(ex)
                buckets[fam].append(ex)
                if len(buckets[fam]) >= per_family_cap:
                    break
        if files % 200 == 0 and verbose:
            done = all(len(v) >= per_family_cap for v in buckets.values())
            print(f"  files={files} elapsed={time.time()-t0:.0f}s "
                  + " ".join(f"{k}={len(v)}" for k, v in buckets.items()))
        if time.time() - t0 > time_budget_s:
            break
        if all(len(v) >= per_family_cap for v in buckets.values()):
            break
    stats = dict(files=files, packages_processed=len(seen_pkgs),
                 elapsed_s=round(time.time() - t0, 1),
                 counts={k: len(v) for k, v in buckets.items()})
    return buckets, stats


def calibrate(n_per_family: int = 30, n_pkgs: int = 40, seed=7) -> dict:
    """Small-scale run: construct examples, validate all (assert), and score
    the no-op random baseline (must be ~0 since every region line changes).
    Samples preferentially from dplyr-using packages so na_rm_propagation
    yields examples."""
    pool = tidy_packages() or list_packages()
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n_pkgs, len(pool)))
    buckets, stats = build_examples(sample, seed=seed,
                                    per_family_cap=n_per_family,
                                    time_budget_s=240)
    report = {"packages_sampled": len(sample), "files": stats["files"],
              "elapsed_s": stats["elapsed_s"], "families": {}}
    for fam, exs in buckets.items():
        assert len(exs) >= 5, f"{fam}: only {len(exs)} examples constructed"
        for ex in exs:
            validate_example(ex)  # every example must pass exactness checks
            r = noop_baseline_score(ex)
            assert r <= 0.5, f"{fam}: no-op baseline scored {r}"
            assert exact_reward(ex["region_new"], ex["region_new"]) == 1.0
        scores = [noop_baseline_score(e) for e in exs]
        report["families"][fam] = {
            "n_constructed": len(exs),
            "noop_baseline_mean": round(sum(scores) / len(scores), 4),
            "noop_baseline_max": round(max(scores), 4),
            "all_valid": True,
        }
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--n-per-family", type=int, default=30)
    ap.add_argument("--time-budget", type=int, default=780)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.calibrate:
        rep = calibrate(n_per_family=args.n_per_family)
        print(json.dumps(rep, indent=1))
        return

    all_pkgs = list_packages()
    rng = random.Random(args.seed)
    n = min(args.packages, len(all_pkgs))
    sample = rng.sample(all_pkgs, n)
    # dplyr-using packages go FIRST (na_rm_propagation / pipe_rewrite need
    # tidyverse-style code, which is scarce in a uniform random sample);
    # the random sample fills breadth for rename_propagation. Families cap
    # independently, so once a family is full its packages just feed the rest.
    tidy = tidy_packages()
    tidy_first = [p for p in tidy if p in set(all_pkgs)]
    rng.shuffle(tidy_first)
    rest = [p for p in sample if p not in set(tidy)]
    package_list = tidy_first + rest
    print(f"building from {len(tidy_first)} tidy + {len(rest)} random "
          f"packages (seed {args.seed}) ...")
    buckets, stats = build_examples(
        package_list, seed=args.seed, time_budget_s=args.time_budget,
        verbose=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for fam, exs in buckets.items():
        p = args.out / f"{fam}.jsonl"
        with p.open("w") as fh:
            for ex in exs:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    # final validation pass over everything written + baseline sampling
    rng2 = random.Random(1)
    for fam, exs in buckets.items():
        for ex in exs:
            validate_example(ex)
        samp = rng2.sample(exs, min(200, len(exs))) if exs else []
        base = round(sum(noop_baseline_score(e) for e in samp) / len(samp), 4) \
            if samp else None
        stats[f"{fam}_noop_baseline_mean(sampled)"] = base
        stats[f"{fam}_written"] = len(exs)
    stats["packages_requested"] = n
    stats["seed"] = args.seed
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
