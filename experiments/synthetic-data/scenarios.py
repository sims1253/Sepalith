#!/usr/bin/env python3
"""Programmatic edit-scenario training examples with exact ground truth.

Five scenario families. The first three are built from the normalized CRAN
corpus (/mnt/h/sepalith/normalized/<pkg>/<ver>/<pkg>/R/*.R) via tree-sitter-r:

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

Two later families (see the "new families" section below):

  format_propagation  diff RAW package tarball members
                      (/mnt/h/sepalith/tarballs/<pkg>_<ver>.tar.gz, python
                      tarfile, only the needed members) against the
                      AIR/JARL-normalized trees: the raw->normalized diff is
                      exactly what `air format` changes. For files with >= 2
                      clean difflib hunks, the event shows ONE hunk
                      reformatted (raw -> formatted) and the target edit
                      reformats the NEXT hunk (region_old = raw hunk lines,
                      region_new = the corresponding formatted lines taken
                      from the normalized file; splice-verified).
  doc_sync            tree-sitter-r over normalized files: for functions
                      with rich roxygen (>= 1 @param), the event appends one
                      benign argument to the signature (', verbose = FALSE'
                      style, chosen deterministically from verbose / call /
                      env, skipped if already present); the target edit
                      documents it by inserting the matching
                      "#' @param <arg> <desc>" line before the @return/@export
                      tag in the roxygen block.

doc_sync DIAGNOSIS (2026-08-19, from results_scenarios_sft_v3_minicpm5
+ intent_suite_v1: 40% validator pass, "duplicates the typed line, never
adds the @param"). All three suspected causes are confirmed:

  * TOO FEW / TOO NARROW EXAMPLES - 739 rows from 88 packages containing
    only TWO distinct inserted lines (722x "@param verbose Show progress
    messages while the function runs.", 17x "@param call ..."). The model
    never sees enough variation to learn "an argument appeared in the
    signature -> emit its @param tag NOW", it only sees one memorised
    string.
  * TARGET SHAPE MISMATCH - the GT inserts the new @param immediately
    before the @return/@export anchor, i.e. AFTER the blank "#'" separator
    line in 318/739 rows. Idiomatic roxygen (and the corpus prior the
    model already carries) puts @param tags together BEFORE the blank, so
    the model's natural answer fails exact match; conversely the
    fail_kind="shape" rows show it re-emitting region_old unchanged -
    the pure-insertion target is rare in the mixture (the sibling
    families all teach "re-emit the region with a small in-line edit").
  * ASSERTION DIFFICULTY - the description string is not derivable from
    anything visible (the function body sits below the roxygen block and
    is absent from the prompt), so the exact-match gate is a pure memory
    test of DOC_DESCS; the observed paraphrases ("(logical) print
    progress messages while the function runs", missing trailing period)
    copy the surrounding block's style instead.

Remedy (doc_sync v2, below): two additional constructions that keep the
exactness gate but make the target DERIVABLE and the insertion the FIRST
thing the model emits - variant="missing_param" (a real signature param
with no @param yet; descriptions come from a fixed name grammar +
punctuation/capitalisation copied from the block's own @param tags) and
variant="version_pair" (REAL upstream doc updates mined from adjacent
versions of the same package in the git mirrors; the normalized corpus
holds exactly one version per package, so the pairs come from
/mnt/h/sepalith/git).

Example JSON shape (the first three families use single-line regions, so
edits are exactly verifiable; the new families may carry multi-line
regions/events with the same keys; cursor_idx is a character offset into
"\n".join(region_old)):

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
  uv run python experiments/synthetic/scenarios.py --calibrate-new
  uv run python experiments/synthetic/scenarios.py --new-only --new-packages 150
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import tarfile
import time
from bisect import bisect_right
from pathlib import Path

import tree_sitter_r
from tree_sitter import Language, Parser

ROOT = Path("/mnt/h/sepalith/normalized")
TAR_DIR = Path("/mnt/h/sepalith/tarballs")
OUT_DIR = Path("/mnt/h/sepalith/datasets/scenarios_v1")
STATS_PATH = OUT_DIR / "stats.json"
FAMILIES = ("rename_propagation", "pipe_rewrite", "na_rm_propagation")
NEW_FAMILIES = ("format_propagation", "doc_sync")
ALL_FAMILIES = FAMILIES + NEW_FAMILIES
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
    import os
    try:  # os.listdir: no per-entry stat (drvfs metadata reads are the
        return sorted(p for p in os.listdir(root))  # bottleneck on the NAS)
    except OSError:
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
    assert ex["family"] in ALL_FAMILIES
    for f in ("prefix", "region_old", "region_new"):
        assert isinstance(ex[f], list) and ex[f], f"{f} must be non-empty list"
        assert all(isinstance(l, str) and "\n" not in l for l in ex[f]), \
            f"{f} must be single-line strings"
    assert ex["region_old"] != ex["region_new"], "GT must change the region"
    joined = "\n".join(ex["region_old"])
    assert isinstance(ex["cursor_idx"], int) and 0 <= ex["cursor_idx"] <= len(joined)
    if ex["family"] in NEW_FAMILIES:
        _validate_new_family(ex)
        return
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
# NEW family 4: format_propagation (raw tarball vs air-normalized trees)
# ---------------------------------------------------------------------------

TAR_MAX_BYTES = 50_000_000     # skip monster tarballs (headers-only pkgs etc.)
FORMAT_HUNK_MAX_RAW = 12       # region size caps, in the spirit of the
FORMAT_HUNK_MAX_NEW = 16       # single-line/small-region convention above
FORMAT_LINE_MAX = 300
FORMAT_EX_PER_FILE = 3


def _group_format_hunks(raw_lines, norm_lines) -> list[tuple[int, int, int, int]]:
    """difflib opcodes grouped into logical hunks: contiguous non-equal
    opcodes merge; hunks are separated by >= 1 equal line by construction."""
    sm = difflib.SequenceMatcher(a=raw_lines, b=norm_lines, autojunk=False)
    groups, cur = [], None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            cur = None
            continue
        if cur is None:
            cur = [i1, i2, j1, j2]
            groups.append(cur)
        else:
            cur[1], cur[3] = i2, j2
    return [tuple(g) for g in groups]


def _fmt_canonical(lines) -> str:
    """Whitespace-free, brace-free canonical form: reformatting only ever
    changes whitespace/line-wrapping (air format also adds {} around bare
    function bodies), so canonical forms must be equal for a genuine hunk
    pair. Catches difflib alignments that pair unrelated code."""
    return re.sub(r"\s+", "", "\n".join(lines)).replace("{", "").replace("}", "")


def _fmt_only_edit(old_lines, new_lines) -> bool:
    if not old_lines or not new_lines or old_lines == new_lines:
        return False
    return _fmt_canonical(old_lines) == _fmt_canonical(new_lines)


def format_pairs_from_lines(package: str, rel: str, raw_lines, norm_lines,
                            cap: int = FORMAT_EX_PER_FILE) -> list[dict]:
    """Build format_propagation examples from one raw/normalized line pair.

    Acceptance per hunk: replace-style (both sides non-empty), within size
    caps, shares NO line (rstripped) between raw and formatted sides (keeps
    the no-op baseline exactly 0), and is a whitespace/brace-only reformat
    (_fmt_only_edit). The event shows hunk k reformatted; the region is
    hunk k+1 raw -> formatted. Splice invariant: applying every grouped hunk
    to raw must rebuild the normalized file exactly."""
    hunks = _group_format_hunks(raw_lines, norm_lines)
    if len(hunks) < 2:
        return []
    rebuilt, pos = [], 0
    for i1, i2, j1, j2 in hunks:
        rebuilt.extend(raw_lines[pos:i1])
        rebuilt.extend(norm_lines[j1:j2])
        pos = i2
    rebuilt.extend(raw_lines[pos:])
    if rebuilt != norm_lines:
        return []  # alignment failed the splice check -> ambiguous, skip file
    ok = []
    for i1, i2, j1, j2 in hunks:
        if i1 >= i2 or j1 >= j2:
            continue  # insert/delete-only: no raw lines to show in region_old
        old, new = raw_lines[i1:i2], norm_lines[j1:j2]
        if len(old) > FORMAT_HUNK_MAX_RAW or len(new) > FORMAT_HUNK_MAX_NEW:
            continue
        if any(len(l) > FORMAT_LINE_MAX for l in old + new):
            continue
        if set(l.rstrip() for l in old) & set(l.rstrip() for l in new):
            continue  # a shared line would let the no-op policy score > 0
        if not _fmt_only_edit(old, new):
            continue  # not a pure reformat -> suspicious alignment, skip
        ok.append((i1, i2, j1, j2))
    out = []
    for (ei1, ei2, ej1, ej2), (ti1, ti2, tj1, tj2) in zip(ok, ok[1:]):
        first = raw_lines[ti1]
        cursor = len(first) - len(first.lstrip())  # first changed line, at
        # its first non-blank column (every line of the hunk changed)
        prefix = raw_lines[max(0, ti1 - 10):ti1]
        note = ("propagate air format to the next changed hunk "
                f"(raw hunk at line {ti1 + 1} -> formatted)")
        out.append(make_multiline_example(
            "format_propagation", package, rel, prefix,
            raw_lines[ti1:ti2], norm_lines[tj1:tj2], cursor,
            raw_lines[ei1:ei2], norm_lines[ej1:ej2], ei1 + 1, note))
        if len(out) >= cap:
            break
    return out


def build_format_examples(package_names, seed=13,
                          per_family_cap=MAX_PER_FAMILY,
                          max_files=MAX_FILES_PER_PKG,
                          time_budget_s=780, verbose=False):
    """Read only the needed members from each package tarball (never
    extracting whole tarballs), diff against the normalized tree."""
    rng = random.Random(seed)
    out: list[dict] = []
    t0 = time.time()
    files_diffed = tarballs_read = files_with_pairs = 0
    seen_pkgs = set()
    for pkg in package_names:
        if len(out) >= per_family_cap or time.time() - t0 > time_budget_s:
            break
        try:
            ver_dir = next((ROOT / pkg).iterdir())
            rdir = ver_dir / pkg / "R"
        except (StopIteration, FileNotFoundError, NotADirectoryError,
                OSError):
            continue
        if not rdir.is_dir():
            continue
        tar_path = TAR_DIR / f"{pkg}_{ver_dir.name}.tar.gz"
        if not tar_path.exists() or tar_path.stat().st_size > TAR_MAX_BYTES:
            continue
        try:
            files = sorted(list(rdir.glob("*.R")) + list(rdir.glob("*.r")))
        except OSError:
            continue
        if not files:
            continue
        if len(files) > max_files:
            files = rng.sample(files, max_files)
        wanted = {}
        for f in files:
            try:
                if f.stat().st_size <= MAX_FILE_BYTES:
                    wanted[f.name] = f
            except OSError:
                continue
        if not wanted:
            continue
        try:
            tf = tarfile.open(tar_path, "r:gz")
        except (tarfile.TarError, OSError):
            continue
        raws = {}
        with tf:
            tarballs_read += 1
            seen_pkgs.add(pkg)
            for m in tf.getmembers():  # stream order: cheap sequential reads
                parts = m.name.split("/")
                if (len(parts) == 3 and parts[1] == "R" and m.isfile()
                        and parts[2] in wanted and m.size <= MAX_FILE_BYTES
                        and parts[2] not in raws):
                    try:
                        fh = tf.extractfile(m)
                        if fh is not None:
                            raws[parts[2]] = fh.read()
                    except (tarfile.TarError, OSError):
                        continue
        for name, norm_path in wanted.items():
            if name not in raws or len(out) >= per_family_cap:
                continue
            try:
                norm = norm_path.read_bytes()
            except OSError:
                continue
            files_diffed += 1
            raw_lines = raws[name].decode("utf-8", "replace").splitlines()
            norm_lines = norm.decode("utf-8", "replace").splitlines()
            exs = format_pairs_from_lines(pkg, f"R/{name}",
                                          raw_lines, norm_lines)
            if exs:
                files_with_pairs += 1
            for ex in exs:
                validate_example(ex)
                out.append(ex)
                if len(out) >= per_family_cap:
                    break
        if verbose and tarballs_read % 25 == 0:
            print(f"  format_propagation: tarballs={tarballs_read} "
                  f"files_diffed={files_diffed} examples={len(out)} "
                  f"elapsed={time.time() - t0:.0f}s")
    stats = dict(tarballs_read=tarballs_read, files_diffed=files_diffed,
                 files_with_pairs=files_with_pairs,
                 packages_processed=len(seen_pkgs),
                 elapsed_s=round(time.time() - t0, 1), count=len(out))
    return out, stats


# ---------------------------------------------------------------------------
# NEW family 5: doc_sync (roxygen @param sync for a new signature argument)
# ---------------------------------------------------------------------------

DOC_ARG_SPECS = (
    ("verbose", "FALSE", "Show progress messages while the function runs."),
    ("call", "caller_env()", "Calling environment captured by rlang."),
    ("env", "parent.frame()", "Environment in which to evaluate expressions."),
)
DOC_DEFAULTS = {a: d for a, d, _ in DOC_ARG_SPECS}
DOC_DESCS = {a: t for a, _, t in DOC_ARG_SPECS}
DOC_REGION_MAX_LINES = 14

ROXY_LINE_RE = re.compile(r"^\s*#'(?:\s|$)")
ROXY_ANCHOR_RE = re.compile(r"^\s*#'\s*@(return|export)\b")
ROXY_PARAM_TAG_RE = re.compile(r"^\s*#'\s*@param\b")
DOC_PARAM_LINE_RE = re.compile(r"^(\s*#') @param (verbose|call|env) (.+)$")


def _signature_param_names(src: bytes, params) -> set[str]:
    names = set()
    for c in params.children:
        if c.type == "identifier":
            names.add(node_text(src, c).decode("utf-8", "replace"))
        elif c.type == "parameter":
            for k in c.children:
                if k.type == "identifier":
                    names.add(node_text(src, k).decode("utf-8", "replace"))
                    break
    return names


def extract_doc_sync(b: Bundle, rng: random.Random, cap: int = 2) -> list[dict]:
    """For each named function whose roxygen block has >= 1 @param and an
    @return/@export anchor: event = signature gains ', <arg> = <default>'
    right before the parameters' closing paren; region = roxygen @param
    block area; region_new = same block plus the matching deterministic
    "#' @param <arg> <desc>" line inserted before the anchor tag."""
    src = b.src
    out = []
    for fn in traverse(b.tree.root_node):
        if len(out) >= cap:
            break
        if fn.type != "function_definition":
            continue
        params = next((c for c in fn.children if c.type == "parameters"), None)
        if params is None:
            continue
        parent = fn.parent
        if (parent is None or parent.type != "binary_operator"
                or not parent.children
                or parent.children[0].type != "identifier"):
            continue  # roxygen attaches to named top-level definitions
        argish = [c for c in params.children
                  if c.type in ("identifier", "parameter", "dots")]
        if not argish:
            continue  # zero-arg function: ', arg' insert would be invalid R
        fn_name = node_text(src, parent.children[0]).decode("utf-8", "replace")
        top_row = min(fn.start_point[0], parent.children[0].start_point[0])
        if top_row <= 0:
            continue
        r, block = top_row - 1, []
        while r >= 0 and ROXY_LINE_RE.match(b.line_str(r)):
            block.append(r)
            r -= 1
        if not block:
            continue
        block.reverse()
        pnames = _signature_param_names(src, params)
        arg = next((a for a, _, _ in DOC_ARG_SPECS if a not in pnames), None)
        if arg is None:
            continue
        if any(re.match(rf"^\s*#'\s*@param\s+{re.escape(arg)}\b",
                        b.line_str(rr)) for rr in block):
            continue  # already documented
        prow = [rr for rr in block if ROXY_PARAM_TAG_RE.match(b.line_str(rr))]
        if not prow:
            continue
        arow = next((rr for rr in block
                     if ROXY_ANCHOR_RE.match(b.line_str(rr))), None)
        if arow is None or arow < prow[0]:
            continue  # need @return/@export AFTER the @param lines
        win = list(range(prow[0], arow + 1))
        if len(win) > DOC_REGION_MAX_LINES:
            continue
        region_old = [b.line_str(rr) for rr in win]
        if any(len(l) > FORMAT_LINE_MAX for l in region_old):
            continue
        pm = re.match(r"\s*#'", region_old[-1])
        ins_line = f"{pm.group(0)} @param {arg} {DOC_DESCS[arg]}"
        pos = win.index(arow)
        region_new = region_old[:pos] + [ins_line] + region_old[pos:]
        cursor = sum(len(l) + 1 for l in region_old[:pos])  # anchor line start
        # event: append ', arg = default' immediately before the closing paren
        end = params.end_byte
        if src[end - 1:end] != b")":
            continue
        ev_row, ev_col = b.rowcol(end - 1)
        lb = b.line_bytes(ev_row)
        ev_old = b.line_str(ev_row)
        cc = len(lb[:ev_col].decode("utf-8", "replace"))
        ev_new = ev_old[:cc] + f", {arg} = {DOC_DEFAULTS[arg]}" + ev_old[cc:]
        if ev_new == ev_old:
            continue
        prefix = [b.line_str(rr) for rr in range(max(0, win[0] - 10), win[0])]
        note = (f"document new argument {arg} in the roxygen block of "
                f"{fn_name} (event added it to the signature)")
        out.append(make_multiline_example(
            "doc_sync", b.package, b.rel, prefix, region_old, region_new,
            cursor, [ev_old], [ev_new], ev_row + 1, note))
    return out


def build_doc_sync_examples(package_names, seed=13,
                            per_family_cap=MAX_PER_FAMILY,
                            time_budget_s=600, verbose=False):
    rng = random.Random(seed)
    out: list[dict] = []
    t0 = time.time()
    files = files_with_ex = 0
    seen_pkgs = set()
    for b in iter_bundles(package_names, rng):
        files += 1
        seen_pkgs.add(b.package)
        if len(out) < per_family_cap:
            exs = extract_doc_sync(b, rng)
            if exs:
                files_with_ex += 1
            for ex in exs:
                validate_example(ex)
                out.append(ex)
                if len(out) >= per_family_cap:
                    break
        if verbose and files % 200 == 0:
            print(f"  doc_sync: files={files} examples={len(out)} "
                  f"elapsed={time.time() - t0:.0f}s")
        if time.time() - t0 > time_budget_s:
            break
        if len(out) >= per_family_cap:
            break
    stats = dict(files=files, files_with_examples=files_with_ex,
                 packages_processed=len(seen_pkgs),
                 elapsed_s=round(time.time() - t0, 1), count=len(out))
    return out, stats


# ---------------------------------------------------------------------------
# doc_sync v2: variant="missing_param" + variant="version_pair"
# (see the DIAGNOSIS block in the module docstring)
# ---------------------------------------------------------------------------

MISSING_MAX_PARAMS = 4          # new @param lines one row may add
MISSING_REGION_MAX_LINES = 14
MISSING_EVENT_MAX_LINES = 10
PAIR_DELTA_MAX = 3              # max params added+removed per version bump
PAIR_EVENT_MAX_LINES = 6
PAIR_REGION_MAX_OLD = 14        # 1 cursor line + up to 13 changed lines
PAIR_REGION_MAX_NEW = 16

GIT_ROOT = Path("/mnt/h/sepalith/git")
META_DIR = Path("/mnt/h/sepalith/meta/cran-to-git")
PKG2GIT_CACHE = Path(__file__).resolve().parent / ".pkg2git_cache.json"
PROV_DIR = Path("/mnt/h/sepalith/provenance")
CRAN_CONTRIB = "https://cran.r-project.org/src/contrib/{}"

# style_tag.py's classifier, duplicated (kept in sync by test) so the rows
# carry the same `style` values the rest of the pipeline expects.
STYLE_TIDY = re.compile(r"%>%|\|>|dplyr::|tidyr::|purrr::|ggplot|mutate\(|"
                        r"summarise\(|summarize\(|filter\(|select\(|"
                        r"group_by\(|across\(|left_join\(|pivot_longer\(|"
                        r"pivot_wider\(|readr::|tibble\(")
STYLE_BASE = re.compile(r"\bapply\(|\bsapply\(|\blapply\(|\btapply\(|"
                        r"\baggregate\(|\bmerge\(|\bsubset\(|\bwith\(|"
                        r"\bwithin\(|\bdata\.frame\(|\bstrsplit\(|\bgrepl\(|"
                        r"\bregexpr\(|do\.call\(")


def _style_of(text: str) -> str:
    t, b = len(STYLE_TIDY.findall(text)), len(STYLE_BASE.findall(text))
    if t >= 2 and t > b * 2:
        return "tidyverse"
    if b >= 2 and b > t * 2:
        return "base"
    if t > b:
        return "tidyverse-lean"
    if b > t:
        return "base-lean"
    return "neutral"


def attach_provenance(rec: dict, derivation: str) -> dict:
    """source_url/license/version/upstream/derivation/style, matching the
    fields the existing scenario rows carry (enrich_provenance.py style)."""
    pkg = rec.get("package", "")
    try:
        p = json.loads((PROV_DIR / f"{pkg}.json").read_text())
    except (OSError, ValueError):
        p = {}
    if p.get("tarball"):
        rec.setdefault("source_url", CRAN_CONTRIB.format(p["tarball"]))
    rec.setdefault("license", p.get("license") or "unknown")
    rec.setdefault("version", p.get("version"))
    rec.setdefault("upstream", p.get("upstream") or "")
    rec["derivation"] = derivation
    rec["style"] = _style_of("\n".join(
        rec.get("prefix", []) + rec.get("region_old", [])
        + rec.get("region_new", [])))
    return rec


# --- deterministic name grammar -> @param description ----------------------
# First matching rule wins; "{0}" is the humanised name prefix. A param name
# that matches no rule yields no row (every target stays derivable from the
# param NAME + the block's own tag style, both visible in the prompt).

DOC_NAME_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^\.\.\.$"), "Further arguments passed on to other methods."),
    (re.compile(r"^verbose$"), "Show progress messages while the function runs."),
    (re.compile(r"^quiet$"), "Suppress messages when TRUE."),
    (re.compile(r"^debug$"), "Print extra debug information when TRUE."),
    (re.compile(r"^progress$"), "Display a progress bar when TRUE."),
    (re.compile(r"^dry_run$"), "Skip applying the changes when TRUE."),
    (re.compile(r"^checks?$"), "Perform consistency checks on the input."),
    (re.compile(r"^na[._]rm$"), "Should missing values be removed?"),
    (re.compile(r"^sep$"), "Separator used between values."),
    (re.compile(r"^collapse$"),
     "Separator used to combine the result into a single string."),
    # prefix rules BEFORE suffix rules so e.g. use_names hits `use_` (not
    # `_names`) and min_size hits `min_` (not `_sizes`)
    (re.compile(r"^n_(.+)$"), "Number of {0}."),
    (re.compile(r"^num_(.+)$"), "Number of {0}."),
    (re.compile(r"^min_(.+)$"), "Minimum {0} allowed."),
    (re.compile(r"^max_(.+)$"), "Maximum {0} allowed."),
    (re.compile(r"^use_(.+)$"), "Use {0} when TRUE."),
    (re.compile(r"^show_(.+)$"), "Show the {0}."),
    (re.compile(r"^(.+)_cols$"), "Names of the columns holding {0}."),
    (re.compile(r"^(.+)_columns$"), "Names of the columns holding {0}."),
    (re.compile(r"^(.+)_col$"), "Name of the column holding {0}."),
    (re.compile(r"^(.+)_colname$"), "Name of the column holding {0}."),
    (re.compile(r"^(.+?)_files?$"), "Path to the {0} file."),
    (re.compile(r"^(.+?)_paths?$"), "Path to the {0} file."),
    (re.compile(r"^(.+?)_dirs?$"), "Directory containing the {0}."),
    (re.compile(r"^(.+?)_directory$"), "Directory containing the {0}."),
    (re.compile(r"^(.+?)_urls?$"), "URL of the {0}."),
    (re.compile(r"^(.+?)_colors?$"), "Color used for the {0}."),
    (re.compile(r"^(.+?)_colours?$"), "Color used for the {0}."),
    (re.compile(r"^(.+?)_sizes?$"), "Size of the {0}."),
    (re.compile(r"^(.+?)_width$"), "Width of the {0}."),
    (re.compile(r"^(.+?)_height$"), "Height of the {0}."),
    (re.compile(r"^(.+?)_fonts?$"), "Font used for the {0}."),
    (re.compile(r"^(.+?)_names$"), "Names of the {0}."),
    (re.compile(r"^(.+?)_name$"), "Name of the {0}."),
    (re.compile(r"^(.+?)_labels$"), "Labels for the {0}."),
    (re.compile(r"^(.+?)_label$"), "Label for the {0}."),
    (re.compile(r"^(.+?)_types?$"), "Type of the {0}."),
    (re.compile(r"^(.+?)_methods?$"), "Method to use for the {0}."),
    (re.compile(r"^(.+?)_funs?$"), "Function to apply to the {0}."),
    (re.compile(r"^(.+?)_patterns?$"), "Regular expression used to match the {0}."),
    (re.compile(r"^(.+?)_prefix$"), "Prefix added to the {0}."),
    (re.compile(r"^(.+?)_suffix$"), "Suffix added to the {0}."),
    (re.compile(r"^(.+?)_formats?$"), "Format used for the {0}."),
    (re.compile(r"^(.+?)_levels?$"), "Level of the {0}."),
    (re.compile(r"^(.+?)_scales?$"), "Scale used for the {0}."),
    (re.compile(r"^(.+?)_seeds?$"), "Random seed used for the {0}."),
    (re.compile(r"^(.+?)_digits$"), "Number of digits used for the {0}."),
    (re.compile(r"^(.+?)_thresholds?$"), "Threshold used for the {0}."),
    (re.compile(r"^(.+?)_time$"), "Time point used for the {0}."),
    (re.compile(r"^(.+?)_env$"), "Environment in which to evaluate the {0}."),
    (re.compile(r"^(.+?)_datas?$"), "Data used for the {0}."),
    (re.compile(r"^(.+?)_args$"), "Arguments passed to the {0}."),
    (re.compile(r"^(.+?)_opts$"), "Options controlling the {0}."),
)

ROXY_PARAM_RE = re.compile(r"^\s*#'\s*@param\s+([.\w]+|\.\.\.)")
ROXY_TAG_RE = re.compile(r"^\s*#'\s*(@\w+)")


def doc_desc_for_name(name: str) -> str | None:
    """Deterministic description for a parameter name (grammar above), or
    None when the name matches no rule (row is then not built)."""
    if not name or not IDENT_RE.match(name) and name != "...":
        return None
    if name in RESERVED:
        return None
    for rx, tpl in DOC_NAME_RULES:
        m = rx.match(name)
        if not m:
            continue
        if m.groups():
            rest = m.group(1).replace("_", " ").strip()
            if not rest or not re.match(r"^[A-Za-z][A-Za-z0-9 ]*$", rest):
                return None
            return tpl.format(rest)
        return tpl
    return None


def _roxy_param_descs(lines) -> list[str]:
    """First-line descriptions of the @param tags in a roxygen region (used
    to copy the block's punctuation/capitalisation style)."""
    out = []
    for l in lines:
        m = re.match(r"^\s*#'\s*@param\s+(?:[.\w]+|\.\.\.)\s+(.+)$", l)
        if m:
            out.append(m.group(1))
    return out


def _doc_style(descs: list[str]) -> tuple[bool, bool]:
    """(capitalise, trailing_period) majority style of the block's @param
    descriptions; ties keep the canonical form (capitalised, with period)."""
    if not descs:
        return True, True
    n = len(descs)
    cap = sum(1 for d in descs if d[:1].isupper())
    per = sum(1 for d in descs if d.endswith("."))
    return cap * 2 >= n, per * 2 >= n


def _styled_desc(desc: str, cap: bool, period: bool) -> str:
    d = (desc[0].upper() + desc[1:]) if cap else (desc[0].lower() + desc[1:])
    if d[-1:] in ".?!":  # already terminal punctuation: never double it
        return d
    return d + "." if period else d


def _roxy_param_names(lines) -> list[str]:
    out = []
    for l in lines:
        m = ROXY_PARAM_RE.match(l)
        if m:
            out.append(m.group(1))
    return out


def _ordered_params(src: bytes, params):
    """Ordered [(name, node)] of a tree-sitter `parameters` node."""
    out = []
    for c in params.children:
        if c.type == "identifier":
            out.append((node_text(src, c).decode("utf-8", "replace"), c))
        elif c.type == "parameter":
            for k in c.children:
                if k.type == "identifier":
                    out.append((node_text(src, k).decode("utf-8", "replace"),
                                c))
                    break
        elif c.type == "dots":
            out.append(("...", c))
    return out


def _params_from_sig_text(text: str) -> list[str] | None:
    """Ordered parameter names from signature SOURCE text such as
    'foo <- function(x, na_rm = TRUE, ...) {' (string-blind, depth-aware)."""
    s = strip_strings(text.encode()).decode("utf-8", "replace")
    m = re.search(r"\bfunction\s*\(", s)
    if not m:
        return None
    i = m.end() - 1
    depth, inner = 0, None
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                inner = s[i + 1:j]
                break
    if inner is None:
        return None
    parts, buf, d = [], [], 0
    for ch in inner:
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
        if ch == "," and d == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    names = []
    for p in parts:
        if not p.strip():
            return None  # trailing comma / empty -> ambiguous
        nm = p.split("=", 1)[0].strip().strip("`")
        if nm == "...":
            names.append(nm)
            continue
        if not IDENT_RE.match(nm) or nm in RESERVED:
            return None
        names.append(nm)
    return names


def _arg_span(line: str, name: str) -> tuple[int, int] | None:
    """(start, end) char span of the argument `name` (with its default) on
    one signature source line, or None. The span ends at the first top-level
    ',' (before it) or at the bracket that closes the argument list level."""
    s = strip_strings(line.encode()).decode("utf-8", "replace")
    for m in re.finditer(rf"(?<![\w.`]){re.escape(name)}(?![\w.`])", s):
        pre = s[:m.start()].rstrip(" \t")
        if pre and pre[-1] not in ",(":
            continue  # not an argument position on this line (callee etc.)
        depth, j = 0, m.end()
        while j < len(s):
            ch = s[j]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                break
            j += 1
        return m.start(), j
    return None


def _strip_prev_trailing_comma(lines: list[str], upto: int) -> bool:
    """Drop the trailing comma of the nearest non-blank line above `upto`
    (used when a wrapped signature's last argument was removed)."""
    for pj in range(upto - 1, -1, -1):
        if not lines[pj].strip():
            continue
        m = re.match(r"^(.*?),[ \t]*$", lines[pj], re.DOTALL)
        if m and m.group(1).strip():
            lines[pj] = m.group(1)
            return True
        return False
    return False


def _remove_named_args(lines: list[str], names: list[str]) -> list[str] | None:
    """Remove the arguments `names` (and one adjacent separator comma) from
    signature source lines; an argument alone on its line takes the whole
    line with it (the previous line loses its trailing comma when the last
    argument goes). Returns the new lines, or None when a name cannot be
    removed cleanly. Shared by the missing_param generator AND its validator,
    so the event is exact by construction."""
    cur = list(lines)
    for name in names:
        hit = None
        for li, line in enumerate(cur):
            sp = _arg_span(line, name)
            if sp is not None:
                hit = (li, sp[0], sp[1])
                break
        if hit is None:
            return None
        li, s, e = hit
        line = cur[li]
        before, after = line[:s], line[e:]
        if not before.strip() and not after.strip(" \t,"):
            nxt = cur[li + 1] if li + 1 < len(cur) else ""
            new = cur[:li] + cur[li + 1:]
            if nxt.lstrip(" \t").startswith(")"):
                if not _strip_prev_trailing_comma(new, li):
                    return None
            cur = new
            continue
        # separator comma after the argument on the same line?
        k = e
        while k < len(line) and line[k] in " \t":
            k += 1
        if k < len(line) and line[k] == ",":
            k2 = k + 1
            while k2 < len(line) and line[k2] in " \t":
                k2 += 1
            new_line = before + line[k2:]
            cur[li] = new_line.rstrip() if not line[k2:] else new_line
            continue
        # separator comma before the argument on the same line?
        k = s
        while k > 0 and line[k - 1] in " \t":
            k -= 1
        if k > 0 and line[k - 1] == ",":
            cur[li] = (line[:k - 1].rstrip()
                       + ("" if after[:1] in ")]}," or not after else " ")
                       + after)
            continue
        # wrapped signature: the separator comma sits on the previous line
        if not before.strip() and after.lstrip(" \t")[:1] == ")":
            cur[li] = before + after
            if not _strip_prev_trailing_comma(cur, li):
                return None
            continue
        return None
    return cur


def _missing_event_lines(b: Bundle, ordered, missing, sig_first, sig_last):
    """(ev_old_lines, ev_new_lines) for a missing_param row: ev_new is the
    file's real signature lines; ev_old is the same signature WITHOUT the
    `missing` arguments (the user 'just added them'). Returns None when the
    removals cannot be expressed exactly or would not parse as the intended
    parameter list."""
    sig_lines = [b.line_str(r) for r in range(sig_first, sig_last + 1)]
    if not sig_lines or any(len(l) > FORMAT_LINE_MAX for l in sig_lines):
        return None
    names = [nm for nm, _ in missing]
    ev_old = _remove_named_args(sig_lines, names)
    if ev_old is None or ev_old == sig_lines:
        return None
    if any(len(l) > FORMAT_LINE_MAX for l in ev_old):
        return None
    want = [nm for nm, _ in ordered if nm not in set(names)]
    got = _params_from_sig_text("\n".join(ev_old))
    if got != want:  # guard against a name removed from the wrong spot
        return None
    return ev_old, sig_lines


def extract_doc_sync_missing(b: Bundle, rng: random.Random,
                             cap: int = 2) -> list[dict]:
    """Single-file variant: REAL signature parameters that have no @param tag
    yet (names must match the grammar). region_old = the block's @param tag
    lines (with continuations); region_new = the same + the missing
    "#' @param <name> <desc>" lines appended after the last tag (signature
    order); the cursor sits at the end of the last existing tag line; the
    event is the real signature with those arguments removed (the user "just
    added them" - the event defines which arguments the target documents).
    Descriptions are a pure function of the param name plus the block's own
    punctuation/capitalisation style."""
    src = b.src
    out = []
    for fn in traverse(b.tree.root_node):
        if len(out) >= cap:
            break
        if fn.type != "function_definition":
            continue
        params = next((c for c in fn.children if c.type == "parameters"),
                      None)
        if params is None:
            continue
        parent = fn.parent
        if (parent is None or parent.type != "binary_operator"
                or not parent.children
                or parent.children[0].type != "identifier"):
            continue
        ordered = _ordered_params(src, params)
        if not ordered:
            continue
        fn_name = node_text(src, parent.children[0]).decode("utf-8", "replace")
        top_row = min(fn.start_point[0], parent.children[0].start_point[0])
        if top_row <= 0:
            continue
        r, block = top_row - 1, []
        while r >= 0 and ROXY_LINE_RE.match(b.line_str(r)):
            block.append(r)
            r -= 1
        if not block:
            continue
        block.reverse()
        block_lines = [b.line_str(rr) for rr in block]
        if any(re.match(r"^\s*#'\s*@inherit", l) for l in block_lines):
            continue  # tags may be inherited on purpose -> ambiguous
        documented = set(_roxy_param_names(block_lines))
        pnames = [nm for nm, _ in ordered]
        if not (documented & set(pnames)):
            continue  # block documents none of the signature -> not doc_sync
        prow = [rr for rr in block if ROXY_PARAM_TAG_RE.match(b.line_str(rr))]
        if not prow:
            continue
        cand = [(nm, node) for nm, node in ordered
                if nm not in documented and doc_desc_for_name(nm)]
        if not cand:
            continue
        # region: first @param tag .. end of the last tag's continuation lines
        rr = prow[-1] + 1
        while rr <= block[-1]:
            t = b.line_str(rr)
            m = ROXY_TAG_RE.match(t)
            if (not ROXY_LINE_RE.match(t) or m
                    or t.strip() == "#'"):
                break
            rr += 1
        win = list(range(prow[0], rr))
        if not (win[0] >= 1 and 0 < len(win) <= MISSING_REGION_MAX_LINES):
            continue  # need >= 1 line above the region for `prefix`
        region_old = [b.line_str(x) for x in win]
        if any(len(l) > FORMAT_LINE_MAX for l in region_old):
            continue
        end = params.end_byte
        if src[end - 1:end] != b")":
            continue
        sig_first = top_row
        sig_last = b.rowcol(end - 1)[0]
        if not (1 <= sig_last - sig_first + 1 <= MISSING_EVENT_MAX_LINES):
            continue
        # document the largest prefix of the undocumented, grammar-covered
        # arguments whose signature removal is exactly expressible
        event = None
        missing = None
        for k in range(min(MISSING_MAX_PARAMS, len(cand)), 0, -1):
            event = _missing_event_lines(b, ordered, cand[:k],
                                         sig_first, sig_last)
            if event is not None:
                missing = cand[:k]
                break
        if missing is None or event is None:
            continue
        ev_old_lines, ev_new_lines = event
        descs = _roxy_param_descs(region_old)
        cap_s, per_s = _doc_style(descs)
        names = [nm for nm, _ in missing]
        ins_lines = []
        pm = re.match(r"\s*#'", region_old[-1])
        for nm in names:
            ins_lines.append(f"{pm.group(0)} @param {nm} "
                             f"{_styled_desc(doc_desc_for_name(nm), cap_s, per_s)}")
        region_new = region_old + ins_lines
        cursor = len("\n".join(region_old))  # end of the last existing tag
        prefix = [b.line_str(x) for x in range(max(0, win[0] - 10), win[0])]
        note = (f"document missing argument(s) {', '.join(names)} of "
                f"{fn_name} (event added them to the signature)")
        ex = make_multiline_example(
            "doc_sync", b.package, b.rel, prefix, region_old, region_new,
            cursor, ev_old_lines, ev_new_lines, sig_first + 1, note)
        ex["variant"] = "missing_param"
        out.append(ex)
    return out


# --- variant="version_pair": real upstream doc sync between releases -------

def _git(repo: Path, *args: str, timeout: int = 120) -> bytes | None:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


_BUMP_VER = re.compile(rb"^([-+]Version:)\s*(\S+)\s*$", re.M)


def version_transitions(repo: Path, cap: int = 6) -> list[tuple[str, str, str, str]]:
    """Adjacent DESCRIPTION version bumps -> [(shaA, shaB, verA, verB)]: the
    state just before bump k (version A's final state) vs just before bump
    k+1 (version B's final state), i.e. everything done during B's cycle.
    Newest pairs first."""
    outp = _git(repo, "log", "-n", "400", "--format=%x00%H", "-p",
                "--unified=0", "--", "DESCRIPTION")
    if not outp:
        return []
    bumps = []  # (commit, old, new), newest first
    for chunk in outp.split(b"\x00")[1:]:
        lines = chunk.split(b"\n", 1)
        if len(lines) < 2:
            continue
        commit = lines[0].decode("ascii", "replace").strip()
        old = new = None
        for m in _BUMP_VER.finditer(lines[1]):
            if m.group(1).startswith(b"-"):
                old = m.group(2).decode("ascii", "replace")
            elif m.group(1).startswith(b"+"):
                new = m.group(2).decode("ascii", "replace")
        if commit and old and new and old != new:
            bumps.append((commit, old, new))
    pairs = []
    for (c_new, _vu, ver_b), (c_old, _vd, ver_a) in zip(bumps, bumps[1:]):
        pairs.append((f"{c_old}^", f"{c_new}^", ver_a, ver_b))
        if len(pairs) >= cap:
            break
    return pairs


def _roxy_param_group_only(lines) -> bool:
    """Every line is a @param tag line or a continuation of one (regions are
    restricted to the @param area of the roxygen block)."""
    in_param = False
    for l in lines:
        if not ROXY_LINE_RE.match(l):
            return False
        m = ROXY_TAG_RE.match(l)
        if m:
            in_param = m.group(1) == "@param"
        elif l.strip() == "#'":
            return False  # blank separator: not part of a tag's description
        elif not in_param:
            return False
    return True


def _fn_table(src: bytes):
    """{fn name: (fn_node, params_node, first_row, last_row, block_lines,
    file_lines)} for named function definitions; duplicated names dropped."""
    tree = parser.parse(src)
    lines = src.decode("utf-8", "replace").splitlines()
    by_name: dict[str, list] = {}
    for fn in traverse(tree.root_node):
        if fn.type != "function_definition":
            continue
        params = next((c for c in fn.children if c.type == "parameters"),
                      None)
        parent = fn.parent
        if (params is None or parent is None
                or parent.type != "binary_operator" or not parent.children
                or parent.children[0].type != "identifier"):
            continue
        name = node_text(src, parent.children[0]).decode("utf-8", "replace")
        first_row = min(fn.start_point[0],
                        parent.children[0].start_point[0])
        last_row = params.end_point[0]
        if first_row <= 0 or last_row >= len(lines):
            continue
        r, block = first_row - 1, []
        while r >= 0 and ROXY_LINE_RE.match(lines[r]):
            block.append(lines[r])
            r -= 1
        if not block:
            continue
        block.reverse()
        by_name.setdefault(name, []).append(
            (fn, params, first_row, last_row, block, lines))
    return {k: v[0] for k, v in by_name.items() if len(v) == 1}


def extract_doc_sync_pair(pkg: str, rel: str, old_src: bytes,
                          new_src: bytes, cap: int = 1) -> list[dict]:
    """Rows from one file's two versions: the prompt keeps the OLD roxygen,
    the event shows the REAL signature change, and the target is the
    maintainer's actual updated @param lines. Accepted only when the
    @param-name delta equals the signature-param delta exactly."""
    if not old_src or not new_src or len(old_src) > MAX_FILE_BYTES \
            or len(new_src) > MAX_FILE_BYTES:
        return []
    old_tab, new_tab = _fn_table(old_src), _fn_table(new_src)
    out = []
    for name in sorted(set(old_tab) & set(new_tab)):
        if len(out) >= cap:
            break
        (fn_a, par_a, fr_a, lr_a, blk_a, lines_a) = old_tab[name]
        (fn_b, par_b, fr_b, lr_b, blk_b, lines_b) = new_tab[name]
        names_a = [nm for nm, _ in _ordered_params(old_src, par_a)]
        names_b = [nm for nm, _ in _ordered_params(new_src, par_b)]
        if not names_a or not names_b:
            continue
        added = [n for n in names_b if n not in names_a]
        removed = [n for n in names_a if n not in names_b]
        if not (added or removed) or len(added) + len(removed) > PAIR_DELTA_MAX:
            continue
        doc_a, doc_b = _roxy_param_names(blk_a), _roxy_param_names(blk_b)
        if set(doc_b) - set(doc_a) != set(added):
            continue
        if set(doc_a) - set(doc_b) != set(removed):
            continue
        sm = difflib.SequenceMatcher(a=blk_a, b=blk_b, autojunk=False)
        ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
        if not ops:
            continue
        s, e = min(o[1] for o in ops), max(o[2] for o in ops)
        j1, j2 = min(o[3] for o in ops), max(o[4] for o in ops)
        if s == 0 or blk_a[s - 1] != blk_b[j1 - 1]:
            continue  # need one shared cursor line above the first change
        region_old = [blk_a[s - 1]] + blk_a[s:e]
        region_new = [blk_a[s - 1]] + blk_b[j1:j2]
        if (len(region_old) > PAIR_REGION_MAX_OLD
                or len(region_new) > PAIR_REGION_MAX_NEW
                or region_old == region_new):
            continue
        if any(len(l) > FORMAT_LINE_MAX for l in region_old + region_new):
            continue
        if not (_roxy_param_group_only(region_old[1:])
                and _roxy_param_group_only(region_new[1:])):
            continue
        if blk_a[:s - 1] + region_new + blk_a[e:] != blk_b:
            continue  # splice invariant: the edit must rebuild the new block
        sig_a, sig_b = lines_a[fr_a:lr_a + 1], lines_b[fr_b:lr_b + 1]
        if (sig_a == sig_b or len(sig_a) > PAIR_EVENT_MAX_LINES
                or len(sig_b) > PAIR_EVENT_MAX_LINES
                or any(len(l) > FORMAT_LINE_MAX for l in sig_a + sig_b)):
            continue
        pa, pb = _params_from_sig_text("\n".join(sig_a)), \
            _params_from_sig_text("\n".join(sig_b))
        if pa is None or pb is None or set(pb) - set(pa) != set(added) \
                or set(pa) - set(pb) != set(removed):
            continue  # event must carry the same param delta as the docs
        prefix = lines_a[max(0, s - 11):s - 1]
        cursor = len(region_old[0])  # end of the first stale line
        delta = ", ".join(f"+{a}" for a in added) + " " + \
            ", ".join(f"-{r}" for r in removed)
        note = (f"sync roxygen of {name} with the signature change between "
                f"versions ({delta.strip()})")
        ex = make_multiline_example(
            "doc_sync", pkg, rel, prefix, region_old, region_new, cursor,
            sig_a, sig_b, fr_a + 1, note)
        ex["variant"] = "version_pair"
        out.append(ex)
    return out


def pkg_to_git_dirs(cache: Path = PKG2GIT_CACHE) -> dict[str, str]:
    """{package: git mirror dir name} for root-level CRAN mirrors that are
    actually materialised (subdir repos are skipped; the drvfs walk of the
    meta jsons is cached)."""
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except (ValueError, OSError):
            pass
    import os
    mapping: dict[str, str] = {}
    try:
        git_dirs = {p for p in os.listdir(GIT_ROOT)}
        meta_files = sorted(os.listdir(META_DIR))
    except OSError:
        return mapping
    for mf in meta_files:
        if not mf.endswith(".json"):
            continue
        try:
            entries = json.loads((META_DIR / mf).read_text())
        except (OSError, ValueError):
            continue
        for e in (entries if isinstance(entries, list) else [entries]):
            pkg, url, sub = e.get("package"), e.get("url", ""), \
                e.get("subdir", "")
            if not pkg or not url or sub:
                continue
            tail = url.rstrip("/").split("github.com/")[-1]
            gd = "__".join(tail.split("/"))
            if gd in git_dirs:
                mapping[pkg] = gd
    try:
        cache.write_text(json.dumps(mapping))
    except OSError:
        pass
    return mapping


def iter_pair_rows(pkg: str, git_dir: str, cap_per_repo: int = 4) -> list[dict]:
    """All doc_sync version-pair rows mined from one git mirror."""
    repo = GIT_ROOT / git_dir
    out = []
    try:
        for sha_a, sha_b, _va, _vb in version_transitions(repo):
            if len(out) >= cap_per_repo:
                break
            names = _git(repo, "diff", "--name-only", sha_a, sha_b, "--", "R")
            if not names:
                continue
            files = [n for n in names.decode("utf-8", "replace").splitlines()
                     if n.endswith((".R", ".r"))][:8]
            for rel in files:
                a = _git(repo, "show", f"{sha_a}:{rel}")
                b = _git(repo, "show", f"{sha_b}:{rel}")
                if a is None or b is None:
                    continue
                if len(a) > MAX_FILE_BYTES or len(b) > MAX_FILE_BYTES:
                    continue
                rel_norm = f"R/{rel.split('/', 1)[-1]}"
                for ex in extract_doc_sync_pair(pkg, rel_norm, a, b):
                    ex["version"] = _vb
                    out.append(ex)
                if len(out) >= cap_per_repo:
                    break
    except Exception:  # never let one flaky repo kill the run
        pass
    return out


# --- builder: resumable, <=6 workers, drvfs-tolerant writes -----------------

def _nas_write_jsonl(path: Path, rows: list[dict], tries: int = 20,
                     wait_s: float = 30.0) -> bool:
    """Write JSONL to the NAS store, riding out drvfs ENOMEM flaps (same
    contract as comment_to_code.py _nas_write_lines)."""
    for attempt in range(tries):
        try:
            with open(path, "w") as fh:
                for ex in rows:
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            return True
        except OSError as e:
            if attempt == tries - 1:
                print(f"  [nas-write] giving up on {path}: {e}", flush=True)
                return False
            print(f"  [nas-write] {e}; retry {attempt + 1}/{tries - 1} "
                  f"in {wait_s:.0f}s", flush=True)
            time.sleep(wait_s)
    return False


def _row_key(ex: dict) -> tuple:
    import hashlib
    h = hashlib.sha1("\n".join(ex["region_new"]).encode()).hexdigest()[:16]
    return (ex["package"], ex["path"], h)


def build_doc_sync_v2(target_new: int = 4700, max_workers: int = 6,
                      seed: int = 13, out_dir: Path = OUT_DIR,
                      time_budget_s: float = 14400.0,
                      sample_repos: int = 0, verbose: bool = True):
    """Grow doc_sync: missing_param rows from the normalized corpus +
    version_pair rows from the git mirrors. Resumable via sidecar progress
    (doc_sync_v2_progress.json) and a partial rows file; every row passes
    validate_example before it is kept (generator-side fixes only)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_dir.mkdir(parents=True, exist_ok=True)
    prog_path = out_dir / "doc_sync_v2_progress.json"
    partial_path = out_dir / "doc_sync_v2_partial.jsonl"
    rows: list[dict] = []
    prog = {"missing_done": [], "pair_done": []}
    try:
        rows = [json.loads(l) for l in partial_path.read_text().splitlines()
                if l.strip()]
        prog = json.loads(prog_path.read_text())
    except (OSError, ValueError):
        pass
    seen_keys = {_row_key(r) for r in rows}
    missing_done, pair_done = set(prog["missing_done"]), set(prog["pair_done"])
    stats = {"resumed_rows": len(rows), "dropped_invalid": 0,
             "missing_units": 0, "pair_units": 0}

    def keep(new_rows: list[dict], unit_kind: str, unit: str):
        kept = 0
        for ex in new_rows:
            if len(rows) >= target_new:
                break
            if _row_key(ex) in seen_keys:
                continue
            try:
                validate_example(ex)
            except AssertionError as e:
                stats["dropped_invalid"] += 1
                if verbose and stats["dropped_invalid"] <= 5:
                    print(f"  [invalid:{ex.get('variant')}] {e} "
                          f"({ex['package']} {ex['path']})", flush=True)
                continue
            attach_provenance(ex, f"doc_sync {ex.get('variant')} "
                                  f"constructor (scenarios.py)")
            seen_keys.add(_row_key(ex))
            rows.append(ex)
            kept += 1
        (missing_done if unit_kind == "missing" else pair_done).add(unit)
        return kept

    t0 = time.time()
    rng = random.Random(seed)

    # phase A: version pairs first (scarcer, harder, mined from git history)
    p2g = pkg_to_git_dirs()
    pair_units = sorted(p2g.items())
    if sample_repos:
        rng.shuffle(pair_units)
        pair_units = pair_units[:sample_repos]
    todo = [(p, g) for p, g in pair_units if p not in pair_done]
    if verbose:
        print(f"doc_sync v2 phase A: {len(todo)} git mirrors to mine "
              f"(target {target_new} new rows)", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(iter_pair_rows, p, g): (p, g) for p, g in todo}
        for fut in as_completed(futs):
            p, g = futs[fut]
            stats["pair_units"] += 1
            try:
                keep(fut.result(), "pair", p)
            except Exception as e:  # noqa: BLE001 - one repo must not kill
                pair_done.add(p)
                if verbose:
                    print(f"  [pair-error] {p}: {e}", flush=True)
            if time.time() - t0 > time_budget_s * 0.4:
                for f in futs:
                    f.cancel()
                break
            if stats["pair_units"] % 50 == 0 and verbose:
                print(f"  pairs: {stats['pair_units']}/{len(todo)} units, "
                      f"rows={len(rows)} elapsed={time.time()-t0:.0f}s",
                      flush=True)

    # phase B: missing-@param rows from the normalized corpus (bulk)
    pkgs = [p for p in list_packages() if p not in missing_done]
    rng.shuffle(pkgs)
    if verbose:
        print(f"doc_sync v2 phase B: {len(pkgs)} packages to scan, "
              f"{len(rows)} rows so far", flush=True)

    def scan_pkg(pkg: str) -> list[dict]:
        import zlib
        got: list[dict] = []
        rng_local = random.Random(seed + zlib.crc32(pkg.encode()) % 10_000)
        try:
            bundles = list(iter_bundles([pkg], rng_local, max_files=8))
        except OSError:
            return got
        for b in bundles:
            if len(got) >= 6:
                break
            got.extend(extract_doc_sync_missing(b, rng_local, cap=3))
        return got

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(scan_pkg, p): p for p in pkgs}
        for fut in as_completed(futs):
            p = futs[fut]
            stats["missing_units"] += 1
            try:
                keep(fut.result(), "missing", p)
            except Exception as e:  # noqa: BLE001
                missing_done.add(p)
                if verbose:
                    print(f"  [missing-error] {p}: {e}", flush=True)
            if stats["missing_units"] % 100 == 0 and verbose:
                print(f"  missing: {stats['missing_units']}/{len(pkgs)} "
                      f"pkgs, rows={len(rows)}/{target_new} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                _nas_write_jsonl(partial_path, rows)
                try:
                    prog_path.write_text(json.dumps(
                        {"missing_done": sorted(missing_done),
                         "pair_done": sorted(pair_done)}))
                except OSError:
                    pass
            if len(rows) >= target_new or time.time() - t0 > time_budget_s:
                for f in futs:
                    f.cancel()
                break

    # final checkpoint
    _nas_write_jsonl(partial_path, rows)
    prog_path.write_text(json.dumps(
        {"missing_done": sorted(missing_done), "pair_done": sorted(pair_done)}))
    stats.update(count=len(rows), elapsed_s=round(time.time() - t0, 1),
                 variants={v: sum(1 for r in rows if r.get("variant") == v)
                           for v in ("missing_param", "version_pair")})
    return rows, stats


def merge_doc_sync(new_rows: list[dict], out_dir: Path = OUT_DIR,
                   sample_n: int = 50, seed: int = 7) -> dict:
    """Rewrite doc_sync.jsonl = old rows (preserved) + new rows appended,
    deduped by (package, path, target-hash). Validates EVERY row, samples
    `sample_n` new rows for the pass-rate report, and merges counts into
    stats.json."""
    path = out_dir / "doc_sync.jsonl"
    old = []
    try:
        old = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except (OSError, ValueError):
        pass
    merged, seen = [], set()
    dropped = 0
    for ex in old + new_rows:
        k = _row_key(ex)
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        merged.append(ex)
    for ex in merged:
        validate_example(ex)  # the whole family file stays validator-clean
    rng = random.Random(seed)
    sample = rng.sample(new_rows, min(sample_n, len(new_rows))) \
        if new_rows else []
    passed = fails = 0
    for ex in sample:
        try:
            validate_example(ex)
            noop_exact_score(ex)
            passed += 1
        except AssertionError:
            fails += 1
    if not _nas_write_jsonl(path, merged):
        raise SystemExit(f"could not write {path}")
    variants = {"canonical": sum(1 for r in merged if not r.get("variant"))}
    for v in ("missing_param", "version_pair"):
        variants[v] = sum(1 for r in merged if r.get("variant") == v)
    rep = {"old_rows": len(old), "new_rows": len(new_rows),
           "merged_rows": len(merged), "dropped_dupes": dropped,
           "variants": variants,
           "sample_size": len(sample), "sample_pass": passed,
           "sample_pass_rate": round(passed / len(sample), 4) if sample else None}
    try:
        stats = json.loads((out_dir / "stats.json").read_text())
    except (OSError, ValueError):
        stats = {}
    if not isinstance(stats.get("counts"), dict):
        stats["counts"] = {}
    stats["counts"]["doc_sync"] = len(merged)
    stats["doc_sync_v2"] = rep
    try:
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    except OSError as e:
        print(f"  [stats] write failed: {e}", flush=True)
    return rep


# ---------------------------------------------------------------------------
# new-family example builder + event diff (multi-line capable) + validators
# ---------------------------------------------------------------------------

EVENT_DIFF_MULTI_RE = re.compile(
    r'^User edited "(?P<path>.+)":\n\n```diff\n@@ (?P<hunk>[^@\n]*)@@\n'
    r"(?P<old>(?:-.*\n)+)(?P<new>(?:\+.*\n)+)```$")


def parse_event_diff_lines(s: str) -> tuple[str, list[str], list[str]]:
    """Parse an event diff whose body may span multiple -/+ lines (the
    single-line format used by event_diff_for is a special case)."""
    m = EVENT_DIFF_MULTI_RE.match(s)
    assert m, "event_diff malformed (multi-line)"
    old = m.group("old").split("\n")[:-1]
    new = m.group("new").split("\n")[:-1]
    return m.group("path"), [l[1:] for l in old], [l[1:] for l in new]


def make_multiline_example(family: str, package: str, path: str,
                           prefix_lines, region_old, region_new, cursor_idx,
                           event_old_lines, event_new_lines, event_lineno,
                           note: str) -> dict:
    """Example builder for families whose region and/or event span multiple
    lines (same JSON keys as make_example)."""
    if len(event_old_lines) == 1 and len(event_new_lines) == 1:
        hdr = f"@@ -{event_lineno} +{event_lineno} @@"
    else:
        hdr = (f"@@ -{event_lineno},{len(event_old_lines)} "
               f"+{event_lineno},{len(event_new_lines)} @@")
    body = "".join(f"-{l}\n" for l in event_old_lines) \
        + "".join(f"+{l}\n" for l in event_new_lines)
    return {
        "family": family,
        "package": package,
        "path": path,
        "prefix": list(prefix_lines),
        "region_old": list(region_old),
        "region_new": list(region_new),
        "cursor_idx": cursor_idx,
        "event_diff": f'User edited "{path}":\n\n```diff\n{hdr}\n{body}```',
        "note": note,
    }


def _assert_doc_region(old_lines, new_lines) -> str:
    """new must be old with exactly ONE inserted line that is the exact
    deterministic @param line, placed immediately before an @return/@export
    anchor. Returns the documented arg name."""
    for i in range(len(new_lines)):
        if new_lines[:i] + new_lines[i + 1:] != old_lines:
            continue
        m = DOC_PARAM_LINE_RE.match(new_lines[i])
        nxt = new_lines[i + 1] if i + 1 < len(new_lines) else ""
        if (m and m.group(3) == DOC_DESCS[m.group(2)]
                and ROXY_ANCHOR_RE.match(nxt)):
            return m.group(2)
    raise AssertionError("doc_sync region is not old + one valid @param line "
                         f"before @return/@export: {old_lines} -> {new_lines}")


def _sig_args_added(ev_old: list[str], ev_new: list[str],
                    names: list[str]) -> bool:
    """True if `ev_new` is `ev_old` as a signature where exactly the
    arguments `names` were added: removing them (one adjacent separator
    comma each) from ev_new must rebuild ev_old, every other argument must
    be identical and in order, and the parameter-list parses must agree."""
    if not ev_old or not ev_new or ev_old == ev_new:
        return False
    if any(l != l.rstrip("\r") for l in ev_old + ev_new):
        return False
    po = _params_from_sig_text("\n".join(ev_old))
    pn = _params_from_sig_text("\n".join(ev_new))
    if po is None or pn is None:
        return False
    if [n for n in pn if n not in names] != [n for n in po if n not in names]:
        return False  # every other argument identical and in order
    if set(pn) - set(po) != set(names) or set(po) & set(names):
        return False
    return _remove_named_args(ev_new, names) == ev_old


def _assert_doc_region_missing(old_lines, new_lines) -> list[str]:
    """missing_param variant: new must be old with 1..MISSING_MAX_PARAMS
    lines APPENDED after the last existing tag, each a deterministic
    "#' @param <name> <desc>" whose description is the name grammar output
    styled after the block's own @param tags. Returns the documented names."""
    k = len(new_lines) - len(old_lines)
    if not (1 <= k <= MISSING_MAX_PARAMS) \
            or new_lines[:len(old_lines)] != old_lines:
        raise AssertionError("doc_sync missing_param region must append the "
                             f"new @param lines: {old_lines} -> {new_lines}")
    cap, per = _doc_style(_roxy_param_descs(old_lines))
    pm = re.match(r"\s*#'", old_lines[-1])
    documented = _roxy_param_names(old_lines)
    names = []
    for l in new_lines[len(old_lines):]:
        m = re.match(r"(\s*#') @param ([.\w]+|\.\.\.) (.+)$", l)
        if not m or (pm and m.group(1) != pm.group(0)):
            raise AssertionError(f"bad @param line: {l!r}")
        name = m.group(2)
        desc = doc_desc_for_name(name)
        if desc is None or name in documented \
                or m.group(3) != _styled_desc(desc, cap, per):
            raise AssertionError(f"non-deterministic @param line: {l!r}")
        names.append(name)
    return names


def _assert_doc_region_pair(old_lines, new_lines, ev_old, ev_new) -> None:
    """version_pair variant: the region is the roxygen @param area of the
    OLD version (plus one shared cursor line above the change); the new lines
    are the maintainer's actual update. The @param-name delta must equal the
    signature-param delta carried by the event."""
    if not old_lines or old_lines[0] != new_lines[0]:
        raise AssertionError("version_pair region must share the cursor line")
    if not (_roxy_param_group_only(old_lines[1:])
            and _roxy_param_group_only(new_lines[1:])):
        raise AssertionError("version_pair region must stay inside the "
                             "@param area of the roxygen block")
    doc_a = _roxy_param_names(old_lines[1:])
    doc_b = _roxy_param_names(new_lines[1:])
    added = set(doc_b) - set(doc_a)
    removed = set(doc_a) - set(doc_b)
    if not (added or removed):
        raise AssertionError("version_pair region changes no @param name")
    pa = _params_from_sig_text("\n".join(ev_old))
    pb = _params_from_sig_text("\n".join(ev_new))
    if pa is None or pb is None:
        raise AssertionError("version_pair event is not a parseable "
                             "signature change")
    if set(pb) - set(pa) != added or set(pa) - set(pb) != removed:
        raise AssertionError("doc delta does not match the signature delta "
                             f"(docs +{sorted(added)}/-{sorted(removed)}; "
                             f"event +{sorted(set(pb)-set(pa))}"
                             f"/-{sorted(set(pa)-set(pb))})")


def _validate_new_family(ex: dict) -> None:
    epath, ev_old, ev_new = parse_event_diff_lines(ex["event_diff"])
    assert epath == ex["path"], "event path mismatch"
    assert ev_old and ev_new and ev_old != ev_new, "event must be a real edit"
    if ex["family"] == "format_propagation":
        assert _fmt_only_edit(ex["region_old"], ex["region_new"]), \
            (ex["region_old"], ex["region_new"])
        assert _fmt_only_edit(ev_old, ev_new), (ev_old, ev_new)
        shared = (set(l.rstrip() for l in ex["region_old"])
                  & set(l.rstrip() for l in ex["region_new"]))
        assert not shared, f"format region shares lines: {shared}"
    elif ex.get("variant") == "missing_param":
        names = _assert_doc_region_missing(ex["region_old"], ex["region_new"])
        assert _sig_args_added(ev_old, ev_new, names), \
            (ev_old, ev_new, names)
        assert ex["cursor_idx"] == len("\n".join(ex["region_old"])), \
            "missing_param cursor must sit at the end of the last tag line"
    elif ex.get("variant") == "version_pair":
        _assert_doc_region_pair(ex["region_old"], ex["region_new"],
                                ev_old, ev_new)
        assert ex["cursor_idx"] == len(ex["region_old"][0]), \
            "version_pair cursor must sit at the end of the first stale line"
    else:  # doc_sync (canonical construction, unchanged)
        arg = _assert_doc_region(ex["region_old"], ex["region_new"])
        assert len(ev_old) == 1 and len(ev_new) == 1, \
            "doc_sync event must be single-line"
        assert _single_insert_before_close(
            ev_old[0], ev_new[0], f", {arg} = {DOC_DEFAULTS[arg]}"), \
            (ev_old[0], ev_new[0])


def noop_exact_score(ex: dict) -> float:
    """1.0 iff the no-op prediction equals the gold exactly. Always 0.0 for
    valid examples; used as the doc_sync calibration gate because a pure
    line-insertion GT scores high under exact_reward's line-F1 (every old
    line matches) - an unavoidable metric artifact for insertions."""
    p = [l.rstrip() for l in ex["region_old"]]
    g = [l.rstrip() for l in ex["region_new"]]
    while p and p[-1] == "":
        p.pop()
    while g and g[-1] == "":
        g.pop()
    return 1.0 if p == g else 0.0


def calibrate_new(n_per_family: int = 30, n_pkgs: int = 40, seed=7) -> dict:
    """Small-scale run for the two new families: construct, validate every
    example (assert), and score no-op baselines. format_propagation's
    line-F1 no-op must be exactly 0.0 (no shared lines). doc_sync is a pure
    insertion so its line-F1 no-op is ~2n/(2n+1) by construction; per the
    agreed convention the gate is the exact-match no-op (0.0) and the
    line-F1 value is reported as a metric artifact."""
    pool = list_packages()
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n_pkgs, len(pool)))
    fmt, fmt_stats = build_format_examples(sample, seed=seed,
                                           per_family_cap=n_per_family,
                                           time_budget_s=300)
    doc, doc_stats = build_doc_sync_examples(sample, seed=seed,
                                             per_family_cap=n_per_family,
                                             time_budget_s=240)
    report = {"packages_sampled": len(sample), "seed": seed,
              "format_stats": fmt_stats, "doc_stats": doc_stats,
              "families": {}}
    for fam, exs in (("format_propagation", fmt), ("doc_sync", doc)):
        assert len(exs) >= 5, f"{fam}: only {len(exs)} examples constructed"
        line_scores, exact_scores = [], []
        for ex in exs:
            validate_example(ex)  # every example must pass exactness checks
            line_scores.append(noop_baseline_score(ex))
            exact_scores.append(noop_exact_score(ex))
            assert exact_reward(ex["region_new"], ex["region_new"]) == 1.0
        rep = {"n_constructed": len(exs),
               "noop_exact_mean": round(sum(exact_scores) / len(exact_scores), 4),
               "all_valid": True}
        if fam == "format_propagation":
            assert max(line_scores) == 0.0, \
                f"{fam}: line-F1 no-op baseline scored {max(line_scores)}"
            rep["noop_baseline_mean"] = round(sum(line_scores) / len(line_scores), 4)
            rep["noop_baseline_max"] = round(max(line_scores), 4)
        else:
            assert max(exact_scores) == 0.0, \
                f"{fam}: exact no-op baseline scored {max(exact_scores)}"
            rep["noop_line_f1_mean"] = round(sum(line_scores) / len(line_scores), 4)
            rep["noop_line_f1_max"] = round(max(line_scores), 4)
            rep["note"] = ("pure-insertion GT: line-F1 no-op is high by "
                           "construction; gate is the exact-match no-op")
        report["families"][fam] = rep
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibrate-new", action="store_true",
                    help="calibrate the format_propagation/doc_sync families")
    ap.add_argument("--new-only", action="store_true",
                    help="build only the new families and merge stats.json")
    ap.add_argument("--new-packages", type=int, default=150,
                    help="package sample size for --new-only")
    ap.add_argument("--n-per-family", type=int, default=30)
    ap.add_argument("--time-budget", type=int, default=780)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--doc-sync-v2", action="store_true",
                    help="grow doc_sync (missing_param + version_pair rows) "
                         "and rewrite doc_sync.jsonl (old rows preserved)")
    ap.add_argument("--doc-sync-target-new", type=int, default=4700,
                    help="new rows to add under --doc-sync-v2")
    ap.add_argument("--doc-sync-workers", type=int, default=6)
    ap.add_argument("--doc-sync-sample-repos", type=int, default=0,
                    help=">0: probe only N git mirrors (yield test)")
    ap.add_argument("--doc-sync-merge-only", action="store_true",
                    help="skip mining; merge the partial file + rewrite")
    args = ap.parse_args()

    if args.calibrate:
        rep = calibrate(n_per_family=args.n_per_family)
        print(json.dumps(rep, indent=1))
        return

    if args.calibrate_new:
        rep = calibrate_new(n_per_family=args.n_per_family)
        print(json.dumps(rep, indent=1))
        return

    if args.doc_sync_v2 or args.doc_sync_merge_only:
        if args.doc_sync_merge_only:
            new_rows = []
            try:
                new_rows = [json.loads(l) for l in
                            (args.out / "doc_sync_v2_partial.jsonl")
                            .read_text().splitlines() if l.strip()]
            except (OSError, ValueError):
                pass
            stats = {"count": len(new_rows), "variants": {
                v: sum(1 for r in new_rows if r.get("variant") == v)
                for v in ("missing_param", "version_pair")}}
        else:
            new_rows, stats = build_doc_sync_v2(
                target_new=args.doc_sync_target_new,
                max_workers=min(args.doc_sync_workers, 6),
                seed=args.seed, out_dir=args.out,
                sample_repos=args.doc_sync_sample_repos)
            print(json.dumps(stats, indent=1))
        rep = merge_doc_sync(new_rows, out_dir=args.out)
        print(json.dumps(rep, indent=1))
        return

    if args.new_only:
        all_pkgs = list_packages()
        rng = random.Random(args.seed)
        n_new = min(args.new_packages, len(all_pkgs))
        sample = rng.sample(all_pkgs, n_new)
        print(f"building new families from {n_new} random packages "
              f"(seed {args.seed}) ...")
        fmt, fmt_stats = build_format_examples(
            sample, seed=args.seed, time_budget_s=args.time_budget,
            verbose=True)
        doc, doc_stats = build_doc_sync_examples(
            sample, seed=args.seed, time_budget_s=args.time_budget,
            verbose=True)
        args.out.mkdir(parents=True, exist_ok=True)
        built = (("format_propagation", fmt), ("doc_sync", doc))
        for fam, exs in built:
            with (args.out / f"{fam}.jsonl").open("w") as fh:
                for ex in exs:
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        # final validation pass + baselines; merge into the existing stats
        rng2 = random.Random(1)
        try:
            stats = json.loads((args.out / "stats.json").read_text())
        except (OSError, ValueError):
            stats = {}
        counts = stats.setdefault("counts", {})
        stats["new_families_build"] = {
            "packages_sampled": n_new, "seed": args.seed,
            "format_propagation": fmt_stats, "doc_sync": doc_stats}
        for fam, exs in built:
            for ex in exs:
                validate_example(ex)
            samp = rng2.sample(exs, min(200, len(exs))) if exs else []
            exact = round(sum(noop_exact_score(e) for e in samp) / len(samp), 4) \
                if samp else None
            linef1 = round(sum(noop_baseline_score(e) for e in samp) / len(samp), 4) \
                if samp else None
            counts[fam] = len(exs)
            stats[f"{fam}_noop_exact_mean(sampled)"] = exact
            stats[f"{fam}_noop_line_f1_mean(sampled)"] = linef1
            stats[f"{fam}_written"] = len(exs)
        (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats, indent=1))
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
