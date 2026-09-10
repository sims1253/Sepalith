#!/usr/bin/env python3
"""TU3: cross-file reference edit-seed pack generator (queue row TU3; the
2026-09-10 design doc, SEED phase).

Task space: the on-disk normalized CRAN mirror
(`/mnt/h/sepalith/normalized/<pkg>/<ver>/<pkg>/R/*.R`, 14,202 packages)
x cross-file families whose target edit is DERIVABLE from the visible
render (backlog rank-4 rule):

  F1 rename_function_xfile   def renamed in file A -> rewrite the call
                             site block in file B                        (cap 700)
  F2 arg_rename_xfile        param renamed in A's signature -> update
                             the named argument at the call in B        (cap 500)
  F3 doc_sync_to_reference   signature in the target render gained a
                             param -> insert the exact @param line in
                             the roxygen block (TU1 fix: deterministic
                             target, fixed desc template family)        (cap 500)
  F4 port_with_reference     sibling file A exhibits idiom X -> port B's
                             block to X (B11 idiom gates + reference-
                             presence check)                            (cap 300)

SEED phase (this file): 500 accepted tasks proportional to caps
(175/125/125/75) + ~800 corrupted-twin eval rows (1 twin pair per accepted
row, capped at 400 pairs; the design's every-5th ratio is the scale-up
setting). F5/F6 stay parked per the design.

Subcommands (the B11 wave-driver shape, reused verbatim where possible):
  mine      cross-file def->use index (tree-sitter R) -> per-family
            candidates with EvidenceRecords, difficulty features,
            2-5 file selection, ref<=120 lines / 4000 chars, target
            prefix<=80 / suffix<=40 -> seeds.jsonl
  author    same backend/queue/resume pattern as B11 (zai glm-5.3 +
            opencode muse-spark free tier, deterministic hash split,
            Breaker, *.done.jsonl / *.stats.json sidecars); every
            region_new must pass the TU3 validator stack (universal
            gates incl. splice + untouched-statement preservation +
            no-op triple, then per-family gates)
  assemble  zeta2 multi-file render under the 16,000-char budget,
            3%-by-package eval split per family (seed 42), corrupted
            twins into eval, license manifest (DESCRIPTION License per
            batch, no-derivatives excluded), seed-gates report
            (validator acceptance + twin separability + glm-probe
            sample) -> nextcoder_r2_cross_v1/{train,eval}.jsonl

Fail-closed: missing API key -> refuse before any request; validator
failure -> row dropped + counted; dedupe collisions logged + counted.
No LLM judges anywhere in the gate stack.

--smoke: synthetic fixtures-only corpus (no NAS, no network), mock
backend, full mine->author->assemble proof into /tmp.

Output default: /mnt/h/sepalith/datasets/nextcoder_r2_cross_v1/
Work dir default: /mnt/h/sepalith/datasets/nextcoder_r2_cross_v1_work/
CPU only (nice -n 19). Python: system python3 (tree_sitter +
tree_sitter_r, the experiments/synthetic-data/cases deps).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # experiments/data-mining
REPO = HERE.parent.parent                              # repo root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "synthetic-data"))
sys.path.insert(0, str(REPO / "experiments" / "eval"))

import build_nextcoder_r as b11                         # noqa: E402  (B11 base)
from build_nextcoder_r import (                         # noqa: E402
    Breaker, SparkFreeAuthorBackend, ZaiAuthorBackend, _append_line,
    _load_done, _load_field, _now, _write_json, assign_backend,
    fragment_clean, is_holdout, load_holdout_audit, n_statements,
    package_split, parse_fragment, struct_key,
)
from tree_sitter import Node                            # noqa: E402
from cases.backends import Backend, BackendError        # noqa: E402
from cases.backends import extract_json_object, strip_fences  # noqa: E402

NAS = Path("/mnt/h/sepalith")
CORPUS = NAS / "normalized"
HOLDOUT_JSON = NAS / "datasets" / "a2" / "holdout_packages.json"
DEFAULT_OUT = NAS / "datasets" / "nextcoder_r2_cross_v1"
DEFAULT_WORK = NAS / "datasets" / "nextcoder_r2_cross_v1_work"

RULE = "tu3-cross@v1"
UPDATED = "\n>>>>>>> UPDATED"

# ---- budgets (design S5) --------------------------------------------------
MAX_CHARS_CROSS = 16_000     # prompt+target budget (2.5x B11; versioned dev)
MAX_BLOCK_LINES = 25         # B11 MAX_BLOCK_LINES, kept
MAX_STATEMENTS_LO, MAX_STATEMENTS_HI = 1, 10
REF_MAX_LINES, REF_MAX_CHARS, REF_WINDOW = 120, 4_000, 20
TARGET_PREFIX_LINES, TARGET_SUFFIX_LINES = 80, 40
FILES_MIN, FILES_MAX = 2, 5
MAX_FILE_BYTES, MAX_FILES_PER_PKG, MAX_PKG_BYTES = 300_000, 40, 2_000_000

SEED_RNG = 42

# ---- seed phase scale (design S7) ----------------------------------------
SEED_TASKS = 500                                     # accepted-task target
FAMILY_CAPS = {                                      # v1 caps (design S7)
    "rename_function_xfile": 700,
    "arg_rename_xfile": 500,
    "doc_sync_to_reference": 500,
    "port_with_reference": 300,
}
TWIN_PAIRS_CAP = 400          # 400 pairs x 2 = ~800 twin eval rows
PROBE_PER_FAMILY = 60         # glm pass@1 band sample rows per family
EXPECT_ACCEPT = 0.45          # B11 acceptance rate (pool sizing only)

FAMILIES = tuple(sorted(FAMILY_CAPS))
PORT_DIRECTIONS = ("tidyverse", "data.table", "base_r", "vectorize")

# ---- per-family instruction phrasings (3 each, B11 convention) -----------
PHRASINGS = {
    "rename_function_xfile": [
        "The function `{old}` in the reference file was renamed to `{new}`. "
        "Update the call site in this block to use the new name; change "
        "nothing else.",
        "A rename refactor renamed `{old}` to `{new}` (see the reference "
        "definition). Rewrite this block's call accordingly.",
        "The definition of `{old}` is now `{new}`. Propagate the rename to "
        "this block, touching only that call.",
    ],
    "arg_rename_xfile": [
        "The parameter `{old}` of `{fn}` was renamed to `{new}` in its "
        "definition (see the reference file). Update the named argument in "
        "this call; keep everything else identical.",
        "Signature change: `{fn}` now spells the argument `{new}` instead of "
        "`{old}`. Fix the named argument in this block.",
        "The argument `{old}` of `{fn}` was renamed to `{new}`. Update this "
        "call site's named argument only.",
    ],
    "doc_sync_to_reference": [
        "The signature of `{fn}` gained the parameter `{p}` (see the event "
        "and the reference call). Insert the roxygen line `{line}` at the "
        "right place; touch nothing else.",
        "Docs lag the code: `{fn}` now takes `{p}`. Add the line `{line}` to "
        "this roxygen block at the correct anchor.",
        "Sync the docs to the reference signature: add `{line}` for the new "
        "parameter `{p}` of `{fn}`.",
    ],
    "port_with_reference": [
        "The reference file shows this package's own `{idiom}` style. Port "
        "this block to the same {idiom} idiom; keep the behavior identical.",
        "Bring this block in line with the package's `{idiom}` precedent "
        "shown in the reference file.",
        "Rewrite this block using the `{idiom}` idiom, matching the "
        "reference file's style.",
    ],
}
IDIOM_NAME = {"tidyverse": "tidyverse", "data.table": "data.table",
              "base_r": "base-R", "vectorize": "vectorized"}

# ---- regex kits (B11 conventions + TU3 additions) ------------------------
TIDY_RE = re.compile(b11.TIDY_RE.pattern)
DT_RE = re.compile(b11.DT_RE.pattern)
FOR_RE = re.compile(b11.FOR_RE.pattern)
CALL_RE = re.compile(b11.CALL_RE.pattern)
APPLY_RE = re.compile(r"\b(vapply|sapply|lapply|apply|mapply|Map|pmax|pmin)"
                      r"\s*\(")

R_RESERVED = {"if", "else", "repeat", "while", "function", "for", "in",
              "next", "break", "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA",
              "NA_integer_", "NA_real_", "NA_character_", "NA_complex_", "T",
              "F", "c", "list", "mean", "sum"}


def word_re(name: str) -> re.Pattern[str]:
    r"""R-symbol-aware boundary regex (design S3.2 F1: `(?<![\w.$])foo(?![\w.$])`)."""
    return re.compile(r"(?<![\w.$])" + re.escape(name) + r"(?![\w.$])")


NO_DERIV_RE = re.compile(r"\bND\b|NoDerivs?|no[-_ ]derivatives?", re.I)
PARAM_TAG_RE = re.compile(r"^#'\s*@param\s+([\w.]+)")
TERMINAL_TAG_RE = re.compile(
    r"^#'\s*@(return|export|exportPattern|examples|keywords|name|rdname|"
    r"seealso|details|section|author|import)\b")

# F3: fixed deterministic description templates (TU2 lesson: identical
# strings teach nothing; pick via content hash)
DOC_TEMPLATES = (
    "passed straight through to `{fn}`; see the reference call.",
    "the `{p}` argument of `{fn}` as shown in the updated signature.",
    "see the `{p}` usage in the reference file.",
    "argument `{p}`; default behaviour unchanged.",
    "carries the new `{p}` value through `{fn}`.",
    "the parameter `{p}` added to the `{fn}` signature.",
    "supplies `{p}`; required by the updated signature.",
)

# F4: which direction a reference file "exhibits", and which statements of
# the target block the port intends to change (preservation exemptions)
REF_IDIOM_RE = {
    "tidyverse": lambda t: bool(TIDY_RE.search(t)),
    "data.table": lambda t: bool(DT_RE.search(t)),
    "base_r": lambda t: bool(CALL_RE.search(t)) and not TIDY_RE.search(t)
    and not DT_RE.search(t),
    "vectorize": lambda t: bool(APPLY_RE.search(t)),
}
PORT_EXEMPT_ALL = {"tidyverse", "data.table"}   # whole-block port intent
PORT_EXEMPT_RE = {
    "vectorize": FOR_RE,
    "base_r": re.compile(b11.TIDY_RE.pattern + "|" + b11.DT_RE.pattern),
}
# reference anchor line = first line exhibiting the TARGET idiom; old-idiom
# markers only exist for the surgical (preservation-checked) directions
REF_ANCHOR_RE = {
    "tidyverse": TIDY_RE, "data.table": DT_RE, "base_r": CALL_RE,
    "vectorize": APPLY_RE,
}
PORT_OLD_RE = {
    "vectorize": FOR_RE,
    "base_r": re.compile(b11.TIDY_RE.pattern + "|" + b11.DT_RE.pattern),
}
OPPOSITE_RE = {
    "tidyverse": lambda t: bool(DT_RE.search(t)),
    "data.table": lambda t: bool(TIDY_RE.search(t)),
    "base_r": lambda t: bool(TIDY_RE.search(t)),
    "vectorize": lambda t: bool(FOR_RE.search(t)),
}

# ---------------------------------------------------------------------------
# AST helpers (B11 machinery + TU3 cross-file additions)
# ---------------------------------------------------------------------------

def _walk(n: Node):
    stack = [n]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def parse_src(text: str):
    """(tree, src_bytes) so node text extraction is GC-safe."""
    src = text.encode("utf-8", "surrogateescape")
    return parse_fragment(text), src


def _text(n: Node, src: bytes) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", "replace")


def _sexp_named(n: Node, src: bytes) -> str:
    """Identifier-INCLUSIVE s-expression. B11's ast_hash is identifier-blind
    (right for dedupe); TU3's no-op gate must SEE a rename, so this variant
    keeps identifier leaf values (design S3.2 F1)."""
    if n.type == "identifier" and n.child_count == 0:
        return f"(id:{_text(n, src)})"
    kids = "".join(_sexp_named(c, src) for c in n.children if c.is_named)
    return f"({n.type}{kids})"


def ast_hash_named(text: str) -> str | None:
    tree, src = parse_src(text)
    if tree.root_node.has_error:
        return None
    parts = [_sexp_named(c, src) for c in tree.root_node.children
             if c.is_named and c.type != "comment"]
    if not parts:
        return None
    return hashlib.sha1("".join(parts).encode()).hexdigest()[:16]


def struct_key_named(text: str) -> tuple:
    """Rename-aware dedupe key: ast-named first, normalized-text fallback."""
    a = ast_hash_named(text)
    return ("astn", a) if a else ("txt", hashlib.sha1(
        "\n".join(l.strip() for l in text.splitlines())
        .encode("utf-8", "surrogateescape")).hexdigest()[:16])


def identifiers(text: str) -> Counter:
    """Multiset of identifier leaf values (untouched-preservation gate)."""
    tree, src = parse_src(text)
    out = Counter()
    for n in _walk(tree.root_node):
        if n.type == "identifier" and n.child_count == 0:
            out[_text(n, src)] += 1
    return out


def top_statements(text: str) -> list[tuple[int, int, str]]:
    """(start_row, end_row, text) of top-level named statements."""
    tree, _ = parse_src(text)
    lines = text.split("\n")
    out = []
    for c in tree.root_node.children:
        if not c.is_named or c.type == "comment":
            continue
        r0, r1 = c.start_point[0], c.end_point[0]
        out.append((r0, r1, "\n".join(lines[r0:r1 + 1])))
    return out


def statement_span(tree, row: int) -> tuple[int, int]:
    """Row span of the top-level statement containing `row`."""
    for c in tree.root_node.children:
        if not c.is_named:
            continue
        if c.start_point[0] <= row <= c.end_point[0]:
            return c.start_point[0], c.end_point[0]
    return row, row


def named_args(call_node: Node, src: bytes) -> list[str]:
    """Named-argument names of a call node (tree-sitter-r `argument`
    children whose second child is the `=` operator)."""
    out = []
    for ch in call_node.children:
        if ch.type != "arguments":
            continue
        for a in ch.children:
            if a.type == "argument" and len(a.children) >= 3 \
                    and a.children[1].type == "=" \
                    and a.children[0].type == "identifier":
                out.append(_text(a.children[0], src))
    return out


def n_positional(call_node: Node) -> int:
    return sum(1 for ch in call_node.children if ch.type == "arguments"
               for a in ch.children
               if a.type == "argument" and not (
                   len(a.children) >= 3 and a.children[1].type == "="))


def calls_in(text: str) -> list[dict]:
    """Identifier-callee calls: {name, row, named, node, src}."""
    tree, src = parse_src(text)
    out = []
    for n in _walk(tree.root_node):
        if n.type != "call" or not n.children or \
                n.children[0].type != "identifier":
            continue
        out.append(dict(name=_text(n.children[0], src), node=n,
                        row=n.start_point[0],
                        named=named_args(n, src)))
    return out

# ---------------------------------------------------------------------------
# license handling (design S4): DESCRIPTION License field per batch,
# no-derivatives excluded deterministically
# ---------------------------------------------------------------------------

def parse_description(path: Path) -> dict:
    """RFC822-ish DESCRIPTION reader (continuation lines indented)."""
    out: dict[str, str] = {}
    key = ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip():
            continue
        if line[:1] in (" ", "\t") and key:
            out[key] += " " + line.strip()
        elif ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            out[key] = val.strip()
    return out


def license_field(desc: dict) -> str:
    return (desc.get("License") or "?").strip()


def is_no_derivatives(license_str: str) -> bool:
    """CC-BY-ND-style / no-derivatives licenses (deterministic field rule)."""
    return bool(NO_DERIV_RE.search(license_str))


# ---------------------------------------------------------------------------
# corpus reading (normalized CRAN mirror) + per-package symbol table
# ---------------------------------------------------------------------------

def latest_pkg_root(pkg_dir: Path) -> Path | None:
    try:
        vers = sorted(d.name for d in pkg_dir.iterdir() if d.is_dir())
    except OSError:
        return None
    if not vers:
        return None
    root = pkg_dir / vers[-1] / pkg_dir.name
    return root if root.is_dir() else None


def file_facts(path: Path, rel: str, pkg: str) -> dict | None:
    """Parse one R file -> defs, calls, idioms, roxygen blocks."""
    try:
        src = path.read_bytes()
    except OSError:
        return None
    if not src or len(src) > MAX_FILE_BYTES:
        return None
    text = src.decode("utf-8", "replace")
    tree, srcb = parse_src(text)
    if tree.root_node.has_error:
        return None                       # parse-clean files only (S3.1.1)
    lines = text.split("\n")
    defs: dict[str, dict] = {}
    for c in tree.root_node.children:
        if c.type != "binary_operator" or len(c.children) < 3:
            continue
        lhs, rhs = c.children[0], c.children[-1]
        if lhs.type != "identifier" or rhs.type != "function_definition":
            continue
        name = _text(lhs, srcb)
        params: list[str] = []
        hdr_end = rhs.start_point[0]
        for ch in rhs.children:
            if ch.type == "parameters":
                hdr_end = ch.end_point[0]
                for p in ch.children:
                    if p.type == "parameter" and p.children \
                            and p.children[0].type == "identifier":
                        params.append(_text(p.children[0], srcb))
        # roxygen block: contiguous `#'` comment rows directly above the def
        r0 = c.start_point[0]
        doc_start = r0
        while doc_start - 1 >= 0 and lines[doc_start - 1].lstrip() \
                .startswith("#'"):
            doc_start -= 1
        roxy = lines[doc_start:r0] if doc_start < r0 else []
        defs[name] = dict(
            name=name, row=r0, end=c.end_point[0], params=params,
            hdr_first=r0, hdr_end=hdr_end, hdr_lines=lines[r0:hdr_end + 1],
            roxygen=roxy, roxy_rows=(doc_start, r0 - 1) if roxy else None)
    calls = []
    for n in _walk(tree.root_node):
        if n.type != "call" or not n.children or \
                n.children[0].type != "identifier":
            continue
        calls.append(dict(name=_text(n.children[0], srcb), node=n,
                          row=n.start_point[0],
                          named=named_args(n, srcb)))
    return dict(pkg=pkg, path=rel, sha=hashlib.sha256(src).hexdigest(),
                text=text, lines=lines, tree=tree, defs=defs, calls=calls,
                tidy=bool(TIDY_RE.search(text)), dt=bool(DT_RE.search(text)))


def analyze_package(pkg: str, root: Path) -> tuple[dict, list[dict]] | None:
    """(description, facts) or None when the package is unusable."""
    desc = parse_description(root / "DESCRIPTION")
    rdir = root / "R"
    if not rdir.is_dir():
        return None
    facts, total = [], 0
    for f in sorted(list(rdir.glob("*.R")) + list(rdir.glob("*.r"))):
        if len(facts) >= MAX_FILES_PER_PKG or total > MAX_PKG_BYTES:
            break
        total += f.stat().st_size if f.exists() else 0
        fa = file_facts(f, f"R/{f.name}", pkg)
        if fa:
            facts.append(fa)
    if len(facts) < 2:                    # cross-file needs >= 2 files (S5)
        return None
    return desc, facts

# ---------------------------------------------------------------------------
# budget trims (design S5: ref <=120 lines / 4000 chars; target prefix <=80,
# suffix <=40; whole file when it fits, else definition +-20 lines)
# ---------------------------------------------------------------------------

def trim_reference(lines: list[str], anchor: int | None = None,
                   max_lines: int = REF_MAX_LINES,
                   max_chars: int = REF_MAX_CHARS,
                   window: int = REF_WINDOW) -> tuple[list[str], int]:
    """(shown_lines, anchor_row_in_shown) — whole file when it fits, else
    the anchor +- window lines, then line/char caps."""
    n = len(lines)
    if n <= max_lines and len("\n".join(lines)) <= max_chars:
        return list(lines), (anchor if anchor is not None else 0)
    a = anchor if anchor is not None else 0
    lo = max(0, a - window)
    hi = min(n, lo + max_lines)
    lo = max(0, hi - max_lines)
    shown = lines[lo:hi]
    while shown and len("\n".join(shown)) > max_chars:
        if a - lo >= len(shown) - (a - lo):      # drop the farther side
            shown.pop(0)
            lo += 1
        else:
            shown.pop()
    return shown, a - lo


def trim_target(lines: list[str], tree, r0: int,
                r1: int) -> tuple[list[str], list[str]]:
    """(prefix, suffix) around region rows [r0, r1] under the cursor-zone
    budget (prefix <=80 lines, suffix <=40), cut at TOP-LEVEL STATEMENT
    boundaries so the rendered window — and therefore the splice gate's
    post-edit text — stays parseable (S3.1.1)."""
    starts, ends = [], []
    for c in tree.root_node.children:
        if c.is_named:
            starts.append(c.start_point[0])
            ends.append(c.end_point[0])
    cand = [st for st in starts
            if st <= r0 and r0 - st <= TARGET_PREFIX_LINES]
    p_start = min(cand) if cand else r0        # longest prefix within budget
    prefix = lines[p_start:r0]
    cand2 = [e for e in ends
             if e > r1 and e - r1 <= TARGET_SUFFIX_LINES]
    s_end = max(cand2) if cand2 else r1        # longest suffix within budget
    suffix = lines[r1 + 1:s_end + 1]
    return prefix, suffix


# ---------------------------------------------------------------------------
# candidate assembly helpers
# ---------------------------------------------------------------------------

def fresh_name(base: str, taken: set[str], start: int = 2) -> str:
    """Deterministic fresh symbol (`base2`, `base3`, ...)."""
    i = start
    while f"{base}{i}" in taken:
        i += 1
    return f"{base}{i}"


def desc_template(key: str, p: str, fn: str) -> str:
    t = DOC_TEMPLATES[int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
                      % len(DOC_TEMPLATES)]
    return t.format(p=p, fn=fn)


def evidence_record(path: str, file_sha: str, shown: list[str],
                    first_line: int, symbol: str, role: str) -> dict:
    """sepalith.edit-context.v1 EvidenceRecord-shaped entry (source kind
    `file`, content identity = file sha) — evals freeze the context exactly."""
    content = "\n".join(shown)
    return dict(source_kind="file", content=content, path=path,
                content_identity=file_sha, symbol=symbol,
                source_range=dict(start=first_line,
                                  end=first_line + len(shown) - 1),
                confidence=1.0,
                metadata=dict(role=role, rule=RULE))


def difficulty_label(region_lines: int, n_args: int) -> tuple[dict, str]:
    """Light deterministic difficulty harness (features + label); feeds the
    seed-gates report, no gate value at author time."""
    feats = dict(region_lines=region_lines, n_args=n_args)
    if region_lines <= 6 and n_args <= 3:
        lab = "simple"
    elif region_lines >= 14 or n_args >= 7:
        lab = "hard"
    else:
        lab = "medium"
    return feats, lab


def make_seed(family: str, pkg: str, lic: str, target: dict, refs: list[dict],
              event_diff: str, meta: dict, twin_material: dict | None) -> dict:
    """One mined candidate -> seed row (seeds.jsonl)."""
    key_src = "|".join([family, pkg, target["path"],
                        target["file_sha"],
                        "\n".join(target["region_old"])] +
                       [r["file_sha"] for r in refs])
    key = "tu3_seed:" + hashlib.sha1(key_src.encode()).hexdigest()[:12]
    sha_a, sha_b = refs[0]["file_sha"], target["file_sha"]
    ev = [evidence_record(r["path"], r["file_sha"], r["render_lines"],
                          r.get("first_line", 0), r.get("symbol", ""),
                          "reference") for r in refs]
    ev.append(evidence_record(target["path"], target["file_sha"],
                              target["prefix"] + target["region_old"]
                              + target["suffix"], 0,
                              meta.get("fn") or meta.get("old") or "",
                              "target"))
    _, lab = difficulty_label(len(target["region_old"]),
                              meta.get("n_args", 0))
    old_text = "\n".join(target["region_old"])
    ref_text = "\n".join(refs[0]["render_lines"])
    seed = dict(
        key=key, family=family, package=pkg, license=lic,
        public_ok=not is_no_derivatives(lic),
        target=target, references=refs, event_diff=event_diff,
        meta=meta, evidence=ev,
        difficulty=dict(region_lines=len(target["region_old"]),
                        n_args=meta.get("n_args", 0), label=lab),
        parent_link=f"{sha_a}|{sha_b}@{RULE}",
        dedupe_key=":".join([family] + list(struct_key_named(ref_text))
                            + list(struct_key_named(old_text))),
        phrasing_pick=int(hashlib.sha1(key.encode()).hexdigest()[:6], 16) % 3,
    )
    if twin_material:
        seed["twin_material"] = twin_material
    # S5 bound: 2 files minimum (target + >=1 reference), 5 cap
    assert FILES_MIN <= 1 + len(refs) <= FILES_MAX, "file-count budget"
    assert len(target["region_old"]) <= MAX_BLOCK_LINES
    return seed


def render_event(lines_before: str, lines_after: str, ctx: str) -> str:
    return f"@@ {ctx} @@\n-{lines_before}\n+{lines_after}"


def task_text(seed: dict, phrasing: int) -> str:
    m = seed["meta"]
    fam = seed["family"]
    if fam == "port_with_reference":
        m = dict(m, idiom=IDIOM_NAME[m["direction"]])
    if fam == "doc_sync_to_reference":
        m = dict(m, line=f"#' @param {m['p']} {m['desc']}")
    return PHRASINGS[fam][phrasing].format(**m)

# ---------------------------------------------------------------------------
# per-family candidate extraction (design S2/S3; every candidate carries
# EvidenceRecords, budget trims, difficulty features, twin material)
# ---------------------------------------------------------------------------

def _target(fact: dict, r0: int, r1: int, cursor: int) -> dict:
    prefix, suffix = trim_target(fact["lines"], fact["tree"], r0, r1)
    return dict(path=fact["path"], file_sha=fact["sha"], prefix=prefix,
                region_old=fact["lines"][r0:r1 + 1], suffix=suffix,
                cursor_idx=cursor)


def _ref(fact: dict, shown: list[str], first_line: int, symbol: str,
         render_lines: list[str] | None = None) -> dict:
    return dict(path=fact["path"], file_sha=fact["sha"], corpus_lines=shown,
                render_lines=render_lines if render_lines is not None
                else list(shown), first_line=first_line, symbol=symbol,
                role="reference")


def _first_match_row(lines: list[str], rx: re.Pattern[str]) -> int:
    for i, l in enumerate(lines):
        if rx.search(l):
            return i
    return 0


def _clean_block_excerpts(facts: list[dict], skip_path: str,
                          pred, limit: int = 2) -> list[list[str]]:
    """Small clean top-level blocks matching `pred` (F4 twin material)."""
    out: list[list[str]] = []
    for f in facts:
        if f["path"] == skip_path or len(out) >= limit:
            continue
        for r0, r1 in b11._fn_blocks(
                f["text"].encode("utf-8", "surrogateescape")):
            block = f["lines"][r0:r1 + 1]
            if pred("\n".join(block)):
                shown, _ = trim_reference(block, (r1 - r0) // 2)
                out.append(shown)
                break
    return out


def extract_f1(pkg: str, lic: str, facts: list[dict]) -> list[dict]:
    """rename_function_xfile: def in A, call-site statement in B. The new
    name is deterministic (old2, bumped past collisions); the render of A
    shows the def post-rename and the event diff documents it."""
    out = []
    def_names = {d["name"] for f in facts for d in f["defs"].values()}
    for fa in facts:
        for d in fa["defs"].values():
            old = d["name"]
            if len(old) < 2 or old in R_RESERVED:
                continue
            # unbound-symbol sweep precondition: the def name occurs in A
            # exactly once (the LHS); recursion/other uses would survive the
            # rename and break the S3.2 sweep gate
            if len(word_re(old).findall(fa["text"])) != 1:
                continue
            shown, first = trim_reference(fa["lines"], d["row"])
            ai = d["row"] - first
            if not (0 <= ai < len(shown)):
                continue
            def_line = shown[ai]
            for fb in facts:
                if fb["path"] == fa["path"]:
                    continue
                calls = [c for c in fb["calls"] if c["name"] == old]
                if not calls:
                    continue
                rows = [i for i, l in enumerate(fb["lines"])
                        if word_re(old).search(l)]
                c = calls[0]
                r0, r1 = statement_span(fb["tree"], c["row"])
                if r1 - r0 + 1 > MAX_BLOCK_LINES:
                    continue
                # every visible occurrence of `old` in B must live inside
                # the region (one-region edit contract)
                if any(not r0 <= r <= r1 for r in rows):
                    continue
                taken = def_names | set(identifiers(fb["text"]))
                new = fresh_name(old, taken)
                wrong = fresh_name(old, taken | {new}, start=3)
                render = list(shown)
                render[ai] = word_re(old).sub(new, def_line)
                wrong_render = list(shown)
                wrong_render[ai] = word_re(old).sub(wrong, def_line)
                ev = render_event(def_line, render[ai],
                                  f"{fa['path']}: renamed {old} -> {new}")
                meta = dict(old=old, new=new, fn=old,
                            n_args=len(c["named"]) + n_positional(c["node"]))
                tgt = _target(fb, r0, r1, c["row"] - r0)
                refs = [_ref(fa, shown, first, old, render)]
                twin = dict(
                    no_op=dict(             # reference unchanged: no rename
                        event_diff="",
                        references=[_ref(fa, shown, first, old)],
                        target_override=None),
                    wrong=dict(             # renamed to a DIFFERENT name
                        event_diff=render_event(
                            def_line, wrong_render[ai],
                            f"{fa['path']}: renamed {old} -> {wrong}"),
                        references=[_ref(fa, shown, first, old, wrong_render)],
                        target_override=None),
                )
                out.append(make_seed("rename_function_xfile", pkg, lic,
                                     tgt, refs, ev, meta, twin))
    return out


def extract_f2(pkg: str, lic: str, facts: list[dict]) -> list[dict]:
    """arg_rename_xfile: param renamed in A's signature -> named arg at the
    call in B. Callee stays fixed (else it is an F1 row, misclassified)."""
    out = []
    for fa in facts:
        for d in fa["defs"].values():
            fn = d["name"]
            if len(fn) < 2 or fn in R_RESERVED or len(d["params"]) < 1:
                continue
            shown, first = trim_reference(fa["lines"], d["row"])
            ai = d["row"] - first
            if not (0 <= ai < len(shown)):
                continue
            hdr_span = d["hdr_end"] - d["hdr_first"]
            if ai + hdr_span >= len(shown):
                continue                    # header must be fully shown
            hdr_rows = list(range(ai, ai + hdr_span + 1))
            for p in d["params"]:
                if not re.fullmatch(r"[a-zA-Z.][\w.]*", p) or p in R_RESERVED:
                    continue
                for fb in facts:
                    if fb["path"] == fa["path"]:
                        continue
                    for c in fb["calls"]:
                        if c["name"] != fn or p not in c["named"]:
                            continue
                        r0, r1 = statement_span(fb["tree"], c["row"])
                        if r1 - r0 + 1 > MAX_BLOCK_LINES:
                            continue
                        tgt = _target(fb, r0, r1, c["row"] - r0)
                        vis = "\n".join(tgt["prefix"] + tgt["suffix"])
                        if word_re(p).search(vis):
                            continue        # `p =` visible outside the region
                        new = fresh_name(p, set(d["params"]))
                        wrong = fresh_name(p, set(d["params"]) | {new},
                                           start=3)
                        before = [shown[i] for i in hdr_rows]
                        after = [word_re(p).sub(new, shown[i])
                                 for i in hdr_rows]
                        render = list(shown)
                        for k, i in enumerate(hdr_rows):
                            render[i] = after[k]
                        ev = render_event(
                            "\n".join(before), "\n".join(after),
                            f"{fa['path']}: {fn}() parameter renamed "
                            f"{p} -> {new}")
                        meta = dict(old=p, new=new, fn=fn,
                                    n_args=len(c["named"])
                                    + n_positional(c["node"]))
                        refs = [_ref(fa, shown, first, fn, render)]
                        wrong_after = [word_re(p).sub(wrong, shown[i])
                                       for i in hdr_rows]
                        wrong_render = list(shown)
                        for k, i in enumerate(hdr_rows):
                            wrong_render[i] = wrong_after[k]
                        twin = dict(
                            no_op=dict(
                                event_diff="",
                                references=[_ref(fa, shown, first, fn)],
                                target_override=None),
                            wrong=dict(
                                event_diff=render_event(
                                    "\n".join(before),
                                    "\n".join(wrong_after),
                                    f"{fa['path']}: {fn}() parameter renamed "
                                    f"{p} -> {wrong}"),
                                references=[_ref(fa, shown, first, fn,
                                                 wrong_render)],
                                target_override=None),
                        )
                        out.append(make_seed("arg_rename_xfile", pkg, lic,
                                             tgt, refs, ev, meta, twin))
    return out


def remove_param(header: str, p: str) -> str | None:
    """Deterministically drop param `p` (with a plain-literal default) from
    a single-line signature; None when the drop is not textually safe
    (defaults with commas/quotes/parens, e.g. c("none","nullify"), are
    rejected rather than corrupted)."""
    wr = word_re(p)
    # a consumed default must END at a `,` or `)` (lookahead), else the
    # regex would eat a partial literal like `=c` and leave `(0.05,` behind
    deflt = r"(?:\s*=\s*[\w.$:+\-*/ ]+)?(?=\s*[,)])"
    m = re.search(r",\s*" + wr.pattern + deflt, header)
    if not m:
        m = re.search(wr.pattern + deflt + r"\s*,\s*", header)
    if not m:
        return None
    out = header[:m.start()] + header[m.end():]
    return None if word_re(p).search(out) else out


def extract_f3(pkg: str, lic: str, facts: list[dict]) -> list[dict]:
    """doc_sync_to_reference: roxygen block (target, file B) lags the def's
    signature; the usage/reference file A plus the event diff make the
    @param line a deterministic function of the visible render (TU1 fix)."""
    out = []
    for fb in facts:
        for d in fb["defs"].values():
            fn = d["name"]
            roxy = d["roxygen"]
            if len(fn) < 2 or fn in R_RESERVED or not roxy \
                    or len(roxy) < 3 or d["hdr_first"] != d["hdr_end"]:
                continue
            documented = set()
            for l in roxy:
                m = PARAM_TAG_RE.match(l.lstrip())
                if m:
                    documented.add(m.group(1))
            if not documented:
                continue
            missing = [p for p in d["params"]
                       if re.fullmatch(r"[a-zA-Z.][\w.]*", p)
                       and p not in documented and p not in R_RESERVED]
            if not missing:
                continue
            p = missing[0]
            hdr_line = fb["lines"][d["hdr_first"]]
            before = remove_param(hdr_line, p)
            if before is None:
                continue
            balanced = before + "}" * max(
                0, before.count("{") - before.count("}"))
            if not fragment_clean(balanced):
                continue
            # usage reference: another file calling fn with `p =` named
            ref_fact = None
            for fa in facts:
                if fa["path"] != fb["path"] and word_re(fn).search(fa["text"]) \
                        and re.search(word_re(p).pattern + r"\s*=", fa["text"]):
                    ref_fact = fa
                    break
            if ref_fact is None:
                continue
            r0, r1 = d["roxy_rows"]
            if r1 - r0 + 1 > MAX_BLOCK_LINES:
                continue
            region = fb["lines"][r0:r1 + 1]
            tgt = _target(fb, r0, r1, 0)
            # def signature must stay visible in the target suffix
            hdr_in_suffix = d["hdr_first"] - (r1 + 1)
            if not 0 <= hdr_in_suffix < len(tgt["suffix"]):
                continue
            last_param = max(i for i, l in enumerate(region)
                             if PARAM_TAG_RE.match(l.lstrip()))
            term = next((i for i in range(last_param + 1, len(region))
                         if TERMINAL_TAG_RE.match(region[i].lstrip())),
                        len(region))
            anchor = last_param + 1
            if anchor > term:
                continue
            key = f"{pkg}|{fb['path']}|{fn}|{p}"
            desc = desc_template(key, p, fn)
            meta = dict(fn=fn, p=p, desc=desc, anchor_idx=anchor,
                        terminal_idx=term, old=p, new=p,
                        n_args=len(d["params"]))
            ev = render_event(before, hdr_line,
                              f"{fn}() signature in {fb['path']}")
            ref_shown, ref_first = trim_reference(
                ref_fact["lines"],
                _first_match_row(ref_fact["lines"], word_re(fn)))
            refs = [_ref(ref_fact, ref_shown, ref_first, fn)]
            # twin (i): docs already synced -> no edit
            noop_region = region[:anchor] + [f"#' @param {p} {desc}"] \
                + region[anchor:]
            noop_tgt = dict(tgt, region_old=noop_region, cursor_idx=anchor)
            # twin (ii): the visible signature gained a DIFFERENT param
            # and does NOT contain p -> the task premise is false
            twin_wrong = None
            others = [q for q in d["params"] if q != p
                      and re.fullmatch(r"[a-zA-Z.][\w.]*", q)]
            if others:
                q = others[0]
                hi = hdr_in_suffix
                wrong_suffix = list(tgt["suffix"])
                wrong_suffix[hi] = before      # signature without p
                q_before = remove_param(before, q)
                wrong_ev = "" if q_before is None else render_event(
                    q_before, before,
                    f"{fn}() signature in {fb['path']}")
                twin_wrong = dict(
                    event_diff=wrong_ev, references=refs,
                    target_override=dict(tgt, suffix=wrong_suffix))
            twin = dict(
                no_op=dict(event_diff=ev, references=refs,
                           target_override=noop_tgt),
                wrong=twin_wrong or dict(
                    event_diff="", references=refs, target_override=None),
            )
            out.append(make_seed("doc_sync_to_reference", pkg, lic,
                                 tgt, refs, ev, meta, twin))
    return out


def extract_f4(pkg: str, lic: str, facts: list[dict]) -> list[dict]:
    """port_with_reference: sibling file A exhibits the target idiom; a
    block in B still uses the other idiom. B11 idiom gates apply at author
    time; the reference-presence check is here (no teaching from a
    reference that does not use it)."""
    out = []
    for direction in PORT_DIRECTIONS:
        refA = next((f for f in facts
                     if REF_IDIOM_RE[direction](f["text"])), None)
        if refA is None:
            continue
        anchor_rx = REF_ANCHOR_RE[direction]
        shown, first = trim_reference(
            refA["lines"], _first_match_row(refA["lines"], anchor_rx))
        refs = [_ref(refA, shown, first, "idiom")]
        for fb in facts:
            if fb["path"] == refA["path"]:
                continue
            for r0, r1 in b11._fn_blocks(
                    fb["text"].encode("utf-8", "surrogateescape")):
                block = fb["lines"][r0:r1 + 1]
                text = "\n".join(block)
                if direction not in b11.tag_applicability(text):
                    continue
                old_rx = PORT_OLD_RE.get(direction)
                idiom_lines = [i for i, l in enumerate(block)
                               if old_rx and old_rx.search(l)] or [0]
                tgt = _target(fb, r0, r1, idiom_lines[0])
                meta = dict(direction=direction,
                            exempt_all=direction in PORT_EXEMPT_ALL,
                            n_args=0, old=direction, new=direction,
                            idiom_lines=idiom_lines)
                noop_pred = (lambda d: lambda t: not REF_IDIOM_RE[d](t))(
                    direction)
                wrong_pred = (lambda d: lambda t: bool(OPPOSITE_RE[d](t)))(
                    direction)
                noop_refs = _clean_block_excerpts(
                    facts, fb["path"], noop_pred, limit=1)
                wrong_refs = _clean_block_excerpts(
                    facts, fb["path"], wrong_pred, limit=1)
                twin = None
                if noop_refs:
                    twin = dict(no_op=dict(
                        event_diff="",
                        references=[dict(refs[0],
                                         render_lines=noop_refs[0])],
                        target_override=None),
                        wrong=dict(
                            event_diff="",
                            references=[dict(
                                refs[0], render_lines=wrong_refs[0])]
                            if wrong_refs else None,
                            target_override=None))
                out.append(make_seed("port_with_reference", pkg, lic,
                                     tgt, refs, "", meta, twin))
    return out


EXTRACTORS = {
    "rename_function_xfile": extract_f1,
    "arg_rename_xfile": extract_f2,
    "doc_sync_to_reference": extract_f3,
    "port_with_reference": extract_f4,
}


def family_targets(total: int, caps: dict | None = None) -> dict[str, int]:
    """Largest-remainder split of `total` tasks across family caps
    (seed: 500 across 700/500/500/300 -> 175/125/125/75)."""
    caps = caps or FAMILY_CAPS
    total_cap = sum(caps.values())
    raw = {f: total * c / total_cap for f, c in caps.items()}
    base = {f: int(v) for f, v in raw.items()}
    for f in sorted(caps, key=lambda f: (-(raw[f] - base[f]), f))[
            :total - sum(base.values())]:
        base[f] += 1
    return base


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------

def cmd_mine(args) -> dict:
    t0 = time.time()
    audit = load_holdout_audit(Path(args.holdout_json))
    targets = family_targets(args.seeds)
    pool_n = {f: max(1, round(targets[f] * args.pool_mult))
              for f in FAMILIES}
    rng = random.Random(args.seed)
    stats = Counter()
    cands: dict[str, list[dict]] = {f: [] for f in FAMILIES}
    licenses: dict[str, dict] = {}
    corpus = Path(args.corpus)
    pkg_dirs = sorted(d for d in corpus.iterdir() if d.is_dir()) \
        if corpus.is_dir() else []
    stats["packages_found"] = len(pkg_dirs)
    if args.max_packages > 0:
        pkg_dirs = pkg_dirs[:args.max_packages]

    def pools_full() -> bool:
        return all(len(cands[f]) >= pool_n[f] for f in FAMILIES)

    stopped_early = False
    for pkg_dir in pkg_dirs:
        if pools_full():
            stopped_early = True
            break
        pkg = pkg_dir.name
        if is_holdout(pkg) or pkg in audit:
            stats["holdout_blocked"] += 1
            continue
        root = latest_pkg_root(pkg_dir)
        if root is None:
            stats["no_r_layout"] += 1
            continue
        desc = parse_description(root / "DESCRIPTION")
        lic = license_field(desc)
        nd = is_no_derivatives(lic)
        licenses[pkg] = dict(license=lic, excluded_no_derivatives=nd,
                             rows=0)
        if nd and args.exclude_no_derivatives:
            stats["license_excluded"] += 1
            continue
        res = analyze_package(pkg, root)
        if res is None:
            stats["pkg_unusable"] += 1
            continue
        _, facts = res
        stats["packages_used"] += 1
        for fam in FAMILIES:
            try:
                seeds_f = EXTRACTORS[fam](pkg, lic, facts)
            except Exception as e:            # mine must never die mid-scan
                stats[f"extract_error:{fam}"] += 1
                print(f"  [mine] extract {fam} error in {pkg}: {e!r}",
                      flush=True)
                continue
            rng.shuffle(seeds_f)
            cap = args.per_package_cap
            for s in seeds_f[:cap]:
                cands[fam].append(s)
            stats[f"cands:{fam}"] += len(seeds_f)

    # structural dedupe + per-family sampling to the pool target
    seeds: list[dict] = []
    for fam in FAMILIES:
        seen: set[str] = set()
        uniq = []
        for s in cands[fam]:
            if s["dedupe_key"] in seen:
                stats["dedupe_collisions"] += 1
                continue
            seen.add(s["dedupe_key"])
            uniq.append(s)
        rng.shuffle(uniq)
        picked = uniq[:pool_n[fam]]
        seeds.extend(picked)
        stats[f"seeds:{fam}"] = len(picked)
        stats[f"unique:{fam}"] = len(uniq)
    rng.shuffle(seeds)
    for s in seeds:
        licenses.setdefault(s["package"], dict(
            license=s["license"], excluded_no_derivatives=False, rows=0))
        licenses[s["package"]]["rows"] += 1
    stats["seeds_total"] = len(seeds)
    stats["packages_sampled"] = len({s["package"] for s in seeds})
    stats["difficulty"] = dict(Counter(s["difficulty"]["label"]
                                       for s in seeds))
    stats["twin_material"] = dict(Counter(
        k for s in seeds for k in (s.get("twin_material") or {})
        if s["twin_material"][k] is not None))

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "seeds.jsonl", "w") as fh:
        for s in seeds:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    _write_json(work / "licenses.json", dict(
        ts=_now(), rule=RULE,
        exclude_no_derivatives=bool(args.exclude_no_derivatives),
        packages=dict(sorted(licenses.items()))))
    _write_json(work / "seeds.jsonl.stats.json", dict(
        ts=_now(), rule=RULE, seed_rng=args.seed,
        family_targets=targets, pool_targets=pool_n,
        stopped_early=stopped_early, stats=dict(stats),
        elapsed_s=round(time.time() - t0, 1)))
    print(f"[mine] {len(seeds)} seeds from {stats['packages_used']} "
          f"packages (holdout-blocked {stats['holdout_blocked']}, "
          f"license-excluded {stats['license_excluded']}, "
          f"dedupe-dropped {stats['dedupe_collisions']}); "
          f"per-family={[ (f, stats[f'seeds:{f}']) for f in FAMILIES ]}",
          flush=True)
    return dict(stats=dict(stats), family_targets=targets)

# ---------------------------------------------------------------------------
# validators (design S3: universal gates + per-family gates; all
# deterministic, no LLM anywhere in the acceptance path)
# ---------------------------------------------------------------------------

def _norm_text(text: str) -> str:
    return " ".join(text.split())


def splice_text(seed: dict, region_new: list[str]) -> str:
    """region_new spliced into the target file at the region position."""
    t = seed["target"]
    return "\n".join(t["prefix"] + list(region_new) + t["suffix"])


def _preserve_untouched(old_text: str, new_text: str,
                        exempt) -> tuple[bool, str]:
    """S3.1.3: every top-level statement the family does not intend to
    change must survive byte-identical (rstrip), in order, count stable."""
    old_st = top_statements(old_text)
    new_st = top_statements(new_text)
    if len(old_st) != len(new_st):
        return False, "statement_count_changed"
    kept = [s.rstrip() for _, _, s in new_st]
    j = 0
    for _, _, s in old_st:
        if exempt(s):
            continue
        want = s.rstrip()
        while j < len(kept) and kept[j] != want:
            j += 1
        if j >= len(kept):
            return False, "untouched_statement_modified"
        j += 1
    return True, ""


def _ident_delta(old_text: str, new_text: str):
    co, cn = identifiers(old_text), identifiers(new_text)
    return cn - co, co - cn        # (added, removed) Counters


def validate_region_new(region_new: list[str], seed: dict) -> tuple[bool, str]:
    """Universal gates first (S3.1.1-3.1.5), then the family gate."""
    text = "\n".join(region_new)
    if not any(l.strip() for l in region_new):
        return False, "empty_region_new"
    if len(region_new) > MAX_BLOCK_LINES:
        return False, f"over_{MAX_BLOCK_LINES}_lines"
    if len(text) > 2500:
        return False, "over_2500_chars"
    if not fragment_clean(text):
        return False, "not_clean_r"
    if "<filename>" in text or "```" in text:
        return False, "edits_reference_file"      # S3.1.4 immutability
    old_text = "\n".join(seed["target"]["region_old"])
    if seed["family"] != "doc_sync_to_reference":
        n = n_statements(text)
        if not MAX_STATEMENTS_LO <= n <= MAX_STATEMENTS_HI:
            return False, f"statements_{n}"
    # no-op triple (B11 semantics; the ast arm is identifier-INCLUSIVE here
    # so a rename is a real edit and a reformat still no-ops)
    if text == old_text:
        return False, "no_op_text"
    if struct_key_named(text) == struct_key_named(old_text):
        return False, "no_op_ast"
    if _norm_text(text) == _norm_text(old_text):
        return False, "no_op_normalized_text"
    if not fragment_clean(splice_text(seed, region_new)):
        return False, "splice_not_clean"
    return FAMILY_VALIDATORS[seed["family"]](region_new, seed, text,
                                             old_text)


def _v_rename(region_new, seed, text, old_text) -> tuple[bool, str]:
    m = seed["meta"]
    old, new = m["old"], m["new"]
    if word_re(old).search(text):
        return False, "old_symbol_still_present"
    if not word_re(new).search(text):
        return False, "new_symbol_missing"
    if struct_key(text) != struct_key(old_text):
        return False, "structure_changed"          # callee shape / arg count
    added, removed = _ident_delta(old_text, text)
    if set(added) != {new} or set(removed) != {old}:
        return False, "other_identifiers_changed"  # S3.2 F1: NO other ids
    return _preserve_untouched(old_text, text,
                               lambda s: bool(word_re(old).search(s)))


def _v_arg_rename(region_new, seed, text, old_text) -> tuple[bool, str]:
    m = seed["meta"]
    old, new, fn = m["old"], m["new"], m["fn"]
    if re.search(word_re(old).pattern + r"\s*=", text):
        return False, "old_arg_still_present"
    if not re.search(word_re(new).pattern + r"\s*=", text):
        return False, "new_arg_missing"
    # the callee must stay fixed (else it is an F1 row, misclassified)
    if not word_re(fn).search(text):
        return False, "callee_missing"
    # named args: same order, only old->new; positional count unchanged
    # (checked BEFORE the generic structure gate so the specific reason
    # stays reachable)
    old_call = next((c for c in calls_in(old_text) if c["name"] == fn), None)
    new_call = next((c for c in calls_in(text) if c["name"] == fn), None)
    if old_call is None or new_call is None:
        return False, "call_not_found"
    want = [new if nm == old else nm for nm in old_call["named"]]
    if new_call["named"] != want:
        return False, "named_args_reordered_or_positional"
    if n_positional(old_call["node"]) != n_positional(new_call["node"]):
        return False, "positional_count_changed"
    if struct_key(text) != struct_key(old_text):
        return False, "structure_changed"
    added, removed = _ident_delta(old_text, text)
    if set(added) != {new} or set(removed) != {old}:
        return False, "other_identifiers_changed"
    return _preserve_untouched(
        old_text, text,
        lambda s: bool(word_re(fn).search(s)) and bool(word_re(old).search(s)))


def _v_doc_sync(region_new, seed, text, old_text) -> tuple[bool, str]:
    m = seed["meta"]
    expected = f"#' @param {m['p']} {m['desc']}"
    old_lines = seed["target"]["region_old"]
    if len(region_new) != len(old_lines) + 1:
        return False, "f3_not_single_line_insertion"
    ins = None
    for i in range(len(region_new)):
        rest = region_new[:i] + region_new[i + 1:]
        if [l.rstrip() for l in rest] == [l.rstrip() for l in old_lines]:
            ins = i
            break
    if ins is None:
        return False, "f3_other_lines_touched"
    if region_new[ins].rstrip() != expected:
        return False, "f3_line_mismatch"            # deterministic target
    if not m["anchor_idx"] <= ins <= m["terminal_idx"]:
        return False, "f3_anchor_violation"         # before @return/@export
    return True, ""


def _v_port(region_new, seed, text, old_text) -> tuple[bool, str]:
    d = seed["meta"]["direction"]
    tidy = bool(TIDY_RE.search(text))
    dt = bool(DT_RE.search(text))
    looping = bool(FOR_RE.search(text))
    # B11 idiom gates verbatim per direction
    if d == "tidyverse" and not tidy:
        return False, "idiom_tidyverse_missing"
    if d == "data.table" and not dt:
        return False, "idiom_data_table_missing"
    if d == "vectorize" and looping:
        return False, "idiom_vectorize_still_looping"
    if d == "base_r" and (tidy or dt):
        return False, "idiom_base_r_contaminated"
    # reference-presence check happened at mine; whole-block port intent
    # directions rewrite their statements freely
    if seed["meta"].get("exempt_all"):
        return True, ""
    rx = PORT_EXEMPT_RE[d]
    return _preserve_untouched(old_text, text,
                               lambda s: bool(rx.search(s)))


FAMILY_VALIDATORS = {
    "rename_function_xfile": _v_rename,
    "arg_rename_xfile": _v_arg_rename,
    "doc_sync_to_reference": _v_doc_sync,
    "port_with_reference": _v_port,
}


# ---------------------------------------------------------------------------
# corrupted twins (design S3.1.7: unchanged-reference no-op + wrong-
# reference; both must FAIL the family gate and be answerable "no edit")
# ---------------------------------------------------------------------------

TWIN_KINDS = ("no_op_reference", "wrong_reference")


def build_twin_exs(seed: dict) -> list[dict]:
    """Eval-only twin render inputs from the seed's twin material; None
    entries (family lacks material for that twin) are skipped."""
    tm = seed.get("twin_material") or {}
    out = []
    for kind, key in zip(TWIN_KINDS, ("no_op", "wrong"), strict=True):
        t = tm.get(key)
        if not t:
            continue
        target = t.get("target_override") or seed["target"]
        out.append(dict(kind=kind, event_diff=t["event_diff"],
                        references=t["references"]
                        or seed["references"], target=target))
    return out


def twin_gate_check(seed: dict, twin: dict) -> bool:
    """The twin's CORRECT answer (region unchanged) must FAIL the family
    gate — that failure is what makes the row answerable `no edit` and the
    blind edit a measurable false-propose."""
    ok, _why = validate_region_new(list(twin["target"]["region_old"]),
                                   dict(seed, target=twin["target"]))
    return not ok


# ---------------------------------------------------------------------------
# multi-file zeta2 render + pack row (design S5)
# ---------------------------------------------------------------------------

CURSOR2 = "<|user_cursor|>"


def _with_cursor(lines: list[str], idx: int) -> list[str]:
    out = list(lines)
    if 0 <= idx < len(out):
        out[idx] = out[idx] + CURSOR2
    return out


def render_zeta2_cross(ex: dict) -> str:
    """zeta2 marker set EXACTLY as shipped; the multi-file delta is
    additive read-only <filename> sections between edit_history and the
    writable target file (last, nearest cursor)."""
    ev_lines = [l for l in (ex.get("event_diff") or "").splitlines()]
    while ev_lines and not ev_lines[0].strip():
        ev_lines.pop(0)
    parts = ["<[fim-suffix]>"] + list(ex["suffix"]) \
        + ["<[fim-prefix]><filename>edit_history"]
    if ev_lines:
        parts += ev_lines + [""]
    for ref in ex.get("references") or []:
        parts += [f"<filename>{ref['path']}"] \
            + list(ref["render_lines"]) + [""]
    parts += [f"<filename>{ex['path']}"] + list(ex["prefix"]) \
        + ["<<<<<<< CURRENT"]
    parts += _with_cursor(ex["region_old"], ex["cursor_idx"])
    parts += ["=======", "<[fim-middle]>"]
    return "\n".join(parts)


def edit_row_cross(ex: dict, family: str, pkg: str,
                   expected_noop: bool = False,
                   twin_kind: str | None = None):
    """Pack row under the 16,000-char cross-file budget."""
    prompt = render_zeta2_cross(ex)
    target = "\n".join(ex["region_new"]).rstrip() + UPDATED
    if len(prompt) + len(target) > MAX_CHARS_CROSS:
        return None, f"over_{MAX_CHARS_CROSS}"
    row = dict(text=prompt + target, prompt=prompt, target=target,
               family=family, package_or_repo=pkg, has_types=False,
               expected_noop=expected_noop, twin_kind=twin_kind)
    return row, ""

# ---------------------------------------------------------------------------
# task space + authoring (B11 wave-driver pattern: backends, hash split,
# Breaker, resume sidecars, warm-started dedupe)
# ---------------------------------------------------------------------------

AUTHOR_BACKENDS = {
    "zai": ZaiAuthorBackend,
    "opencode-spark-free": SparkFreeAuthorBackend,
    "mock": None,          # bound below (MockAuthorBackend)
}


def preflight_backend(name: str):
    """Fail-closed: refuse before any request when the key is missing."""
    cls = AUTHOR_BACKENDS[name]
    env_key = getattr(cls, "env_key", None)
    if env_key and not os.environ.get(env_key):
        sys.exit(f"refusing: backend {name!r} needs ${env_key}; "
                 f"export it or use --backends mock")
    return cls()


PROMPT_TMPL_CROSS = """You are editing R code across files of one package. The render shows a read-only reference file, then the WRITABLE target file. Apply EXACTLY this task to the CURRENT block of the target file and return the edited block.

Task: {task}

Rules:
- return ONLY the lines that replace the block in the target file
- never edit the reference file; the edit belongs to the target file only
- keep every line you do not need to change byte-identical
- respond ONLY with a JSON object: {{"region_new": ["line 1", "line 2"], "note": "one short sentence"}}

REFERENCE FILE ({ref_path}, read-only):
```r
{ref_text}
```

TARGET FILE ({target_path}) CURRENT BLOCK:
```r
{code}
```"""


def build_prompt(seed: dict, phrasing: int) -> str:
    t = seed["target"]
    ref = seed["references"][0]
    return PROMPT_TMPL_CROSS.format(
        task=task_text(seed, phrasing), ref_path=ref["path"],
        ref_text="\n".join(ref["render_lines"]), target_path=t["path"],
        code="\n".join(t["region_old"]))


def _mock_port(lines: list[str], direction: str, idiom_lines: list[int],
               h: int) -> list[str]:
    """Deterministic idiom rewrite of the marked spans (mock/smoke/tests):
    surgical for the preservation-checked directions, whole-block single
    statement for the exempt-all port directions."""
    if direction == "vectorize":
        out = list(lines)
        for i in sorted(set(idiom_lines), reverse=True):
            if not FOR_RE.search(out[i]):
                continue
            j, depth, started = i, 0, False
            while j < len(out):
                depth += out[j].count("{") - out[j].count("}")
                if "{" in out[j]:
                    started = True
                if started and depth <= 0:
                    break
                j += 1
            out = out[:i] + [f"out_mock_{h % 89} <- vapply("
                             f"xs, function(z) z + {h % 7}, numeric(1))"] \
                + out[j + 1:]
        return out
    if direction == "base_r":
        out = list(lines)
        for i in idiom_lines:
            if TIDY_RE.search(out[i]) or DT_RE.search(out[i]):
                out[i] = f"res_mock_{h % 89} <- as.numeric(x)"
        return out
    line = {
        "tidyverse": f"res_mock_{h % 89} <- dplyr::mutate(d, v = {h % 13})",
        "data.table": f"dt_mock_{h % 89} <- data.table::as.data.table("
                      f"x)[, v := {h % 13}]",
    }[direction]
    return [line]


class MockAuthorBackend(Backend):
    """Deterministic no-network author (smoke/tests). Derives a valid
    region_new from the machine-readable hint appended for mock runs only.
    Env knobs: TU3_MOCK_FAIL_EVERY (unparseable every Nth request),
    TU3_MOCK_INVALID (schema-ok but gate-failing)."""
    name = "mock"
    model = "mock-0"

    def __init__(self):
        super().__init__()
        self._n = 0

    def _complete_once(self, prompt: str) -> str:
        self._n += 1
        self._bump("ok", 0.0)
        fe = int(os.environ.get("TU3_MOCK_FAIL_EVERY", "0") or 0)
        if fe and self._n % fe == 0:
            return "not json, sorry"
        hint = json.loads(prompt.rsplit("#MOCK-HINT ", 1)[1])
        m = hint["meta"]
        lines = list(hint["region_old"])
        fam = hint["family"]
        h = int(hashlib.sha1(hint["task_key"].encode()).hexdigest()[:6], 16)
        if fam == "rename_function_xfile":
            lines = [word_re(m["old"]).sub(m["new"], l) for l in lines]
        elif fam == "arg_rename_xfile":
            lines = [re.sub(word_re(m["old"]).pattern + r"\s*=\s*",
                            f"{m['new']} = ", l) for l in lines]
        elif fam == "doc_sync_to_reference":
            a = m["anchor_idx"]
            lines = lines[:a] + [hint["expected_line"]] + lines[a:]
        else:
            lines = _mock_port(lines, m["direction"],
                               m.get("idiom_lines") or [0], h)
        if os.environ.get("TU3_MOCK_INVALID"):
            lines = lines + ["this is << not R"]
        return json.dumps({"region_new": lines, "note": f"mock {fam}"})


AUTHOR_BACKENDS["mock"] = MockAuthorBackend


def build_task_list(seeds: list[dict], n_phrasings: int) -> list[dict]:
    tasks = []
    for s in seeds:
        phrs = [s["phrasing_pick"]] if n_phrasings <= 1 \
            else list(range(n_phrasings))
        for p in phrs:
            tasks.append(dict(
                task_key=f"{s['key']}:{s['family']}:p{p}", seed=s,
                phrasing=p, note=PHRASINGS[s["family"]][p]))
    return tasks


def corpus_signature_set(seeds: list[dict]) -> set[tuple]:
    """Rename-aware AST dedupe reference: every seed's region_old and every
    rendered reference (authored output must not duplicate corpus content)."""
    out: set[tuple] = set()
    for s in seeds:
        out.add(struct_key_named("\n".join(s["target"]["region_old"])))
        for r in s["references"]:
            out.add(struct_key_named("\n".join(r["render_lines"])))
    return out


def author_row(task: dict, region_new: list[str], backend_name: str,
               model: str, prompt: str) -> dict:
    s = task["seed"]
    t = s["target"]
    chash = hashlib.sha1(
        f"{RULE}\x00{task['task_key']}\x00{backend_name}".encode()
    ).hexdigest()
    return dict(
        family=s["family"], key=task["task_key"], phrasing=task["phrasing"],
        note=task["note"], package=s["package"], path=t["path"],
        license=s["license"], prefix=t["prefix"], region_old=t["region_old"],
        region_new=region_new, suffix=t["suffix"],
        cursor_idx=t["cursor_idx"], event_diff=s["event_diff"],
        references=s["references"], evidence=s["evidence"], meta=s["meta"],
        difficulty=s["difficulty"],
        case="build_tu3", backend=backend_name, model=model,
        generated_at=_now(), full_prompt=prompt,
        parent_link=s["parent_link"], seed_key=s["key"],
        transform_id=f"tu3/{s['family']}@1",
        derivation=dict(rule_id=f"tu3/{s['family']}", rule_version=1,
                        phrasing=task["phrasing"], seed=s["key"]),
        content_hash=chash, determinism="D3 author-LLM (gated)")


def args_backends(args) -> list[str]:
    names = [b.strip() for b in args.backends.split(",") if b.strip()]
    return names or ["mock"]


def cmd_author(args) -> int:
    work = Path(args.work)
    seeds_path = work / "seeds.jsonl"
    if not seeds_path.exists():
        sys.exit(f"no seed pool at {seeds_path}; run `mine` first")
    seeds = [json.loads(l) for l in seeds_path.read_text().splitlines()
             if l.strip()]
    backend_names = args_backends(args)
    for b in backend_names:
        if b not in AUTHOR_BACKENDS or AUTHOR_BACKENDS[b] is None:
            sys.exit(f"unknown backend {b!r}; known: {sorted(AUTHOR_BACKENDS)}")
    backends = {b: preflight_backend(b) for b in backend_names}

    out_path = work / "authored.jsonl"
    done_path = Path(str(out_path) + ".done.jsonl")
    done = _load_done(done_path)
    hashes = _load_field(out_path, "content_hash")

    tasks = build_task_list(seeds, args.phrasings)
    if args.max > 0:
        tasks = tasks[:args.max]
    pending = [t for t in tasks
               if assign_backend(t["task_key"], backend_names) in backends
               and t["task_key"] not in done]
    print(f"[author] backends={backend_names} pool={len(tasks)} "
          f"pending={len(pending)} done_keys={len(done)}", flush=True)

    corpus_keys = corpus_signature_set(seeds)
    stats = dict(attempted=0, accepted=0, dropped=Counter(),
                 dups_corpus=0, dups_batch=0, backend_error=0,
                 per_family=Counter(), attempted_family=Counter(),
                 per_backend=Counter(), per_phrasing=Counter())
    stats_lock, batch_lock = threading.Lock(), threading.Lock()
    batch_keys: set[tuple] = set()
    if out_path.exists():               # warm-start dedupe (resume safety)
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                prior = json.loads(line)
                batch_keys.add((prior["family"],
                                struct_key_named("\n".join(
                                    prior.get("region_new") or [])),
                                struct_key_named("\n".join(
                                    (prior.get("references") or [{}])[0]
                                    .get("render_lines") or []))))
            except Exception:
                continue
    breaker = Breaker()
    rows_written = [0]

    def process(task: dict) -> dict:
        s = task["seed"]
        bname = assign_backend(task["task_key"], backend_names)
        backend = backends[bname]
        prompt = build_prompt(s, task["phrasing"])
        if isinstance(backend, MockAuthorBackend):
            m = s["meta"]
            expected = (f"#' @param {m['p']} {m['desc']}"
                        if s["family"] == "doc_sync_to_reference" else None)
            hint = dict(task_key=task["task_key"], family=s["family"],
                        meta=m, region_old=s["target"]["region_old"],
                        expected_line=expected)
            prompt += "\n#MOCK-HINT " + json.dumps(hint)
        with stats_lock:
            stats["attempted"] += 1
            stats["attempted_family"][s["family"]] += 1
        if not breaker.wait_turn():
            return dict(kind="aborted", task=task)
        try:
            raw = backend.complete(prompt)
            breaker.report(ok=True, rate_error=False)
        except BackendError as e:
            breaker.report(ok=False, rate_error=(e.kind == "rate"))
            with stats_lock:
                stats["backend_error"] += 1
            return dict(kind="backend_error", task=task, err=str(e)[:160])
        obj = extract_json_object(strip_fences(raw))
        rn = (obj or {}).get("region_new")
        if not isinstance(rn, list) or not rn or \
                not all(isinstance(l, str) for l in rn):
            return dict(kind="rejected", task=task, reason="layer2_schema")
        ok, why = validate_region_new(rn, s)
        if not ok:
            return dict(kind="rejected", task=task, reason=why)
        key = (s["family"], struct_key_named("\n".join(rn)),
               struct_key_named("\n".join(s["references"][0]
                                           ["render_lines"])))
        with batch_lock:
            if key in corpus_keys:
                with stats_lock:
                    stats["dups_corpus"] += 1
                return dict(kind="rejected", task=task,
                            reason="dedupe_corpus")
            if key in batch_keys:
                with stats_lock:
                    stats["dups_batch"] += 1
                return dict(kind="rejected", task=task, reason="dedupe_batch")
            batch_keys.add(key)
        row = author_row(task, rn, bname, backend.model, prompt)
        return dict(kind="accepted", task=task, row=row)

    def flush_stats(final: bool = False):
        rep = dict(ts=_now(), final=final, rule=RULE,
                   backends={n: backends[n].stats_summary() for n in backends},
                   pending=len(pending), rows_total=rows_written[0],
                   counts={k: (dict(v) if isinstance(v, Counter) else v)
                           for k, v in stats.items()},
                   done_keys=len(done))
        _write_json(Path(str(out_path) + ".stats.json"), rep)

    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=args.workers)
    outstanding: set = set()
    it = iter(pending)
    n_done = 0
    try:
        while True:
            if time.time() - t0 > args.time_budget or breaker.stop.is_set():
                break
            while len(outstanding) < args.workers * 2:
                try:
                    t = next(it)
                except StopIteration:
                    break
                outstanding.add(ex.submit(process, t))
            if not outstanding:
                break
            done_set, _ = wait(outstanding, timeout=30,
                               return_when=FIRST_EXCEPTION)
            for fut in done_set:
                outstanding.discard(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  [worker-exception] {e!r}", flush=True)
                    continue
                n_done += 1
                task = res["task"]
                if res["kind"] == "accepted":
                    row = res["row"]
                    if row["content_hash"] in hashes:
                        with stats_lock:
                            stats["dups_batch"] += 1
                        rec = dict(key=task["task_key"], ok=True,
                                   rows=0, dup=True, ts=_now())
                    else:
                        hashes.add(row["content_hash"])
                        _append_line(out_path, row)
                        rows_written[0] += 1
                        with stats_lock:
                            stats["accepted"] += 1
                            stats["per_family"][task["seed"]["family"]] += 1
                            stats["per_phrasing"][task["phrasing"]] += 1
                            stats["per_backend"][row["backend"]] += 1
                        rec = dict(key=task["task_key"], ok=True, rows=1,
                                   ts=_now())
                elif res["kind"] == "rejected":
                    with stats_lock:
                        stats["dropped"][res["reason"]] += 1
                    rec = dict(key=task["task_key"], ok=False,
                               reason=res["reason"][:200], ts=_now())
                else:                       # backend_error / aborted: retry
                    continue
                _append_line(done_path, rec)
                done[task["task_key"]] = rec
            if n_done % 50 == 0 and n_done:
                flush_stats()
    finally:
        for fut in outstanding:
            fut.cancel()
        ex.shutdown(wait=True)
        flush_stats(final=True)
    print(f"[author] FINISHED done={n_done}/{len(pending)} "
          f"accepted={stats['accepted']} drops={dict(stats['dropped'])} "
          f"dups(corpus/batch)={stats['dups_corpus']}/"
          f"{stats['dups_batch']} in {time.time()-t0:.0f}s", flush=True)
    return 0

# ---------------------------------------------------------------------------
# assemble: pack + license manifest + seed-gates report + glm-probe sample
# ---------------------------------------------------------------------------

PACK_SCHEMA = {"text", "prompt", "target", "family", "package_or_repo",
               "has_types", "expected_noop", "twin_kind"}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pack_ex(row: dict, refs: list[dict]) -> dict:
    return dict(path=row["path"], prefix=row["prefix"],
                suffix=row["suffix"], region_old=row["region_old"],
                region_new=row["region_new"],
                cursor_idx=row["cursor_idx"],
                event_diff=row.get("event_diff") or "",
                references=refs)


def _author_acceptance(work: Path) -> dict:
    side = work / "authored.jsonl.stats.json"
    try:
        rep = json.loads(side.read_text())
    except (OSError, ValueError):
        return {}
    c = rep.get("counts") or {}
    fams = c.get("attempted_family") or {}
    acc = c.get("per_family") or {}
    drops = c.get("dropped") or {}
    per = {}
    for f in FAMILIES:
        att, ok = fams.get(f, 0), acc.get(f, 0)
        per[f] = dict(attempted=att, accepted=ok,
                      rate=round(ok / att, 3) if att else None)
    return dict(per_family=per, overall=dict(
        attempted=c.get("attempted", 0), accepted=c.get("accepted", 0)),
        drops=drops)


def cmd_assemble(args) -> int:
    work = Path(args.work)
    out = Path(args.out)
    authored = work / "authored.jsonl"
    if not authored.exists():
        sys.exit(f"no authored rows at {authored}; run `author` first")
    audit = load_holdout_audit(Path(args.holdout_json))
    seeds = {s["key"]: s for s in _read_jsonl(work / "seeds.jsonl")}
    targets = family_targets(args.seeds)
    stats = Counter()

    rows_raw = []
    for r in _read_jsonl(authored):
        pkg = r.get("package") or "?"
        if is_holdout(pkg) or pkg in audit:       # belt-and-braces
            stats["holdout_blocked"] += 1
            continue
        if r.get("content_hash") and any(
                x["content_hash"] == r["content_hash"] for x in rows_raw):
            stats["drop:dup_content_hash"] += 1
            continue
        rows_raw.append(r)

    rng = random.Random(SEED_RNG)
    by_seed: dict[str, dict] = {}
    for r in rows_raw:
        by_seed.setdefault(r["seed_key"], r)
    fam_cands: dict[str, list[dict]] = {f: [] for f in FAMILIES}
    for r in by_seed.values():
        fam_cands.setdefault(r["family"], []).append(r)

    accepted: list[dict] = []
    max_chars_seen = 0
    for fam in FAMILIES:
        cands = sorted(fam_cands[fam], key=lambda r: r["seed_key"])
        rng.shuffle(cands)
        for r in cands[:targets[fam]]:
            ex = _pack_ex(r, r.get("references") or [])
            prow, why = edit_row_cross(ex, fam, r["package"])
            if prow is None:
                stats[f"drop:{why}"] += 1
                continue
            max_chars_seen = max(max_chars_seen, len(prow["text"]))
            prow["_seed_key"] = r["seed_key"]
            accepted.append(prow)
            stats[f"kept:{fam}"] += 1

    eval_pkgs = package_split(
        [dict(family=r["family"], package_or_repo=r["package_or_repo"])
         for r in accepted], frac=args.eval_frac)
    train = [r for r in accepted
             if r["package_or_repo"] not in eval_pkgs[r["family"]]]
    evals = [r for r in accepted
             if r["package_or_repo"] in eval_pkgs[r["family"]]]

    # ---- corrupted twins (S3.1.7): eval-only, ground truth = no edit -----
    twin_rows: list[dict] = []
    twin_stats = Counter()
    pairs = 0
    ordered = sorted(accepted, key=lambda r: r["_seed_key"])
    for i, prow in enumerate(ordered):
        if pairs >= args.twin_pairs_cap:
            break
        if args.twin_every > 1 and i % args.twin_every:
            continue
        seed = seeds.get(prow["_seed_key"])
        if seed is None:
            twin_stats["no_seed"] += 1
            continue
        tws = build_twin_exs(seed)
        if not tws:
            twin_stats["no_material"] += 1
            continue
        made = 0
        for twin in tws:
            tgt = twin["target"]
            if not twin_gate_check(seed, twin):
                twin_stats["gate_leak"] += 1   # must fail the family gate
                continue
            ex = dict(path=seed["target"]["path"], prefix=tgt["prefix"],
                      suffix=tgt["suffix"], region_old=tgt["region_old"],
                      region_new=tgt["region_old"],
                      cursor_idx=tgt["cursor_idx"],
                      event_diff=twin["event_diff"],
                      references=twin["references"])
            trow, why = edit_row_cross(ex, seed["family"], seed["package"],
                                       expected_noop=True,
                                       twin_kind=twin["kind"])
            if trow is None:
                twin_stats[f"drop:{why}"] += 1
                continue
            trow["_seed_key"] = prow["_seed_key"]
            twin_rows.append(trow)
            twin_stats[f"twin:{twin['kind']}:{seed['family']}"] += 1
            made += 1
        if made:
            pairs += 1
    assert twin_stats["gate_leak"] == 0, "twin leaked through family gate"

    eval_final = evals + twin_rows
    # schema + split validation (B11 conventions)
    for r in train + eval_final:
        assert set(r) - {"_seed_key"} == PACK_SCHEMA, \
            f"schema drift: {sorted(r)}"
        assert r["target"].endswith(UPDATED)
        assert r["text"] == r["prompt"] + r["target"]
    tr_pkgs: dict[str, set[str]] = {}
    for r in train:
        tr_pkgs.setdefault(r["family"], set()).add(r["package_or_repo"])
    for fam, ev in eval_pkgs.items():
        assert not (ev & tr_pkgs.get(fam, set())), \
            f"eval/train package overlap: {fam}: {ev & tr_pkgs.get(fam, set())}"
    assert all(r["expected_noop"] is True for r in twin_rows)
    random.Random(SEED_RNG).shuffle(train)

    out.mkdir(parents=True, exist_ok=True)
    for split, data in (("train", train), ("eval", eval_final)):
        with open(out / f"{split}.jsonl", "w") as fh:
            for r in data:
                fh.write(json.dumps(
                    {k: v for k, v in r.items() if k != "_seed_key"},
                    ensure_ascii=False) + "\n")

    # ---- glm-probe sample (difficulty gate measurement, lead-fired) ------
    probe = []
    for fam in FAMILIES:
        pool = ([r for r in evals if r["family"] == fam]
                + [r for r in train if r["family"] == fam])
        for r in pool[:args.probe_per_family]:
            probe.append(dict(prompt=r["prompt"], target=r["target"],
                              family=fam,
                              package_or_repo=r["package_or_repo"],
                              expected_noop=False, twin_kind=None,
                              split="probe_regular"))
    for r in twin_rows[:200]:
        probe.append(dict(prompt=r["prompt"], target=r["target"],
                          family=r["family"],
                          package_or_repo=r["package_or_repo"],
                          expected_noop=True, twin_kind=r["twin_kind"],
                          split="probe_twin"))
    with open(out / "probe_sample.jsonl", "w") as fh:
        for r in probe:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- license manifest (per batch; propagation from mine) -------------
    lic_path = work / "licenses.json"
    licenses: dict = {}
    if lic_path.exists():
        try:
            licenses = json.loads(lic_path.read_text()).get("packages") or {}
        except ValueError:
            licenses = {}
    row_counts: Counter = Counter()
    for r in train + eval_final:
        row_counts[r["package_or_repo"]] += 1
    lic_manifest = dict(
        ts=_now(), rule=RULE,
        excluded_no_derivatives=sorted(
            p for p, v in licenses.items()
            if v.get("excluded_no_derivatives")),
        packages={p: dict(license=v.get("license", "?"),
                          rows=row_counts.get(p, 0))
                  for p, v in sorted(licenses.items())})
    _write_json(out / "licenses.json", lic_manifest)

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    difficulty = dict(Counter(
        (by_seed[r["_seed_key"]].get("difficulty") or {}).get("label", "?")
        for r in train + evals))
    manifest = dict(
        rule=RULE, generated_at=_now(), phase="seed",
        sources=dict(workdir=str(work), authored_file=str(authored),
                     authored_sha256=sha(authored),
                     holdout_json=str(args.holdout_json)),
        targets=dict(seed_tasks=args.seeds, family_targets=targets,
                     family_caps=FAMILY_CAPS),
        split=dict(eval_frac=args.eval_frac, seed=SEED_RNG,
                   eval_packages={f: sorted(v) for f, v in
                                  sorted(eval_pkgs.items())}),
        counts=dict(train=len(train), eval=len(eval_final),
                    eval_regular=len(evals), eval_twins=len(twin_rows),
                    twin_pairs=pairs,
                    per_family_train=dict(Counter(
                        r["family"] for r in train)),
                    per_family_eval=dict(Counter(
                        r["family"] for r in eval_final)),
                    difficulty=difficulty),
        budget=dict(max_chars_cross=MAX_CHARS_CROSS,
                    max_prompt_target_seen=max_chars_seen,
                    ref_max_lines=REF_MAX_LINES,
                    ref_max_chars=REF_MAX_CHARS,
                    target_prefix_lines=TARGET_PREFIX_LINES,
                    target_suffix_lines=TARGET_SUFFIX_LINES,
                    files_per_row=f"{FILES_MIN}-{FILES_MAX}"),
        licenses=dict(excluded_no_derivatives=len(
            lic_manifest["excluded_no_derivatives"])),
        drops=dict(stats),
        hashes=dict(train_sha256=sha(out / "train.jsonl"),
                    eval_sha256=sha(out / "eval.jsonl")),
    )
    _write_json(out / "manifest.json", manifest)

    gates = dict(
        rule=RULE, phase="seed", generated_at=_now(),
        validator_acceptance=_author_acceptance(work),
        twins=dict(pairs=pairs, rows=len(twin_rows),
                   per_kind=dict(twin_stats),
                   gate_rejects_noop_answer=True,   # asserted above
                   expected_noop=True,
                   note="twins are eval rows; correct answer is no edit; "
                        "blindly applying the instructed edit is the "
                        "pre-registered false-propose miss"),
        glm_probe=dict(sample_file="probe_sample.jsonl",
                       rows=len(probe),
                       regular=sum(1 for r in probe
                                   if r["split"] == "probe_regular"),
                       twin=sum(1 for r in probe
                                if r["split"] == "probe_twin"),
                       pass_at1_band=[0.30, 0.85],
                       random_policy_target=0.0,
                       status="ready for the lead-fired probe"),
        budget=dict(max_seen=max_chars_seen,
                    dropped={k: stats[k] for k in stats
                             if k.startswith("drop:")}),
        licenses=dict(excluded_no_derivatives=len(
            lic_manifest["excluded_no_derivatives"])),
        counts=dict(train=len(train), eval=len(eval_final)),
    )
    _write_json(out / "gates_report.json", gates)
    print(f"[assemble] train={len(train)} eval={len(eval_final)} "
          f"(regular {len(evals)} + twins {len(twin_rows)}, "
          f"{pairs} pairs) -> {out}")
    print(json.dumps(gates["validator_acceptance"].get("per_family", {}),
                     indent=1), flush=True)
    return 0

# ---------------------------------------------------------------------------
# fixtures-only smoke corpus (the --smoke default: no NAS, no network)
# ---------------------------------------------------------------------------

SMOKE_PKG_A = dict(
    DESCRIPTION="Package: smokeA\nVersion: 1.0\nLicense: GPL-3\n",
    files={
        "R/base_ref.R": """quick_max <- function(xs) {
  max(vapply(xs, function(z) z, numeric(1)))
}

pipe_sum <- function(d) {
  d %>% dplyr::summarise(s = sum(v))
}
""",
        "R/defs.R": """prep_data <- function(x, weight = 1) {
  x * weight
}

calc_score <- function(v, k) {
  sum(v) + k
}
""",
        "R/docs.R": """#' Compute the thing
#'
#' @param x input vector
#' @return numeric
#' @export
make_thing <- function(x, scale = 1) {
  x * scale
}

slow_copy <- function(xs) {
  out <- c()
  for (v in xs) {
    out <- c(out, v)
  }
  out
}
""",
        "R/ports.R": """loop_total <- function(xs) {
  tot <- 0
  for (i in seq_along(xs)) {
    tot <- tot + xs[i]
  }
  tot
}
""",
        "R/usage.R": """res_one <- prep_data(df, weight = w)
res_two <- calc_score(vals, 2)
out_thing <- make_thing(vals, scale = 2)

dt_roll <- function(df) {
  data.table::as.data.table(df)[, v := sum(v)]
}
""",
    })
SMOKE_PKG_B = dict(
    DESCRIPTION="Package: smokeB\nVersion: 1.0\nLicense: MIT + file LICENSE\n",
    files={
        "R/defs2.R": """norm_vec <- function(x, center = 0) {
  (x - center) / max(abs(x))
}
""",
        "R/use2.R": """scaled <- norm_vec(v, center = mu)
""",
    })
SMOKE_PKG_ND = dict(          # no-derivatives: must be excluded
    DESCRIPTION="Package: smokeND\nVersion: 1.0\n"
                "License: CC BY-NC-ND 4.0\n",
    files={
        "R/d.R": """nd_fn <- function(x) {
  x + 1
}
""",
        "R/u.R": """nd_res <- nd_fn(3)
""",
    })


def build_smoke_corpus(root: Path) -> Path:
    """Normalized-mirror-shaped fixture corpus (three packages: GPL-3,
    MIT, CC-BY-NC-ND); package names dodge the holdout hash rule so the
    smoke run is deterministic."""
    root.mkdir(parents=True, exist_ok=True)
    for spec in (SMOKE_PKG_A, SMOKE_PKG_B, SMOKE_PKG_ND):
        name = spec["DESCRIPTION"].split("Package: ")[1].split()[0]
        while is_holdout(name):
            name += "x"
        pkg_root = root / name / "1.0" / name
        (pkg_root / "R").mkdir(parents=True, exist_ok=True)
        (pkg_root / "DESCRIPTION").write_text(
            spec["DESCRIPTION"].replace(
                spec["DESCRIPTION"].split("Package: ")[1].split()[0], name, 1))
        for rel, text in spec["files"].items():
            (pkg_root / rel).write_text(text)
    return root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("mine", "author", "assemble", "all"))
    ap.add_argument("--smoke", action="store_true",
                    help="fixtures-only corpus (built under the work dir), "
                         "mock backend, tiny defaults into the work dir; "
                         "explicit flags still override")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--holdout-json", default=str(HOLDOUT_JSON))
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=None,
                    help="accepted-task target (default 500 = the seed "
                         "phase; families proportional to caps)")
    ap.add_argument("--pool-mult", type=float, default=None,
                    help="mined candidates per family target (attrition "
                         "headroom for the author wave)")
    ap.add_argument("--seed", type=int, default=SEED_RNG)
    ap.add_argument("--per-package-cap", type=int, default=None)
    ap.add_argument("--max-packages", type=int, default=0)
    ap.add_argument("--phrasings", type=int, default=None, choices=(1, 2, 3))
    ap.add_argument("--backends", default=None)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=3600.0)
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--twin-every", type=int, default=None,
                    help="twin pair every N accepted rows (seed phase: 1; "
                         "design v1 scale-up ratio: 5)")
    ap.add_argument("--twin-pairs-cap", type=int, default=None)
    ap.add_argument("--probe-per-family", type=int, default=None)
    ap.add_argument("--include-no-derivatives", action="store_true",
                    help="keep no-derivatives packages (flagged "
                         "public_ok=false) instead of excluding them")
    args = ap.parse_args(argv)

    if args.smoke:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        args.work = args.work or f"/tmp/tu3_smoke_{stamp}"
        args.out = args.out or args.work + "/out"
        corpus = Path(args.work) / "corpus_fixture"
        if not corpus.exists():
            build_smoke_corpus(corpus)
        args.corpus = str(corpus)
        args.seeds = args.seeds if args.seeds is not None else 12
        args.pool_mult = args.pool_mult if args.pool_mult is not None \
            else 1.0
        args.backends = args.backends or "mock"
        args.workers = args.workers if args.workers is not None else 1
        args.per_package_cap = args.per_package_cap \
            if args.per_package_cap is not None else 3
        args.phrasings = args.phrasings if args.phrasings is not None else 1
        args.twin_every = args.twin_every if args.twin_every is not None \
            else 1
        args.twin_pairs_cap = args.twin_pairs_cap \
            if args.twin_pairs_cap is not None else 8
        args.probe_per_family = args.probe_per_family \
            if args.probe_per_family is not None else 4
        args.include_no_derivatives = False
        print(f"[smoke] work={args.work} out={args.out} "
              f"corpus={args.corpus}")
    else:
        args.work = args.work or str(DEFAULT_WORK)
        args.out = args.out or str(DEFAULT_OUT)
        args.seeds = args.seeds if args.seeds is not None else SEED_TASKS
        args.pool_mult = args.pool_mult if args.pool_mult is not None \
            else 2.5
        args.backends = args.backends or "zai,opencode-spark-free"
        args.workers = args.workers if args.workers is not None else 4
        args.per_package_cap = args.per_package_cap \
            if args.per_package_cap is not None else 2
        args.phrasings = args.phrasings if args.phrasings is not None else 1
        args.twin_every = args.twin_every if args.twin_every is not None \
            else 1
        args.twin_pairs_cap = args.twin_pairs_cap \
            if args.twin_pairs_cap is not None else TWIN_PAIRS_CAP
        args.probe_per_family = args.probe_per_family \
            if args.probe_per_family is not None else PROBE_PER_FAMILY
    args.exclude_no_derivatives = not args.include_no_derivatives

    if args.command in ("mine", "all"):
        cmd_mine(args)
    if args.command in ("author", "all"):
        cmd_author(args)
    if args.command in ("assemble", "all"):
        cmd_assemble(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
