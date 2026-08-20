"""Registered corpus selectors: turn a spec's corpus_source into a list of
prompt-bearing items (each: key, prefix, block, suffix, package, path).

  c2c_cache_blocks     dataset_file — the .c2c_cache.json candidate blocks
                       (comment-free CRAN function-body blocks) that back the
                       comment families
  tidyselect_helpers    normalized_corpus — real tidyselect helper calls
                       nested inside dplyr selection verbs; the helper text
                       is elided from the line and becomes the completion
                       target (corpus original kept for provenance)

Selectors are seeded (the harness owns the rng) and cache their scan on the
local disk (drvfs corpus scans are slow; `.cases_cache/` beside the package)
so resumed/extended runs skip the rescan.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scenarios/
import build_astfim as ASTFIM          # version_key/pick_version_dir helpers
import scenarios as S
from scenarios import Bundle, iter_bundles, node_text

from cases.validators import (TIDYSELECT_HELPERS, _walk, fragment_clean,
                              parse_fragment)

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cases_cache"
TAR_DIR = S.TAR_DIR
MAX_MEMBER_BYTES = 300_000
MAX_TEST_FILES_PER_TARBALL = 4
REGISTRY: dict = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def _apply_difficulty(item: dict, spec) -> dict:
    """Difficulty knobs shared by every selector: context_lines trims the
    prefix to its last N lines; target_lines_min/max bound the block."""
    d = spec.difficulty or {}
    ctx = int(d.get("context_lines", 0) or 0)
    if ctx > 0 and len(item.get("prefix") or []) > ctx:
        item = dict(item, prefix=list(item["prefix"][-ctx:]))
    return item


def _filter_block_length(items: list, spec) -> list:
    d = spec.difficulty or {}
    lo = int(d.get("target_lines_min", 0) or 0)
    hi = int(d.get("target_lines_max", 0) or 0)
    if not lo and not hi:
        return items
    out = []
    for it in items:
        n = len(it.get("block") or [])
        if lo and n < lo:
            continue
        if hi and n > hi:
            continue
        out.append(it)
    return out


def select_corpus(spec, rng: random.Random, want: int) -> list[dict]:
    cs = spec.corpus_source
    name = cs["selector"]
    if name not in REGISTRY:
        raise KeyError(f"unknown corpus selector {name!r}; "
                       f"registered: {sorted(REGISTRY)}")
    items = REGISTRY[name](spec, rng, want)
    items = _filter_block_length(items, spec)
    out = []
    for it in items:
        it = _apply_difficulty(it, spec)
        if it.get("prefix") and it.get("block"):
            out.append(it)
    return out


def _resolve_path(cs: dict, spec) -> Path:
    p = Path(cs["path"])
    if not p.is_absolute():
        p = (Path(spec.source_path).parent / p) if spec.source_path and \
            spec.source_path != "<dict>" else (HERE.parent / p)
    return p


def _params_hash(blob: dict) -> str:
    return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# dataset_file selectors
# ---------------------------------------------------------------------------

@register("c2c_cache_blocks")
def sel_c2c_cache(spec, rng: random.Random, want: int) -> list[dict]:
    """Comment-free candidate blocks from the .c2c_cache.json built by
    comment_to_code.py (prefix lines + the real code block)."""
    cs = spec.corpus_source
    path = _resolve_path(cs, spec)
    params = dict(cs.get("params") or {})
    blob = json.loads(path.read_text())
    cands = blob["cands"] if isinstance(blob, dict) else blob
    order = list(range(len(cands)))
    rng.shuffle(order)
    cap = int(params.get("max_items") or (want * 6 + 50))
    out = []
    for i in order[:cap]:
        c = cands[i]
        block = [str(l) for l in (c.get("block") or [])]
        prefix = [str(l) for l in (c.get("prefix_lines") or c.get("prefix") or [])]
        if not block or not prefix:
            continue
        out.append(dict(key=f"c2c:{i}", prefix=prefix, block=block,
                        suffix=[str(l) for l in (c.get("suffix") or [])],
                        package=c.get("package", "?"), path=c.get("path", "?"),
                        corpus_target="\n".join(block)))
    return out


# ---------------------------------------------------------------------------
# normalized_corpus selectors
# ---------------------------------------------------------------------------

def _callee(src: bytes, call_node) -> str | None:
    return S.callee_name(src, call_node)


def extract_tidyselect(b: Bundle, verbs: set, helpers: dict,
                       context_lines: int = 8, cap: int = 8) -> list[dict]:
    """tidyselect helper calls nested inside dplyr selection verbs, from one
    parsed bundle. The helper text is cut out of its (single) line: the item
    prefix ends right where the helper began (mid-line cursor) and the suffix
    carries the rest of the line; the elided original is corpus_target."""
    src = b.src
    out, seen = [], set()
    for n in _walk(b.tree.root_node):
        if n.type != "call" or _callee(src, n) not in verbs:
            continue
        for d in _walk(n):
            if d is n or d.type != "call":
                continue
            name = _callee(src, d)
            if name not in helpers:
                continue
            if d.start_point[0] != d.end_point[0]:
                continue  # single-line helpers only (exact cursor semantics)
            named_args = [c for c in d.children[1:] if c.is_named]
            if not named_args:
                continue
            row, col = b.rowcol(d.start_byte)
            erow, ecol = b.rowcol(d.end_byte)
            if erow != row:
                continue
            line = b.line_str(row)
            helper_text = line[col:ecol].strip()
            prefix_part, suffix = line[:col].rstrip(), line[ecol:].strip()
            if not helper_text or not prefix_part:
                continue
            k = (b.rel, row, helper_text)
            if k in seen:
                continue
            seen.add(k)
            prefix = [b.line_str(r) for r in
                      range(max(0, row - context_lines), row)] + [prefix_part]
            out.append(dict(key=None, prefix=prefix, block=[helper_text],
                            suffix=[suffix] if suffix else [],
                            package=b.package, path=b.rel, row=row,
                            corpus_target=helper_text))
            if len(out) >= cap:
                return out
    return out


@register("tidyselect_helpers")
def sel_tidyselect(spec, rng: random.Random, want: int) -> list[dict]:
    """Scan dplyr-using packages of the normalized CRAN corpus for real
    tidyselect helper calls (cached; the drvfs scan is slow)."""
    cs = spec.corpus_source
    params = dict(cs.get("params") or {})
    verbs = set(params.get("enclosing_verbs", ["select", "relocate", "across"]))
    helpers = dict(params.get("helpers") or TIDYSELECT_HELPERS)
    context = int(params.get("context_lines", 8))
    max_items = int(params.get("max_items") or (want * 6 + 50))
    time_budget = float(params.get("time_budget_s", 600))
    n_packages = int(params.get("sample_packages", 400))

    CACHE_DIR.mkdir(exist_ok=True)
    meta = dict(selector="tidyselect_helpers", verbs=sorted(verbs),
                helpers=sorted(helpers), context_lines=context,
                sample_packages=n_packages)
    cache = CACHE_DIR / "tidyselect_helpers.json"
    if cache.exists():
        try:
            blob = json.loads(cache.read_text())
            if blob.get("params_hash") == _params_hash(meta):
                items = blob["items"]
                rng.shuffle(items)
                for i, it in enumerate(items):
                    it["key"] = f"ts:{i}"
                print(f"  [corpus] tidyselect cache: {len(items)} items "
                      f"(scan {blob.get('scan')})")
                return items[:max_items]
        except (ValueError, KeyError, OSError):
            pass

    pool = S.tidy_packages() if params.get("packages", "tidy") == "tidy" \
        else S.list_packages()
    order = list(pool)
    rng.shuffle(order)
    order = order[:n_packages] if n_packages > 0 else order
    items: list[dict] = []
    t0, files, pkgs = time.time(), 0, set()
    for b in iter_bundles(order, rng):
        files += 1
        pkgs.add(b.package)
        for it in extract_tidyselect(b, verbs, helpers,
                                     context_lines=context):
            it["key"] = None
            items.append(it)
        if len(items) >= max_items or time.time() - t0 > time_budget:
            break
        if files % 250 == 0:
            print(f"  [corpus] files={files} pkgs={len(pkgs)} "
                  f"items={len(items)} elapsed={time.time()-t0:.0f}s",
                  flush=True)
    # unique keys, stable by (package, path, row)
    items.sort(key=lambda it: (it["package"], it["path"], it.get("row", 0)))
    for i, it in enumerate(items):
        it["key"] = f"ts:{i}"
    scan = dict(files=files, packages=len(pkgs), elapsed_s=round(time.time() - t0, 1),
                items=len(items))
    try:
        cache.write_text(json.dumps(dict(params_hash=_params_hash(meta),
                                         scan=scan, items=items)))
    except OSError:
        pass
    print(f"  [corpus] tidyselect scan: {scan}")
    return items[:max_items]


# ---------------------------------------------------------------------------
# wave-1 case selectors (proposals_v1 top-5)
#
# Shared conventions:
#   * every item carries corpus_target (the verbatim ground-truth text from
#     the corpus — reverse-strip / cut constructions, tier-1 exact GT);
#   * prefix ends AT the cursor (its last line is the partial line when the
#     cut is mid-line); suffix holds the file below the target (never the
#     target itself);
#   * block == corpus_target lines so difficulty.target_lines_* bounds apply.
# ---------------------------------------------------------------------------

def _finish_item(it: dict, prefix_key: str, i: int) -> dict:
    it["key"] = f"{prefix_key}:{i}"
    return it


def _scan_normalized(spec, rng: random.Random, cache_name: str, meta: dict,
                     extract, pool: list[str], params: dict) -> list[dict]:
    """Cached scan of the normalized corpus (drvfs scans are slow; the
    tidyselect_helpers convention). `extract(bundle, rng, params)` returns
    candidate items without keys."""
    max_items = int(params.get("max_items") or 2500)
    time_budget = float(params.get("time_budget_s", 1200))
    n_packages = int(params.get("sample_packages", 400))
    per_package = int(params.get("per_package_cap", 3))

    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / cache_name
    meta = dict(meta, max_items=max_items, sample_packages=n_packages,
                per_package_cap=per_package)
    if cache.exists():
        try:
            blob = json.loads(cache.read_text())
            if blob.get("params_hash") == _params_hash(meta):
                items = blob["items"]
                rng.shuffle(items)
                for i, it in enumerate(items):
                    it["key"] = f"{meta['selector']}:{i}"
                print(f"  [corpus] {cache_name} cache: {len(items)} items "
                      f"(scan {blob.get('scan')})")
                return items[:max_items]
        except (ValueError, KeyError, OSError):
            pass

    order = list(pool)
    rng.shuffle(order)
    order = order[:n_packages] if n_packages > 0 else order
    items: list[dict] = []
    pkg_counts: dict[str, int] = {}
    t0, files = time.time(), 0
    for b in iter_bundles(order, rng):
        files += 1
        if pkg_counts.get(b.package, 0) >= per_package:
            continue
        got = extract(b, rng, params)
        if got and pkg_counts.get(b.package, 0) + len(got) > per_package:
            got = got[:per_package - pkg_counts.get(b.package, 0)]
        if got:
            pkg_counts[b.package] = pkg_counts.get(b.package, 0) + len(got)
            items.extend(got)
        if len(items) >= max_items or time.time() - t0 > time_budget:
            break
        if files % 250 == 0:
            print(f"  [corpus] {cache_name}: files={files} "
                  f"items={len(items)} elapsed={time.time()-t0:.0f}s",
                  flush=True)
    items.sort(key=lambda it: (it.get("package", ""),
                               it.get("path", ""), it.get("row", 0)))
    for i, it in enumerate(items):
        it["key"] = f"{meta['selector']}:{i}"
    scan = dict(files=files, packages=len(pkg_counts),
                elapsed_s=round(time.time() - t0, 1), items=len(items))
    try:
        cache.write_text(json.dumps(dict(params_hash=_params_hash(meta),
                                         scan=scan, items=items)))
    except OSError:
        pass
    print(f"  [corpus] {cache_name} scan: {scan}")
    return items[:max_items]


def _scan_tarballs(spec, rng: random.Random, cache_name: str, meta: dict,
                   handle, params: dict) -> list[dict]:
    """Cached streaming scan of the CRAN tarballs (format_propagation's
    python-tarfile pattern: getmembers once, extractfile only the members
    the case wants). `handle(pkg, member_rel, text)` -> candidate items."""
    max_items = int(params.get("max_items") or 2500)
    time_budget = float(params.get("time_budget_s", 2400))
    n_tarballs = int(params.get("sample_tarballs", 2500))
    per_package = int(params.get("per_package_cap", 2))
    wanted = tuple(params.get("members", ()))

    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / cache_name
    meta = dict(meta, max_items=max_items, sample_tarballs=n_tarballs,
                per_package_cap=per_package, members=sorted(wanted))
    if cache.exists():
        try:
            blob = json.loads(cache.read_text())
            if blob.get("params_hash") == _params_hash(meta):
                items = blob["items"]
                rng.shuffle(items)
                for i, it in enumerate(items):
                    it["key"] = f"{meta['selector']}:{i}"
                print(f"  [corpus] {cache_name} cache: {len(items)} items "
                      f"(scan {blob.get('scan')})")
                return items[:max_items]
        except (ValueError, KeyError, OSError):
            pass

    try:
        tars = sorted(TAR_DIR.glob("*.tar.gz"))
    except OSError:
        tars = []
    order = list(tars)
    rng.shuffle(order)
    order = order[:n_tarballs] if n_tarballs > 0 else order
    items: list[dict] = []
    pkg_counts: dict[str, int] = {}
    extras: dict = dict(meta.get("extras") or {})
    t0, opened = time.time(), 0
    for tp in order:
        if time.time() - t0 > time_budget or len(items) >= max_items:
            break
        try:
            tf = tarfile.open(tp, "r:gz")
        except (tarfile.TarError, OSError):
            continue
        with tf:
            opened += 1
            members = [m for m in tf.getmembers()
                       if m.isfile() and m.size <= MAX_MEMBER_BYTES]
            for m in members:
                parts = m.name.split("/")
                if len(parts) < 2:
                    continue
                pkg, rel = parts[0], "/".join(parts[1:])
                if not _member_wanted(rel, wanted, parts):
                    continue
                if pkg_counts.get(pkg, 0) >= per_package:
                    continue
                try:
                    fh = tf.extractfile(m)
                    if fh is None:
                        continue
                    text = fh.read().decode("utf-8", "replace")
                except (tarfile.TarError, OSError, UnicodeDecodeError):
                    continue
                got = handle(pkg, rel, text, extras) or []
                if pkg_counts.get(pkg, 0) + len(got) > per_package:
                    got = got[:max(0, per_package - pkg_counts.get(pkg, 0))]
                if got:
                    pkg_counts[pkg] = pkg_counts.get(pkg, 0) + len(got)
                    items.extend(got)
        if opened % 250 == 0 and opened:
            print(f"  [corpus] {cache_name}: tarballs={opened} "
                  f"items={len(items)} elapsed={time.time()-t0:.0f}s",
                  flush=True)
    items.sort(key=lambda it: (it.get("package", ""),
                               it.get("path", ""), it.get("row", 0)))
    for i, it in enumerate(items):
        it["key"] = f"{meta['selector']}:{i}"
    scan = dict(tarballs=opened, packages=len(pkg_counts),
                elapsed_s=round(time.time() - t0, 1), items=len(items),
                **extras)
    try:
        cache.write_text(json.dumps(dict(params_hash=_params_hash(meta),
                                         scan=scan, items=items)))
    except OSError:
        pass
    print(f"  [corpus] {cache_name} scan: {scan}")
    return items[:max_items]


def _member_wanted(rel: str, wanted: tuple, parts: list[str]) -> bool:
    if not wanted:
        return False
    if wanted == ("*",):                        # tests/testthat/*.R
        return (len(parts) == 4 and parts[1] == "tests"
                and parts[2] == "testthat" and rel.endswith(".R")
                and parts[3].startswith("test-"))
    return rel in wanted                        # NAMESPACE / DESCRIPTION


def _mk_item(package: str, path: str, row: int, prefix: list[str],
             target_lines: list[str], suffix: list[str], **extra) -> dict:
    it = dict(package=package, path=path, row=row,
              prefix=[str(l) for l in prefix],
              block=[str(l) for l in target_lines],
              suffix=[str(l) for l in suffix],
              corpus_target="\n".join(str(l) for l in target_lines))
    it.update(extra)
    return it


# ---------------------------------------------------------------------------
# case 1: namespace_qualify_propagation (reverse-strip; corpus-side exact)
# ---------------------------------------------------------------------------

_NS_RE = re.compile(r"(?<![\w.$@])([A-Za-z][\w.]*)::")
_QUAL_HEAD_RE = re.compile(r"^([A-Za-z][\w.]*)::([A-Za-z.][\w.]*)\s*\(")


def _strip_ns(line: str, p: str) -> str:
    return re.sub(rf"(?<![\w.$@]){re.escape(p)}::", "", line)


def extract_namespace_qualifiers(b: Bundle, rng: random.Random,
                                 params: dict) -> list[dict]:
    """Files using one namespace >= 2x: every `p::` is stripped from the
    prompt-side state; the event occurrence (an earlier one) keeps its
    qualifier, the target occurrence is cut right before the bare callee so
    the completion re-qualifies it. GT = the original line text from the
    cursor (exact by construction)."""
    window = int(params.get("window_lines", 12))
    cap = int(params.get("per_file_cap", 2))
    occ: dict[str, list[tuple[int, int]]] = {}
    for r in range(b.nlines()):
        line = b.line_str(r)
        if not line or line.lstrip().startswith("#"):
            continue
        for m in _NS_RE.finditer(line):
            occ.setdefault(m.group(1), []).append((r, m.start()))
    out = []
    for p, positions in sorted(occ.items()):
        if len(positions) < 2 or len(out) >= cap:
            break
        picks = list(range(1, len(positions)))
        rng.shuffle(picks)
        for i in picks[:cap]:
            row, col = positions[i]
            line = b.line_str(row)
            if any(pr == row and pc < col for pr, pc in positions):
                continue          # earlier same-line occurrence: cut col shifts
            target_text = line[col:].rstrip()
            if not _QUAL_HEAD_RE.match(target_text) or len(target_text) > 240:
                continue
            ev = positions[i - 1] if rng.random() < 0.7 else \
                positions[rng.randrange(i)]
            prefix = []
            for r in range(max(0, row - window), row):
                l = b.line_str(r)
                # the event line keeps its qualifier (the user's restore)
                prefix.append(l if r == ev[0] else _strip_ns(l, p))
            prefix.append(line[:col])
            suffix = [_strip_ns(b.line_str(r), p)
                      for r in range(row + 1, min(b.nlines(), row + 1 + window))]
            out.append(_mk_item(
                b.package, b.rel, row, prefix, [target_text], suffix,
                qualify_package=p,
                corpus_line=line,
                note=f"re-qualify the next bare call with {p}:: "
                     f"(event: earlier {p}:: occurrence restored above)"))
    return out


@register("namespace_qualifiers")
def sel_namespace_qualifiers(spec, rng: random.Random, want: int) -> list[dict]:
    cs = spec.corpus_source
    params = dict(cs.get("params") or {})
    params.setdefault("max_items", want * 6 + 50)
    return _scan_normalized(
        spec, rng, "namespace_qualifiers.json",
        dict(selector="namespace_qualifiers",
             window_lines=params.get("window_lines", 12)),
        extract_namespace_qualifiers,
        S.list_packages(), params)


# ---------------------------------------------------------------------------
# case 2: pipe_chain_link (cut at the pipe boundary; corpus-side exact GT,
# agy-generated completion gated by exact match)
# ---------------------------------------------------------------------------

PIPE_OPS = (b"%>%", b"|>")


def _is_pipe_op(src: bytes, n) -> bool:
    return n.type in ("special", "|>") and node_text(src, n) in PIPE_OPS


def extract_pipe_links(b: Bundle, rng: random.Random, params: dict) -> list[dict]:
    """Multi-line pipe chains: the cursor sits right after the `|>`/`%>%`
    that ends a line; the target is the NEXT chain link line verbatim (the
    rest of the chain stays in the suffix)."""
    cap = int(params.get("per_file_cap", 2))
    cut = params.get("cut", "line")
    window = int(params.get("window_lines", 10))
    src = b.src
    cands, seen_rows = [], set()
    for n in _walk(b.tree.root_node):
        if n.type != "binary_operator" or len(n.children) < 3:
            continue
        lhs, op, rhs = n.children[0], n.children[1], n.children[2]
        if not _is_pipe_op(src, op) or rhs.type != "call":
            continue
        op_row, op_end_col = op.end_point[0], op.end_point[1]
        if b.line_str(op_row)[op_end_col:].strip():
            continue              # operator must END its line (cursor at EOL)
        if rhs.start_point[0] != op_row + 1:
            continue              # next link must begin the following line
        if op_row in seen_rows:
            continue
        t_row = rhs.start_point[0]
        target_line = b.line_str(t_row)
        if (not target_line.strip() or len(target_line) > 220
                or target_line.lstrip().startswith("#")):
            continue
        # walk down the LHS to the chain head so the data context is visible
        head, node = lhs, lhs
        while node.type == "binary_operator" and len(node.children) >= 3 \
                and _is_pipe_op(src, node.children[1]):
            head = node.children[0]
            node = head
        head_row = head.start_point[0]
        prefix = [b.line_str(r)
                  for r in range(max(head_row, op_row - window), op_row + 1)]
        suffix = [b.line_str(r) for r in
                  range(t_row + 1, min(b.nlines(), t_row + 1 + window))]
        target_lines = [target_line.rstrip()]
        if cut == "verb":         # deeper cut: through the callee's open paren
            m = re.match(r"^(\s*[\w.]+\()", target_line)
            if not m:
                continue
            k = len(m.group(1))
            prefix.append(target_line[:k])
            target_lines = [target_line[k:].rstrip()]
            if not target_lines[0].strip():
                continue
        seen_rows.add(op_row)
        chain_pos = ("mid" if target_line.rstrip().endswith(("%>%", "|>"))
                     else "final")
        cands.append(_mk_item(
            b.package, b.rel, t_row, prefix, target_lines, suffix,
            chain_pos=chain_pos,
            note=f"next link of a {chain_pos}-position pipe chain "
                 f"(cut after the {node_text(src, op).decode()} operator)"))
    # the tree walk is pre-order (outermost/last boundary first): sample so
    # the per-file cap does not bias toward the chain-final link
    rng.shuffle(cands)
    return cands[:cap]


@register("pipe_chain_links")
def sel_pipe_chain_links(spec, rng: random.Random, want: int) -> list[dict]:
    cs = spec.corpus_source
    params = dict(cs.get("params") or {})
    params.setdefault("max_items", want * 6 + 50)
    pool = S.tidy_packages() if params.get("packages", "tidy") == "tidy" \
        else S.list_packages()
    # the tidy pool alone (~440 pkgs) is too thin for 400+ accepted rows:
    # top it up with random packages from the full corpus (breadth stratum)
    extra = int(params.get("extra_random_packages", 0) or 0)
    if extra > 0:
        rest = [p for p in S.list_packages() if p not in set(pool)]
        rng.shuffle(rest)
        pool = list(pool) + rest[:extra]
    return _scan_normalized(
        spec, rng, "pipe_chain_links.json",
        dict(selector="pipe_chain_links", cut=params.get("cut", "line"),
             window_lines=params.get("window_lines", 10),
             extra_random_packages=extra),
        extract_pipe_links, pool, params)


# ---------------------------------------------------------------------------
# case 3: pkg_metadata_sync (tarball NAMESPACE/DESCRIPTION reverse-strip)
# ---------------------------------------------------------------------------

NS_DIRECTIVE_RE = re.compile(r"^(export|importFrom|S3method)\((.*)\)\s*$")
# DESCRIPTION continuation entries: `    pkg`, `    pkg (>= 1.0)`, with the
# trailing comma that separates entries inside one field block
DSC_ENTRY_RE = re.compile(r"^\s+([A-Za-z][\w.]*)\s*(\([^)]*\))?\s*,?\s*$")


def ns_entry_key(line: str):
    m = NS_DIRECTIVE_RE.match(line)
    if not m:
        return None
    return m.group(1), " ".join(m.group(2).lower().split())


def dsc_entry_key(line: str):
    m = DSC_ENTRY_RE.match(line)
    if not m:
        return None
    return (m.group(1).lower(),)


def _handle_namespace(pkg: str, text: str, extras: dict) -> list[dict]:
    lines = text.splitlines()
    extras["ns_files"] = extras.get("ns_files", 0) + 1
    groups: dict[str, list[tuple[int, tuple]]] = {}
    for i, l in enumerate(lines):
        k = ns_entry_key(l)
        if k is not None:
            groups.setdefault(k[0], []).append((i, k))
    if not groups:
        return []
    sorted_groups = {d for d, g in groups.items()
                     if all(g[j][1] < g[j + 1][1] for j in range(len(g) - 1))}
    extras["ns_files_fully_sorted"] = extras.get("ns_files_fully_sorted", 0) \
        + (1 if sorted_groups == set(groups) else 0)
    out = []
    for d, g in sorted(groups.items()):
        if d not in sorted_groups or not g:
            continue
        i, _k = g[len(g) // 2] if len(g) > 2 else g[0]
        if i > 80:               # keep the whole file head visible
            continue
        out.append(_mk_item(
            pkg, "NAMESPACE", i, lines[:i], [lines[i]], lines[i + 1:i + 41],
            entry_kind=f"namespace:{d}",
            note=f"re-insert the reverse-stripped {d}() directive at its "
                 f"alphabetically sorted slot"))
    return out[:1]


DSC_ENTRY_FULL_RE = re.compile(r"^([A-Za-z][\w.]*)\s*(\([^)]*\))?$")


def _handle_description(pkg: str, text: str, extras: dict) -> list[dict]:
    """Imports: entries, wherever they physically sit (on the field line
    itself, on one-entry-per-line continuations, or several entries per
    continuation line). The chosen entry is reverse-stripped at a LOCALLY
    sorted slot (its immediate neighbours are in order): cut at the entry
    start - own line for the per-line style, mid-line for inline entries
    (the target then carries the entry plus the rest of the physical line,
    exact from the file)."""
    lines = text.splitlines()
    h = next((i for i, l in enumerate(lines) if l.startswith("Imports:")),
             None)
    if h is None or len(lines[h]) > 200:
        return []
    frags = []                     # (line_idx, non-empty text of the line)
    head_txt = lines[h][len("Imports:"):].strip()
    if head_txt:
        frags.append((h, head_txt))
    i = h + 1
    while i < len(lines):
        l = lines[i]
        if not l.strip() or not l[0].isspace():
            break                 # blank line / next field header ends block
        frags.append((i, l))
        i += 1
    entries = []                   # (key, line_idx, col, entry_text)
    for idx, frag in frags:
        base, pos = lines[idx], 0
        for part in frag.split(","):
            t = part.strip()
            if not t:
                continue
            m = DSC_ENTRY_FULL_RE.match(t)
            if not m:
                continue
            col = base.find(t, pos)
            if col < 0:
                continue
            entries.append(((m.group(1).lower(),), idx, col, t))
            pos = col + len(t)
    if len(entries) < 3:
        return []
    loc = [j for j in range(1, len(entries) - 1)
           if entries[j - 1][0] < entries[j][0] < entries[j + 1][0]]
    if not loc:
        return []
    j = loc[len(loc) // 2]
    _key, idx, col, t = entries[j]
    if len(lines[idx]) > 200:
        return []
    if lines[idx].strip().rstrip(",").strip() == t:   # own-line entry
        prefix, target_lines = lines[:idx], [lines[idx]]
    else:                          # inline entry: mid-line cut at its start
        prefix = lines[:idx] + [lines[idx][:col]]
        target_lines = [lines[idx][col:]]
    return [_mk_item(
        pkg, "DESCRIPTION", idx, prefix, target_lines,
        lines[idx + 1:idx + 31],
        entry_kind="description:imports",
        note="re-insert the reverse-stripped Imports: entry at its "
             "locally sorted slot")]


@register("pkg_metadata")
def sel_pkg_metadata(spec, rng: random.Random, want: int) -> list[dict]:
    cs = spec.corpus_source
    params = dict(cs.get("params") or {})
    params.setdefault("max_items", want * 6 + 50)
    params.setdefault("members", ("NAMESPACE", "DESCRIPTION"))

    def handle(pkg: str, rel: str, text: str, extras: dict) -> list[dict]:
        return (_handle_namespace(pkg, text, extras) if rel == "NAMESPACE"
                else _handle_description(pkg, text, extras))

    return _scan_tarballs(spec, rng, "pkg_metadata.json",
                          dict(selector="pkg_metadata"), handle, params)


# ---------------------------------------------------------------------------
# case 4: expectation_completion (tarball tests/testthat/*.R)
# ---------------------------------------------------------------------------

def _paren_scan(line: str, start: int) -> tuple[int, bool]:
    """Scan from `start` (the open paren) to where depth first returns to 0:
    (end_col, balanced). String/char literals are skipped, escape aware."""
    depth, q, i = 0, None, start
    while i < len(line):
        c = line[i]
        if q:
            if c == "\\":
                i += 1             # skip the escaped char
            elif c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i, True
            if depth < 0:
                return i, False
        i += 1
    return len(line), False


def _expect_cut(line: str, mode: str) -> int | None:
    m = re.search(r"(?<![\w.])expect_", line)
    if not m:
        return None
    if mode == "expect_":
        return m.end()
    op = line.find("(", m.end())
    if op == -1:
        return None
    if mode == "open_paren":
        return op + 1
    if mode == "first_arg":       # after the first top-level comma
        depth, q = 0, None
        for i in range(op + 1, len(line)):
            c = line[i]
            if q:
                if c == "\\":
                    continue
                if c == q:
                    q = None
                continue
            if c in "\"'":
                q = c
            elif c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0:
                    return None
                depth -= 1
            elif c == "," and depth == 1:
                return i + 1
        return None
    return None


def extract_expectations(pkg: str, rel: str, text: str,
                         params: dict) -> list[dict]:
    """Single-line expect_* statements inside test_that blocks: the file is
    cut right after the typed `expect_` partial; the target is the author's
    exact line remainder."""
    mode = params.get("cut", "expect_")
    window = int(params.get("window_lines", 12))
    cap = int(params.get("per_file_cap", 3))
    lines = text.splitlines()
    out = []
    for row, line in enumerate(lines):
        if len(out) >= cap:
            break
        s = line.strip()
        if not s.startswith("expect_") or "(" not in s or len(line) > 250:
            continue
        cut = _expect_cut(line, mode)
        if cut is None:
            continue
        target = line[cut:].rstrip()
        if not target.strip():
            continue
        _, ok = _paren_scan(line, line.index("("))
        if not ok:                # not a balanced single-line call
            continue
        # a test_that header must sit above with >= 1 setup line between
        h = next((r for r in range(max(0, row - window), row)
                  if "test_that(" in lines[r]), None)
        if h is None or not any(lines[r].strip() for r in range(h + 1, row)):
            continue
        prefix = [lines[r] for r in range(max(h, row - window), row)]
        prefix.append(line[:cut])
        suffix = [lines[r] for r in
                  range(row + 1, min(len(lines), row + 1 + window))]
        out.append(_mk_item(
            pkg, rel, row, prefix, [target], suffix,
            cut_mode=mode,
            note=f"complete the author's expect_* assertion (cut after the "
                 f"typed partial, mode={mode})"))
    return out


@register("testthat_expectations")
def sel_testthat_expectations(spec, rng: random.Random, want: int) -> list[dict]:
    cs = spec.corpus_source
    params = dict(cs.get("params") or {})
    params.setdefault("max_items", want * 6 + 50)
    params.setdefault("members", ("*",))
    seen: dict[str, int] = {}

    def handle(pkg: str, rel: str, text: str, extras: dict) -> list[dict]:
        n = seen.get(pkg, 0)
        if n >= MAX_TEST_FILES_PER_TARBALL:
            return []
        got = extract_expectations(pkg, rel, text, params)
        if got:
            seen[pkg] = n + 1
        return got

    return _scan_tarballs(spec, rng, "testthat_expectations.json",
                          dict(selector="testthat_expectations",
                               cut=params.get("cut", "expect_")),
                          handle, params)


# ---------------------------------------------------------------------------
# case 5: trycatch_handler_completion (suffix convention: complete the
# handler clauses of an existing tryCatch — never re-emit/wrap above)
# ---------------------------------------------------------------------------

HANDLER_NAMES = ("error", "warning", "message", "finally")
_HANDLER_HEAD_RE = re.compile(r"^\s*(error|warning|message|finally)\s*=",
                              re.IGNORECASE)


def extract_trycatch_handlers(b: Bundle, rng: random.Random,
                              params: dict) -> list[dict]:
    """tryCatch/withCallingHandlers calls: the cut sits right after the
    comma that follows the guarded expression (start of the second
    argument); the target is the author's handler clauses verbatim through
    the call's closing paren."""
    max_lines = int(params.get("max_handler_lines", 4))
    window = int(params.get("window_lines", 8))
    cap = int(params.get("per_file_cap", 2))
    src = b.src
    out = []
    for n in _walk(b.tree.root_node):
        if n.type != "call" or S.callee_name(src, n) not in \
                ("tryCatch", "withCallingHandlers"):
            continue
        if n.end_point[0] - n.start_point[0] < 2:
            continue              # call must span >= 3 lines
        args = next((c for c in n.children if c.type == "arguments"), None)
        arg_nodes = [a for a in (args.children if args is not None else [])
                     if a.type == "argument"]
        if len(arg_nodes) < 2:
            continue
        cut = arg_nodes[1].start_byte
        target = src[cut:n.end_byte].decode("utf-8", "replace")
        if not _HANDLER_HEAD_RE.match(target):
            continue
        t_lines = [l.rstrip() for l in target.split("\n")]
        if not (1 <= len(t_lines) <= max_lines):
            continue
        if any(len(l) > 200 for l in t_lines) or len(target) > 500:
            continue
        crow, ccol = b.rowcol(cut)
        prefix = [b.line_str(r)
                  for r in range(max(0, n.start_point[0] - window), crow)]
        partial = b.line_str(crow)[:ccol]
        if partial.strip():
            prefix.append(partial)
        erow = n.end_point[0]
        suffix = [b.line_str(r) for r in
                  range(erow + 1, min(b.nlines(), erow + 1 + window))]
        out.append(_mk_item(
            b.package, b.rel, crow, prefix, t_lines, suffix,
            note="complete the condition handlers of the tryCatch (cut "
                 "after the guarded expression's closing comma)"))
        if len(out) >= cap:
            break
    return out


@register("trycatch_handlers")
def sel_trycatch_handlers(spec, rng: random.Random, want: int) -> list[dict]:
    cs = spec.corpus_source
    params = dict(cs.get("params") or {})
    params.setdefault("max_items", want * 6 + 50)
    return _scan_normalized(
        spec, rng, "trycatch_handlers.json",
        dict(selector="trycatch_handlers",
             max_handler_lines=params.get("max_handler_lines", 4)),
        extract_trycatch_handlers, S.list_packages(), params)


# ---------------------------------------------------------------------------
# case 6: mid_body_edit (ONE deterministic line-level mutation inside a
# function body; the target is only the changed line, the suffix pins the
# post-change remainder of the function — the scope-aware-context rule of
# docs/prompt-format.md: the enclosing function is always complete)
# ---------------------------------------------------------------------------

MID_BODY_STAT_FNS = ("mean", "sd", "var", "median")
MID_BODY_KINDS = ("arg_edit", "na_rm_insert", "rename_once", "insert_line")
_STAT_SHADOW_RE = re.compile(rb"(?<![\w.])(?:mean|sd|var|median)\s*<-")
_LITERAL_RE = re.compile(r"^(?:TRUE|FALSE|\d+(?:\.\d+)?L?)$")
_ASSIGN_LINE_RE = re.compile(r"^(\s*)([A-Za-z.][A-Za-z0-9._]*)\s*<-\s*(\S.*)$")
NA_RM_INSERT = b", na.rm = TRUE"


def _mutate_literal(text: str) -> str | None:
    """Deterministic constant edit (the whole point: GT is exact by
    construction): TRUE/FALSE flip; integer +1 (an L suffix survives);
    float with the last decimal digit shifted +1 (9 wraps to 8)."""
    if text in ("TRUE", "FALSE"):
        return "FALSE" if text == "TRUE" else "TRUE"
    if not _LITERAL_RE.match(text):
        return None
    if "." not in text:
        if text.endswith("L"):
            return f"{int(text[:-1]) + 1}L"
        return str(int(text) + 1)
    d = text[-1]
    return text[:-1] + (str(int(d) + 1) if d < "9" else "8")


def _fn_body(b: Bundle, fn) -> tuple | None:
    """(body_node, head_row, r0, r1, non_blank_body_rows) of a function with
    a braced body; r0/r1 are the `{`/`}` rows."""
    body = next((c for c in fn.children if c.type == "braced_expression"),
                None)
    if body is None:
        return None
    r0, r1 = body.start_point[0], body.end_point[0]
    nb = [r for r in range(r0 + 1, r1) if b.line_str(r).strip()]
    return body, fn.start_point[0], r0, r1, nb


def _arg_edit_cands(b: Bundle, body) -> list[tuple]:
    """Call arguments holding a plain numeric/logical constant; mutating the
    constant rewrites exactly one token of one line (target = that line)."""
    src = b.src
    out = []
    for n in _walk(body):
        if n.type != "argument" or n.parent is None \
                or n.parent.type != "arguments":
            continue
        val = next((c for c in n.children if c.is_named), None)
        if val is None or val.type not in ("float", "true", "false"):
            continue
        if val.start_point[0] != val.end_point[0]:
            continue          # single-line edit only
        old = node_text(src, val).decode("utf-8", "replace")
        new = _mutate_literal(old)
        if new is None or new == old:
            continue
        out.append(("arg_edit", val.start_point[0],
                    dict(sb=val.start_byte, old_b=old.encode(),
                         new_b=new.encode(), old_tok=old, new_tok=new)))
    return out


def _na_rm_cands(b: Bundle, body) -> list[tuple]:
    """mean/sd/var/median calls (stats::-qualified or bare) lacking na.rm:
    `, na.rm = TRUE` lands right before the closing paren (the scenarios
    extract_na_rm splice)."""
    src = b.src
    out = []
    for n in _walk(body):
        if n.type != "call" or S.callee_name(src, n) not in MID_BODY_STAT_FNS:
            continue
        if n.start_point[0] != n.end_point[0]:
            continue          # single-line calls only (exact line-region GT)
        call_txt = S.strip_strings(node_text(src, n))
        if S.NA_RM_EQ.search(call_txt):
            continue          # already has na.rm
        args_txt = call_txt[call_txt.find(b"(") + 1:-1].strip()
        if not args_txt or args_txt.endswith(b","):
            continue          # zero-arg call / trailing comma -> ambiguous
        ins = n.end_byte - 1  # the closing ')'
        while ins > 0 and src[ins - 1:ins] in (b" ", b"\t"):
            ins -= 1
        fn = S.callee_name(src, n)
        out.append(("na_rm_insert", n.start_point[0],
                    dict(ins_b=ins, stat_fn=fn)))
    return out


def _rename_cands(b: Bundle, fn_node, body) -> list[tuple]:
    """Declared locals (scenarios extract_rename conventions) with >= 2
    occurrences: ONE mid-body occurrence is renamed — single-site, the other
    occurrences keep the old name in the visible prefix/suffix."""
    src = b.src
    params = next((c for c in fn_node.children if c.type == "parameters"),
                  None)
    declared: set[str] = set()
    if params is not None:
        for p in _walk(params):
            if p.type == "identifier":
                declared.add(node_text(src, p).decode("utf-8", "replace"))
    for n in _walk(body):
        if n.type == "binary_operator" and n.children and \
                n.children[0].type == "identifier":
            declared.add(node_text(src, n.children[0])
                         .decode("utf-8", "replace"))

    id_occ: dict[str, list] = {}
    for n in _walk(body):
        if n.type != "identifier":
            continue
        if S.parent_is_caller(n):
            continue          # callee position: renaming changes the call
        p = n.parent
        if p is not None and p.type == "argument" and any(
                c.type == "=" for c in p.children) and \
                next(c for c in p.children if c.type == "identifier") is n:
            continue          # named-argument slot: renaming breaks the call
        tok = node_text(src, n).decode("utf-8", "replace")
        if (len(tok) < 3 or tok in S.RESERVED or not S.IDENT_RE.match(tok)
                or tok not in declared):
            continue
        id_occ.setdefault(tok, []).append((n.start_byte, n.end_byte))

    out = []
    for tok, occs in sorted(id_occ.items()):
        if len(occs) < 2:
            continue          # the old name must survive elsewhere (context)
        new_tok = S.derive_new_name(tok)
        if new_tok is None or new_tok in b.id_names or new_tok in b.str_contents:
            continue          # collision anywhere in file -> ambiguous
        for sb, eb in occs:
            out.append(("rename_once", b.rowcol(sb)[0],
                        dict(sb=sb, old_b=tok.encode(), new_b=new_tok.encode(),
                             old_tok=tok, new_tok=new_tok)))
    return out


def _insert_cands(b: Bundle, r0: int, r1: int) -> list[tuple]:
    """Insert-one-line: a corpus-attested assignment line of the SAME function
    (the neighbour pattern), re-emitted with a fresh LHS (scenarios
    derive_new_name) at the site's own indentation — every RHS variable is
    attested in-function by construction."""
    attested: list[tuple[str, str]] = []      # (lhs, rhs)
    for r in range(r0 + 1, r1):
        m = _ASSIGN_LINE_RE.match(b.line_str(r))
        if m and len(b.line_str(r)) <= 200:
            attested.append((m.group(2), m.group(3)))
    if not attested:
        return []
    out = []
    for anchor_row in range(r0 + 1, r1):
        line = b.line_str(anchor_row)
        if not line.strip():
            continue
        indent = line[:len(line) - len(line.lstrip())]
        for lhs, rhs in attested:
            lhs2 = S.derive_new_name(lhs)
            if lhs2 is None or lhs2 in b.id_names or lhs2 in b.str_contents:
                continue
            new_line = f"{indent}{lhs2} <- {rhs}"
            if len(new_line) > 220:
                continue
            out.append(("insert_line", anchor_row,
                        dict(new_line=new_line, insert_base=f"{indent}{lhs} <- {rhs}")))
    return out


def _one_clean_statement(text: str) -> bool:
    """The target line parses as clean R with exactly one top-level
    statement (the same tree-sitter floor the validator puts on draws)."""
    if not text.strip():
        return False
    if not fragment_clean(text):
        return False
    return len([c for c in parse_fragment(text).root_node.children
                if c.is_named]) == 1


def _mid_body_item(b: Bundle, kind: str, row: int, payload: dict,
                   window: int, head_row: int, r1: int) -> dict | None:
    """Assemble the suffix-convention item: prefix through the line before
    the change (typed-partial tail), target = the one changed/new line,
    suffix = the post-change remainder of the function (through the closing
    brace) plus a window of file-below lines."""
    lb = b.line_bytes(row)
    if kind == "insert_line":
        new_line, old_line = payload["new_line"], ""
    else:
        if kind == "na_rm_insert":
            new_lb = lb[:payload["ins_b"] - b.starts[row]] + NA_RM_INSERT + \
                lb[payload["ins_b"] - b.starts[row]:]
            old_tok = new_tok = ""
        else:
            col = payload["sb"] - b.starts[row]
            new_lb = lb[:col] + payload["new_b"] + lb[col + len(payload["old_b"]):]
            old_tok, new_tok = payload["old_tok"], payload["new_tok"]
        new_line = new_lb.decode("utf-8", "replace").rstrip("\r")
        old_line = b.line_str(row)
        if new_line == old_line:
            return None
    if not new_line.strip() or len(new_line) > 220 \
            or new_line.lstrip().startswith("#") \
            or not _one_clean_statement(new_line):
        return None
    cut = row + 1 if kind == "insert_line" else row
    prefix = [b.line_str(r) for r in range(max(0, head_row - window), cut)]
    if not prefix or not prefix[-1].strip():
        return None          # the cursor must sit at the end of a typed line
    suffix = [b.line_str(r) for r in
              range(row + 1, min(b.nlines(), r1 + 1 + window))]
    if not suffix:
        return None
    carry = dict(mutation_kind=kind, corpus_line=old_line,
                 fn_head=b.line_str(head_row),
                 note=f"{kind}: emit only the changed line; the post-change "
                      f"function remainder stays visible below (scope pin)")
    if kind in ("arg_edit", "rename_once"):
        carry.update(old_tok=old_tok, new_tok=new_tok)
    elif kind == "na_rm_insert":
        carry.update(stat_fn=payload["stat_fn"])
    elif kind == "insert_line":
        carry.update(insert_base=payload["insert_base"])
    return _mk_item(b.package, b.rel, row, prefix, [new_line], suffix, **carry)


def extract_mid_body_edits(b: Bundle, rng: random.Random,
                           params: dict) -> list[dict]:
    """Functions of 3+ body lines from one bundle: one deterministic
    mutation per function (kind shuffled), at a site that is strictly
    mid-body (a non-blank statement above AND below inside the braces)."""
    kinds = [k for k in (params.get("kinds") or MID_BODY_KINDS)
             if k in MID_BODY_KINDS]
    window = int(params.get("window_lines", 10))
    min_body = int(params.get("min_body_lines", 3))
    max_body = int(params.get("max_body_lines", 40))
    cap_file = int(params.get("per_file_cap", 2))
    cap_fn = int(params.get("per_function_cap", 1))
    shadowed = bool(_STAT_SHADOW_RE.search(b.src))
    out: list[dict] = []
    for fn in (n for n in _walk(b.tree.root_node)
               if n.type == "function_definition"):
        if len(out) >= cap_file:
            break
        geom = _fn_body(b, fn)
        if geom is None:
            continue
        body, head_row, r0, r1, nb = geom
        if not min_body <= len(nb) <= max_body:
            continue
        cands: dict[str, list] = {}
        if "arg_edit" in kinds:
            cands["arg_edit"] = _arg_edit_cands(b, body)
        if "na_rm_insert" in kinds and not shadowed:
            cands["na_rm_insert"] = _na_rm_cands(b, body)
        if "rename_once" in kinds:
            cands["rename_once"] = _rename_cands(b, fn, body)
        if "insert_line" in kinds:
            cands["insert_line"] = _insert_cands(b, r0, r1)
        cands = {k: lst for k, lst in cands.items() if lst}
        for lst in cands.values():
            rng.shuffle(lst)
        emitted, seen_rows = 0, set()
        while emitted < cap_fn and len(out) < cap_file and cands:
            # one KIND per emission slot (uniform over the kinds that still
            # have candidates) so the abundant insert/rename sites cannot
            # starve the rarer na_rm/arg_edit mutations
            kind = rng.choice(sorted(cands))
            _k, row, payload = cands[kind].pop()
            if not cands[kind]:
                del cands[kind]
            if row <= r0 or row >= r1 or row in seen_rows:
                continue
            if not (any(b.line_str(r).strip() for r in range(r0 + 1, row))
                    and any(b.line_str(r).strip()
                            for r in range(row + 1, r1))):
                continue          # strictly mid-body, never the last statement
            item = _mid_body_item(b, kind, row, payload, window, head_row, r1)
            if item is None:
                continue
            out.append(item)
            emitted += 1
            seen_rows.add(row)
    return out


def _resolve_pkg_versions(pool: list[str]) -> dict[str, str]:
    """{pkg: highest version dir} for the sampled packages only — the
    build_astfim versions.json cache pattern, scoped to what this selector
    touches (the full-corpus walk costs ~8 minutes on drvfs)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "mid_body_versions.json"
    versions: dict[str, str] = {}
    if cache.exists():
        try:
            blob = json.loads(cache.read_text())
            if blob.get("root") == str(S.ROOT):
                versions = dict(blob.get("versions") or {})
        except (ValueError, OSError):
            versions = {}
    dirty = False
    for pkg in pool:
        if pkg in versions:
            continue
        ver = ASTFIM.pick_version_dir(S.ROOT / pkg)
        if ver is not None:
            versions[pkg] = str(ver)
            dirty = True
    if dirty:
        try:
            cache.write_text(json.dumps(dict(root=str(S.ROOT),
                                             versions=versions)))
        except OSError:
            pass
    return versions


def iter_bundles_highest(package_names, rng: random.Random,
                         max_files: int = S.MAX_FILES_PER_PKG):
    """iter_bundles, but the version directory is the HIGHEST version of the
    package (build_astfim pick_version_dir + src_root_for), not whichever
    iterdir happens to yield first."""
    versions = _resolve_pkg_versions(list(package_names))
    for pkg in package_names:
        vd = versions.get(pkg)
        rdir = ASTFIM.src_root_for(Path(vd), pkg) if vd else None
        if rdir is None:
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
            if not src or len(src) > S.MAX_FILE_BYTES:
                continue
            yield Bundle(pkg, f"R/{f.name}", src)


def _scan_normalized_versions(spec, rng: random.Random, cache_name: str,
                              meta: dict, extract, params: dict,
                              pool: list[str] | None = None) -> list[dict]:
    """_scan_normalized with the highest-version-per-package iterator
    (mid_body_edit reads ONE canonical version of each package)."""
    max_items = int(params.get("max_items") or 2500)
    time_budget = float(params.get("time_budget_s", 2400))
    n_packages = int(params.get("sample_packages", 500))
    per_package = int(params.get("per_package_cap", 3))

    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / cache_name
    meta = dict(meta, max_items=max_items, sample_packages=n_packages,
                per_package_cap=per_package,
                pool=params.get("packages", "all"))
    if cache.exists():
        try:
            blob = json.loads(cache.read_text())
            if blob.get("params_hash") == _params_hash(meta):
                items = blob["items"]
                rng.shuffle(items)
                for i, it in enumerate(items):
                    it["key"] = f"{meta['selector']}:{i}"
                print(f"  [corpus] {cache_name} cache: {len(items)} items "
                      f"(scan {blob.get('scan')})")
                return items[:max_items]
        except (ValueError, KeyError, OSError):
            pass

    order = list(pool if pool is not None else S.list_packages())
    rng.shuffle(order)
    order = order[:n_packages] if n_packages > 0 else order
    items: list[dict] = []
    pkg_counts: dict[str, int] = {}
    t0, files = time.time(), 0
    for b in iter_bundles_highest(order, rng):
        files += 1
        if pkg_counts.get(b.package, 0) >= per_package:
            continue
        got = extract(b, rng, params)
        if got and pkg_counts.get(b.package, 0) + len(got) > per_package:
            got = got[:per_package - pkg_counts.get(b.package, 0)]
        if got:
            pkg_counts[b.package] = pkg_counts.get(b.package, 0) + len(got)
            items.extend(got)
        if len(items) >= max_items or time.time() - t0 > time_budget:
            break
        if files % 250 == 0:
            print(f"  [corpus] {cache_name}: files={files} "
                  f"items={len(items)} elapsed={time.time()-t0:.0f}s",
                  flush=True)
    items.sort(key=lambda it: (it.get("package", ""),
                               it.get("path", ""), it.get("row", 0)))
    for i, it in enumerate(items):
        it["key"] = f"{meta['selector']}:{i}"
    scan = dict(files=files, packages=len(pkg_counts),
                elapsed_s=round(time.time() - t0, 1), items=len(items))
    try:
        cache.write_text(json.dumps(dict(params_hash=_params_hash(meta),
                                         scan=scan, items=items)))
    except OSError:
        pass
    print(f"  [corpus] {cache_name} scan: {scan}")
    return items[:max_items]


@register("mid_body_sites")
def sel_mid_body_sites(spec, rng: random.Random, want: int) -> list[dict]:
    cs = spec.corpus_source
    params = dict(cs.get("params") or {})
    params.setdefault("max_items", want * 6 + 50)
    # the na_rm_insert mutation needs tidyverse-style code (mean/sd calls
    # lacking na.rm), which is rare in a uniform sample: the tidy pool first
    # (scenarios tidy_packages DESCRIPTION pre-scan), topped up with random
    # packages from the full corpus for breadth (pipe_chain_links convention)
    params.setdefault("packages", "tidy")
    pool = S.tidy_packages() if params.get("packages") == "tidy" \
        else S.list_packages()
    extra = int(params.get("extra_random_packages", 0) or 0)
    if extra > 0:
        rest = [p for p in S.list_packages() if p not in set(pool)]
        rng.shuffle(rest)
        pool = list(pool) + rest[:extra]
    return _scan_normalized_versions(
        spec, rng, "mid_body_sites.json",
        dict(selector="mid_body_sites",
             kinds=list(params.get("kinds") or MID_BODY_KINDS),
             window_lines=params.get("window_lines", 10),
             min_body_lines=params.get("min_body_lines", 3),
             max_body_lines=params.get("max_body_lines", 40)),
        extract_mid_body_edits, params, pool=pool)
