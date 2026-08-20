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
import scenarios as S
from scenarios import Bundle, iter_bundles, node_text

from cases.validators import TIDYSELECT_HELPERS, _walk

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
