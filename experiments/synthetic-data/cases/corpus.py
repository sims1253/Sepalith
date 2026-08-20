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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scenarios/
import scenarios as S
from scenarios import Bundle, iter_bundles, node_text

from cases.validators import TIDYSELECT_HELPERS, _walk

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cases_cache"
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
