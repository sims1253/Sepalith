#!/usr/bin/env python3
"""Two SUFFIX-CONVENTION scenario families from the normalized CRAN corpus.

SFT v5 target convention: the model only ever predicts what comes AFTER the
cursor. Both families below are constructed natively in that convention.

no_op (the eagerness fix): teach the model to STOP instead of hallucinating
the next construct. Five kinds, emitted to no_op.jsonl:

  after_close_brace : cursor immediately after a top-level function's
      closing-brace line. region_old = ["}"] (+ cursor marker), target EMPTY
      (the assembler renders prompt + "\\n>>>>>>> UPDATED").
  blank_between     : cursor at the end of the FIRST blank line of a run of
      >= 2 blank lines between two top-level functions. region_old = [""]
      (+ cursor), target = the following blank line only.
  interior_line_end : cursor at the end of a complete statement line
      mid-body whose next line is blank (the natural pause between
      statements) — v8 geometry (eval_noop_fp: untrained = 100% FP).
  mid_identifier    : cursor INSIDE an identifier being typed (midtyping;
      rest of the line not typed yet) — the only sane suggestion at a
      mid-word pause is nothing — v8 geometry.
  blank_in_body     : cursor on a blank line strictly inside a body —
      the intra-function sibling of blank_between — v8 geometry.
  file_end          : cursor at the last non-blank line of the file, all
      code complete (eval class a_file_end, 100% FP on v7) — v8 geometry.

mid_roxygen: cursor at the END of line k INSIDE a roxygen block; k is now
drawn UNIFORMLY from the whole block interior (the positional-realism fix
— the old 3-candidate set left a dead zone at head-fraction 0.25-0.67).
region_old = the roxygen lines so far (+ cursor),
region_new = the REMAINING roxygen lines (suffix convention: nothing already
above the cursor is re-emitted), suffix = the function below the block.

Corpus walk: ONE version per package — highest version wins
(build_astfim.py's version_key/pick_version_dir; the astfim versions.json
map is reused when present, else resolved and cached here).

Row shape (the edit-pair schema the assembler consumes):

  {"prefix": [file lines above, front-truncated to budget],
   "region_old": [...], "cursor_idx": <line index the cursor follows>,
   "region_new": [...], "suffix": [...], "event_diff": "",
   "family": "no_op" | "mid_roxygen", "kind": ..., "package": ...,
   "path": "R/<file>.R", "note": ...}

Output: /mnt/h/sepalith/datasets/scenarios_v1/{no_op,mid_roxygen}.jsonl
Resume: no_op.done.txt / mid_roxygen.done.txt sidecars list processed
packages (roxygen_drafting.py convention); the JSONL is appended per
package and the sidecar updated after each package, so an interrupted run
restarts where it stopped (a crash between the two appends can duplicate
one package's rows; the assembler dedups on rendered text anyway).

Row targets: no_op 3-8k (kind mix ~60/40), mid_roxygen 5-10k; per-package
caps keep big packages from dominating, packages are visited in a seeded
shuffle so the sample spans the alphabet, and the run stops early once the
quotas fill.

Resource-polite: CPU-only, <= 6 process workers, no network, no API keys.
Run it as:

  nice -n 19 uv run python experiments/synthetic-data/suffix_scenarios.py
  (add --limit 20 for a smoke test, --restart to rebuild from scratch)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import accumulate
from pathlib import Path

try:
    import tree_sitter_r
    from tree_sitter import Language, Parser
except ImportError as e:  # no fallback by design (siblings do the same)
    sys.exit(f"tree-sitter-r unavailable ({e}); refusing to run without it")

ROOT = Path("/mnt/h/sepalith/normalized")
OUT_DIR = Path("/mnt/h/sepalith/datasets/scenarios_v1")
VERSION_CACHE = Path("/mnt/h/sepalith/datasets/astfim_v1/versions.json")
LOCAL_VERSION_CACHE = OUT_DIR / "suffix_scenarios.versions.json"

NO_OP_OUT = OUT_DIR / "no_op.jsonl"
NO_OP_DONE = OUT_DIR / "no_op.done.txt"
MID_OUT = OUT_DIR / "mid_roxygen.jsonl"
MID_DONE = OUT_DIR / "mid_roxygen.done.txt"

MAX_FILE_BYTES = 400_000   # finish_block.py convention
MAX_PREFIX_LINES = 30      # comment_to_code.py convention
MAX_RECORD_CHARS = 5_850   # + ~130 zeta2 marker chars -> assembler's 6000 gate
MIN_ROXY_LINES = 3
WORKERS = 6
SHUFFLE_SEED = 7           # package visit order (stable across reruns)

# row quotas and per-package caps. The three v8 geometries
# (interior_line_end / mid_identifier / blank_in_body) are the untrained
# stop geometries from eval_noop_fp (100% false-suggestion on v7) —
# restraint must generalize across cursor positions, not just the two
# geometries v6/v7 trained on.
NO_OP_QUOTA = dict(after_close_brace=4200, blank_between=2800,
                   interior_line_end=2400, mid_identifier=1200,
                   blank_in_body=1200, file_end=800)   # 12.6k total
NO_OP_PKG_CAP = dict(after_close_brace=3, blank_between=2,
                     interior_line_end=3, mid_identifier=2, blank_in_body=2,
                     file_end=1)
MID_QUOTA = 8000
MID_PKG_CAP = 4
BATCH = 240                # packages submitted per executor wave

_parser: Parser | None = None


def get_parser() -> Parser:
    """Per-process parser (workers fork; created lazily in each)."""
    global _parser
    if _parser is None:
        _parser = Parser(Language(tree_sitter_r.language()))
    return _parser


# ---------------------------------------------------------------------------
# corpus walk: ONE version per package, highest version wins (build_astfim.py)
# ---------------------------------------------------------------------------

def version_key(v: str):
    """Semver-ish CRAN version key: numeric components compared as ints
    ('1.2-0' == '1.2.0' > '1.2'), non-numeric residue as a string tiebreak."""
    parts = []
    for p in re.split(r"[.-]", v):
        m = re.match(r"(\d+)(.*)", p)
        parts.append((int(m.group(1)), m.group(2)) if m else (-1, p))
    return tuple(parts)


def pick_version_dir(pkg_dir: Path) -> Path | None:
    """Highest version directory of one package."""
    try:
        vers = [v for v in pkg_dir.iterdir() if v.is_dir()]
    except OSError:
        return None
    vers = [v for v in vers if version_key(v.name) != (-1,)]
    if not vers:
        return None
    return max(vers, key=lambda v: version_key(v.name))


def src_root_for(ver_dir: Path, pkg: str) -> Path | None:
    root = ver_dir / pkg / "R"
    if root.is_dir():
        return root
    try:  # case-mismatched directory names
        for child in ver_dir.iterdir():
            if child.name.lower() == pkg.lower() and (child / "R").is_dir():
                return child / "R"
    except OSError:
        pass
    return None


def resolve_versions(rescan: bool = False) -> dict:
    """{pkg: version dir path}; reuses the astfim cache when valid."""
    for cache in (VERSION_CACHE, LOCAL_VERSION_CACHE):
        if cache.exists() and not rescan:
            try:
                cached = json.loads(cache.read_text())
                if isinstance(cached, dict) and cached.get("root") == str(ROOT) \
                        and "versions" in cached:
                    return cached["versions"]
            except (ValueError, OSError):
                pass
    print("  resolving highest version per package (drvfs walk)...",
          flush=True)
    t0 = time.time()
    versions = {}
    for pkg in sorted(os.listdir(ROOT)):
        p = ROOT / pkg
        if not p.is_dir():
            continue
        ver = pick_version_dir(p)
        if ver is not None:
            versions[pkg] = str(ver)
    try:
        LOCAL_VERSION_CACHE.write_text(json.dumps(
            dict(root=str(ROOT), versions=versions)))
        print(f"  version map cached ({len(versions)} packages, "
              f"{time.time() - t0:.0f}s)", flush=True)
    except OSError as e:
        print(f"  [version-cache] write failed: {e}", flush=True)
    return versions


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def top_level_functions(children, src):
    """[(lhs_name, node, fn_node)] for named top-level function assignments."""
    out = []
    for node in children:
        if node.type != "binary_operator":
            continue
        kids = node.children
        rhs = [c for c in kids if c.type == "function_definition"]
        if not rhs or not any(c.type in ("<-", "=", "<<-") for c in kids):
            continue
        if kids[0].type != "identifier":
            continue
        name = src[kids[0].start_byte:kids[0].end_byte].decode(
            "utf-8", "replace")
        out.append((name, node, rhs[0]))
    return out


def trim_prefix(lines: list[str], budget: int, st: Counter, tag: str):
    """Last MAX_PREFIX_LINES lines, front-truncated to the char budget."""
    head = lines[-MAX_PREFIX_LINES:]
    used = 0
    for i, l in enumerate(head):
        used += len(l) + 1
    while head and used > budget:
        used -= len(head[0]) + 1
        head.pop(0)
    if head and used > budget:
        st[f"skip_{tag}_too_long"] += 1
        return None
    return head


def extract_no_op(pkg: str, fname: str, src: bytes, rows: list,
                  caps: Counter, st: Counter):
    """At most one after_close_brace + one blank_between row per file."""
    tree = get_parser().parse(src)
    text = [l.decode("utf-8", "replace").rstrip("\r")
            for l in src.split(b"\n")]
    starts = list(accumulate((len(l) + 1 for l in src.split(b"\n")),
                             initial=0))

    def row(byte: int) -> int:
        return bisect_right(starts, byte) - 1

    fns = top_level_functions(tree.root_node.children, src)
    base = dict(suffix=[], event_diff="", family="no_op", package=pkg,
                path=f"R/{fname}")

    # (a) cursor right after a top-level function's lone closing-brace line
    if caps["after_close_brace"] < NO_OP_PKG_CAP["after_close_brace"]:
        for name, node, fn in fns:
            body = next((c for c in fn.children
                         if c.type == "braced_expression"), None)
            if body is None:
                continue
            r_close = row(body.end_byte - 1)
            if r_close >= len(text) or text[r_close].strip() != "}":
                continue
            head = trim_prefix(text[:r_close], MAX_RECORD_CHARS, st, "a")
            if head is None:
                continue
            st["emitted_a"] += 1
            caps["after_close_brace"] += 1
            rows.append(dict(base, kind="after_close_brace",
                             prefix=head, region_old=[text[r_close]],
                             cursor_idx=0, region_new=[],
                             note=f"{name}: stop after closing brace"))
            break

    # (b) cursor at the end of the first blank line of a >=2 blank run
    #     between two top-level functions; target = the next blank line
    if caps["blank_between"] < NO_OP_PKG_CAP["blank_between"]:
        for (n1, e1, _), (n2, e2, _) in zip(fns, fns[1:]):
            r1, r2 = row(e1.end_byte - 1), row(e2.start_byte)
            gap = text[r1 + 1:r2]
            if len(gap) < 2 or any(g.strip() for g in gap):
                continue
            head = trim_prefix(text[:r1 + 2], MAX_RECORD_CHARS, st, "b")
            if head is None:
                continue
            st["emitted_b"] += 1
            caps["blank_between"] += 1
            rows.append(dict(base, kind="blank_between",
                             prefix=head, region_old=[""],
                             cursor_idx=0, region_new=[],
                             note=f"{n1}/{n2}: one more blank line, "
                                  f"then stop"))
            break

    # (c)-(e): the three untrained stop geometries (eval_noop_fp: v7
    # proposes on 100% of them; the trained two sit at ~30%). Same row
    # shape as (a)/(b): region_old anchors the cursor line, region_new
    # empty = the sanctioned suggestion is NOTHING.
    def _body_rows(fn):
        body = next((c for c in fn.children
                     if c.type == "braced_expression"), None)
        if body is None:
            return None
        return row(body.start_byte), row(body.end_byte - 1)

    # (c) interior_line_end: cursor at the end of a complete statement
    #     line INSIDE a body, next line blank (the natural mid-function
    #     pause). Not a finish_block cut: nothing was removed, the user
    #     is between statements.
    if caps["interior_line_end"] < NO_OP_PKG_CAP["interior_line_end"]:
        for name, node, fn in fns:
            geom = _body_rows(fn)
            if geom is None:
                continue
            r0, r1 = geom
            cut = next((r for r in range(r0 + 1, r1 - 1)
                        if text[r].strip()
                        and not text[r].lstrip().startswith("#")
                        and r + 1 < r1 and not text[r + 1].strip()
                        and any(text[rr].strip()
                                for rr in range(r0 + 1, r))), None)
            if cut is None:
                continue
            head = trim_prefix(text[:cut], MAX_RECORD_CHARS, st, "c")
            if head is None:
                continue
            st["emitted_c"] += 1
            caps["interior_line_end"] += 1
            rows.append(dict(base, kind="interior_line_end",
                             prefix=head, region_old=[text[cut]],
                             cursor_idx=0, region_new=[],
                             note=f"{name}: statement end mid-body, "
                                  f"blank line follows — stop"))
            break

    # (d) mid_identifier: cursor INSIDE an identifier the user is typing
    #     (not at a word boundary). The rest of the line is not typed
    #     yet — midtyping convention; the only sane suggestion at a
    #     mid-word pause is nothing. Cut position is seeded per site.
    if caps["mid_identifier"] < NO_OP_PKG_CAP["mid_identifier"]:
        idre = re.compile(r"[A-Za-z.][A-Za-z0-9._]{5,}")
        for name, node, fn in fns:
            geom = _body_rows(fn)
            if geom is None:
                continue
            r0, r1 = geom
            for r in range(r0 + 1, r1):
                line = text[r]
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                m = idre.search(line)
                if not m:
                    continue
                rng = random.Random(f"{pkg}:{fname}:{name}:{r}:midid")
                cutcol = m.start() + rng.randint(2, max(2, m.end()
                                                       - m.start() - 2))
                head = trim_prefix(text[:r], MAX_RECORD_CHARS, st, "d")
                if head is None:
                    continue
                st["emitted_d"] += 1
                caps["mid_identifier"] += 1
                rows.append(dict(base, kind="mid_identifier",
                                 prefix=head,
                                 region_old=[line[:cutcol]],
                                 cursor_idx=0, region_new=[],
                                 note=f"{name}: mid-identifier pause at "
                                      f"'{m.group()[:cutcol - m.start()]}' "
                                      f"— stop"))
                break
            else:
                continue
            break

    # (e) blank_in_body: cursor on a blank line strictly inside a body
    #     (statements above and below) — the intra-function sibling of
    #     blank_between.
    if caps["blank_in_body"] < NO_OP_PKG_CAP["blank_in_body"]:
        for name, node, fn in fns:
            geom = _body_rows(fn)
            if geom is None:
                continue
            r0, r1 = geom
            cut = next((r for r in range(r0 + 1, r1)
                        if not text[r].strip()
                        and any(text[rr].strip()
                                for rr in range(r0 + 1, r))
                        and any(text[rr].strip()
                                for rr in range(r + 1, r1))), None)
            if cut is None:
                continue
            head = trim_prefix(text[:cut + 1], MAX_RECORD_CHARS, st, "e")
            if head is None:
                continue
            st["emitted_e"] += 1
            caps["blank_in_body"] += 1
            rows.append(dict(base, kind="blank_in_body",
                             prefix=head, region_old=[""],
                             cursor_idx=0, region_new=[],
                             note=f"{name}: blank line mid-body — stop"))
            break

    # (f) file_end: cursor at the last non-blank line of the file —
    #     everything is complete, there is nothing left to propose
    if caps["file_end"] < NO_OP_PKG_CAP["file_end"]:
        last = max((r for r in range(len(text)) if text[r].strip()),
                   default=None)
        if last is not None and not text[last].lstrip().startswith("#") \
                and len(text) - 1 - last <= 3:
            head = trim_prefix(text[:last], MAX_RECORD_CHARS, st, "f")
            if head is not None:
                st["emitted_f"] += 1
                caps["file_end"] += 1
                rows.append(dict(base, kind="file_end",
                                 prefix=head, region_old=[text[last]],
                                 cursor_idx=0, region_new=[],
                                 note=f"end of file (last non-blank "
                                      f"line {last + 1}) — stop"))


def extract_mid_roxygen(pkg: str, fname: str, src: bytes, rows: list,
                        caps: Counter, st: Counter):
    """1-2 rows per roxygen block (varied k), per-package capped."""
    tree = get_parser().parse(src)
    text = [l.decode("utf-8", "replace").rstrip("\r")
            for l in src.split(b"\n")]
    starts = list(accumulate((len(l) + 1 for l in src.split(b"\n")),
                             initial=0))

    def row(byte: int) -> int:
        return bisect_right(starts, byte) - 1

    children = tree.root_node.children
    for i, (name, node, fn) in enumerate(top_level_functions(children, src)):
        if caps["mid"] >= MID_PKG_CAP:
            return
        # roxygen block: contiguous #' comment lines directly above (the
        # bottom run only, as in roxygen_drafting.py)
        roxy_rows: list[int] = []
        j, prev = i - 1, None
        while j >= 0 and children[j].type == "comment":
            r = row(children[j].start_byte)
            if prev is not None and r != prev - 1:
                break
            if not text[r].lstrip().startswith("#'"):
                break
            roxy_rows.insert(0, r)
            prev, j = r, j - 1
        if len(roxy_rows) < MIN_ROXY_LINES:
            continue
        roxy = text[roxy_rows[0]:roxy_rows[-1] + 1]
        L = len(roxy)
        if L < MIN_ROXY_LINES + 1:      # need k >= 0 and a non-empty rest
            continue
        suffix = text[row(node.start_byte):row(fn.end_byte - 1) + 1]
        if not suffix:
            continue
        # k drawn UNIFORMLY from the whole block interior (the
        # positional-realism fix: the old 3-candidate set left a dead
        # zone at head-fraction 0.25-0.67); seeded per base sample, so
        # the same block always yields the same cut on reruns
        rng = random.Random(f"{pkg}:{fname}:{name}")
        k = rng.randrange(0, L - 1)
        core = len("\n".join(roxy)) + len("\n".join(suffix))
        if core > MAX_RECORD_CHARS:
            st["skip_mid_too_long"] += 1
            continue
        head = trim_prefix(text[:roxy_rows[0]], MAX_RECORD_CHARS - core,
                           st, "mid")
        if head is None:
            continue
        st["emitted_mid"] += 1
        caps["mid"] += 1
        rows.append(dict(
            prefix=head,
            region_old=roxy[:k + 1],
            cursor_idx=k,
            region_new=roxy[k + 1:],   # suffix convention: only the rest
            suffix=suffix,
            event_diff="",
            family="mid_roxygen",
            package=pkg,
            path=f"R/{fname}",
            note=f"{name}: roxygen line {k + 1}/{L}, "
                 f"{L - k - 1} lines remain",
        ))


def process_package(pkg: str, ver_dir: str):
    """Worker: one package -> (no_op rows, mid_roxygen rows, counters)."""
    st: Counter = Counter()
    no_rows: list = []
    mid_rows: list = []
    root = src_root_for(Path(ver_dir), pkg)
    if root is None:
        st["no_source"] = 1
        return no_rows, mid_rows, st
    try:
        # dict-by-name: drvfs is case-insensitive, *.R and *.r can overlap
        files = sorted({f.name: f for f in list(root.glob("*.R"))
                        + list(root.glob("*.r"))}.values(),
                       key=lambda f: f.name)
    except OSError:
        st["no_source"] = 1
        return no_rows, mid_rows, st
    caps_a = Counter()
    caps_m = Counter()
    for f in files:
        try:
            src = f.read_bytes()
        except OSError:
            st["read_errors"] += 1
            continue
        if not src or len(src) > MAX_FILE_BYTES:
            st["size_skips"] += 1
            continue
        st["files"] += 1
        try:
            if sum(caps_a.values()) < sum(NO_OP_PKG_CAP.values()):
                extract_no_op(pkg, f.name, src, no_rows, caps_a, st)
            if caps_m["mid"] < MID_PKG_CAP:
                extract_mid_roxygen(pkg, f.name, src, mid_rows, caps_m, st)
        except Exception:
            st["extract_errors"] += 1
    return no_rows, mid_rows, st


# ---------------------------------------------------------------------------
# drvfs-tolerant output handling (roxygen_drafting.py conventions)
# ---------------------------------------------------------------------------

def append_all(path: Path, chunk: str, tries: int = 20, wait: float = 10.0):
    for attempt in range(tries):
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(chunk)
                fh.flush()
            return True
        except OSError as e:
            if attempt == tries - 1:
                print(f"[nas-write] giving up on {path}: {e}", flush=True)
                return False
            print(f"[nas-write] {e}; retry {attempt + 1}/{tries - 1} "
                  f"in {wait:.0f}s", flush=True)
            time.sleep(wait)
    return False


def trim_partial_tail(path: Path) -> None:
    """Drop a trailing partial JSONL line left by a crash mid-append."""
    size = path.stat().st_size
    if size == 0:
        return
    with open(path, "rb") as fh:
        fh.seek(max(0, size - 65536))
        tail = fh.read()
    if tail.endswith(b"\n"):
        return
    cut = tail.rfind(b"\n")
    keep = (size - len(tail)) + (cut + 1 if cut >= 0 else 0)
    with open(path, "r+b") as fh:
        fh.truncate(keep)


def count_existing(path: Path) -> tuple[int, Counter]:
    """(rows, per-kind counts) already written, for quota resume."""
    n, kinds = 0, Counter()
    if not path.exists():
        return n, kinds
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        n += 1
        kinds[r.get("kind", "?")] += 1
    return n, kinds


def load_done(path: Path) -> set:
    if path.exists():
        return set(path.read_text().split())
    return set()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0,
                    help="scan only the first N pending packages (smoke test)")
    ap.add_argument("--restart", action="store_true",
                    help="clear previous output + progress and rebuild")
    args = ap.parse_args()
    try:  # resource-polite even when launched without nice
        os.nice(19)
    except OSError:
        pass
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.restart:
        for p in (NO_OP_OUT, NO_OP_DONE, MID_OUT, MID_DONE):
            p.unlink(missing_ok=True)
    for out in (NO_OP_OUT, MID_OUT):
        if out.exists():
            trim_partial_tail(out)
    # stale sidecar without its output file
    if NO_OP_DONE.exists() != NO_OP_OUT.exists():
        NO_OP_DONE.unlink(missing_ok=True)
    if MID_DONE.exists() != MID_OUT.exists():
        MID_DONE.unlink(missing_ok=True)

    done = load_done(NO_OP_DONE) | load_done(MID_DONE)
    versions = resolve_versions()
    pkgs = sorted(versions)
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(pkgs)  # sample spans the alphabet, stable across reruns
    todo = [p for p in pkgs if p not in done]
    if args.limit:
        todo = todo[: args.limit]
    # in-memory quotas (seeded once from the existing outputs, then counted
    # down per appended row; avoids re-parsing the JSONL per package)
    _, kinds = count_existing(NO_OP_OUT)
    n_mid, _ = count_existing(MID_OUT)
    left = {kind: max(0, quota - kinds.get(kind, 0))
            for kind, quota in NO_OP_QUOTA.items()}
    left["mid_roxygen"] = max(0, MID_QUOTA - n_mid)
    print(f"packages: {len(pkgs)} total | {len(done)} done (skipped) | "
          f"{len(todo)} todo | workers={args.workers} | quotas left "
          f"{dict(left)}", flush=True)

    totals: Counter = Counter()
    written = Counter()
    n_done = 0

    def take(rows: list, kind: str | None, quota: int) -> list:
        if kind is None:
            return rows[:quota]
        return [r for r in rows if r["kind"] == kind][:quota]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        try:
            for i in range(0, len(todo), BATCH):
                if not any(left.values()):
                    print("quotas filled; stopping early", flush=True)
                    break
                futs = {ex.submit(process_package, p, versions[p]): p
                        for p in todo[i:i + BATCH]}
                for fut in as_completed(futs):
                    pkg = futs[fut]
                    try:
                        no_rows, mid_rows, st = fut.result()
                    except Exception as e:
                        totals["worker_errors"] += 1
                        print(f"[error] {pkg}: {type(e).__name__}: {e}",
                              flush=True)
                        continue  # stays pending; a rerun retries it
                    totals.update(st)
                    takes = [take(no_rows, kind, left[kind])
                             for kind in NO_OP_QUOTA]
                    for t in takes:
                        for r in t:
                            left[r["kind"]] = max(0, left[r["kind"]] - 1)
                    take_m = take(mid_rows, None, left["mid_roxygen"])
                    left["mid_roxygen"] = max(0, left["mid_roxygen"]
                                              - len(take_m))
                    ok = (not any(takes) or append_all(
                        NO_OP_OUT, "".join(
                            json.dumps(r, ensure_ascii=False) + "\n"
                            for t in takes for r in t)))
                    ok = ok and (not take_m or append_all(
                        MID_OUT, "".join(
                            json.dumps(r, ensure_ascii=False) + "\n"
                            for r in take_m)))
                    if ok and append_all(NO_OP_DONE, pkg + "\n") \
                            and append_all(MID_DONE, pkg + "\n"):
                        written["no_op"] += sum(len(t) for t in takes)
                        written["mid"] += len(take_m)
                        n_done += 1
                    else:
                        totals["write_failures"] += 1
                    if n_done and n_done % 50 == 0:
                        el = time.time() - t0
                        print(f"[{n_done}] written={dict(written)} "
                              f"left={dict(left)} elapsed={el:.0f}s "
                              f"last={pkg}", flush=True)
        except KeyboardInterrupt:
            print("interrupted; rerun to resume from the sidecars",
                  flush=True)
            raise

    # final counts + parse check over the outputs
    report = {}
    for name, out in (("no_op", NO_OP_OUT), ("mid_roxygen", MID_OUT)):
        n_lines = n_bad = 0
        kinds = Counter()
        pkgs_out: set = set()
        if out.exists():
            for line in open(out, encoding="utf-8"):
                n_lines += 1
                try:
                    r = json.loads(line)
                    kinds[r.get("kind", "?")] += 1
                    pkgs_out.add(r["package"])
                except (ValueError, KeyError):
                    n_bad += 1
        report[name] = dict(rows=n_lines, kinds=dict(kinds),
                            packages=len(pkgs_out), bad_lines=n_bad)
        assert n_bad == 0, f"{out}: {n_bad} unparseable lines"
    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "suffix_scenarios.stats.json").write_text(
        json.dumps(dict(report=report, totals=dict(totals)), indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
