#!/usr/bin/env python3
"""Roxygen-drafting family: function -> docs, mined from the CRAN corpus.

For every top-level named function definition whose immediately preceding
roxygen block is RICH (contains at least one @param or @return tag), emit
one edit-pair example: cursor on the empty line where the docs belong, the
full function (signature + body) visible BELOW the cursor as the suffix,
target = the roxygen block the author wrote. The model reads the code and
drafts its documentation. Bare `#` comment runs above functions are NOT
docs and never emit.

Corpus: /mnt/h/sepalith/normalized/<pkg>/<version>/<pkg>/R/*.R (same store
finish_block.py / scenarios.py / comment_to_code.py mine).

Extraction is tree-sitter-r ONLY, reusing the finish_block.py query
pattern (root-level binary_operator with a <- / = / <<- operator and a
function_definition RHS, plus the backwards comment-node walk). There is
deliberately NO line-based fallback: tree_sitter_r ships in the uv venv
(finish_block.py already depends on it), and if the import ever fails the
script exits nonzero instead of silently degrading to regex parsing.

Skips: anonymous / complex assignment targets (lhs must be a plain
identifier), functions without a braced body, functions with < 3 body
lines, roxygen blocks < 3 lines, non-rich roxygen, and records whose
target + suffix alone exceed the char budget.

Budget: prefix + region_new + suffix kept <= MAX_RECORD_CHARS (5850) by
dropping prefix lines from the FRONT; the assembler's zeta2 renderer adds
~130 marker chars and drops rendered rows over 6000, so 5850 keeps rows
inside that gate. The prefix is additionally capped at the last 30 lines
above the block (comment-family convention, comment_to_code.py).

Record shape (the edit-pair schema the assembler already consumes, cf.
comment_drafting.jsonl):

  {"prefix": [file head above the block, front-truncated to budget],
   "region_old": [""], "cursor_idx": 0,
   "region_new": [roxygen block lines],          # the target
   "suffix": [name <- function(...) { body }],   # visible below the cursor
   "event_diff": "", "family": "roxygen_drafting",
   "package": ..., "path": "R/<file>.R", "note": "<fn>: N roxygen lines ..."}

Output: /mnt/h/sepalith/datasets/scenarios_v1/roxygen_drafting.jsonl
Resume: packages already listed in the roxygen_drafting.done.txt sidecar
are skipped, the JSONL is appended per package and the sidecar updated
after each package, so an interrupted run restarts where it stopped
(crash between the two appends can duplicate one package's rows; the
assembler dedups on rendered text anyway).

Resource-polite: CPU-only, <= 6 process workers, no network, no API keys.
Run it as:

  nice -n 19 uv run python experiments/synthetic-data/roxygen_drafting.py
  (add --limit 20 for a smoke test, --restart to rebuild from scratch)
"""
from __future__ import annotations

import argparse
import json
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
except ImportError as e:  # no fallback by design (see docstring)
    sys.exit(f"tree-sitter-r unavailable ({e}); refusing to run without it")

ROOT = Path("/mnt/h/sepalith/normalized")
OUT_DIR = Path("/mnt/h/sepalith/datasets/scenarios_v1")
OUT = OUT_DIR / "roxygen_drafting.jsonl"
DONE = OUT_DIR / "roxygen_drafting.done.txt"

MAX_FILE_BYTES = 400_000   # finish_block.py convention
MAX_PREFIX_LINES = 30      # comment_to_code.py convention
MAX_RECORD_CHARS = 5_850   # + ~130 zeta2 marker chars -> assembler's 6000 gate
MIN_ROXY_LINES = 3
MIN_BODY_LINES = 3
WORKERS = 6

_parser: Parser | None = None


def get_parser() -> Parser:
    """Per-process parser (workers fork; created lazily in each)."""
    global _parser
    if _parser is None:
        _parser = Parser(Language(tree_sitter_r.language()))
    return _parser


def extract_file(pkg: str, fname: str, src: bytes, rows: list, st: Counter):
    """Append one record per (rich roxygen block, function) pair in one file."""
    tree = get_parser().parse(src)
    lines_b = src.split(b"\n")
    starts = list(accumulate((len(l) + 1 for l in lines_b), initial=0))
    text = [l.decode("utf-8", "replace").rstrip("\r") for l in lines_b]

    def row(byte: int) -> int:
        return bisect_right(starts, byte) - 1

    children = tree.root_node.children
    for i, node in enumerate(children):
        if node.type != "binary_operator":
            continue
        kids = node.children
        rhs = [c for c in kids if c.type == "function_definition"]
        if not rhs or not any(c.type in ("<-", "=", "<<-") for c in kids):
            continue
        fn = rhs[0]
        st["functions"] += 1
        if kids[0].type != "identifier":
            st["skip_anon"] += 1  # x$y <- function, backtick names, etc.
            continue
        name = src[kids[0].start_byte:kids[0].end_byte].decode("utf-8", "replace")

        # roxygen block: contiguous #' comment lines directly above (a blank
        # line between block and function is allowed, as in finish_block.py)
        roxy_rows: list[int] = []
        j, prev = i - 1, None
        while j >= 0 and children[j].type == "comment":
            r = row(children[j].start_byte)
            if prev is not None and r != prev - 1:
                break  # blank line inside the run: keep only the bottom run
            if not text[r].lstrip().startswith("#'"):
                break  # plain # comment terminates the block
            roxy_rows.insert(0, r)
            prev, j = r, j - 1
        if not roxy_rows:
            st["skip_no_roxy"] += 1
            continue
        st["with_roxygen"] += 1
        roxy = text[roxy_rows[0]:roxy_rows[-1] + 1]
        roxy_text = "\n".join(roxy)
        if len(roxy) < MIN_ROXY_LINES:
            st["skip_short_roxy"] += 1
            continue
        if "@param" not in roxy_text and "@return" not in roxy_text:
            st["skip_not_rich"] += 1  # bare #' description blocks are not rich
            continue
        st["rich_roxygen"] += 1

        body = next((c for c in fn.children if c.type == "braced_expression"),
                    None)
        if body is None:
            st["skip_no_body"] += 1
            continue
        inner = src[body.start_byte + 1:body.end_byte - 1].decode(
            "utf-8", "replace")
        n_body = inner.count("\n") + 1 if inner.strip() else 0
        if n_body < MIN_BODY_LINES:
            st["skip_short_body"] += 1
            continue

        suffix = text[row(node.start_byte):row(fn.end_byte - 1) + 1]
        head = text[max(0, roxy_rows[0] - MAX_PREFIX_LINES):roxy_rows[0]]
        t_len = len("\n".join(roxy)) + len("\n".join(suffix))
        while head and t_len + len("\n".join(head)) > MAX_RECORD_CHARS:
            head.pop(0)  # truncate from the START, keep lines near the cursor
        if t_len + len("\n".join(head)) > MAX_RECORD_CHARS:
            st["skip_too_long"] += 1
            continue

        st["emitted"] += 1
        rows.append(dict(
            prefix=head,
            region_old=[""],
            cursor_idx=0,
            region_new=roxy,
            suffix=suffix,
            event_diff="",
            family="roxygen_drafting",
            package=pkg,
            path=f"R/{fname}",
            note=f"{name}: {len(roxy)} roxygen lines above {n_body}-line body",
        ))


def process_package(pkg: str):
    """Mine one package (worker entry point). Returns (rows, counters)."""
    st: Counter = Counter()
    rows: list = []
    rdir = None
    try:
        ver_dirs = sorted((d for d in (ROOT / pkg).iterdir() if d.is_dir()),
                          key=lambda d: d.name)
        if ver_dirs:
            rdir = ver_dirs[0] / pkg / "R"
    except OSError:
        rdir = None
    if rdir is None or not rdir.is_dir():
        st["no_source"] = 1
        return rows, st
    try:
        # dict-by-name: drvfs is case-insensitive, *.R and *.r can overlap
        files = sorted({f.name: f for f in list(rdir.glob("*.R"))
                        + list(rdir.glob("*.r"))}.values(),
                       key=lambda f: f.name)
    except OSError:
        st["no_source"] = 1
        return rows, st
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
            extract_file(pkg, f.name, src, rows, st)
        except Exception:
            st["extract_errors"] += 1
    return rows, st


def append_all(path: Path, chunk: str, tries: int = 20, wait: float = 10.0):
    """Append with retries (drvfs intermittently throws ENOMEM on open)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0,
                    help="scan only the first N pending packages (smoke test)")
    ap.add_argument("--restart", action="store_true",
                    help="clear previous output + progress and rebuild")
    args = ap.parse_args()
    t0 = time.time()

    # no per-entry is_dir() here: 14k drvfs stats cost ~200 s; a stray
    # non-directory is handled per worker (NotADirectoryError -> no_source)
    all_pkgs = sorted(p.name for p in ROOT.iterdir())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.restart:
        OUT.unlink(missing_ok=True)
        DONE.unlink(missing_ok=True)
    if OUT.exists():
        trim_partial_tail(OUT)

    done: set = set()
    if DONE.exists() and OUT.exists():
        done = set(DONE.read_text().split())
    elif DONE.exists():  # stale sidecar without its output file
        DONE.unlink(missing_ok=True)
    todo = [p for p in all_pkgs if p not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"packages: {len(all_pkgs)} total | {len(done)} done (skipped) | "
          f"{len(todo)} todo | workers={args.workers}", flush=True)

    totals: Counter = Counter()
    written = n_done = 0
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_package, p): p for p in todo}
            try:
                for fut in as_completed(futs):
                    pkg = futs[fut]
                    try:
                        rows, st = fut.result()
                    except Exception as e:
                        totals["worker_errors"] += 1
                        print(f"[error] {pkg}: {type(e).__name__}: {e}",
                              flush=True)
                        continue  # stays pending; a rerun retries it
                    totals.update(st)
                    ok = not rows or append_all(
                        OUT, "".join(json.dumps(r, ensure_ascii=False) + "\n"
                                     for r in rows))
                    if ok and append_all(DONE, pkg + "\n"):
                        written += len(rows)
                        n_done += 1
                    else:
                        totals["write_failures"] += 1
                    if n_done and n_done % 25 == 0:
                        el = time.time() - t0
                        print(f"[{n_done}/{len(todo)}] records={written} "
                              f"elapsed={el:.0f}s "
                              f"({n_done / max(el, 1):.2f} pkg/s) "
                              f"rich={totals['rich_roxygen']} last={pkg}",
                              flush=True)
            except KeyboardInterrupt:
                print("interrupted; rerun to resume from the sidecar",
                      flush=True)
                for f in futs:
                    f.cancel()
                raise

    # final counts + parse check over the whole output file
    n_lines = n_bad = 0
    pkgs_out: set = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                n_lines += 1
                try:
                    pkgs_out.add(json.loads(line)["package"])
                except ValueError:
                    n_bad += 1
    totals["records_in_file"] = n_lines
    totals["unparseable_lines"] = n_bad
    totals["packages_in_file"] = len(pkgs_out)
    totals["written_this_run"] = written
    totals["packages_done_sidecar"] = n_done
    totals["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(dict(totals), indent=1), flush=True)
    print(f"records -> {OUT} ({n_lines})", flush=True)


if __name__ == "__main__":
    main()
