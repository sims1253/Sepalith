#!/usr/bin/env python3
"""AST-FIM midtraining corpus builder (data only) for the base-model bake-off.

Fill-in-the-middle over SEMANTIC spans of the normalized CRAN corpus
(/mnt/h/sepalith/normalized/<pkg>/<version>/<pkg>/R/*.R): the missing middle
is always a tree-sitter-delimited span -- a function body (predict-the-body:
the prefix keeps the signature line through `{`), a top-level block, or an
argument list -- never a random character cut.

Row shape mirrors experiments/post-processing/format_sft_v1.py (text =
prompt + target concatenated), but renders the SEPALITH PSM markers from
docs/prompt-format.md instead of the Zeta-2 markers:

  prompt = <|context|>pkg/R/foo.R
           ... file content above the span ...          (prefix)
           <|history|>
                                                             (no events)
           <|cursor|><|suffix|>                          (span removed)
           ... file content below the span ...           (suffix)
           <|end|>
  target = <span text>
           <|end|>                                       (model stops here)

The model fills the empty zone between <|cursor|> and <|suffix|>; it stops
when it emits <|end|> (docs/prompt-format.md "The prompt"). Prefix/suffix are
byte-exact file slices: prefix + span + suffix == source, asserted per span.

Corpus hygiene (see bake-off spec):
  - ONE version per package: highest version wins (semver-ish sort), which
    removes the 4-7x multi-version duplication.
  - max 6000 chars per row (drop longer), max 4 spans per file (largest
    first, mutually non-overlapping), files > 2000 lines skipped.

Output: /mnt/h/sepalith/datasets/astfim_v1/train-<shard>.jsonl (~100k rows
per shard) + stats.json {files, spans, rows, dropped, est_tokens (chars/3.5)}
+ progress.jsonl (one line per package, written BEFORE the package's rows so
a crash truncates cleanly on resume; the last entry's shard is rolled back to
its recorded byte offset).

drvfs: every write goes through the _nas_write retry pattern from
comment_to_code.py (OSError errno 12 flaps), extended to roll back partial
appends by truncating to the pre-write size. CPU-only; run under
`nice -n 19`, at most 6 workers.

Usage:
  nice -n 19 python3 experiments/synthetic-data/build_astfim.py --workers 6
  python3 experiments/synthetic-data/build_astfim.py --limit 5 --out /tmp/astfim_smoke
  python3 experiments/synthetic-data/build_astfim.py --split-eval  # after build
  python3 experiments/synthetic-data/build_astfim.py --verify      # shards
  python3 experiments/synthetic-data/build_astfim.py --spot 3      # 3 rows
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import re
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/mnt/h/sepalith/normalized")
OUT = Path("/mnt/h/sepalith/datasets/astfim_v1")

# PSM markers (docs/prompt-format.md "The prompt")
CONTEXT, HISTORY, CURSOR, SUFFIX, END = (
    "<|context|>", "<|history|>", "<|cursor|>", "<|suffix|>", "<|end|>")

MAX_ROW_CHARS = 6000     # prompt + target budget (~1.7k tokens)
MIN_SPAN_CHARS = 30      # tiny spans (empty bodies, `c(1, 2)`) carry no signal
MAX_SPANS_PER_FILE = 4   # prefer largest, mutually non-overlapping
MAX_FILE_LINES = 2000
MAX_FILE_BYTES = 2_000_000
ROWS_PER_SHARD = 100_000
CHUNK = 24               # packages per executor.map batch
STATS_EVERY = 100        # packages between stats.json rewrites
NAS_TRIES, NAS_WAIT_S = 20, 30.0
TOP_LEVEL_TYPES = {
    "binary_operator", "call", "if_statement", "for_statement",
    "while_statement", "repeat_statement", "unary_operator",
}

_PARSER = None


def _init_worker():
    global _PARSER
    import tree_sitter_r
    from tree_sitter import Language, Parser
    _PARSER = Parser(Language(tree_sitter_r.language()))


def traverse(n):
    stack = [n]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(x.children)


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


# ---------------------------------------------------------------------------
# span extraction
# ---------------------------------------------------------------------------

def collect_candidates(src: bytes, tree) -> list[dict]:
    """(start, end, kind) byte ranges of semantic spans, one dict each."""
    out = []

    def add(start, end, kind):
        out.append(dict(start=start, end=end, kind=kind,
                        size=end - start))

    for fn in traverse(tree.root_node):
        if fn.type != "function_definition":
            continue
        body = next((c for c in fn.children if c.type == "braced_expression"),
                    None)
        if body is None or body.end_byte - body.start_byte < 3:
            continue
        # predict-the-body: span is the content between the braces; the
        # prefix keeps everything through `{`, the suffix starts at `}`.
        add(body.start_byte + 1, body.end_byte - 1, "function_body")

    for node in tree.root_node.children:
        if not node.is_named or node.type not in TOP_LEVEL_TYPES:
            continue
        if node.type == "binary_operator" and \
                any(c.type == "function_definition" for c in node.children):
            continue  # covered by its function_body span
        add(node.start_byte, node.end_byte, "top_level_block")

    for node in traverse(tree.root_node):
        if node.type != "arguments" or node.parent is None \
                or node.parent.type != "call":
            continue
        if sum(1 for c in node.children if c.type == "argument") < 2:
            continue
        if node.end_byte - node.start_byte < 3:
            continue  # "()" -- nothing inside the delimiters
        # the arguments node includes its parens; mirror the function-body
        # convention and keep the delimiters in the prefix/suffix
        add(node.start_byte + 1, node.end_byte - 1, "argument_list")

    return out


def render(pkg: str, rel: str, src: bytes, span: dict) -> tuple[str, str]:
    """(prompt, target) in PSM marker order; prefix+span+suffix == src.
    CRLF is folded to LF (drvfs corpus keeps both). Raises ValueError if the
    source itself contains a PSM marker string (such a row would break
    marker-count invariants at serving time)."""
    pre = src[:span["start"]].decode("utf-8", "replace")\
        .replace("\r\n", "\n").rstrip("\n")
    mid = src[span["start"]:span["end"]].decode("utf-8", "replace")\
        .replace("\r\n", "\n").strip("\n")
    suf = src[span["end"]:].decode("utf-8", "replace")\
        .replace("\r\n", "\n").strip("\n")
    for marker in (CONTEXT, HISTORY, CURSOR, SUFFIX, END):
        if marker in pre or marker in mid or marker in suf:
            raise ValueError(marker)
    prompt = (f"{CONTEXT}{pkg}/{rel}\n{pre}\n{HISTORY}\n\n"
              f"{CURSOR}{SUFFIX}\n{suf}\n{END}\n")
    target = f"{mid}\n{END}"
    return prompt, target


def select_spans(spans: list[dict], dropped: dict) -> list[dict]:
    """At most MAX_SPANS_PER_FILE per file, preferring the largest. One slot
    is reserved for the largest argument_list: function bodies are almost
    always bigger, so a pure size sort would starve that span type entirely
    (it would never appear in the corpus despite being a named objective)."""
    def ok(c, occupied):
        return not any(c["start"] < e and c["end"] > s for s, e in occupied)

    arg = [c for c in spans if c["kind"] == "argument_list"]
    rest = [c for c in spans if c["kind"] != "argument_list"]
    picked, occupied = [], []
    if arg:
        c = max(arg, key=lambda c: c["size"])
        picked.append(c)
        occupied.append((c["start"], c["end"]))
    for c in sorted(rest, key=lambda c: c["size"], reverse=True):
        if len(picked) >= MAX_SPANS_PER_FILE:
            break
        if ok(c, occupied):
            picked.append(c)
            occupied.append((c["start"], c["end"]))
    dropped["span_cap"] = dropped.get("span_cap", 0) + (len(spans) - len(picked))
    return picked


def process_file(src: bytes, pkg: str, rel: str,
                 dropped: dict) -> list[dict]:
    tree = _PARSER.parse(src)
    cands = collect_candidates(src, tree)
    spans = []
    for c in cands:  # min-size gate
        text = src[c["start"]:c["end"]].decode("utf-8", "replace").strip()
        if len(text) < MIN_SPAN_CHARS:
            dropped["span_small"] = dropped.get("span_small", 0) + 1
        else:
            spans.append(c)
    picked = select_spans(spans, dropped)
    rows = []
    for c in picked:
        assert src[:c["start"]] + src[c["start"]:c["end"]] + \
            src[c["end"]:] == src  # byte-exact splice invariant
        try:
            prompt, target = render(pkg, rel, src, c)
        except ValueError:
            dropped["markers_in_src"] = dropped.get("markers_in_src", 0) + 1
            continue
        row = dict(text=prompt + target, prompt=prompt, target=target,
                   kind=c["kind"], package=pkg, path=rel)
        if len(row["text"]) > MAX_ROW_CHARS:
            dropped["row_chars"] = dropped.get("row_chars", 0) + 1
            continue
        rows.append(row)
    return rows


def process_package(pkg: str, ver_dir: str) -> dict:
    """Worker: one package -> rendered JSON lines + counters."""
    dropped: dict = {}
    lines, files, spans_n, chars = [], 0, 0, 0
    root = src_root_for(Path(ver_dir), pkg)
    if root is None:
        return dict(pkg=pkg, lines=lines, files=0, spans=0, chars=0,
                    rows=0, dropped=dropped)
    try:
        files_list = sorted(list(root.glob("*.R")) + list(root.glob("*.r")))
    except OSError:
        return dict(pkg=pkg, lines=lines, files=0, spans=0, chars=0,
                    rows=0, dropped=dropped)
    for f in files_list:
        try:
            src = f.read_bytes()
        except OSError:
            dropped["file_read"] = dropped.get("file_read", 0) + 1
            continue
        if not src:
            continue
        if src.count(b"\n") + 1 > MAX_FILE_LINES or len(src) > MAX_FILE_BYTES:
            dropped["file_lines"] = dropped.get("file_lines", 0) + 1
            continue
        rows = process_file(src, pkg, f"R/{f.name}", dropped)
        files += 1
        spans_n += len(rows)
        for row in rows:
            chars += len(row["text"])
            lines.append(json.dumps(row, ensure_ascii=False))
    return dict(pkg=pkg, lines=lines, files=files, spans=spans_n,
                chars=chars, rows=len(lines), dropped=dropped)


# ---------------------------------------------------------------------------
# drvfs-tolerant appends (comment_to_code.py _nas_write_lines pattern,
# extended to roll back partial appends before retrying)
# ---------------------------------------------------------------------------

def nas_append(path: Path, data: str, tries: int = NAS_TRIES,
               wait_s: float = NAS_WAIT_S) -> bool:
    for attempt in range(tries):
        good = 0
        try:
            good = path.stat().st_size
        except OSError:
            pass
        try:
            with open(path, "ab") as fh:
                fh.write(data.encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
            return True
        except OSError as e:
            try:  # roll back any partial bytes from the failed attempt
                with open(path, "r+b") as fh:
                    fh.truncate(good)
            except OSError:
                pass
            if attempt == tries - 1:
                print(f"  [nas-write] giving up on {path}: {e}", flush=True)
                return False
            print(f"  [nas-write] {e}; retry {attempt + 1}/{tries - 1} "
                  f"in {wait_s:.0f}s", flush=True)
            time.sleep(wait_s)
    return False


# ---------------------------------------------------------------------------
# builder main loop
# ---------------------------------------------------------------------------

def _agg(stats: dict, res: dict):
    stats["files"] += res["files"]
    stats["spans"] += res["spans"]
    stats["rows"] += res["rows"]
    stats["chars"] += res["chars"]
    stats["packages"] += 1
    for k, v in res["dropped"].items():
        stats["dropped"][k] = stats["dropped"].get(k, 0) + v


def _write_stats(out: Path, stats: dict, complete: bool):
    stats = dict(stats)
    stats["est_tokens"] = int(stats["chars"] / 3.5)
    stats["packages_done"] = stats.pop("packages")
    stats["complete"] = complete
    for attempt in range(NAS_TRIES):
        try:
            (out / "stats.json").write_text(json.dumps(stats, indent=1))
            return
        except OSError as e:
            if attempt == NAS_TRIES - 1:
                print(f"  [stats] giving up: {e}", flush=True)
                return
            time.sleep(10)


def load_progress(out: Path) -> tuple[set, dict, list]:
    """Resume state: done packages, aggregated counters (excluding the last,
    rolled-back entry), and the valid entry list."""
    stats = dict(files=0, spans=0, rows=0, chars=0, packages=0, dropped={})
    entries = []
    path = out / "progress.jsonl"
    if not path.exists():
        return set(), stats, entries
    try:
        raw = path.read_text().splitlines()
    except OSError:
        return set(), stats, entries
    for line in raw:
        try:
            entries.append(json.loads(line))
        except ValueError:
            print(f"  [resume] ignoring partial progress line", flush=True)
    if not entries:
        return set(), stats, entries
    # the last entry's rows may be absent or partial: roll its shard back to
    # the pre-write offset and reprocess that package. A zero-row entry
    # (e.g. an eval-holdout pin written by --split-eval) never wrote shard
    # bytes, so it is trusted as-is -- dropping it would reprocess a held-out
    # package and leak its rows back into the train shards.
    last = entries[-1]
    if last.get("rows", 0) == 0:
        for e in entries:
            stats["files"] += e.get("files", 0)
            stats["spans"] += e.get("spans", 0)
            stats["rows"] += e.get("rows", 0)
            stats["chars"] += e.get("chars", 0)
            stats["packages"] += 1
            for k, v in e.get("dropped", {}).items():
                stats["dropped"][k] = stats["dropped"].get(k, 0) + v
        return {e["pkg"] for e in entries}, stats, entries
    shard = out / f"train-{last['shard']:03d}.jsonl"
    try:
        if shard.stat().st_size > last["off"]:
            with open(shard, "r+b") as fh:
                fh.truncate(last["off"])
            print(f"  [resume] rolled back {last['pkg']} "
                  f"({shard.name} -> {last['off']} bytes)", flush=True)
    except OSError as e:
        print(f"  [resume] rollback failed ({e}); continuing", flush=True)
    entries = entries[:-1]
    for e in entries:
        stats["files"] += e.get("files", 0)
        stats["spans"] += e.get("spans", 0)
        stats["rows"] += e.get("rows", 0)
        stats["chars"] += e.get("chars", 0)
        stats["packages"] += 1
        for k, v in e.get("dropped", {}).items():
            stats["dropped"][k] = stats["dropped"].get(k, 0) + v
    return {e["pkg"] for e in entries}, stats, entries


def shard_row_counts(out: Path) -> dict[int, int]:
    """Rows already in each shard (newline-terminated lines only)."""
    counts = {}
    for p in sorted(out.glob("train-*.jsonl")):
        idx = int(p.stem.split("-")[1])
        n, tail = 0, b""
        try:
            with open(p, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    tail = (tail + chunk)[-1:]
                    n += chunk.count(b"\n")
        except OSError:
            pass
        counts[idx] = n
    return counts


def _work1(item):
    return process_package(*item)


def resolve_versions(root: Path, cache_path: Path,
                     rescan: bool = False) -> dict:
    """{pkg: version dir path} for the whole corpus. The drvfs metadata walk
    costs ~8 minutes, so it is cached next to the progress file; --rescan
    forces a fresh walk."""
    if cache_path.exists() and not rescan:
        try:
            cached = json.loads(cache_path.read_text())
            if isinstance(cached, dict) and cached.get("root") == str(root) \
                    and "versions" in cached:
                return cached["versions"]
        except (ValueError, OSError):
            pass
    print("  resolving highest version per package "
          "(drvfs walk, cached afterwards)...", flush=True)
    t0 = time.time()
    versions = {}
    for pkg in sorted(os.listdir(root)):
        p = root / pkg
        if not p.is_dir():
            continue
        ver = pick_version_dir(p)
        if ver is not None:
            versions[pkg] = str(ver)
    try:
        cache_path.write_text(json.dumps(dict(root=str(root),
                                              versions=versions)))
        print(f"  version map cached ({len(versions)} packages, "
              f"{time.time() - t0:.0f}s)", flush=True)
    except OSError as e:
        print(f"  [version-cache] write failed: {e}", flush=True)
    return versions


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    versions = resolve_versions(ROOT, out / "versions.json", args.rescan)
    pkgs = sorted(versions)  # cache keys == valid package dirs (no re-stat)
    done, stats, entries = load_progress(out)
    if done:
        print(f"  [resume] {len(done)} packages already done", flush=True)

    todo = [(p, versions[p]) for p in pkgs if p not in done]
    print(f"{len(pkgs)} packages, {len(todo)} to process "
          f"({args.workers} workers)", flush=True)
    if args.limit:
        todo = todo[:args.limit]

    counts = shard_row_counts(out)
    if counts:
        last = max(counts)
        shard = last if counts[last] < args.shard_rows else last + 1
    else:
        shard = 0
    shard_rows = counts.get(shard, 0)
    print(f"  active shard train-{shard:03d}.jsonl "
          f"({shard_rows} rows)", flush=True)

    stop = {"flag": False}

    def _sig(signum, frame):
        # flag only -- print() here can fire reentrantly inside another
        # print's buffered writer and crash the loop (observed on SIGTERM)
        stop["flag"] = True
        stop["signum"] = signum

    prev_int, prev_term = signal.getsignal(signal.SIGINT), \
        signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    processed = 0
    state = {"remaining": len(todo)}

    def _final2():
        _write_stats(out, stats, complete=state["remaining"] == 0)
        print(f"stats: files={stats['files']} spans={stats['spans']} "
              f"rows={stats['rows']} "
              f"est_tokens={int(stats['chars'] / 3.5)} "
              f"dropped={stats['dropped']} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)

    atexit.register(_final2)

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as pool:
        for i in range(0, len(todo), CHUNK):
            if stop["flag"]:
                break
            batch = todo[i:i + CHUNK]
            try:
                results = list(pool.map(_work1, batch, chunksize=1))
            except KeyboardInterrupt:
                state["remaining"] = len(todo) - i
                break
            if stop["flag"]:
                print(f"\n  [signal {stop.get('signum')}] stopping "
                      f"(resume-safe)", flush=True)
                state["remaining"] = len(todo) - i
                break
            state["remaining"] = len(todo) - (i + len(batch))
            for res in results:
                if stop["flag"]:
                    break
                if shard_rows >= args.shard_rows:
                    shard += 1
                    shard_rows = 0
                off = 0
                try:
                    off = (out / f"train-{shard:03d}.jsonl").stat().st_size
                except OSError:
                    pass
                entry = dict(pkg=res["pkg"], shard=shard, off=off,
                             files=res["files"], spans=res["spans"],
                             rows=res["rows"], chars=res["chars"],
                             dropped=res["dropped"])
                # progress line FIRST: a crash between the two writes rolls
                # this package back cleanly on resume
                if not nas_append(out / "progress.jsonl",
                                  json.dumps(entry) + "\n"):
                    continue  # rows unwritten -> package reprocessed later
                if res["lines"]:
                    ok = nas_append(out / f"train-{shard:03d}.jsonl",
                                    "\n".join(res["lines"]) + "\n")
                    if not ok:
                        continue
                shard_rows += res["rows"]
                _agg(stats, res)
            processed += len(batch)
            if processed % STATS_EVERY < CHUNK:
                _write_stats(out, stats, complete=False)
                print(f"  [{processed}/{len(todo)}] pkgs "
                      f"rows={stats['rows']} "
                      f"est_tokens={int(stats['chars'] / 3.5)} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)

    signal.signal(signal.SIGINT, prev_int)
    signal.signal(signal.SIGTERM, prev_term)
    atexit.unregister(_final2)
    _final2()


# ---------------------------------------------------------------------------
# eval holdout (package-level, from the finished shards)
# ---------------------------------------------------------------------------

def _nas_tmp_replace(path: Path, write_fn):
    """Write via tmp + rename, riding out drvfs flaps; write_fn(fh) must be
    re-runnable (each attempt starts a fresh file)."""
    tmp = path.with_name(path.name + ".tmp")
    for attempt in range(NAS_TRIES):
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                write_fn(fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return True
        except OSError as e:
            try:
                tmp.unlink()
            except OSError:
                pass
            if attempt == NAS_TRIES - 1:
                print(f"  [nas-write] giving up on {path}: {e}", flush=True)
                return False
            print(f"  [nas-write] {e}; retry {attempt + 1}/{NAS_TRIES - 1} "
                  f"in {NAS_WAIT_S:.0f}s", flush=True)
            time.sleep(NAS_WAIT_S)
    return False


def split_eval(args):
    """Move a seeded package-level holdout out of the shards into eval.jsonl
    (same convention as format_sft_v1: split by PACKAGES, not rows).

    The sample is drawn from the canonical package universe (versions
    cache), so reruns are idempotent. Content-driven: groups are recovered
    from the rows' own `package` fields (progress.jsonl can carry stale
    rollback entries from interrupted runs, so entry-count bookkeeping
    cannot be trusted), and the held-out rows are REGENERATED
    deterministically from the source packages. progress.jsonl is rebuilt
    with corrected byte offsets so builder resume stays consistent."""
    out = Path(args.out)
    meta: dict[str, dict] = {}  # pkg -> newest progress entry (files/... )
    for line in (out / "progress.jsonl").read_text(
            encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            meta[e["pkg"]] = e  # last occurrence wins (stale entries lose)
        except ValueError:
            pass
    # the sample universe is the CANONICAL package list (versions cache),
    # NOT progress.jsonl -- progress reflects the current train/eval split,
    # so sampling from it would move the holdout on every rerun
    versions = resolve_versions(ROOT, out / "versions.json")
    pkgs = sorted(set(versions))
    rng = random.Random(args.split_seed)
    n_eval = max(1, round(len(pkgs) * args.eval_frac))
    eval_pkgs = set(rng.sample(pkgs, n_eval))
    print(f"holdout: {n_eval}/{len(pkgs)} packages "
          f"(seed {args.split_seed}, frac {args.eval_frac})", flush=True)

    # 1. regenerate the held-out rows from source (deterministic extraction)
    todo = [(p, versions[p]) for p in sorted(eval_pkgs) if p in versions]
    eval_lines: list[str] = []
    eval_chars = eval_files = eval_spans = 0
    eval_dropped: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as pool:
        for res in pool.map(_work1, todo, chunksize=1):
            eval_lines.extend(res["lines"])
            eval_chars += res["chars"]
            eval_files += res["files"]
            eval_spans += res["spans"]
            for k, v in res["dropped"].items():
                eval_dropped[k] = eval_dropped.get(k, 0) + v

    def write_eval(fh):
        for ln in eval_lines:
            fh.write(ln + "\n")
    if not _nas_tmp_replace(out / "eval.jsonl", write_eval):
        return 1
    print(f"  eval.jsonl: {len(eval_lines)} rows regenerated", flush=True)

    # 2. strip eval packages' rows from every shard; rebuild progress from
    #    the content that actually lands in the files
    shards = sorted(out.glob("train-*.jsonl"))
    new_entries: list[dict] = []
    seen: set[str] = set()
    last_off = 0
    for s_idx, path in enumerate(shards):
        lines = path.read_text(encoding="utf-8").splitlines()
        staged: list[dict] = []

        def consume(fh):  # fresh pass per write attempt
            staged.clear()
            pos = 0
            for ln in lines:
                row = json.loads(ln)
                if row["package"] in eval_pkgs:
                    continue
                fh.write(ln + "\n")
                if not staged or staged[-1]["pkg"] != row["package"]:
                    staged.append(dict(pkg=row["package"], shard=s_idx,
                                       off=pos, rows=0, chars=0))
                g = staged[-1]
                g["rows"] += 1
                g["chars"] += len(row["text"])
                pos += len(ln.encode("utf-8")) + 1
            return pos

        if not _nas_tmp_replace(path, consume):
            return 1
        for g in staged:
            e = dict(meta[g["pkg"]])
            e.update(shard=g["shard"], off=g["off"], rows=g["rows"],
                     chars=g["chars"])
            new_entries.append(e)
            seen.add(g["pkg"])
        print(f"  rewrote {path.name} (kept {sum(g['rows'] for g in staged)} "
              f"rows)", flush=True)
    last_off = shards[-1].stat().st_size if shards else 0

    # packages with no train rows (the eval holdout, plus the ~11% of
    # packages whose every file/span was filtered) get zero-row entries
    # pinned to the END of the last shard so the resume done-set stays
    # complete and a rollback truncation is a no-op
    for p in sorted(set(meta) - seen):
        e = dict(meta[p])
        e.update(shard=len(shards) - 1, off=last_off, rows=0, chars=0)
        new_entries.append(e)

    def write_progress(fh):
        for e in new_entries:
            fh.write(json.dumps(e) + "\n")
    if not _nas_tmp_replace(out / "progress.jsonl", write_progress):
        return 1

    stats = {}
    try:
        stats = json.loads((out / "stats.json").read_text())
    except (ValueError, OSError):
        pass
    train_rows = sum(e["rows"] for e in new_entries)
    train_chars = sum(e["chars"] for e in new_entries)
    stats.update(dict(rows=train_rows, chars=train_chars,
                      est_tokens=int(train_chars / 3.5)))
    stats["eval"] = dict(seed=args.split_seed, frac=args.eval_frac,
                         packages=n_eval, rows=len(eval_lines),
                         chars=eval_chars,
                         est_tokens=int(eval_chars / 3.5),
                         files=eval_files, spans=eval_spans,
                         dropped=eval_dropped)
    stats["total_rows"] = train_rows + len(eval_lines)
    # keep files/spans corpus-wide (train + eval); guard so reruns of an
    # already-split stats.json do not double-add the eval counters
    if not stats.get("split_files_merged"):
        stats["files"] = stats.get("files", 0) + eval_files
        stats["spans"] = stats.get("spans", 0) + eval_spans
        stats["split_files_merged"] = True
    for attempt in range(NAS_TRIES):
        try:
            (out / "stats.json").write_text(json.dumps(stats, indent=1))
            break
        except OSError as e:
            if attempt == NAS_TRIES - 1:
                print(f"  [stats] giving up: {e}", flush=True)
            time.sleep(10)
    print(json.dumps(dict(train_rows=train_rows,
                          train_est_tokens=int(train_chars / 3.5),
                          eval_rows=len(eval_lines),
                          eval_est_tokens=int(eval_chars / 3.5))), flush=True)
    return 0


def verify(args):
    out = Path(args.out)
    fails, rows, chars, kinds = 0, 0, 0, {}
    order_fail = splice_fail = leak_warn = 0
    files = sorted(out.glob("train-*.jsonl"))
    if (out / "eval.jsonl").exists():
        files = files + [out / "eval.jsonl"]  # same row shape
    for p in files:
        with open(p, "rb") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    fails += 1
                    continue
                rows += 1
                chars += len(row["text"])
                kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
                prompt, target = row["prompt"], row["target"]
                ok = (row["text"] == prompt + target
                      and prompt.count(CONTEXT) == 1
                      and prompt.count(HISTORY) == 1
                      and prompt.count(CURSOR) == 1
                      and prompt.count(SUFFIX) == 1
                      and prompt.count(END) == 1
                      and target.count(END) == 1
                      and target.endswith("\n" + END)
                      and prompt.startswith(
                          f"{CONTEXT}{row['package']}/{row['path']}\n")
                      and CURSOR + SUFFIX in prompt  # empty cursor zone
                      and prompt.index(CONTEXT)
                      < prompt.index(HISTORY)
                      < prompt.index(CURSOR)
                      < prompt.index(SUFFIX)
                      < prompt.index(END))
                if not ok:
                    order_fail += 1
                    continue
                mid = target[:-(len(END) + 1)]
                pre = prompt.split("\n", 1)[1]
                pre = pre[:pre.index("\n" + HISTORY)]
                # byte-exact splice (prefix+span+suffix == src) is asserted
                # at build time; `mid in prompt` here is only a warning --
                # duplicated code elsewhere in the file can false-positive.
                if mid in prompt:
                    leak_warn += 1
                # predict-the-body convention
                if row["kind"] == "function_body" and not \
                        pre.rstrip().endswith("{"):
                    splice_fail += 1
    report = dict(shards=[f.name for f in files], rows=rows,
                  parse_failures=fails, marker_order_failures=order_fail,
                  fnbody_prefix_failures=splice_fail,
                  span_leak_warnings=leak_warn, kinds=kinds,
                  chars=chars, est_tokens=int(chars / 3.5))
    print(json.dumps(report, indent=1))
    return 0 if fails == 0 and order_fail == 0 and splice_fail == 0 else 1


def spot(args):
    out = Path(args.out)
    rng = random.Random(7)
    want_kinds = ["function_body", "top_level_block", "argument_list"]
    pools = {k: [] for k in want_kinds}
    seen = {k: 0 for k in want_kinds}
    K = 50  # per-kind reservoir
    for p in sorted(out.glob("train-*.jsonl")):
        with open(p, "rb") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                k = row.get("kind")
                if k in pools:
                    seen[k] += 1
                    if len(pools[k]) < K:
                        pools[k].append(row)
                    else:
                        j = rng.randrange(seen[k])
                        if j < K:
                            pools[k][j] = row
    picks = [rng.choice(pools[k]) for k in want_kinds if pools[k]]
    nonempty = [k for k in want_kinds if pools[k]]
    while len(picks) < args.spot and nonempty:
        picks.append(rng.choice(pools[rng.choice(nonempty)]))
    picks = picks[:args.spot]
    ok_all = True
    for i, r in enumerate(picks, 1):
        prompt, target = r["prompt"], r["target"]
        mid = target[:-(len(END) + 1)]
        ctx_line, rest = prompt.split("\n", 1)
        pre = rest[:rest.index("\n" + HISTORY)]
        suf = prompt.split(SUFFIX + "\n", 1)[1]
        suf = suf[:suf.index("\n" + END)]
        src_path = ROOT / r["package"]
        src = None
        try:
            ver = pick_version_dir(src_path)
            root = src_root_for(ver, r["package"])
            src = (root / Path(r["path"]).name).read_bytes()\
                .decode("utf-8", "replace").replace("\r\n", "\n")
        except (OSError, AttributeError, TypeError):
            pass
        print(f"--- spot #{i} [{r['kind']}] {r['package']}/{r['path']}")
        print("  ctx   |", ctx_line)
        print("  pre   |", (pre[:100] + ("..." if len(pre) > 100 else ""))
              .replace("\n", "\\n"))
        print("  cursor| <|cursor|><|suffix|>  (span removed)")
        print("  suf   |", (suf[:100] + ("..." if len(suf) > 100 else ""))
              .replace("\n", "\\n"))
        print("  target|", (mid[:100] + ("..." if len(mid) > 100 else ""))
              .replace("\n", "\\n"))
        checks = {
            "marker_order": prompt.index(CONTEXT) < prompt.index(HISTORY)
            < prompt.index(CURSOR) < prompt.index(SUFFIX)
            < prompt.index(END),
            "empty_cursor_zone": CURSOR + SUFFIX in prompt,
            "target_not_in_prompt": mid not in prompt,
            "text_eq_prompt_target": r["text"] == prompt + target,
        }
        if src is not None:
            checks["prefix_is_file_head"] = src.startswith(pre)
            checks["suffix_is_file_tail"] = src.rstrip("\n").endswith(
                suf.rstrip("\n")) if suf else True
            checks["target_in_source"] = mid in src
        ok = all(checks.values())
        ok_all &= ok
        print("  checks |", {k: bool(v) for k, v in checks.items()},
              "OK" if ok else "FAIL")
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N pending packages")
    ap.add_argument("--shard-rows", type=int, default=ROWS_PER_SHARD)
    ap.add_argument("--rescan", action="store_true",
                    help="force a fresh version-resolution walk")
    ap.add_argument("--split-eval", action="store_true",
                    help="after the build: move a seeded package-level "
                         "holdout out of the shards into eval.jsonl")
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--split-seed", type=int, default=20260819)
    ap.add_argument("--verify", action="store_true",
                    help="full pass over written shards: 0 parse failures, "
                         "marker order, span-leak checks")
    ap.add_argument("--spot", type=int, default=0,
                    help="spot-check N rows against the source files")
    args = ap.parse_args()
    globals()["ROOT"] = args.root
    if args.split_eval:
        sys.exit(split_eval(args))
    if args.verify:
        sys.exit(verify(args))
    if args.spot:
        sys.exit(spot(args))
    build(args)


if __name__ == "__main__":
    main()
