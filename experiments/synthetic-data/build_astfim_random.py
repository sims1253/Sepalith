#!/usr/bin/env python3
"""Random-cursor FIM corpus builder (the v8 positional-realism pretraining
stream — docs/research/v8-noop-random-cursor.md layer 2).

The astfim_v1 corpus puts EVERY pretraining cursor at a tree-sitter span
boundary (function body / top-level block / argument list): the model never
sees a cursor resting at an arbitrary position. This builder renders the
SAME normalized CRAN corpus with RANDOM cut positions instead:

  - cut placement: uniform over the file's lines (any depth, not clustered
    at function heads); 2/3 at line starts, 1/3 at a random column INSIDE
    the line (mid-identifier / mid-expression cursors included)
  - span: from the cut to the END of the nearest following named-node end
    (the rest of the current or next statement — corpus-exact, like every
    family), truncated at a line boundary if it would blow the row budget
  - seeded per package:file — deterministic rebuilds, the astfim convention

Row shape is identical to build_astfim.py's PSM layout (prefix+span+suffix
== source asserted per row); kind is "random_cut" with a `cut` subkind
("line" | "midline") and `head_frac` for position audits. In the A2-prime
mixture this stream doses at 10-15% of the FIM share; astfim_v1 proper
stays untouched and frozen.

Output: /mnt/h/sepalith/datasets/astfim_random_v1/train-<shard>.jsonl +
stats.json + progress.jsonl (same resume/rollback protocol as astfim_v1).

Usage:
  nice -n 19 python3 experiments/synthetic-data/build_astfim_random.py --workers 6
  python3 experiments/synthetic-data/build_astfim_random.py --limit 5 --out /tmp/rc_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_astfim as A  # version resolution, markers, nas_append, walk

ROOT = A.ROOT
OUT = Path("/mnt/h/sepalith/datasets/astfim_random_v1")

MAX_ROW_CHARS = 6000     # astfim budget
MIN_SPAN_CHARS = 20      # rest-of-statement floor (tighter than astfim's
                         # 30: midtyping continuations can be short)
MAX_SPAN_CHARS = 2400    # leave room for prefix+suffix in the row budget
MAX_CUTS_PER_FILE = 3
MAX_FILE_LINES = A.MAX_FILE_LINES
MAX_FILE_BYTES = A.MAX_FILE_BYTES
ROWS_PER_SHARD = 100_000
CHUNK = 24               # packages per executor.map batch
STATS_EVERY = 200
MIDLINE_WEIGHT = 1 / 3   # share of cuts placed at a random interior column


def _node_ends(tree) -> list[int]:
    """Sorted end-bytes of every named node (statement-ish ends)."""
    ends = set()
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.is_named:
            ends.add(n.end_byte)
        stack.extend(n.children)
    return sorted(ends)


def random_cuts(text: str, ends: list[int],
                rng: random.Random) -> list[dict]:
    """Up to MAX_CUTS_PER_FILE non-overlapping random-cut spans in text
    space (the source is decoded + CRLF-folded once, so text offsets equal
    the parse tree's byte offsets)."""
    lines = text.split("\n")
    n = len(lines)
    if n < 4:
        return []
    line_starts = []
    acc = 0
    for l in lines:
        line_starts.append(acc)
        acc += len(l) + 1
    total = len(text)
    spans, occupied = [], []
    tries = 0
    while len(spans) < MAX_CUTS_PER_FILE and tries < 12:
        tries += 1
        r = rng.randrange(1, n - 1)          # never the first/last line
        if rng.random() < MIDLINE_WEIGHT:
            # interior column: strictly inside the line's non-ws content
            line = lines[r]
            lo = len(line) - len(line.lstrip())
            hi = len(line.rstrip())
            if hi - lo < 6:
                continue                      # too short to cut inside
            col = rng.randrange(lo + 2, hi - 1)
            start = line_starts[r] + col
            cut = "midline"
        else:
            start = line_starts[r]
            cut = "line"
        if start >= total - MIN_SPAN_CHARS:
            continue
        if any(start < e and start + MIN_SPAN_CHARS > s
               for s, e in occupied):
            continue                          # overlaps a picked span
        # span end: nearest named-node end clearing the floor — the cursor
        # is at `start`, the span runs to the end of the statement being
        # typed (midline cuts) or the next complete statement (line cuts)
        end = next((e for e in ends if e > start + MIN_SPAN_CHARS), None)
        if end is None:
            continue
        if end - start > MAX_SPAN_CHARS:
            approx = text.find("\n", start + MAX_SPAN_CHARS - 400,
                               start + MAX_SPAN_CHARS + 400)
            if approx == -1:
                continue
            end = approx
        pre = text[:start]
        if not any(l.strip() for l in pre.split("\n")):
            continue                          # nothing typed above
        if len(pre) + (end - start) + 200 > MAX_ROW_CHARS - 200:
            continue                          # row-budget guard
        occupied.append((start, end))
        spans.append(dict(start=start, end=end, kind="random_cut",
                          cut=cut,
                          head_frac=round(start / max(1, total), 3)))
    return spans


def render_random(pkg: str, rel: str, text: str,
                  span: dict) -> tuple[str, str]:
    """PSM rendering with the astfim marker layout (text-space sibling of
    build_astfim.render; same marker-leak and newline-strip conventions)."""
    pre = text[:span["start"]].rstrip("\n")
    mid = text[span["start"]:span["end"]].strip("\n")
    suf = text[span["end"]:].strip("\n")
    for marker in (A.CONTEXT, A.HISTORY, A.CURSOR, A.SUFFIX, A.END):
        if marker in pre or marker in mid or marker in suf:
            raise ValueError(marker)
    prompt = (f"{A.CONTEXT}{pkg}/{rel}\n{pre}\n{A.HISTORY}\n\n"
              f"{A.CURSOR}{A.SUFFIX}\n{suf}\n{A.END}\n")
    target = f"{mid}\n{A.END}"
    return prompt, target


def process_file(text: str, pkg: str, rel: str, dropped: dict) -> list[dict]:
    src = text.encode("utf-8", "replace")
    tree = A._PARSER.parse(src)
    ends = _node_ends(tree)
    rng = random.Random(f"rc@1:{pkg}:{rel}")
    out = []
    for span in random_cuts(text, ends, rng):
        try:
            prompt, target = render_random(pkg, rel, text, span)
        except ValueError:
            dropped["marker_leak"] = dropped.get("marker_leak", 0) + 1
            continue
        if len(prompt) + len(target) > MAX_ROW_CHARS:
            dropped["row_budget"] = dropped.get("row_budget", 0) + 1
            continue
        assert (text[:span["start"]] + text[span["start"]:span["end"]]
                + text[span["end"]:]) == text
        out.append(dict(prompt=prompt, target=target,
                        kind=span["kind"], cut=span["cut"],
                        head_frac=span["head_frac"],
                        package=pkg, path=rel))
    return out


def process_package(pkg: str, ver_dir: str) -> dict:
    res = dict(files=0, rows=0, chars=0, dropped={},
               pkg=pkg, rows_out=[])
    root = A.src_root_for(Path(ver_dir), pkg)
    if root is None:
        res["dropped"]["no_source"] = 1
        return res
    try:
        files = sorted({f.name: f for f in list(root.glob("*.R"))
                        + list(root.glob("*.r"))}.values(),
                       key=lambda f: f.name)
    except OSError:
        res["dropped"]["no_source"] = 1
        return res
    # breadth over depth: a seeded per-package shuffle picks WHICH files
    # visit (not always the alphabet-first); astfim's one-file cap
    # generalized to three for mixture volume
    random.Random(f"rcf@1:{pkg}").shuffle(files)
    per_file_cap = 3
    for f in files:
        if res["files"] >= per_file_cap:
            break
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        if not raw or len(raw) > MAX_FILE_BYTES:
            continue
        text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
        if not (4 <= text.count("\n") + 1 <= MAX_FILE_LINES):
            continue
        res["files"] += 1
        for row in process_file(text, pkg, f"R/{f.name}", res["dropped"]):
            res["rows"] += 1
            res["chars"] += len(row["prompt"]) + len(row["target"])
            res["rows_out"].append(row)
    return res


def _work1(item):
    return process_package(*item)


def _agg(stats: dict, res: dict):
    for k in ("files", "rows", "chars"):
        stats[k] = stats.get(k, 0) + res.get(k, 0)
    for k, v in res.get("dropped", {}).items():
        stats.setdefault("dropped", {})
        stats["dropped"][k] = stats["dropped"].get(k, 0) + v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    try:
        os.nice(19)
    except OSError:
        pass
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume: astfim protocol (done pkgs + rolled-back last entry)
    entries_raw = []
    done_pkgs = set()
    stats = dict(files=0, rows=0, chars=0, packages=0, dropped={})
    if (out_dir / "progress.jsonl").exists():
        try:
            for line in (out_dir / "progress.jsonl").read_text().splitlines():
                try:
                    entries_raw.append(json.loads(line))
                except ValueError:
                    print("  [resume] ignoring partial progress line",
                          flush=True)
        except OSError:
            entries_raw = []
    if entries_raw:
        last = entries_raw[-1]
        # roll the last entry's shard back to its recorded offset
        if last.get("rows", 0) > 0:
            shard_p = out_dir / f"train-{last['shard']:03d}.jsonl"
            try:
                if shard_p.stat().st_size > last["off"]:
                    with open(shard_p, "r+b") as fh:
                        fh.truncate(last["off"])
                    print(f"  [resume] rolled back {last['pkg']} "
                          f"({shard_p.name} -> {last['off']} bytes)",
                          flush=True)
            except OSError as e:
                print(f"  [resume] rollback failed ({e}); continuing",
                      flush=True)
            entries_raw = entries_raw[:-1]
        for e in entries_raw:
            done_pkgs.add(e["pkg"])
            for k in ("files", "rows", "chars"):
                stats[k] += e.get(k, 0)
            for k, v in e.get("dropped", {}).items():
                stats["dropped"][k] = stats["dropped"].get(k, 0) + v
    stats["packages"] = len(done_pkgs)

    # reuse astfim_v1's versions cache (same corpus, ~8-min walk avoided)
    versions = A.resolve_versions(
        ROOT, Path("/mnt/h/sepalith/datasets/astfim_v1/versions.json"))
    pkgs = sorted(p for p in versions if p not in done_pkgs)
    if args.limit:
        pkgs = pkgs[: args.limit]
    rng = random.Random(7)
    rng.shuffle(pkgs)
    print(f"[rc] packages total={len(versions)} done={len(done_pkgs)} "
          f"todo={len(pkgs)} workers={args.workers}", flush=True)

    # current shard = highest existing + its byte size
    shard = 0
    off = 0
    for p in sorted(out_dir.glob("train-*.jsonl")):
        idx = int(p.stem.split("-")[1])
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        if idx >= shard:
            shard, off = idx, size if idx == shard else 0
    shard_rows_in = 0
    if off:
        try:
            with open(out_dir / f"train-{shard:03d}.jsonl", "rb") as fh:
                shard_rows_in = sum(c.count(b"\n")
                                    for c in iter(lambda: fh.read(1 << 20),
                                                  b""))
        except OSError:
            shard_rows_in = 0

    t0 = time.time()
    items = [(p, versions[p]) for p in pkgs]

    def handle(res):
        nonlocal shard, off, shard_rows_in
        _agg(stats, res)
        stats["packages"] += 1
        rows = res.pop("rows_out", [])
        if rows:
            entry = dict(pkg=res["pkg"], shard=shard, off=off,
                         rows=res["rows"], files=res["files"],
                         chars=res["chars"], dropped=res["dropped"])
            blob = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in rows)
            A.nas_append(out_dir / f"train-{shard:03d}.jsonl", blob)
            off += len(blob.encode("utf-8"))
            shard_rows_in += len(rows)
            A.nas_append(out_dir / "progress.jsonl", json.dumps(entry) + "\n")
            if shard_rows_in >= ROWS_PER_SHARD:
                shard += 1
                off = 0
                shard_rows_in = 0

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=A._init_worker) as ex:
        n = 0
        for res in ex.map(_work1, items, chunksize=CHUNK):
            handle(res)
            n += 1
            if n % STATS_EVERY == 0:
                stats["elapsed_s"] = round(time.time() - t0, 1)
                stats["complete"] = False
                stats["est_tokens"] = int(stats.get("chars", 0) / 3.5)
                A._write_stats(out_dir, stats, False)
                print(f"[rc] {n}/{len(pkgs)} pkgs rows={stats['rows']} "
                      f"est_tok={stats['est_tokens']} "
                      f"elapsed={stats['elapsed_s']}s", flush=True)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["complete"] = True
    stats["est_tokens"] = int(stats.get("chars", 0) / 3.5)
    A._write_stats(out_dir, stats, True)
    print(f"[rc] DONE rows={stats.get('rows')} files={stats.get('files')} "
          f"est_tokens={stats['est_tokens']} "
          f"dropped={stats.get('dropped')} in {stats['elapsed_s']}s",
          flush=True)


if __name__ == "__main__":
    main()
