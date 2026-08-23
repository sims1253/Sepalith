#!/usr/bin/env python3
"""Stack v2 R acquisition — Phase B: license-gated CONTENT FETCH from the
Software Heritage S3 bucket into stack_staging.

  1. rebuilds the deterministic accept set from the metadata parquets
     (permissive-only per the 2026-08-20/21 ruling; vendor/generated/tiny/
     oversized/cran-bioc-official-mirror excluded+ledgered; exact-dedup by
     content_id, best-star row wins; ordered stars DESC so caps keep the best),
  2. fetches gzip'd content per blob_id from
     https://softwareheritage.s3.amazonaws.com/content/<blob_id> (the dataset
     card's documented mechanism; anonymous reads verified; sha256(content)
     is checked against content_id on a sample),
  3. shards accepted files -> /mnt/h/sepalith/stack_staging/files/shard-*.jsonl
     (~500MB), written+flushed locally per line, bulk-copied to the NAS on
     completion; on restart any unshipped local shard is recovered (parsed,
     trailing partial line dropped, shipped, its blobs marked done),
  4. sqlite state (logs/state.db) marks every fetched/missing blob for resume;
     ledger aggregates appended at start/30min/end.

Politeness: bounded worker pool (default 32), per-request retries with backoff,
metadata read from the once-mirrored NAS parquets (no HF re-hammering).
Caps: STACK_MAX_BYTES (default 15GB) and STACK_DEADLINE (unix ts).
"""
import gzip, hashlib, json, os, queue, shutil, sqlite3, threading, time
import urllib.error, urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

STAGE = Path("/mnt/h/sepalith/stack_staging")
META, FILES, LOGS = STAGE / "metadata", STAGE / "files", STAGE / "logs"
LEDGER = STAGE / "license_ledger.jsonl"
LOCAL = Path("/tmp/stack_r_fetch")
SHARD_DIR = LOCAL / "shards"
DB = LOGS / "state.db"
CONFIGS = ["R", "RDoc", "RMarkdown"]
S3 = "https://softwareheritage.s3.amazonaws.com/content/{}"
UA = "sepalith-stackv2-r-acquisition/0.1 (license-filtered R subset, bounded pool)"

MAX_TOTAL_BYTES = int(os.environ.get("STACK_MAX_BYTES", 15 * 10**9))
DEADLINE = float(os.environ.get("STACK_DEADLINE", time.time() + 5 * 3600))
SHARD_BYTES = 500 * 10**6
WORKERS = int(os.environ.get("STACK_WORKERS", 32))
MIN_BYTES, MAX_FILE_BYTES = 100, 10 * 10**6

PERMISSIVE = {
    "mit", "mit-0", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-4-clause",
    "0bsd", "bsd-2-clause-views", "bsd-3-clause-clear", "unlicense", "isc",
    "zlib", "bsl-1.0", "x11", "ncsa", "ecl-2.0", "postgresql",
    "gpl-1.0-only", "gpl-1.0-or-later", "gpl-2.0-only", "gpl-2.0-or-later",
    "gpl-3.0-only", "gpl-3.0-or-later",
    "lgpl-2.0-only", "lgpl-2.0-or-later", "lgpl-2.1-only", "lgpl-2.1-or-later",
    "lgpl-3.0-only", "lgpl-3.0-or-later",
    "agpl-3.0-only", "agpl-3.0-or-later",
    "cc0-1.0", "cc-by-1.0", "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
    "artistic-1.0", "artistic-1.0-clause8", "artistic-1.0-perl", "artistic-2.0",
    "mpl-1.1", "mpl-2.0", "mpl-2.0-no-copyleft-exception",
    "epl-1.0", "epl-2.0", "cpl-1.0",
    "cecill-1.0", "cecill-1.1", "cecill-2.0", "cecill-2.1", "cecill-b", "cecill-c",
}
PERMISSIVE_PREFIX = ("gpl-", "lgpl-", "agpl-", "bsd-", "apache-", "mit-",
                     "artistic-", "cecill-", "cc-by-", "cc0-")
BENIGN_REFS = {
    "licenseref-scancode-warranty-disclaimer",
    "licenseref-scancode-public-domain",
    "licenseref-scancode-public-domain-disclaimer",
    "licenseref-scancode-other-permissive",
    "licenseref-scancode-generic-cla",
}
HARD_EXCLUDE_PREFIX = ("cc-by-nc", "cc-by-nd", "cc-by-sa", "cc-nc", "gfdl", "json")


def classify(licenses):
    if not licenses:
        return "exclude", "no-detected-licenses"
    for lic in licenses:
        lid = (lic or "").strip().lower()
        if not lid or lid in BENIGN_REFS:
            continue
        if lid in PERMISSIVE or (lid.startswith(PERMISSIVE_PREFIX)
                                 and not lid.startswith(HARD_EXCLUDE_PREFIX)):
            continue
        return "exclude", "non-permissive-or-unknown"
    return "include", ""


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log_line(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(LOGS / "progress.log", "a") as f:
        f.write(line + "\n")


def ledger(rec):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def build_accept_set():
    rows = {}
    exc = Counter(); lic_inc = Counter(); per_cfg = Counter()
    for cfg in CONFIGS:
        pf = pq.ParquetFile(META / f"{cfg}.parquet")
        md = pf.metadata
        for rg in range(md.num_row_groups):
            tbl = pf.read_row_group(rg, columns=[
                "blob_id", "content_id", "path", "detected_licenses", "repo_name",
                "star_events_count", "length_bytes", "src_encoding",
                "is_vendor", "is_generated", "extension", "revision_date", "github_id"])
            cols = [tbl.column(k).to_pylist() for k in
                    ("blob_id", "content_id", "path", "detected_licenses", "repo_name",
                     "star_events_count", "length_bytes", "src_encoding",
                     "is_vendor", "is_generated", "extension", "revision_date", "github_id")]
            for blob, cid, path, dl, repo, stars, lb, enc, v, g, ext, rdate, gid in zip(*cols):
                decision, reason = classify(dl)
                if decision == "exclude":
                    exc[reason] += 1
                    if reason != "no-detected-licenses":
                        exc[f"{reason}: {'|'.join(sorted(x.lower() for x in dl))[:100]}"] += 1
                    continue
                if v or g:
                    exc["vendor-or-generated"] += 1; continue
                if lb < MIN_BYTES:
                    exc["tiny-under-100b"] += 1; continue
                if lb > MAX_FILE_BYTES:
                    exc["oversized-over-10mb"] += 1; continue
                if repo.lower().startswith(("cran/", "bioconductor/")):
                    exc["cran-bioc-official-mirror"] += 1; continue
                key = cid or blob
                rec = (blob, cid, cfg, path, tuple(dl), repo, stars, lb, enc, ext,
                       str(rdate)[:10] if rdate else "", gid)
                cur = rows.get(key)
                if cur is None or stars > cur[6]:
                    rows[key] = rec
                lic_inc["|".join(sorted(x.lower() for x in dl))] += 1
                per_cfg[cfg] += 1
            if rg % 10 == 0:
                log_line(f"accept-set {cfg} rg {rg}/{md.num_row_groups} unique={len(rows):,}")
    ordered = sorted(rows.values(), key=lambda r: (-r[6], -r[7], r[5], r[3]))
    return ordered, exc, lic_inc, per_cfg


def init_db():
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS blobs (blob_id TEXT PRIMARY KEY, "
                "status TEXT, bytes INTEGER, ts TEXT)")
    con.commit()
    return con


def fetch_blob(blob_id, tries=4):
    url = S3.format(blob_id)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return gzip.decompress(r.read()), None
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):
                return None, f"http-{e.code}"
            if attempt == tries - 1:
                return None, f"http-{e.code}"
            time.sleep(min(60, 2 * (attempt + 1) ** 2))
        except Exception as e:
            if attempt == tries - 1:
                return None, f"err-{type(e).__name__}"
            time.sleep(min(60, 2 * (attempt + 1) ** 2))
    return None, "exhausted"


def recover_local_shards(con):
    """Ship any local shard files left by a crash; mark their blobs fetched."""
    recovered = 0
    for p in sorted(SHARD_DIR.glob("shard-*.jsonl")):
        blob_ids, keep_lines = [], []
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    keep_lines.append(line)
                    blob_ids.append(rec["blob_id"])
                except json.JSONDecodeError:
                    break  # trailing partial line -> dropped
        dest = FILES / p.name
        if keep_lines:
            dest.write_text("".join(keep_lines), encoding="utf-8")
            con.executemany("INSERT OR REPLACE INTO blobs VALUES (?,?,?,?)",
                            [(b, "fetched", None, now()) for b in blob_ids])
            con.commit()
            recovered += len(blob_ids)
        p.unlink()
        log_line(f"recovered local shard {p.name} -> NAS ({len(blob_ids):,} blobs)")
    return recovered


def main():
    for d in (FILES, LOGS, SHARD_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log_line(f"phase B start workers={WORKERS} cap={MAX_TOTAL_BYTES:,}B "
             f"deadline={time.strftime('%H:%M:%S', time.localtime(DEADLINE))}")
    accept, exc, lic_inc, per_cfg = build_accept_set()
    est = sum(r[7] for r in accept)
    log_line(f"accept set: {len(accept):,} unique blobs est {est:,}B "
             f"({est/1e9:.2f}GB); top exclusions: {dict(exc.most_common(8))}")
    ledger({"type": "phase-b-accept-set", "ts": now(),
            "unique_accepted_blobs": len(accept),
            "estimated_content_bytes": est,
            "accepted_rows_by_config": dict(per_cfg),
            "included_license_sets_top25": [{"licenses": k, "files": v}
                                            for k, v in lic_inc.most_common(25)],
            "exclusion_reasons": dict(exc.most_common(200))})

    con = init_db()
    recover_local_shards(con)
    done = {r[0]: r[1] for r in con.execute("SELECT blob_id, status FROM blobs")}
    already = sum(1 for s in done.values() if s == "fetched")
    log_line(f"resume: {already:,} fetched, "
             f"{sum(1 for s in done.values() if s == 'missing'):,} missing")
    total_bytes = con.execute(
        "SELECT COALESCE(SUM(bytes),0) FROM blobs WHERE status='fetched'").fetchone()[0]

    shard_idx = len(list(FILES.glob("shard-*.jsonl")))
    shard_fh = None; shard_written = 0
    counts = Counter(); db_batch = []
    sha_match = sha_check = 0
    last_progress = last_ledger = time.time()
    stop_reason = None

    def open_shard():
        nonlocal shard_fh, shard_written
        shard_fh = open(SHARD_DIR / f"shard-{shard_idx:05d}.jsonl", "w",
                        encoding="utf-8")
        shard_written = 0

    def close_ship():
        nonlocal shard_fh, shard_written, shard_idx
        if shard_fh is None:
            return
        shard_fh.close()
        src = SHARD_DIR / f"shard-{shard_idx:05d}.jsonl"
        if shard_written > 0:
            shutil.copyfile(src, FILES / src.name)
        src.unlink()
        shard_fh = None; shard_written = 0
        shard_idx += 1
        log_line(f"shipped {src.name}")

    def flush_db():
        nonlocal db_batch
        if db_batch:
            con.executemany("INSERT OR REPLACE INTO blobs VALUES (?,?,?,?)", db_batch)
            con.commit()
            db_batch = []

    open_shard()
    q_in = queue.Queue(maxsize=WORKERS * 8)
    q_out = queue.Queue()

    def worker():
        while True:
            item = q_in.get()
            if item is None:
                return
            data, err = fetch_blob(item[0])
            q_out.put((item, data, err))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for t in threads:
        t.start()

    fed = 0
    for item in accept:
        if item[0] in done:
            continue
        if total_bytes > MAX_TOTAL_BYTES:
            stop_reason = "size-cap"; break
        if time.time() > DEADLINE:
            stop_reason = "deadline"; break
        q_in.put(item)
        fed += 1
        # drain completed results without starving the feed
        drained = 0
        while drained < 64:
            try:
                (blob, cid, cfg, path, dl, repo, stars, lb, enc, ext, rdate, gid), data, err = \
                    q_out.get(timeout=0.02 if drained == 0 else 0.0)
            except queue.Empty:
                break
            drained += 1
            if err:
                counts[f"missing/{err}"] += 1
                db_batch.append((blob, "missing", 0, now()))
            else:
                sha = hashlib.sha256(data).hexdigest()
                if cid:
                    sha_check += 1
                    if sha == cid:
                        sha_match += 1
                try:
                    text = data.decode(enc or "utf-8", errors="replace")
                except LookupError:
                    text = data.decode("utf-8", errors="replace")
                rec = dict(source="stackv2", config=cfg, path=path, repo=repo,
                           blob_id=blob, content_id=cid, licenses=list(dl),
                           star_events_count=stars, extension=ext,
                           revision_date=rdate, github_id=gid, src_encoding=enc,
                           length_bytes_meta=lb, bytes=len(data), sha256=sha,
                           content=text)
                shard_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                shard_written += len(data)
                total_bytes += len(data)
                counts["fetched"] += 1
                db_batch.append((blob, "fetched", len(data), now()))
            if shard_written >= SHARD_BYTES:
                flush_db(); close_ship(); open_shard()
        if len(db_batch) >= 5000:
            flush_db()
        if time.time() - last_progress > 300:
            flush_db()
            rate = counts["fetched"] / max(time.time() - t0, 1)
            remaining = len(accept) - already - counts["fetched"]
            log_line(f"progress fetched={counts['fetched']:,} "
                     f"missing={sum(v for k, v in counts.items() if k.startswith('missing')):,} "
                     f"bytes={total_bytes:,} ({total_bytes/1e9:.2f}GB) rate={rate:.0f}/s "
                     f"eta={remaining/max(rate,.01)/3600:.1f}h fed={fed:,}")
            last_progress = time.time()
        if time.time() - last_ledger > 1800:
            ledger({"type": "phase-b-progress", "ts": now(), **dict(counts),
                    "total_content_bytes": total_bytes})
            last_ledger = time.time()

    # graceful drain: stop feeding, wait for workers, then final drain
    if stop_reason is None:
        stop_reason = "complete"
    for _ in threads:
        q_in.put(None)
    for t in threads:
        t.join(timeout=120)
    while True:
        try:
            (blob, cid, cfg, path, dl, repo, stars, lb, enc, ext, rdate, gid), data, err = \
                q_out.get(timeout=1)
        except queue.Empty:
            break
        if err:
            counts[f"missing/{err}"] += 1
            db_batch.append((blob, "missing", 0, now()))
        else:
            sha = hashlib.sha256(data).hexdigest()
            try:
                text = data.decode(enc or "utf-8", errors="replace")
            except LookupError:
                text = data.decode("utf-8", errors="replace")
            shard_fh.write(json.dumps(dict(
                source="stackv2", config=cfg, path=path, repo=repo, blob_id=blob,
                content_id=cid, licenses=list(dl), star_events_count=stars,
                extension=ext, revision_date=rdate, github_id=gid,
                src_encoding=enc, length_bytes_meta=lb, bytes=len(data),
                sha256=sha, content=text), ensure_ascii=False) + "\n")
            shard_written += len(data); total_bytes += len(data)
            counts["fetched"] += 1
            db_batch.append((blob, "fetched", len(data), now()))
    flush_db(); close_ship()
    summary = dict(counts)
    summary["total_content_bytes"] = total_bytes
    summary["stop_reason"] = stop_reason
    summary["sha256_vs_content_id"] = f"{sha_match}/{sha_check}"
    summary["accept_set_size"] = len(accept)
    ledger({"type": "phase-b-summary", "ts": now(), **summary})
    log_line(f"PHASE B COMPLETE stop={stop_reason} fetched={counts['fetched']:,} "
             f"bytes={total_bytes:,} sha_ok={sha_match}/{sha_check} "
             f"elapsed={(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
