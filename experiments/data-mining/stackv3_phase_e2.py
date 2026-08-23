#!/usr/bin/env python3
"""Phase E gate v2 — PARALLEL (coordinator pipelining directive).

Same semantics as stackv3_phase_e.py stage2 (sha1-join delta + SPDX-header
license gate + ODC-By class), but work units are per-parquet-file with a
6-process pool; each worker writes shard files named by source part
(files_v3/shard-<lang>-<partidx>.jsonl), a per-part .done marker enables
resume, a multiprocessing.Value enforces the global byte cap, and content_ids
already staged by the earlier serial run are skipped (dedup with
files_v3/shard-*.jsonl).
"""
import hashlib, json, os, sys, time
from collections import Counter
from multiprocessing import Pool, Value
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from stackv3_phase_e import (header_license, log, ledger, V3_LOCAL, FILES_V3,
                             MIN_BYTES, MAX_FILE_BYTES)

MAX_TOTAL_BYTES = int(os.environ.get("STACKV3_MAX_BYTES", 8 * 10**9))
LANGS = os.environ.get("STACKV3_LANGS_GATE", "R,RDoc,RMarkdown").split(",")
_counter = Value("L", 0)
_skip_ids = set()


def _init_worker(join_v2, join_corpus, skip_ids):
    """Forkserver-safe state injection: Python 3.14 pool workers do NOT
    inherit runtime-populated module globals — pass them per worker."""
    global JOIN_V2, JOIN_CORPUS, _skip_ids
    JOIN_V2.update(join_v2)
    JOIN_CORPUS.update(join_corpus)
    _skip_ids.update(skip_ids)


def load_skip_ids():
    for p in FILES_V3.glob("shard-*.jsonl"):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    _skip_ids.add(json.loads(line)["content_id"])
                except Exception:
                    pass


def gate_part(args):
    lang, part_idx, path, cap = args
    marker = FILES_V3 / f".done-{lang}-{part_idx:05d}"
    if marker.exists():
        return (lang, part_idx, "already-done", 0, 0)
    counts = Counter()
    shard_bytes = 0
    out_local = V3_LOCAL / "shards" / f"shard-{lang}-{part_idx:05d}.jsonl"
    with open(out_local, "w", encoding="utf-8") as out:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["content_id", "content",
                                                 "size_bytes", "dedup_cluster",
                                                 "repo_ids"])
            for cid, content, sz, dc, rids in zip(
                    tbl.column("content_id").to_pylist(),
                    tbl.column("content").to_pylist(),
                    tbl.column("size_bytes").to_pylist(),
                    tbl.column("dedup_cluster").to_pylist(),
                    tbl.column("repo_ids").to_pylist()):
                counts["rows"] += 1
                if cid in _skip_ids:
                    counts["already-staged"] += 1
                    continue
                from stackv3_phase_e import build_join_sets  # noqa cached module-level
                if counts["rows"] == 1 and False:
                    pass
                if cid in JOIN_V2 or cid in JOIN_CORPUS:
                    counts["dup-of-v2-or-corpus"] += 1
                    continue
                if sz < MIN_BYTES or sz > MAX_FILE_BYTES:
                    counts["size-out-of-range"] += 1
                    continue
                with _counter.get_lock():
                    if _counter.value > cap:
                        counts["cap-stop"] += 1
                        break
                cls, reason = header_license(content)
                if cls is None:
                    counts[f"excluded/{reason.split(':')[0]}"] += 1
                    continue
                out.write(json.dumps(dict(
                    source="stackv3", language=lang, content_id=cid,
                    dedup_cluster=dc, repo_ids=rids, bytes=sz,
                    dataset_license="ODC-By-1.0", license_class=f"v3-header:{cls}",
                    license_evidence=reason,
                    sha256=hashlib.sha256(content.encode("utf-8", "replace")).hexdigest(),
                    content=content), ensure_ascii=False) + "\n")
                shard_bytes += sz
                with _counter.get_lock():
                    _counter.value += sz
                counts["accepted"] += 1
            else:
                continue
            break
    # ship to NAS + mark done
    import shutil
    if shard_bytes > 0:
        shutil.copyfile(out_local, FILES_V3 / out_local.name)
    out_local.unlink()
    marker.write_text(str(shard_bytes))
    return (lang, part_idx, "ok", counts["accepted"], shard_bytes)


JOIN_V2 = set()
JOIN_CORPUS = set()


def main():
    global JOIN_V2, JOIN_CORPUS
    t0 = time.time()
    log("parallel gate: building join sets")
    import stackv3_phase_e as e
    JOIN_V2, JOIN_CORPUS = e.build_join_sets()
    load_skip_ids()
    log(f"parallel gate: skip_ids={len(_skip_ids):,}")
    units = []
    for lang in LANGS:
        for i, p in enumerate(sorted((V3_LOCAL / lang).glob("*.parquet"))):
            units.append((lang, i, str(p), MAX_TOTAL_BYTES))
    log(f"parallel gate: {len(units)} parts, cap={MAX_TOTAL_BYTES:,}B, "
        f"procs={os.environ.get('STACKV3_PROCS', '6')}")
    with Pool(int(os.environ.get("STACKV3_PROCS", "6")), initializer=_init_worker,
              initargs=(JOIN_V2, JOIN_CORPUS, _skip_ids)) as pool:
        totals = Counter(); total_bytes = 0
        for lang, idx, status, acc, sb in pool.imap_unordered(gate_part, units):
            totals[f"{lang}/{status}"] += 1
            totals[f"{lang}/accepted"] += acc
            total_bytes += sb
            log(f"part {lang}-{idx:05d} {status} accepted={acc:,} bytes={sb:,} "
                f"elapsed={time.time()-t0:.0f}s")
    ledger({"type": "phase-e-delta-gate-parallel", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "parts": dict(totals), "accepted_bytes_new": total_bytes,
            "cap": MAX_TOTAL_BYTES})
    log(f"PARALLEL GATE COMPLETE parts={dict(totals)} new_bytes={total_bytes:,} "
        f"elapsed={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
