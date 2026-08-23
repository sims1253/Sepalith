#!/usr/bin/env python3
"""Stack v2 R acquisition — Phase C: FULL-SCALE cross-corpus MinHash/LSH dedup
(user directive: prevent LARGE-SCALE duplication of existing material).

Same instrument as the A2 minhash audit (experiments/data_pipeline/
minhash_audit.py): word 5-gram shingles -> blake2b-8 -> 128-perm MinHash
(seed 1234), LSH 16 bands x 8 rows (candidate gate ~J>=0.7), verification by
signature Jaccard. Documents:

  - STACK: every staged shard in /mnt/h/sepalith/stack_staging/files/*.jsonl
    (content inline; docs = fetched permissive files)
  - CORPUS baseline: R-family source files (.R/.Rmd/.qmd/.Rd) of the CRAN
    store (/mnt/h/sepalith/normalized) + Bioconductor staging
    (/mnt/h/sepalith/normalized_bioc), read from local tar copies made once
    (sequential SMB, no per-file round trips).

Reports (logs/phase_c_report.json):
  (a) intra-Stack collapse rate at J>=0.9 and J>=0.8 (union-find clusters),
  (b) Stack-vs-corpus overlap at J>=0.9 / >=0.8 — totals, by Stack config,
      by corpus store, by corpus top-level dir area,
  (c) net-new files/bytes AFTER cross-corpus dedup (keep-list written to
      datasets/stack_keep_ids.jsonl: blob_id + decision + reason).

Method notes: exact-duplicate collapse by sha256 first; LSH buckets >500
members are boilerplate — handled by star-verification (representative vs up
to 40 members), never chain-unioned blindly. P(candidate miss) at J=0.8 is
~5% (16x8 banding) — reported bands are lower bounds within that slack.
"""
import hashlib, json, re, sys, tarfile, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

NUM_PERM = 128
MERSENNE = (1 << 61) - 1
BANDS, ROWS = 16, 8
SHINGLE_K = 5
_word_re = re.compile(r"\S+")

STAGE = Path("/mnt/h/sepalith/stack_staging")
FILES = STAGE / "files"
LOGS = STAGE / "logs"
LOCAL_CORPUS = Path("/tmp/stack_r_fetch/corpus")
EXTS = {".R", ".Rmd", ".qmd", ".Rd"}

_rng = np.random.default_rng(1234)
_A = _rng.integers(1, MERSENNE, size=NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, MERSENNE, size=NUM_PERM, dtype=np.uint64)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(LOGS / "phase_c.log", "a") as f:
        f.write(line + "\n")


def doc_signature(text: str):
    words = _word_re.findall(text)
    if len(words) < SHINGLE_K:
        return None
    sh = np.frombuffer(b"".join(
        hashlib.blake2b(" ".join(words[i:i + SHINGLE_K]).encode("utf-8", "replace"),
                        digest_size=8).digest()
        for i in range(len(words) - SHINGLE_K + 1)), dtype=np.uint64)
    proj = (_A[:, None] * sh[None, :] + _B[:, None]) % np.uint64(MERSENNE)
    return proj.min(axis=1)


def work_chunk(chunk):
    """chunk: list of (doc_id, text). -> list of (doc_id, sig_bytes|None)."""
    out = []
    for doc_id, text in chunk:
        sig = doc_signature(text)
        out.append((doc_id, sig.tobytes() if sig is not None else None))
    return out


class Doc:
    __slots__ = ("kind", "sub", "area", "sha", "bytes", "blob")

    def __init__(self, kind, sub, area, sha, nbytes, blob=None):
        self.kind, self.sub, self.area, self.sha, self.bytes, self.blob = \
            kind, sub, area, sha, nbytes, blob


def iter_corpus_docs(tarpath, store):
    """Yield (meta, raw_bytes) for R-family files in a corpus tar.

    Member names are relative to the store root (tar was created with cwd at
    the store): <pkg>/<ver>/<pkg>/rest for CRAN; "./<pkg>/..." for Bioc."""
    n = 0
    with tarfile.open(tarpath, "r") as tf:
        for m in tf:
            name = m.name[2:] if m.name.startswith("./") else m.name
            if not m.isfile() or Path(name).suffix not in EXTS:
                continue
            if m.size > 20 * 10**6:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            parts = name.split("/")
            pkg = parts[0] if parts else "?"
            rest = "/".join(parts[1:])
            area = rest.split("/")[0] if "/" in rest else "."
            n += 1
            yield dict(store=store, pkg=pkg, area=area,
                       sha=hashlib.sha256(data).hexdigest(), bytes=len(data)), data
    log(f"corpus {store}: {n:,} R-family docs from {tarpath}")


def main():
    t0 = time.time()
    docs = []            # Doc metadata in doc-id order
    texts_feed = []      # deferred (id, text) chunks for the pool
    sigs = {}            # doc_id -> signature bytes
    pool = Pool(processes=10)
    pending = []

    def drain(limit=None):
        nonlocal pending
        while pending and (limit is None or len(pending) > limit):
            for doc_id, sigb in pending.pop(0).get():
                sigs[doc_id] = sigb

    def feed(doc_id, text):
        nonlocal pending
        texts_feed.append((doc_id, text))
        if len(texts_feed) >= 200:
            pending.append(pool.apply_async(work_chunk, (texts_feed[:],)))
            texts_feed.clear()
            if len(pending) > 40:
                drain(limit=8)

    # ---- corpus side
    corpus_files = sorted(LOCAL_CORPUS.glob("*.tar"))
    if not corpus_files:
        log("FATAL: no corpus tars in " + str(LOCAL_CORPUS))
        sys.exit(1)
    for tp in corpus_files:
        store = "cran" if "cran" in tp.name else "bioc"
        for meta, data in iter_corpus_docs(tp, store):
            doc_id = len(docs)
            docs.append(Doc("corpus", store, meta["area"], meta["sha"], meta["bytes"]))
            feed(doc_id, data.decode("utf-8", errors="replace"))
    n_corpus = len(docs)
    log(f"corpus indexed: {n_corpus:,} docs")

    # ---- stack side (staged shards, may be partial if phase B still runs)
    n_stack = 0
    for shard in sorted(FILES.glob("shard-*.jsonl")):
        with open(shard, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = len(docs)
                docs.append(Doc("stack", rec.get("config", "?"),
                                rec.get("path", "/").lstrip("/").split("/")[0] or ".",
                                rec.get("sha256"), rec.get("bytes", 0),
                                rec.get("blob_id")))
                feed(doc_id, rec.get("content", ""))
                n_stack += 1
    log(f"stack indexed: {n_stack:,} docs from staged shards")
    if texts_feed:
        pending.append(pool.apply_async(work_chunk, (texts_feed[:],)))
        texts_feed.clear()
    drain(limit=0)
    pool.close(); pool.join()
    log(f"signatures built: {len(sigs):,} in {(time.time()-t0)/60:.1f} min "
        f"({sum(1 for v in sigs.values() if v is None):,} too-short-for-minhash)")

    # ---- LSH: band buckets -> candidate pairs
    sig_arr = np.zeros((len(docs), NUM_PERM), dtype=np.uint64)
    has_sig = np.zeros(len(docs), dtype=bool)
    for i, b in sigs.items():
        if b is not None:
            sig_arr[i] = np.frombuffer(b, dtype=np.uint64)
            has_sig[i] = True
    # persist for phase E (v3 delta) reuse — AFTER filling (v1 saved zeros: bug)
    np.save(LOCAL_CORPUS / "all_sigs.npy", sig_arr)
    with open(LOCAL_CORPUS / "all_docs.jsonl", "w") as f:
        for i, d in enumerate(docs):
            f.write(json.dumps(dict(doc_id=i, kind=d.kind, sub=d.sub, area=d.area,
                                    sha=d.sha, bytes=d.bytes, blob=d.blob,
                                    sig_ok=bool(has_sig[i]))) + "\n")

    pairs = set()          # (i, j) i<j candidate pairs (both with sigs)
    boiler_buckets = 0
    for band in range(BANDS):
        buckets = defaultdict(list)
        sl = sig_arr[:, band * ROWS:(band + 1) * ROWS]
        for i in range(len(docs)):
            if has_sig[i]:
                buckets[sl[i].tobytes()].append(i)
        for members in buckets.values():
            L = len(members)
            if L < 2:
                continue
            if L <= 200:
                for x in range(L):
                    for y in range(x + 1, L):
                        pairs.add((members[x], members[y]) if members[x] < members[y]
                                  else (members[y], members[x]))
            else:
                boiler_buckets += 1
                rep = members[0]
                for other in members[1:41]:   # star-verify a sample
                    pairs.add((rep, other) if rep < other else (other, rep))
        log(f"band {band + 1}/{BANDS}: cumulative candidate pairs {len(pairs):,}, "
            f"boilerplate buckets {boiler_buckets:,}")

    # ---- verify
    jacc = {}
    for i, j in pairs:
        if has_sig[i] and has_sig[j]:
            est = float(np.count_nonzero(sig_arr[i] == sig_arr[j])) / NUM_PERM
            if est >= 0.8:
                jacc[(i, j)] = est
    log(f"verified pairs >=0.8: {len(jacc):,}")

    # exact-dup edges by sha
    by_sha = defaultdict(list)
    for i, d in enumerate(docs):
        if d.sha:
            by_sha[d.sha].append(i)
    exact_edges = 0
    for group in by_sha.values():
        for j in group[1:]:
            exact_edges += 1

    # ---- (a) intra-Stack collapse (union-find over stack-stack edges >= thr)
    parent = list(range(len(docs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    stack_ids = [i for i, d in enumerate(docs) if d.kind == "stack"]
    for g in by_sha.values():
        sg = [x for x in g if docs[x].kind == "stack"]
        for j in sg[1:]:
            union(sg[0], j)
    intra_pairs_by_thr = {0.9: 0, 0.8: 0}
    for (i, j), est in jacc.items():
        if docs[i].kind == "stack" and docs[j].kind == "stack":
            if est >= 0.9:
                intra_pairs_by_thr[0.9] += 1
                union(i, j)
            intra_pairs_by_thr[0.8] += 1
    clusters = defaultdict(int)
    for i in stack_ids:
        clusters[find(i)] += 1
    n_clusters = len(clusters)
    # 0.8-collapse measured WITHOUT re-using 0.9 unions: recompute fresh
    parent8 = list(range(len(docs)))

    def find8(x):
        while parent8[x] != x:
            parent8[x] = parent8[parent8[x]]
            x = parent8[x]
        return x

    def union8(a, b):
        ra, rb = find8(a), find8(b)
        if ra != rb:
            parent8[rb] = ra

    for g in by_sha.values():
        sg = [x for x in g if docs[x].kind == "stack"]
        for j in sg[1:]:
            union8(sg[0], j)
    for (i, j), est in jacc.items():
        if docs[i].kind == "stack" and docs[j].kind == "stack" and est >= 0.8:
            union8(i, j)
    clusters8 = defaultdict(int)
    for i in stack_ids:
        clusters8[find8(i)] += 1

    # ---- (b) cross overlap: best corpus J per stack doc
    best = {}   # stack doc -> (best_est, corpus_doc)
    for (i, j), est in jacc.items():
        if docs[i].kind == "stack":
            a, other = i, j
        elif docs[j].kind == "stack":
            a, other = j, i
        else:
            continue  # corpus-corpus pair: irrelevant for cross overlap
        if docs[other].kind != "corpus":
            continue
        cur = best.get(a)
        if cur is None or est > cur[0]:
            best[a] = (est, other)
    # exact sha overlap
    sha_corpus = {d.sha for d in docs if d.kind == "corpus" and d.sha}
    exact_overlap = sum(1 for i in stack_ids if docs[i].sha in sha_corpus)

    def bucket_stats(thr):
        tot = sum(1 for i in stack_ids if i in best and best[i][0] >= thr)
        by_cfg = defaultdict(int)
        by_store = defaultdict(int)
        by_area = defaultdict(int)
        for i in stack_ids:
            if i in best and best[i][0] >= thr:
                by_cfg[docs[i].sub] += 1
                by_store[docs[best[i][1]].sub] += 1
                by_area[docs[best[i][1]].area] += 1
        return dict(total=tot, by_stack_config=dict(by_cfg),
                    by_corpus_store=dict(by_store), by_corpus_area=dict(by_area))

    # ---- (c) net-new keep set: exclude corpus-overlap >=0.9 and intra dups
    keep = []
    members_by_root = defaultdict(list)
    for i in stack_ids:
        members_by_root[find(i)].append(i)
    reps = set()
    for root, members in members_by_root.items():
        members.sort(key=lambda i: (-docs[i].bytes, i))
        reps.add(members[0])
    for i in stack_ids:
        if i not in reps:
            continue  # intra-stack duplicate (largest file per cluster kept)
        if i in best and best[i][0] >= 0.9:
            continue  # corpus near-dup
        keep.append(i)
    keep_bytes = sum(docs[i].bytes for i in keep)

    report = dict(
        generated=now(),
        docs=dict(corpus=n_corpus, stack=n_stack,
                  corpus_by_store={s: sum(1 for d in docs if d.kind == "corpus" and d.sub == s)
                                   for s in ("cran", "bioc")},
                  stack_by_config=defaultdict(int)),
        exact_dup=dict(stack_exact_edges=exact_edges,
                       stack_exact_sha_overlap_with_corpus=exact_overlap),
        intra_stack=dict(
            files=n_stack,
            clusters_j09=n_clusters,
            collapse_j09=round(1 - n_clusters / max(n_stack, 1), 4),
            clusters_j08=len(clusters8),
            collapse_j08=round(1 - len(clusters8) / max(n_stack, 1), 4),
            verified_pairs=dict(intra_j09=intra_pairs_by_thr[0.9],
                                intra_j08=intra_pairs_by_thr[0.8])),
        cross_overlap=dict(
            j09=bucket_stats(0.9), j08=bucket_stats(0.8),
            boilerplate_buckets=boiler_buckets,
            method_note="16x8 banding: P(candidate miss) ~5% at J=0.8, ~0.2% at J=0.9; "
                        "buckets >200 members star-verified (40-sample); exact sha "
                        "overlap counted separately"),
        net_new=dict(files=len(keep), bytes=keep_bytes,
                     share_of_staged=round(len(keep) / max(n_stack, 1), 4)),
    )
    cfgc = defaultdict(int)
    for i in stack_ids:
        cfgc[docs[i].sub] += 1
    report["docs"]["stack_by_config"] = dict(cfgc)
    (LOGS / "phase_c_report.json").write_text(json.dumps(report, indent=1))

    with open(STAGE / "datasets" / "stack_keep_ids.jsonl", "w") as f:
        for i in keep:
            d = docs[i]
            f.write(json.dumps(dict(doc_id=i, kind=d.kind, config=d.sub, area=d.area,
                                    blob_id=d.blob, sha256=d.sha, bytes=d.bytes)) + "\n")
    log(f"PHASE C COMPLETE corpus={n_corpus:,} stack={n_stack:,} "
        f"intra-collapse@0.9={report['intra_stack']['collapse_j09']} "
        f"cross@0.9={report['cross_overlap']['j09']['total']:,} "
        f"net-new={len(keep):,} ({keep_bytes/1e9:.2f}GB) "
        f"elapsed={(time.time()-t0)/60:.0f}min")


if __name__ == "__main__":
    main()
