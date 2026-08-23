#!/usr/bin/env python3
"""Phase E REMEDY — corrects two compounding bugs in the v3 delta numbers:
  1. stackv3_phase_e2.py pool workers ran WITHOUT the join/skip sets
     (Python 3.14 forkserver: runtime-populated module globals are not
     inherited) -> parallel-accepted records contaminated with corpus/v2
     exact dups (measured 227/500).
  2. phase C persisted all_sigs.npy BEFORE filling it (zeros) -> stage3's
     cross-vs-base LSH join was meaningless (cross=0).
v2 headline numbers are unaffected (computed in-process in phase C).

Remedy (all data already local):
  a. consolidate v3 shard records, dedup by content_id;
  b. EXACT cross-dedup by sha256 vs corpus+v2 universe (all_docs.jsonl);
  c. MinHash signatures for the clean set (10-proc pool, worker-local state);
  d. intra-v3 collapse @J>=0.9 (union-find, largest-file-per-cluster kept);
  e. near-dup cross-check @J>=0.9 vs v2-KEEP signatures (recomputed from the
     NAS shards — corpus near-dups <1.0 unverifiable tonight, tars deleted;
     exact corpus dups are caught by (b); limitation stated in the report);
  f. corrected keep-list, token estimate (qwen), report overwrite.
"""
import hashlib, json, random, sys, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stackv2_phase_c import doc_signature, NUM_PERM, BANDS, ROWS

STAGE = Path("/mnt/h/sepalith/stack_staging")
FILES_V3, FILES_V2 = STAGE / "files_v3", STAGE / "files"
LOGS = STAGE / "logs"
CORPUS = Path("/tmp/stack_r_fetch/corpus")
QWEN = "/home/m0hawk/Documents/Sepalith/experiments/models/qwen3.5-2b-base-text-hf"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(m):
    print(f"{now()} {m}", flush=True)


def work_chunk(chunk):
    return [(i, doc_signature(t).tobytes() if doc_signature(t) is not None else None)
            for i, t in chunk]


def main():
    t0 = time.time()
    # (a) consolidate + id-dedup
    by_id = {}
    for p in sorted(FILES_V3.glob("shard-*.jsonl")):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cur = by_id.get(r["content_id"])
                if cur is None or r["bytes"] > cur["bytes"]:
                    by_id[r["content_id"]] = r
    log(f"v3 unique accepted: {len(by_id):,}")
    # (b) exact cross-dedup vs corpus+v2 sha256 universe
    base_sha = set()
    for line in open(CORPUS / "all_docs.jsonl"):
        d = json.loads(line)
        if d.get("sha"):
            base_sha.add(d["sha"])
    clean = {}
    n_dup = 0
    for cid, r in by_id.items():
        sha = hashlib.sha256(r["content"].encode("utf-8", "replace")).hexdigest()
        if sha in base_sha:
            n_dup += 1
        else:
            r["sha256"] = sha
            clean[cid] = r
    log(f"exact cross-dedup: removed {n_dup:,} corpus/v2 dups -> clean {len(clean):,}")
    # (c) signatures (chunked tasks; state passed via task data, not globals)
    items = list(clean.items())
    idx_texts = [(i, r["content"]) for i, (_, r) in enumerate(items)]
    pool = Pool(10)
    tasks = [idx_texts[i:i + 200] for i in range(0, len(idx_texts), 200)]
    sigs = {}
    for res in pool.imap(work_chunk2, tasks):
        for i, sb in res:
            sigs[i] = sb
    pool.close(); pool.join()
    log(f"signatures: {len(sigs):,} in {(time.time()-t0)/60:.1f} min")
    # (d) intra-v3 collapse
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_sha = defaultdict(list)
    for i, (_, r) in enumerate(items):
        by_sha[r["sha256"]].append(i)
    for g in by_sha.values():
        for j in g[1:]:
            union(g[0], j)
    buckets = defaultdict(list)
    for i in range(len(items)):
        sb = sigs.get(i)
        if sb is None:
            continue
        sig = np.frombuffer(sb, dtype=np.uint64)
        for b in range(BANDS):
            buckets[(b, sig[b * ROWS:(b + 1) * ROWS].tobytes())].append(i)
    npairs = 0
    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:
            if len(members) > 200:
                rep = members[0]
                for other in members[1:41]:
                    sa = np.frombuffer(sigs[rep], dtype=np.uint64)
                    so = np.frombuffer(sigs[other], dtype=np.uint64)
                    if np.count_nonzero(sa == so) / NUM_PERM >= 0.9:
                        npairs += 1; union(rep, other)
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                sa = np.frombuffer(sigs[a], dtype=np.uint64)
                so = np.frombuffer(sigs[b], dtype=np.uint64)
                if np.count_nonzero(sa == so) / NUM_PERM >= 0.9:
                    npairs += 1; union(a, b)
    members_by_root = defaultdict(list)
    for i in range(len(items)):
        members_by_root[find(i)].append(i)
    reps = set()
    for members in members_by_root.values():
        members.sort(key=lambda i: (-items[i][1]["bytes"], i))
        reps.add(members[0])
    log(f"intra-v3: {len(members_by_root):,} clusters from {len(items):,} "
        f"(collapse {1 - len(members_by_root)/len(items):.3f}, pairs {npairs:,})")
    # (e) near-dup vs v2-KEEP signatures (recomputed from NAS shards)
    keep_ids = {json.loads(l)["blob_id"] for l in open(STAGE / "datasets/stack_keep_ids.jsonl")}
    v2_texts = []
    for shard in sorted(FILES_V2.glob("shard-*.jsonl")):
        with open(shard, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("blob_id") in keep_ids:
                    v2_texts.append((len(v2_texts), r.get("content", "")))
    log(f"v2 keep texts for cross-check: {len(v2_texts):,}")
    pool = Pool(10)
    tasks = [v2_texts[i:i + 200] for i in range(0, len(v2_texts), 200)]
    v2_sigs = {}
    for res in pool.imap(work_chunk2, tasks):
        for i, sb in res:
            v2_sigs[i] = sb
    pool.close(); pool.join()
    v2_buckets = defaultdict(list)
    for i, sb in v2_sigs.items():
        if sb is None:
            continue
        sig = np.frombuffer(sb, dtype=np.uint64)
        for b in range(BANDS):
            v2_buckets[(b, sig[b * ROWS:(b + 1) * ROWS].tobytes())].append(i)
    v2_sig_arr = {i: np.frombuffer(sb, dtype=np.uint64) for i, sb in v2_sigs.items() if sb}
    overlapped = set()
    for i in range(len(items)):
        if i not in reps:
            continue
        sb = sigs.get(i)
        if sb is None:
            continue
        sig = np.frombuffer(sb, dtype=np.uint64)
        for b in range(BANDS):
            for j in v2_buckets.get((b, sig[b * ROWS:(b + 1) * ROWS].tobytes()), [])[:50]:
                if np.count_nonzero(sig == v2_sig_arr[j]) / NUM_PERM >= 0.9:
                    overlapped.add(i)
                    break
            if i in overlapped:
                break
    log(f"near-dup vs v2-keep @0.9: {len(overlapped):,}")
    keep = [i for i in reps if i not in overlapped]
    keep_bytes = sum(items[i][1]["bytes"] for i in keep)
    # (f) tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    rng = random.Random(20260822)
    sample = rng.sample(keep, min(300, len(keep)))
    rates = [len(tok.encode(items[i][1]["content"])) / max(items[i][1]["bytes"], 1)
             for i in sample]
    mean_rate = sum(rates) / len(rates)
    # rmd register split on the clean keep set
    rmd = [i for i in keep if items[i][1]["language"] == "RMarkdown"]
    rmd_split = {}
    if rmd:
        import re
        chunk_re = re.compile(r"```[\w-]*\{[rR]", re.I)
        pt = ct = pb = cb = 0
        for i in rng.sample(rmd, min(150, len(rmd))):
            parts, in_code, buf = [], False, []
            for line in items[i][1]["content"].splitlines(keepends=True):
                if not in_code and chunk_re.search(line):
                    in_code = True; parts.append(("prose", "".join(buf))); buf = []
                    continue
                if in_code and line.strip().startswith("```"):
                    in_code = False; parts.append(("code", "".join(buf))); buf = []
                    continue
                buf.append(line)
            parts.append(("code" if in_code else "prose", "".join(buf)))
            prose = "".join(t for k, t in parts if k == "prose")
            code = "".join(t for k, t in parts if k == "code")
            pt += len(tok.encode(prose)); ct += len(tok.encode(code))
            pb += len(prose.encode("utf-8", "replace")); cb += len(code.encode("utf-8", "replace"))
        if pb + cb:
            rmd_split = dict(rmd_kept=len(rmd), sampled=min(150, len(rmd)),
                             prose_bytes_frac=round(pb / (pb + cb), 3),
                             prose_tokens_frac=round(pt / max(pt + ct, 1), 3))
    report = dict(
        ts=now(), corrected=True,
        bug_notes=["e2 forkserver: join/skip sets empty in workers -> contaminated "
                   "accepts, fixed by exact sha256 filter here",
                   "phase C persisted all_sigs.npy pre-fill (zeros) -> stage3 cross "
                   "join void; v2 numbers unaffected (in-process)"],
        v3_accepted_unique=len(by_id),
        exact_dups_vs_corpus_v2=n_dup,
        clean=len(clean),
        intra_v3=dict(clusters=len(members_by_root), pairs=npairs,
                      collapse=round(1 - len(members_by_root) / len(items), 4)),
        near_dup_vs_v2_keep_j09=len(overlapped),
        net_new=dict(files=len(keep), bytes=keep_bytes),
        tokens=dict(mean_tokens_per_byte=round(mean_rate, 4),
                    estimate=int(mean_rate * keep_bytes),
                    sample=dict(n=len(sample), seed=20260822)),
        rmd_register_split=rmd_split,
        limitations="corpus near-dups (0.8<=J<1.0) unverifiable tonight (corpus "
                    "tars deleted after exact-hash join); exact corpus dups "
                    "removed by sha256; v2-keep near-dups removed at J>=0.9")
    (LOGS / "phase_e_report.json").write_text(json.dumps(report, indent=1))
    with open(STAGE / "datasets" / "stack_v3_keep_ids.jsonl", "w") as f:
        for i in keep:
            r = items[i][1]
            f.write(json.dumps(dict(content_id=r["content_id"], language=r["language"],
                                    bytes=r["bytes"], license_class=r["license_class"],
                                    sha256=r["sha256"])) + "\n")
    with open(STAGE / "license_ledger.jsonl", "a") as f:
        f.write(json.dumps({"type": "phase-e-remedy", **report}) + "\n")
    log(f"REMEDY COMPLETE net-new={len(keep):,} ({keep_bytes/1e6:.0f}MB) "
        f"tokens~{int(mean_rate*keep_bytes):,} elapsed={(time.time()-t0)/60:.0f}min")


def work_chunk2(chunk):
    out = []
    for i, t in chunk:
        s = doc_signature(t)
        out.append((i, s.tobytes() if s is not None else None))
    return out


if __name__ == "__main__":
    main()
