#!/usr/bin/env python3
"""Stack v3 DELTA acquisition — Phase E (user-greenlit, coordinator-approved).

Vehicle: HuggingFaceCode stack-v3-full storage bucket, contents partitioned by
language (hf://buckets/HuggingFaceCode/stack-v3-full/contents/language=R +
RDoc). stack-v3-train itself is repo-grouped/mixed-language (15.9TB — no
targeted R pull possible); the bucket's language partitions ARE targeted.

Delta method: v3 content_id (sha1 of content, verified) joined against
  (a) v2-accepted staged contents (sha1 recomputed from shards),
  (b) CRAN + Bioc corpus files (sha1 from the local corpus tars),
blob not in either -> candidate delta.

License gate (tonight's conservative posture, per ruling): STRICT in-content
SPDX/license-header scan — only explicit permissive declarations pass
(SPDX-License-Identifier honored first); no-header/ambiguous -> excluded and
ledgered. This is a strict LOWER BOUND of v2's ScanCode treatment; the proper
per-file license pass requires the metadata/ part (3,495GB) — sized in the
report for a user decision. ODC-By (dataset-level) ledgered as its own class
on every accepted row.

Outputs:
  /mnt/h/sepalith/stack_staging/metadata_v3/        immutable parquet originals
  /mnt/h/sepalith/stack_staging/files_v3/shard-*.jsonl
  ledger entries type=phase-e-*
  logs/phase_e.log + phase_e_report.json
"""
import hashlib, json, os, re, sys, tarfile, time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

STAGE = Path("/mnt/h/sepalith/stack_staging")
FILES_V3 = STAGE / "files_v3"
META_V3 = STAGE / "metadata_v3"
LOGS = STAGE / "logs"
LEDGER = STAGE / "license_ledger.jsonl"
LOCAL = Path("/tmp/stack_r_fetch")
V3_LOCAL = LOCAL / "v3"
CORPUS = LOCAL / "corpus"
FILES_V2 = STAGE / "files"
EXTS = {".R", ".Rmd", ".qmd", ".Rd"}

MAX_TOTAL_BYTES = int(os.environ.get("STACKV3_MAX_BYTES", 8 * 10**9))
MIN_BYTES, MAX_FILE_BYTES = 100, 10 * 10**6

# --- strict in-content license header detection (ruling-aligned) ---
SPDX_ID = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.+-]+)")
PERMISSIVE_IDS = {
    "MIT", "MIT-0", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSD-4-Clause",
    "0BSD", "Unlicense", "ISC", "Zlib", "BSL-1.0", "X11", "NCSA", "ECL-2.0",
    "PostgreSQL", "GPL-1.0-only", "GPL-1.0-or-later", "GPL-2.0-only",
    "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later",
    "LGPL-2.0-only", "LGPL-2.0-or-later", "LGPL-2.1-only", "LGPL-2.1-or-later",
    "LGPL-3.0-only", "LGPL-3.0-or-later", "AGPL-3.0-only", "AGPL-3.0-or-later",
    "CC0-1.0", "CC-BY-1.0", "CC-BY-2.0", "CC-BY-2.5", "CC-BY-3.0", "CC-BY-4.0",
    "CC-BY-SA-1.0", "CC-BY-SA-2.0", "CC-BY-SA-2.5", "CC-BY-SA-3.0", "CC-BY-SA-4.0",
    "Artistic-2.0", "Artistic-1.0", "MPL-1.1", "MPL-2.0", "EPL-1.0", "EPL-2.0",
    "CPL-1.0", "CeCILL-2.0", "CeCILL-2.1",
}
HARD_BAD_SUB = ("NC", "ND", "GFDL", "NON-COMMERCIAL")  # in SPDX id or text markers

_TEXT_MARKERS = [
    # (class, regex, flags) — canonical license text fragments
    ("mit", re.compile(r"Permission is hereby granted, free of charge, to any (person|one)", re.I)),
    ("apache-2.0", re.compile(r"Apache License\b.{0,80}Version 2\.0, January 2004", re.I | re.S)),
    ("apache-2.0", re.compile(r"Licensed under the Apache License, Version 2\.0", re.I)),
    ("gpl-3", re.compile(r"GNU (LESSER |AFFERO )?GENERAL PUBLIC LICENSE.{0,200}Version 3", re.I | re.S)),
    ("gpl-2", re.compile(r"GNU (LESSER |AFFERO )?GENERAL PUBLIC LICENSE.{0,200}Version 2", re.I | re.S)),
    ("gpl-family", re.compile(r"GNU (Lesser )?General Public License", re.I)),
    ("bsd", re.compile(r"Redistribution and use in source and binary forms", re.I)),
    ("cc0", re.compile(r"CC0 1\.0 Universal|creativecommons\.org/publicdomain/zero", re.I)),
    ("cc-by", re.compile(r"Creative Commons Attribution[- ](?:4\.0|3\.0|2\.5|2\.0|1\.0)", re.I)),
    ("cc-by-sa", re.compile(r"Creative Commons Attribution-ShareAlike", re.I)),
    ("artistic-2.0", re.compile(r"The Artistic License.{0,120}Version 2\.0", re.I | re.S)),
    ("mpl-2.0", re.compile(r"Mozilla Public License, version 2\.0|Mozilla Public License 2\.0", re.I)),
    ("unlicense", re.compile(r"unlicense\.org|This is free and unencumbered software released into the public domain", re.I)),
    ("isc", re.compile(r"ISC License|Permission to use, copy, modify, and/or distribute this software for any purpose", re.I)),
    ("zlib", re.compile(r"zlib/libpng License|This software is provided 'as-is'", re.I)),
    ("cecill", re.compile(r"CeCILL.{0,80}(FREE SOFTWARE LICENSE|Licence Libre)", re.I | re.S)),
]
_NC_ND_TEXT = re.compile(
    r"noncommercial|non-commercial|noderivatives|no-derivatives|"
    r"creative commons attribution[^\n]{0,60}noncommercial|cc[\s-]?by[\s-]?nc", re.I)


def header_license(text: str, maxscan=20000):
    """-> (license_class|None, reason). Strict: only explicit declarations."""
    head = text[:maxscan]
    m = SPDX_ID.search(head)
    if m:
        lid = m.group(1).strip()
        if lid in PERMISSIVE_IDS:
            return lid.lower(), "spdx-id"
        if any(b in lid.upper() for b in ("NC", "ND")):
            return None, f"spdx-id-non-permissive:{lid}"
        return None, f"spdx-id-unknown:{lid}"
    # NC/ND in an explicit cc context beats any permissive marker
    for cls, rx in _TEXT_MARKERS:
        if rx.search(head):
            if _NC_ND_TEXT.search(head):
                return None, "nc-nd-terms-in-header"
            return cls, "license-text-marker"
    return None, "no-explicit-license"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(LOGS / "phase_e.log", "a") as f:
        f.write(line + "\n")


def ledger(rec):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def stage1_download():
    from huggingface_hub import HfFileSystem
    import shutil
    langs = os.environ.get("STACKV3_LANGS", "R,RDoc,RMarkdown").split(",")
    V3_LOCAL.mkdir(parents=True, exist_ok=True)
    META_V3.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for lang in langs:
        dstdir = V3_LOCAL / lang
        dstdir.mkdir(exist_ok=True)
        files = sorted(fs.ls(f"buckets/HuggingFaceCode/stack-v3-full/contents/language={lang}",
                             detail=False))
        for i, p in enumerate(files):
            dest = dstdir / p.split("/")[-1]
            if dest.exists() and dest.stat().st_size > 0:
                continue
            t0 = time.time()
            with fs.open(p, "rb") as f, open(dest, "wb") as o:
                shutil.copyfileobj(f, o, 1 << 22)
            log(f"downloaded {lang} {i + 1}/{len(files)} {dest.name} "
                f"{dest.stat().st_size:,}B in {time.time()-t0:.0f}s")
    # ship originals to NAS (bulk, one dir)
    if not os.environ.get("STACKV3_NO_SHIP"):
        t0 = time.time()
        shutil.copytree(V3_LOCAL, META_V3 / "contents", dirs_exist_ok=True)
        log(f"shipped v3 parquets to NAS in {time.time()-t0:.0f}s")
        ledger({"type": "phase-e-download", "ts": now(),
                "note": "stack-v3-full contents partitions language=R,RDoc,RMarkdown "
                        "(immutable originals in metadata_v3/contents/)"})


def build_join_sets():
    """sha1 sets: v2-accepted staged contents + corpus (CRAN+Bioc) files."""
    v2_sha1, corpus_sha1 = set(), set()
    for shard in sorted(FILES_V2.glob("shard-*.jsonl")):
        with open(shard, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = rec.get("content", "")
                v2_sha1.add(hashlib.sha1(c.encode("utf-8", "replace")).hexdigest())
    for tp in sorted(CORPUS.glob("*.tar")):
        with tarfile.open(tp, "r") as tf:
            for m in tf:
                if not m.isfile() or Path(m.name).suffix not in EXTS or m.size > 20 * 10**6:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                corpus_sha1.add(hashlib.sha1(f.read()).hexdigest())
    log(f"join sets: v2={len(v2_sha1):,} corpus={len(corpus_sha1):,} sha1s")
    return v2_sha1, corpus_sha1


def stage2_delta_gate():
    FILES_V3.mkdir(parents=True, exist_ok=True)
    (V3_LOCAL / "shards").mkdir(parents=True, exist_ok=True)
    v2_sha1, corpus_sha1 = build_join_sets()
    counts = Counter(); lic_hist = Counter()
    total_bytes = 0
    shard_idx = len(list(FILES_V3.glob("shard-*.jsonl")))
    fh = None; shard_written = 0
    import shutil
    verified = mismatched = 0

    def open_shard():
        nonlocal fh, shard_written
        fh = open(V3_LOCAL / "shards" / f"shard-{shard_idx:05d}.jsonl", "w",
                  encoding="utf-8")
        shard_written = 0

    def close_ship():
        nonlocal fh, shard_written, shard_idx
        if fh is None:
            return
        fh.close()
        src = V3_LOCAL / "shards" / f"shard-{shard_idx:05d}.jsonl"
        if shard_written > 0:
            shutil.copyfile(src, FILES_V3 / src.name)
        src.unlink()
        fh = None; shard_written = 0
        shard_idx += 1
        log(f"shipped v3 {src.name}")

    open_shard()
    langs_gate = os.environ.get("STACKV3_LANGS_GATE", "R,RDoc,RMarkdown").split(",")
    for lang in langs_gate:
        for p in sorted((V3_LOCAL / lang).glob("*.parquet")):
            pf = pq.ParquetFile(p)
            for rg in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["content_id", "content", "size_bytes",
                                                     "dedup_cluster", "repo_ids"])
                for cid, content, sz, dc, rids in zip(
                        tbl.column("content_id").to_pylist(),
                        tbl.column("content").to_pylist(),
                        tbl.column("size_bytes").to_pylist(),
                        tbl.column("dedup_cluster").to_pylist(),
                        tbl.column("repo_ids").to_pylist()):
                    counts["rows"] += 1
                    if counts["rows"] <= 200:
                        if hashlib.sha1(content.encode("utf-8", "replace")).hexdigest() == cid:
                            verified += 1
                        else:
                            mismatched += 1
                    if cid in v2_sha1 or cid in corpus_sha1:
                        counts["dup-of-v2-or-corpus"] += 1
                        continue
                    if sz < MIN_BYTES or sz > MAX_FILE_BYTES:
                        counts["size-out-of-range"] += 1
                        continue
                    if total_bytes > MAX_TOTAL_BYTES:
                        if fh:
                            close_ship()
                        counts["cap-stop"] += 1
                        break
                    cls, reason = header_license(content)
                    if cls is None:
                        counts[f"excluded/{reason.split(':')[0]}"] += 1
                        continue
                    lic_hist[cls] += 1
                    fh.write(json.dumps(dict(
                        source="stackv3", language=lang, content_id=cid,
                        dedup_cluster=dc, repo_ids=rids, bytes=sz,
                        dataset_license="ODC-By-1.0", license_class=f"v3-header:{cls}",
                        license_evidence=reason,
                        sha256=hashlib.sha256(content.encode("utf-8", "replace")).hexdigest(),
                        content=content), ensure_ascii=False) + "\n")
                    shard_written += sz; total_bytes += sz
                    counts["accepted"] += 1
                    if shard_written >= 500 * 10**6:
                        close_ship(); open_shard()
                else:
                    continue
                break  # cap-stop inner break propagates per file
            if counts["cap-stop"]:
                break
        if counts["cap-stop"]:
            break
    close_ship()
    ledger({"type": "phase-e-delta-gate", "ts": now(),
            "sha1_verified": f"{verified}/{verified + mismatched}",
            "counts": dict(counts), "license_histogram": dict(lic_hist),
            "accepted_bytes": total_bytes})
    log(f"PHASE E2 COMPLETE counts={dict(counts)} bytes={total_bytes:,}")


def stage3_dedup_report():
    """MinHash v3 accepted docs vs phase-C persisted corpus+v2 signatures."""
    import numpy as np
    NUM_PERM, BANDS, ROWS = 128, 16, 8
    sys.path.insert(0, str(Path(__file__).parent))
    from stackv2_phase_c import doc_signature  # same instrument
    base = np.load(CORPUS / "all_sigs.npy", mmap_mode="r")
    base_docs = [json.loads(l) for l in open(CORPUS / "all_docs.jsonl")]
    v3_docs = []; v3_sigs = []
    for shard in sorted(FILES_V3.glob("shard-*.jsonl")):
        with open(shard, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sig = doc_signature(rec.get("content", ""))
                # metadata only — content stays on disk (tokens stage re-reads)
                v3_docs.append({k: rec[k] for k in
                                ("content_id", "language", "bytes", "license_class")})
                v3_sigs.append(sig)
    log(f"v3 accepted docs: {len(v3_docs):,}")
    # exact sha dedup vs base + intra-v3
    base_sha = {d["sha"] for d in base_docs}
    # bucket v3 + base signatures together; only need cross + intra-v3 pairs
    buckets = {}
    def add(idx, sig, tag):
        if sig is None:
            return
        for b in range(BANDS):
            k = (tag, b, sig[b * ROWS:(b + 1) * ROWS].tobytes())
            buckets.setdefault(k, []).append(idx)
    for i, d in enumerate(base_docs):
        if d["sig_ok"]:
            add(i, base[i], "base")
    for j, s in enumerate(v3_sigs):
        add(j, s, "v3")
    cross09 = cross08 = intra_pairs = 0
    overlapped = set()
    v3_sig_arr = np.array([s if s is not None else np.zeros(NUM_PERM, dtype=np.uint64)
                           for s in v3_sigs], dtype=np.uint64)
    v3_has = np.array([s is not None for s in v3_sigs])
    # union-find for intra-v3 0.9 collapse
    parent = list(range(len(v3_docs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    base_sig_cache = {}
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        tag = key[0]
        if tag == "v3":
            if len(members) > 200:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    a, b = members[x], members[y]
                    if v3_has[a] and v3_has[b]:
                        est = float(np.count_nonzero(v3_sig_arr[a] == v3_sig_arr[b])) / NUM_PERM
                        if est >= 0.9:
                            intra_pairs += 1
                            union(a, b)
        else:
            pass  # base-only buckets: no pairs needed
    # cross buckets: build separate map of base members for join
    base_bucket_members = defaultdict(list)
    for key, members in buckets.items():
        if key[0] == "base":
            base_bucket_members[(key[1], key[2])].extend(members)
    for key, members in buckets.items():
        if key[0] != "v3" or not members:
            continue
        band, chunk = key[1], key[2]
        base_members = base_bucket_members.get((band, chunk), [])
        for j in members[:200]:
            if not v3_has[j]:
                continue
            best = 0.0
            for i in base_members[:400]:
                if i not in base_sig_cache:
                    base_sig_cache[i] = np.asarray(base[i])
                est = float(np.count_nonzero(v3_sig_arr[j] == base_sig_cache[i])) / NUM_PERM
                best = max(best, est)
            if best >= 0.9:
                cross09 += 1; overlapped.add(j)
            if best >= 0.8:
                cross08 += 1
    clusters = defaultdict(int)
    for j in range(len(v3_docs)):
        clusters[find(j)] += 1
    keep = []
    seen_root = set()
    for j in range(len(v3_docs)):
        r = find(j)
        if r in seen_root:
            continue
        seen_root.add(r)
        if j not in overlapped:
            keep.append(j)
    keep_bytes = sum(v3_docs[j]["bytes"] for j in keep)
    rep = dict(ts=now(), v3_accepted_docs=len(v3_docs),
               intra_v3=dict(clusters_j09=len(clusters),
                             collapse_j09=round(1 - len(clusters) / max(len(v3_docs), 1), 4),
                             verified_pairs_j09=intra_pairs),
               cross=dict(j09=cross09, j08=cross08),
               net_new=dict(files=len(keep), bytes=keep_bytes,
                            share=round(len(keep) / max(len(v3_docs), 1), 4)))
    (LOGS / "phase_e_report.json").write_text(json.dumps(rep, indent=1))
    with open(STAGE / "datasets" / "stack_v3_keep_ids.jsonl", "w") as f:
        for j in keep:
            r = v3_docs[j]
            f.write(json.dumps(dict(content_id=r["content_id"], language=r["language"],
                                    bytes=r["bytes"],
                                    license_class=r["license_class"])) + "\n")
    ledger({"type": "phase-e-dedup", **rep})
    log(f"PHASE E3 COMPLETE net-new={len(keep):,} ({keep_bytes/1e9:.2f}GB)")


def stage4_tokens():
    from transformers import AutoTokenizer
    import random
    tok = AutoTokenizer.from_pretrained(
        "/home/m0hawk/Documents/Sepalith/experiments/models/qwen3.5-2b-base-text-hf")
    keep_ids = {json.loads(l)["content_id"]
                for l in open(STAGE / "datasets" / "stack_v3_keep_ids.jsonl")}
    rng = random.Random(20260822)
    pool = []
    for shard in sorted(FILES_V3.glob("shard-*.jsonl")):
        with open(shard, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec["content_id"] in keep_ids:
                    pool.append(rec)
    sample = rng.sample(pool, min(300, len(pool)))
    rates = [len(tok.encode(r["content"])) / max(r["bytes"], 1) for r in sample]
    mean_rate = sum(rates) / len(rates)
    total = sum(r["bytes"] for r in pool)
    # prose-vs-code register split for RMarkdown net-new (the register's value)
    chunk_re = re.compile(r"```[\w-]*\{[rR]", re.I)
    rmd = [r for r in pool if r["language"] == "RMarkdown"]
    prose_code = {}
    if rmd:
        rsample = rng.sample(rmd, min(150, len(rmd)))
        pt = ct = pb = cb = 0
        for r in rsample:
            text = r["content"]
            parts, in_code, buf = [], False, []
            for line in text.splitlines(keepends=True):
                if not in_code and chunk_re.search(line):
                    in_code = True
                    parts.append(("prose", "".join(buf))); buf = []
                    continue
                if in_code and line.strip().startswith("```"):
                    in_code = False
                    parts.append(("code", "".join(buf))); buf = []
                    continue
                buf.append(line)
            parts.append(("code" if in_code else "prose", "".join(buf)))
            prose = "".join(t for k, t in parts if k == "prose")
            code = "".join(t for k, t in parts if k == "code")
            pt += len(tok.encode(prose)); ct += len(tok.encode(code))
            pb += len(prose.encode("utf-8", "replace")); cb += len(code.encode("utf-8", "replace"))
        if pb + cb:
            prose_code = dict(
                rmd_files_total=len(rmd),
                sampled=len(rsample),
                prose_bytes_frac=round(pb / (pb + cb), 3),
                prose_tokens_frac=round(pt / max(pt + ct, 1), 3),
                note="code = ```{r} chunk bodies, prose = rest")
    rep = json.load(open(LOGS / "phase_e_report.json"))
    rep["net_new_tokens_estimate"] = dict(
        mean_tokens_per_byte=round(mean_rate, 4), net_new_bytes=total,
        tokens=int(mean_rate * total),
        sample=dict(n=len(sample), seed=20260822))
    rep["rmd_register_split"] = prose_code
    (LOGS / "phase_e_report.json").write_text(json.dumps(rep, indent=1))
    log(f"PHASE E4 tokens: ~{int(mean_rate * total):,} "
        f"({int(mean_rate * total) / 1e9:.2f}B) from {len(pool):,} net-new files; "
        f"rmd split: {prose_code}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("download", "all"):
        stage1_download()
    if stage in ("gate", "all"):
        stage2_delta_gate()
    if stage in ("dedup", "all"):
        stage3_dedup_report()
    if stage in ("tokens", "all"):
        stage4_tokens()
