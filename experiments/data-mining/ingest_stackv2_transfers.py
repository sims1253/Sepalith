#!/usr/bin/env python3
"""the-stack-v2 transfer-strata acquisition (post click-through, 2026-08-25).

The v2 parquets are METADATA-ONLY; content is fetched per blob_id from
Software Heritage's S3 (the dataset-card mechanism our stackv2_phase_b
already used, sha256-verified on samples). Per stratum: select rows from
the language parquets (permissive license filter, star-sorted best-first,
de-vendored/generated), fetch blobs politely, STREAM-tokenize to 32K
blocks (earlyoom-hardened), memmap-write (no copies).

Strata and char budgets (full A2 draw targets, ~3.8 chars/tok):
  python 10.0GB (2.8B tok) | c_cpp 2.2GB | js_ts 1.9GB | sql 1.5GB
  julia 1.1GB | matlab 0.9GB
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.request
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "bigcode/the-stack-v2"
TOK = "/mnt/h/sepalith/datasets/a2_tokenizer_v1/tokenizer.json"
OUT = Path("/mnt/h/sepalith/a2_transfers")
S3 = "https://softwareheritage.s3.amazonaws.com/content/{}"
UA = "sepalith-research/0.1 (dataset preparation; contact via repo)"
SEQ = 1025

STRATA = {
    "python_v2": (["Python"], 10_000_000_000),
    "c_cpp_v2": (["C", "C++"], 2_200_000_000),
    "js_ts_v2": (["JavaScript", "TypeScript"], 1_900_000_000),
    "sql_v2": (["SQL", "TSQL", "PLSQL", "SQLPL", "PLpgSQL"], 1_500_000_000),
    "julia_v2": (["Julia"], 1_100_000_000),
    "matlab_v2": (["MATLAB"], 900_000_000),
}
PERMISSIVE_HINTS = ("mit", "apache", "bsd", "isc", "postgresql",
                    "public domain", "mozilla", "gpl", "lgpl", "unlicense",
                    "zlib", "boost")


def fetch_blob(blob_id: str, sha: str) -> str | None:
    """SWH blobs are GZIPPED (the phase_b contract): decompress, then
    sha256 the PLAIN content against content_id."""
    import gzip
    for attempt in range(3):
        try:
            req = urllib.request.Request(S3.format(blob_id),
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = gzip.decompress(r.read())
            # content_id is the GIT blob sha1 (verified live; phase_b's
            # "sha256" note was wrong): sha1("blob <len>\0" + content)
            git_sha = hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if git_sha != sha:
                return None
            return data.decode("utf-8", "replace")
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def select_rows(langs: list[str], budget: int) -> list[dict]:
    """Per-parquet quota selection (BOUNDED MEMORY — the first version
    accumulated ALL rows and JavaScript's 8.7M-rows/parquet OOM'd it):
    each parquet's filtered rows are sorted by stars desc and the top
    per-parquet quota is kept; quotas fill the byte budget across files."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(REPO)
    files = []
    for lang in langs:
        files += [s.rfilename for s in info.siblings
                  if s.rfilename.startswith(f"data/{lang}/")
                  and s.rfilename.endswith(".parquet")]
    # budget split: bigger share to languages with more files (proxy for
    # corpus size), then per-file quotas
    est_avg = 6_000                       # bytes/blob planning figure
    budget_rows = int(budget / est_avg)
    per_file = max(50_000, budget_rows // max(1, len(files)))
    kept: list[dict] = []
    est = 0
    for rf in files:
        local = hf_hub_download(REPO, rf, repo_type="dataset")
        t = pq.read_table(local, columns=[
            "blob_id", "content_id", "path", "detected_licenses",
            "repo_name", "star_events_count", "is_vendor",
            "is_generated", "length_bytes"])
        rows = []
        for r in t.to_pylist():
            if r["is_vendor"] or r["is_generated"]:
                continue
            if not (0 < (r["length_bytes"] or 0) < 300_000):
                continue
            lic = " ".join(r["detected_licenses"] or []).lower()
            if lic and not any(h in lic for h in PERMISSIVE_HINTS):
                continue
            rows.append(r)
        del t
        rows.sort(key=lambda r: -(r["star_events_count"] or 0))
        quota = per_file
        take = []
        for r in rows:
            if len(take) >= quota or est >= budget:
                break
            take.append(r)
            est += r["length_bytes"]
        # breadth inside the kept slice (star-sorted take, seeded shuffle)
        rng = random.Random(f"sel:{rf}")
        rng.shuffle(take)
        kept.extend(take)
        print(f"  [sel] {rf.split('/')[1]}: kept {len(take)} "
              f"(est {est/1e9:.2f} GB)", flush=True)
        del rows
        if est >= budget:
            break
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strata", default=",".join(STRATA))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    tok = Tokenizer.from_file(TOK)
    eos = tok.token_to_id("<|endoftext|>")

    for name in args.strata.split(","):
        if name not in STRATA:
            continue
        out_dir = OUT / name
        if (out_dir / "blocks.npy").exists():
            print(f"[{name}] present — skip", flush=True)
            continue
        langs, budget = STRATA[name]
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = select_rows(langs, budget)
        est = sum(r["length_bytes"] for r in rows)
        print(f"[{name}] {len(rows)} blobs selected "
              f"(~{est/1e9:.1f} GB est)", flush=True)

        stream = array("i")
        n_docs = n_bytes = n_err = 0
        done_path = out_dir / "fetch_done.txt"
        done_keys = set()
        if done_path.exists():
            done_keys = {l.split("\t")[0] for l in
                         done_path.read_text().splitlines() if l}

        def work(r):
            blob, sha = r["blob_id"], r["content_id"]
            if blob in done_keys:
                return r, ""
            txt = fetch_blob(blob, sha)
            time.sleep(0.05)          # polite per-thread pace
            return r, txt or ""

        t0 = time.time()
        pending_txt = []
        with open(done_path, "a") as done_fh, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r, txt in ex.map(work, rows):
                if not txt:
                    n_err += 1
                    continue
                n_docs += 1
                n_bytes += len(txt.encode("utf-8", "replace"))
                pending_txt.append(txt[:100_000])
                done_fh.write(f"{r['blob_id']}\t{len(txt)}\n")
                if len(pending_txt) >= 4000:
                    for e in tok.encode_batch(pending_txt):
                        stream.extend(e.ids)
                        stream.append(eos)
                    pending_txt = []
                if n_docs % 5000 == 0:
                    print(f"[{name}] {n_docs} docs {n_bytes/1e9:.1f} GB "
                          f"{len(stream)/1e6:.0f}M tok err={n_err} "
                          f"{time.time()-t0:.0f}s", flush=True)
        if pending_txt:
            for e in tok.encode_batch(pending_txt):
                stream.extend(e.ids)
                stream.append(eos)

        n_blocks = (len(stream) - 1) // SEQ
        if n_blocks == 0:
            print(f"[{name}] EMPTY", flush=True)
            continue
        mm = np.lib.format.open_memmap(out_dir / "blocks.npy", mode="w+",
                                       dtype=np.int32,
                                       shape=(n_blocks, SEQ))
        flat = np.frombuffer(stream, dtype=np.int32)
        for b0 in range(0, n_blocks, 4096):
            b1 = min(b0 + 4096, n_blocks)
            seg = flat[b0 * 1024:(b1 - 1) * 1024 + SEQ]
            view = np.lib.stride_tricks.as_strided(
                seg, shape=(b1 - b0, SEQ),
                strides=(seg.strides[0] * 1024, seg.strides[0]))
            mm[b0:b1] = view
        mm.flush()
        del mm
        (out_dir / "stats.json").write_text(json.dumps(
            dict(stratum=name, langs=langs, docs=n_docs, bytes=n_bytes,
                 tokens=len(stream), blocks=n_blocks, fetch_errors=n_err),
            indent=1))
        print(f"[{name}] DONE {n_docs} docs {len(stream)/1e6:.0f}M tok "
              f"err={n_err}", flush=True)


if __name__ == "__main__":
    main()
