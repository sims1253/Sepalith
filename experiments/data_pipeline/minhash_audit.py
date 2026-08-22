#!/usr/bin/env python3
"""KT-4 (A2 §9): MinHash/LSH unique-R audit on /mnt/h/sepalith/normalized.

Question (fromscratch-design-A2 §3.1/§10.1): the designs claim 5-8B qwen-tokens
of unique post-dedup R on acquisition; our in-hand CRAN corpus measured
3.0-3.5B raw. What does near-dup collapse do to the in-hand number?

Method. Deterministic seeded sample of N packages (ALL version dirs per
package — the corpus is <pkg>/<version>/<pkg>/...). Documents = individual
text files (LM-corpus whitelist). Per document:
  - qwen token count (local qwen3.5-2b-base-text-hf tokenizer; the reference
    frame of the 3.0-3.5B figure), byte size;
  - word-level 5-gram shingles -> 128-perm MinHash signature (numpy
    one-hash-per-shingle + (a*h+b) mod p projections).

Dedup, document level, exactly the two semantics the design cares about:
  1. exact dup (content hash) dies first;
  2. near-dup clusters via LSH banding (16 bands x 8 rows, ~J>=0.7 candidate
     recall) + signature-Jaccard verification + union-find, thresholded at
     J>=0.9 and J>=0.8 (two collapse levels).
A cluster contributes ONE exemplar (max-token doc) to unique tokens.

Strata reported separately (§3.4: version chains are near-dups BY DESIGN —
one exemplar feeds the causal stratum, deltas feed edit-diff):
  - ALL: every version of every sampled package (the causal-stratum pool);
  - LATEST: one version per package (max version dir) — isolates the
    cross-package dup rate (r-universe mirrors, boilerplate) from
    version-chain dup.

Extrapolation. Mean per-package raw/unique tokens x 14,202 packages, with a
bootstrap CI over package means (1k resamples). Caveat carried in the report:
sampling 300/14202 packages under-counts GLOBAL cross-package duplication
(the sampled ratio is an upper bound on unique fraction); version-chain and
within-package structure is fully represented per package.

Usage (detached):
  nohup nice -n 10 .venv/bin/python3 -u experiments/data_pipeline/minhash_audit.py \
      --n-packages 300 > /tmp/a2gates/gate2_minhash.log 2>&1 &

Writes (experiments/eval/):
  results_minhash_uniquer_sample.jsonl  one row per sampled package
  results_minhash_uniquer_summary.jsonl final summary row (also stdout)
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

CORPUS = Path("/mnt/h/sepalith/normalized")
OUT_DIR = Path("/home/m0hawk/Documents/Sepalith/experiments/eval")
SAMPLE_OUT = OUT_DIR / "results_minhash_uniquer_sample.jsonl"
SUMMARY_OUT = OUT_DIR / "results_minhash_uniquer_summary.jsonl"

TEXT_EXT = {".R", ".r", ".Rd", ".Rmd", ".Rnw", ".qmd", ".md", ".txt",
            ".cpp", ".c", ".cc", ".h", ".hpp", ".sql", ".py", ".sh",
            ".yaml", ".yml", ".toml", ".cfg", ".html", ".css", ".js"}
TEXT_NAMES = {"DESCRIPTION", "NAMESPACE", "LICENSE", "LICENCE", "NEWS",
              "NEWS.md", "CITATION", "AUTHORS", "README", "README.md"}
# LM-corpus relevant classes for the census split (rest = aux text)
R_CODE = {".R", ".r"}
R_DOCS = {".Rd", ".Rmd", ".Rnw", ".qmd"}
SRC_C = {".cpp", ".c", ".cc", ".h", ".hpp"}

NUM_PERM = 128
MERSENNE = (1 << 61) - 1
BANDS, ROWS = 16, 8          # b*r = 128; ~J>=0.7 candidate gate
SHINGLE_K = 5                # word 5-grams
MIN_DOC_TOKENS = 20          # below this a "document" is noise for dedup stats

_word_re = re.compile(r"\S+")


def is_text(p: Path) -> bool:
    if p.suffix in TEXT_EXT or p.name in TEXT_NAMES:
        return True
    return p.suffix == "" and p.name in TEXT_NAMES


def verkey(v: str):
    try:
        from packaging.version import Version
        return (1, Version(v))
    except Exception:
        return (0, v)


def doc_shingles(text: str):
    words = _word_re.findall(text)
    if len(words) < SHINGLE_K:
        return None
    sh = []
    for i in range(len(words) - SHINGLE_K + 1):
        sh.append(hashlib.blake2b(
            " ".join(words[i:i + SHINGLE_K]).encode("utf-8", "replace"),
            digest_size=8).digest())
    return np.frombuffer(b"".join(sh), dtype=np.uint64).copy()


class MinHasher:
    def __init__(self, seed=1234):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE, size=NUM_PERM, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE, size=NUM_PERM, dtype=np.uint64)

    def signature(self, shingles: np.ndarray) -> np.ndarray:
        h = shingles.astype(np.uint64)
        # (a_i * h + b_i) mod (2^61-1), vectorized over perms x shingles
        proj = (self.a[:, None] * h[None, :] + self.b[:, None]) % np.uint64(MERSENNE)
        return proj.min(axis=1)


def lsh_buckets(sig: np.ndarray):
    """(band, bytes-of-rows) keys -> this doc can be bucketed by them."""
    out = []
    for b in range(BANDS):
        chunk = sig[b * ROWS:(b + 1) * ROWS]
        out.append((b, chunk.tobytes()))
    return out


def sig_jaccard(s1: np.ndarray, s2: np.ndarray) -> float:
    return float(np.count_nonzero(s1 == s2)) / NUM_PERM


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster(docs, threshold):
    """docs: list of (sig, content_hash). Union-find over verified pairs."""
    # exact-dup groups first (same content hash -> same cluster)
    by_hash = defaultdict(list)
    for i, (_, ch) in enumerate(docs):
        if ch is not None:
            by_hash[ch].append(i)
    uf = UnionFind(len(docs))
    for group in by_hash.values():
        for j in group[1:]:
            uf.union(group[0], j)
    # LSH banding -> candidate pairs -> verify estimated jaccard
    buckets = defaultdict(list)
    for i, (sig, _) in enumerate(docs):
        for key in lsh_buckets(sig):
            buckets[key].append(i)
    seen_pairs = set()
    for key, members in buckets.items():
        if len(members) < 2 or len(members) > 200:   # huge bucket = boilerplate; still checked below
            if len(members) > 200:
                # conservative: chain-union consecutive members (all pairwise
                # would be O(n^2) on boilerplate like LICENSE texts)
                for x, y in zip(members, members[1:]):
                    uf.union(x, y)
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                pk = (a, b)
                if pk in seen_pairs:
                    continue
                seen_pairs.add(pk)
                if sig_jaccard(docs[a][0], docs[b][0]) >= threshold:
                    uf.union(a, b)
    clusters = defaultdict(list)
    for i in range(len(docs)):
        clusters[uf.find(i)].append(i)
    return list(clusters.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-packages", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/home/m0hawk/Documents/Sepalith/experiments/models/qwen3.5-2b-base-text-hf")

    packages = sorted(p.name for p in CORPUS.iterdir() if p.is_dir())
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(packages, min(args.n_packages, len(packages))))
    print(json.dumps(dict(corpus_packages=len(packages),
                          sampled=len(sample), seed=args.seed)), flush=True)

    mh = MinHasher()
    out = open(SAMPLE_OUT, "w")
    t_start = time.time()

    for pi, pkg in enumerate(sample):
        t0 = time.time()
        pkg_dir = CORPUS / pkg
        versions = sorted((v for v in pkg_dir.iterdir() if v.is_dir()),
                          key=lambda v: verkey(v.name))
        latest = versions[-1].name if versions else None
        docs_all = []       # (tokens, bytes, cls, ver, sig, content_hash)
        n_files = n_skip = 0
        for ver in versions:
            for f in ver.rglob("*"):
                if not f.is_file() or f.is_symlink():
                    continue
                n_files += 1
                if not is_text(f):
                    n_skip += 1
                    continue
                try:
                    raw = f.read_bytes()
                except OSError:
                    n_skip += 1
                    continue
                if b"\x00" in raw[:4096]:          # binary guard
                    n_skip += 1
                    continue
                text = raw.decode("utf-8", "replace")
                if len(text) < 64:
                    n_skip += 1
                    continue
                nt = len(tok(text, add_special_tokens=False)["input_ids"])
                if nt < MIN_DOC_TOKENS:
                    n_skip += 1
                    continue
                sh = doc_shingles(text)
                if sh is None or len(sh) < 8:
                    n_skip += 1
                    continue
                sfx = f.suffix if f.suffix else f.name
                if sfx in R_CODE:
                    cls = "R"
                elif sfx in R_DOCS:
                    cls = "Rdocs"
                elif sfx in SRC_C:
                    cls = "C"
                elif sfx in TEXT_NAMES and sfx not in TEXT_EXT:
                    cls = "meta"
                else:
                    cls = "other"
                docs_all.append((nt, len(raw), cls, ver.name,
                                 mh.signature(sh),
                                 hashlib.sha1(raw).digest()))
        rec = dict(package=pkg, versions=[v.name for v in versions],
                   latest=latest, files_seen=n_files, files_used=len(docs_all),
                   files_skipped=n_skip,
                   elapsed_s=round(time.time() - t0, 1))
        if docs_all:
            def stats(sel):
                toks = [d[0] for d in sel]
                raw_t = sum(toks)
                clusters = cluster([(d[4], d[5]) for d in sel], 0.9)
                uniq09 = sum(max(sel[i][0] for i in c) for c in clusters)
                clusters8 = cluster([(d[4], d[5]) for d in sel], 0.8)
                uniq08 = sum(max(sel[i][0] for i in c) for c in clusters8)
                # version-chain component: dups that vanish in ALL but exist
                # across DIFFERENT version dirs of the same package
                return dict(docs=len(sel), raw_tokens=raw_t,
                            unique_tokens_j09=uniq09, unique_tokens_j08=uniq08,
                            clusters_j09=len(clusters), clusters_j08=len(clusters8))

            rec["all_versions"] = stats(docs_all)
            latest_docs = [d for d in docs_all if latest and d[3] == latest]
            rec["latest_only"] = stats(latest_docs) if latest_docs else None
            by_cls = defaultdict(int)
            for d in docs_all:
                by_cls[d[2]] += d[0]
            rec["tokens_by_class"] = dict(by_cls)
            rec["bytes_total"] = sum(d[1] for d in docs_all)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if (pi + 1) % 10 == 0 or pi == len(sample) - 1:
            el = time.time() - t_start
            print(json.dumps(dict(progress=pi + 1, total=len(sample),
                                  elapsed_s=round(el, 1))), flush=True)
    out.close()

    # ---- extrapolation ----
    rows = [json.loads(l) for l in open(SAMPLE_OUT)]
    have = [r for r in rows if r.get("all_versions")]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def boot_ci(xs, n=1000, seed=7):
        r = random.Random(seed)
        ms = []
        for _ in range(n):
            ms.append(mean([xs[r.randrange(len(xs))] for _ in range(len(xs))]))
        ms.sort()
        return ms[int(0.025 * n)], ms[int(0.975 * n)]

    raw_all = [r["all_versions"]["raw_tokens"] for r in have]
    u09_all = [r["all_versions"]["unique_tokens_j09"] for r in have]
    u08_all = [r["all_versions"]["unique_tokens_j08"] for r in have]
    raw_lat = [r["latest_only"]["raw_tokens"] for r in have if r.get("latest_only")]
    u09_lat = [r["latest_only"]["unique_tokens_j09"] for r in have if r.get("latest_only")]
    u08_lat = [r["latest_only"]["unique_tokens_j08"] for r in have if r.get("latest_only")]
    N = len(packages)
    summary = dict(
        gate="minhash_unique_R_audit",
        corpus_packages=N, sampled=len(have), seed=args.seed,
        num_perm=NUM_PERM, shingle_k=SHINGLE_K,
        thresholds=dict(jaccard_09=0.9, jaccard_08=0.8),
        extrapolated_tokens=dict(
            all_versions=dict(
                raw=mean(raw_all) * N, raw_ci=boot_ci(raw_all),
                unique_j09=mean(u09_all) * N, unique_j09_ci=boot_ci(u09_all),
                unique_j08=mean(u08_all) * N, unique_j08_ci=boot_ci(u08_all)),
            latest_only=dict(
                raw=mean(raw_lat) * N, raw_ci=boot_ci(raw_lat),
                unique_j09=mean(u09_lat) * N, unique_j09_ci=boot_ci(u09_lat),
                unique_j08=mean(u08_lat) * N, unique_j08_ci=boot_ci(u08_lat)),
        ),
        collapse_ratio=dict(
            all_versions=dict(
                j09=mean(u09_all) / mean(raw_all),
                j08=mean(u08_all) / mean(raw_all)),
            latest_only=dict(
                j09=mean(u09_lat) / mean(raw_lat) if raw_lat else None,
                j08=mean(u08_lat) / mean(raw_lat) if raw_lat else None)),
        version_chain_share_of_dup=None,
        elapsed_s=round(time.time() - t_start, 1),
    )
    ra, ua9 = mean(raw_all), mean(u09_all)
    rl, ul9 = mean(raw_lat), mean(u09_lat)
    # dup decomposition at J>=0.9: total dup = raw_all - unique_all;
    # cross-version dup = raw_all - (raw_latest - unique_latest) - unique_all
    total_dup = ra - ua9
    cross_pkg_dup = rl - ul9            # dup surviving within latest versions
    version_dup = total_dup - cross_pkg_dup
    summary["version_chain_share_of_dup"] = dict(
        total_dup_per_pkg=total_dup,
        version_chain_dup_per_pkg=max(0.0, version_dup),
        cross_package_dup_per_pkg=cross_pkg_dup,
        version_share=round(max(0.0, version_dup) / total_dup, 4) if total_dup > 0 else None)
    with open(SUMMARY_OUT, "w") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
