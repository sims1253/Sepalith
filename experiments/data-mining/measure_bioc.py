#!/usr/bin/env python3
"""Bioconductor ingest measurement — run after (or during) ingest_bioc.py.

Produces /mnt/h/sepalith/bioc_staging/measure_report.json with:
  1. ledger summary: decision counts, included license-class histogram,
     excluded packages + reasons (ruling: permissive-only, no NC/ND)
  2. overlap check: Bioc release names vs the CRAN store
     (/mnt/h/sepalith/normalized) — full join + the scout's 50-name sample
  3. token estimate: 40-package stratified sample tokenized with the qwen
     tokenizer (CPU, tokenizers lib), extrapolated to all included packages

Zero GPU. Reads only the staging tree + the CRAN store directory listing.
"""
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/mnt/h/sepalith")
ST = ROOT / "bioc_staging"
LEDGER = ST / "bioc_license_ledger.jsonl"
MANIFEST = ST / "manifest.jsonl"
CRAN_STORE = ROOT / "normalized"
BIOC_TREE = ROOT / "normalized_bioc"
TOKENIZER_JSON = Path("/home/m0hawk/Documents/Sepalith/experiments/models/"
                       "qwen3.5-2b-base-text-hf/tokenizer.json")
SRC_EXTS = (".R", ".Rmd", ".qmd", ".Rd")
SAMPLE_N = 40


def ledger_summary():
    last = {}
    with open(LEDGER) as f:
        for line in f:
            r = json.loads(line)
            last[r["package"]] = r
    decisions = Counter(r["decision"] for r in last.values())
    included = [r for r in last.values() if r["decision"] == "include"]
    hist = Counter(r["license_class"] for r in included)
    excluded = [(r["package"], r["license_field"][:60], r["reason"])
                for r in last.values() if r["decision"] == "exclude"]
    errors = [(r["package"], r["reason"][:100])
              for r in last.values() if r["decision"] in ("error", "no-tarball")]
    return last, decisions, hist, excluded, errors


def overlap_check(bioc_names):
    cran = {p.name for p in CRAN_STORE.iterdir() if p.is_dir()}
    overlap = sorted(set(bioc_names) & cran)
    random.seed(1234)
    sample = random.sample(sorted(bioc_names), min(50, len(bioc_names)))
    sample_hits = [s for s in sample if s in cran]
    return len(cran), overlap, sample, sample_hits


def pick_sample(manifest):
    """40 included packages, stratified proportionally by license_class."""
    pkgs = [json.loads(l) for l in open(manifest)]
    random.seed(1234)
    by_class = defaultdict(list)
    for m in pkgs:
        by_class[m.get("license_class", "?")].append(m["package"])
    quota = {}
    total = sum(len(v) for v in by_class.values())
    for cls, names in sorted(by_class.items()):
        quota[cls] = max(1, round(SAMPLE_N * len(names) / total))
    chosen = []
    for cls, names in sorted(by_class.items()):
        random.shuffle(names)
        chosen += [(n, cls) for n in names[:quota[cls]]]
    return chosen[:SAMPLE_N + len(by_class)], pkgs


def tokenize_estimate(chosen):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    per_pkg = {}
    for pkg, cls in chosen:
        vdir = next((BIOC_TREE / pkg).iterdir())
        src = vdir / pkg
        files = [f for f in src.rglob("*") if f.suffix in SRC_EXTS and f.is_file()]
        texts = [f.read_text(errors="replace") for f in files]
        enc = tok.encode_batch(texts, add_special_tokens=False)
        per_pkg[pkg] = dict(cls=cls, n_files=len(files),
                            tokens=sum(len(e.ids) for e in enc),
                            bytes=sum(len(t.encode("utf-8", "replace")) for t in texts))
        print(f"  {pkg:30s} {cls:12s} {per_pkg[pkg]['n_files']:4d} files "
              f"{per_pkg[pkg]['tokens']:9,d} tok", flush=True)
    return per_pkg


def main():
    last, decisions, hist, excluded, errors = ledger_summary()
    idx = json.load(open(ST / "PACKAGES_index.json"))
    n_cran, overlap, sample, sample_hits = overlap_check(sorted(idx))
    print(f"ledger decisions: {dict(decisions)}")
    print(f"license-class histogram (included): {hist.most_common()}")
    print(f"excluded ({len(excluded)}): {excluded}")
    if errors:
        print(f"errors/no-tarball ({len(errors)}): {errors[:20]}")
    print(f"CRAN store packages: {n_cran}; Bioc index: {len(idx)}; "
          f"name overlap: {len(overlap)} {overlap[:10]}")
    print(f"50-sample verification: {len(sample_hits)} hits {sample_hits}")

    report = dict(
        ledger_packages=len(last), decisions=dict(decisions),
        license_hist_included=dict(hist.most_common()),
        excluded=excluded, errors=errors,
        overlap_full=dict(cran_packages=n_cran, bioc_packages=len(idx),
                          overlap_count=len(overlap), overlap_names=overlap[:50]),
        overlap_sample50=dict(sample=sample, hits=sample_hits))

    if MANIFEST.exists():
        chosen, all_manifest = pick_sample(MANIFEST)
        per_pkg = tokenize_estimate(chosen)
        toks = [v["tokens"] for v in per_pkg.values()]
        n_inc = decisions.get("include", 0)
        mean = sum(toks) / len(toks)
        srt = sorted(toks)
        est_mean = mean * n_inc
        # robust alternative: extrapolate from per-class means
        by_cls = defaultdict(list)
        for v in per_pkg.values():
            by_cls[v["cls"]].append(v["tokens"])
        cls_means = {c: sum(t) / len(t) for c, t in by_cls.items()}
        cls_counts = Counter(m.get("license_class", "?") for m in all_manifest)
        est_strat = sum(cls_means[c] * cls_counts.get(c, 0) for c in cls_means)
        report["tokens"] = dict(
            sample_n=len(toks), mean_per_pkg=round(mean), median_per_pkg=srt[len(srt)//2],
            p90_per_pkg=srt[int(len(srt)*0.9)-1], min=srt[0], max=srt[-1],
            included_packages=n_inc,
            est_total_mean_extrap=round(est_mean),
            est_total_stratified=round(est_strat),
            scout_estimate="0.35-0.55B",
            tokenizer=str(TOKENIZER_JSON), file_scope=list(SRC_EXTS),
            per_pkg={k: v for k, v in per_pkg.items()})
        print(f"tokens: mean/pkg {mean:,.0f} median {srt[len(srt)//2]:,} "
              f"p90 {srt[int(len(srt)*0.9)-1]:,}")
        print(f"NET-NEW TOKEN ESTIMATE: mean-extrap {est_mean/1e9:.2f}B | "
              f"stratified {est_strat/1e9:.2f}B over {n_inc} included pkgs "
              f"(scout: 0.35-0.55B)")
    ST.joinpath("measure_report.json").write_text(json.dumps(report, indent=1))
    print("report -> /mnt/h/sepalith/bioc_staging/measure_report.json")


if __name__ == "__main__":
    main()
