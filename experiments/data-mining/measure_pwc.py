#!/usr/bin/env python3
"""pwc-archive measurement — run after ingest_pwc.py.

Produces /mnt/h/sepalith/pwc_staging/measure_report.json:
  1. funnel + ledger summary (candidates -> lang-R -> accepted/excluded,
     license histogram, exclusion reasons) per the no-NC/permissive ruling
  2. token estimate: 300-file sample, qwen tokenizer (same one as
     measure_bioc.py), two extrapolations (per-file mean x total files;
     per-repo mean x total repos) + byte-based sanity
  3. near-dup overlap vs CRAN + Bioc stores (100-file sample):
     exact sha256, air-normalized sha256 (CRAN/Bioc shards are
     post-normalization, so raw pwc files need the same treatment to
     compare), and repo-name/CRAN-package overlap
  4. distinctive-value composition: what pwc adds that CRAN lacks
     (paper-linked methods implementations, simulations, analysis scripts)

Zero GPU. Reads staging + the CRAN/Bioc inventory shards.
"""
import hashlib, json, random, re, shutil, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/mnt/h/sepalith")
ST = ROOT / "pwc_staging"
LEDGER = ST / "pwc_license_ledger.jsonl"
MANIFEST = ST / "manifest.jsonl"
PROV = ST / "provenance"
SHARDS = ST / "datasets" / "packages"
REPOS = ST / "repos"
CRAN_SHARDS = ROOT / "datasets" / "packages"
BIOC_SHARDS = ROOT / "bioc_staging" / "datasets" / "packages"
CRAN_STORE = ROOT / "normalized"
TOKENIZER = Path("/home/m0hawk/Documents/Sepalith/experiments/models/"
                 "qwen3.5-2b-base-text-hf/tokenizer.json")
R_EXTS = (".r", ".rmd", ".rmarkdown", ".qmd", ".rd", ".rnw", ".rhtml")
SHA_CACHE = Path("/tmp/pwc/cranbioc_shas.txt")
SAMPLE_SEED = 1234


def load_ledger():
    last = {}
    for l in open(LEDGER):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue  # concurrent-append race on the last line
        last[r["repo"]] = r
    return last


def build_sha_index():
    if SHA_CACHE.exists():
        return set(SHA_CACHE.read_text().split())
    shas = set()
    for d in (CRAN_SHARDS, BIOC_SHARDS):
        shards = sorted(d.glob("*.jsonl"))
        for i, sh in enumerate(shards, 1):
            for l in open(sh):
                try:
                    shas.add(json.loads(l)["sha256"])
                except Exception:
                    pass
            if i % 2000 == 0:
                print(f"  sha index: {i}/{len(shards)} {d.name}", flush=True)
    SHA_CACHE.write_text("\n".join(sorted(shas)))
    return shas


def manifest_stats():
    rows = [json.loads(l) for l in open(MANIFEST)]
    return rows


def staged_files(rows):
    """(repo_key, rel_path, abs_path) for every R-ext file, from shards."""
    out = []
    for sh in sorted(SHARDS.glob("*.jsonl")):
        key = sh.stem
        for l in open(sh):
            r = json.loads(l)
            if r["ext"] in R_EXTS:
                p = REPOS / key / r["path"]
                if p.is_file():
                    out.append((key, r["path"], p))
    return out


def token_estimate(files, n_repos):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    random.seed(SAMPLE_SEED)
    sample = random.sample(files, min(300, len(files)))
    enc = tok.encode_batch([f[2].read_text(errors="replace")
                            for f in sample], add_special_tokens=False)
    toks = [len(e.ids) for e in enc]
    total_files = len(files)
    est_file = sum(toks) / len(toks) * total_files
    # per-repo robust alternative: whole-repo tokenization of ~40 repos
    by_repo = defaultdict(list)
    for k, rel, p in files:
        by_repo[k].append(p)
    repo_keys = sorted(by_repo)
    random.shuffle(repo_keys)
    chosen = repo_keys[:40]
    repo_toks = []
    for rk in chosen:
        enc = tok.encode_batch([p.read_text(errors="replace")
                                for p in by_repo[rk]], add_special_tokens=False)
        repo_toks.append(sum(len(e.ids) for e in enc))
    est_repo = sum(repo_toks) / len(repo_toks) * n_repos
    srt = sorted(toks)
    return dict(
        sample_n=len(toks), mean_per_file=round(sum(toks) / len(toks), 1),
        median_per_file=srt[len(srt) // 2], min=srt[0], max=srt[-1],
        total_staged_files=total_files,
        est_total_per_file_extrap=round(est_file),
        per_repo_sample_n=len(repo_toks),
        est_total_per_repo_extrap=round(est_repo),
        tokenizer=str(TOKENIZER))


def overlap_check(files, shas, rows):
    random.seed(SAMPLE_SEED + 1)
    sample = random.sample(files, min(100, len(files)))
    exact_hits = []
    for k, rel, p in sample:
        if hashlib.sha256(p.read_bytes()).hexdigest() in shas:
            exact_hits.append(f"{k}/{rel}")
    # air-normalized comparison (CRAN/Bioc shas are post-air-format)
    tmp = Path("/tmp/pwc/aircmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    norm_hits = []
    for i, (k, rel, p) in enumerate(sample):
        dst = tmp / f"f{i}_{p.name}"
        shutil.copyfile(p, dst)
    air = subprocess.run(["air", "format", str(tmp)], capture_output=True,
                         text=True, timeout=600)
    jarl = subprocess.run(["jarl", "check", "--fix", "--allow-no-vcs",
                           str(tmp)], capture_output=True, text=True,
                          timeout=600)
    if air.returncode == 0:
        for i, (k, rel, p) in enumerate(sample):
            f = tmp / f"f{i}_{p.name}"
            if f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest() \
                    in shas:
                norm_hits.append(f"{k}/{rel}")
    else:
        norm_hits = [f"air-failed: {air.stderr[:100]}"]
    shutil.rmtree(tmp, ignore_errors=True)
    # repo-name overlap with CRAN package store
    cran_pkgs = {p.name.lower() for p in CRAN_STORE.iterdir() if p.is_dir()}
    name_hits = [r["repo"] for r in rows
                 if r["repo"].split("/")[-1].lower() in cran_pkgs
                 or r["repo"].split("/")[0].lower() == "cran"]
    return dict(
        sample_n=len(sample), exact_dup_hits=len(exact_hits),
        exact_dup_examples=exact_hits[:10],
        air_normalized_dup_hits=len(norm_hits)
        if not (norm_hits and isinstance(norm_hits[0], str)
                and norm_hits[0].startswith("air-failed")) else norm_hits,
        air_normalized_dup_examples=norm_hits[:10],
        cran_name_overlap_repos=len(name_hits),
        cran_name_overlap_examples=name_hits[:15])


def composition(files, rows):
    provs = {p.stem: json.loads(p.read_text()) for p in PROV.glob("*.json")}
    exts = Counter(rel.rsplit("/", 1)[-1].rsplit(".", 1)[-1].lower()
                   for _, rel, _ in files)
    areas = Counter(rel.split("/")[0].lower() if "/" in rel else "(root)"
                    for _, rel, _ in files)
    inc = [p for p in provs.values() if p.get("included")]
    pkg_shaped = sum(1 for p in inc if p.get("has_DESCRIPTION"))
    sim_kw = re.compile(r"sim|replicat|monte|synthetic|bootstrap", re.I)
    sim_files = sum(1 for _, rel, _ in files if sim_kw.search(rel))
    stars = sorted((p.get("stars") or 0) for p in inc)
    n_papers = sum(len(p.get("arxiv_ids", [])) for p in inc)
    official = sum(1 for p in inc if (p.get("official_links") or 0) > 0)
    return dict(
        ext_hist=dict(exts.most_common()),
        area_hist=dict(areas.most_common(15)),
        repos_with_DESCRIPTION=pkg_shaped,
        repos_script_only=len(inc) - pkg_shaped,
        simulation_flavored_files=sim_files,
        paper_linked_repos=official, distinct_arxiv_papers=n_papers,
        stars_median=stars[len(stars) // 2] if stars else 0,
        stars_p90=stars[int(len(stars) * 0.9) - 1] if stars else 0)


def main():
    last = load_ledger()
    decisions = Counter(r["decision"] for r in last.values())
    included = [r for r in last.values() if r["decision"] == "accepted"]
    hist = Counter(r["license_class"] for r in included)
    reasons = Counter(r.get("license_reason") or r.get("reason", "?")[:60]
                      for r in last.values() if r["decision"] == "exclude")
    rows = manifest_stats()
    files = staged_files(rows)
    n_repos = len(rows)

    meta = {}
    gmeta = ST / "meta" / "github_meta.jsonl"
    if gmeta.exists():
        for l in open(gmeta):
            m = json.loads(l)
            meta[m["repo"]] = m
    lang_hist = Counter(str(m.get("lang")) for m in meta.values()
                        if m.get("status") == "ok")

    report = dict(
        funnel=dict(candidates=0, meta_rows=len(meta),
                    lang_hist=dict(lang_hist.most_common(12)),
                    lang_r=sum(1 for m in meta.values()
                               if m.get("lang") == "R"),
                    ledgered_repos=len(last), decisions=dict(decisions),
                    accepted=len(included)),
        license_hist_included=dict(hist.most_common()),
        exclusion_reasons=dict(reasons.most_common(20)),
        staging=dict(repos_staged=n_repos,
                     r_files_staged=len(files),
                     bytes_staged=sum(r["n_bytes"] for r in rows)),
    )
    cands = Path("/tmp/pwc/candidates.jsonl")
    if cands.exists():
        report["funnel"]["candidates"] = sum(1 for _ in open(cands))

    print(f"decisions: {dict(decisions)}")
    print(f"license hist (accepted): {hist.most_common()}")
    print(f"staged: {n_repos} repos, {len(files)} R files, "
          f"{report['staging']['bytes_staged']/1e6:.1f} MB")
    if files:
        report["tokens"] = token_estimate(files, n_repos)
        t = report["tokens"]
        print(f"tokens: per-file extrap {t['est_total_per_file_extrap']/1e6:.1f}M | "
              f"per-repo extrap {t['est_total_per_repo_extrap']/1e6:.1f}M")
        shas = build_sha_index()
        print(f"CRAN+Bioc sha index: {len(shas)} file hashes")
        report["overlap"] = overlap_check(files, shas, rows)
        o = report["overlap"]
        print(f"overlap: exact {o['exact_dup_hits']}/{o['sample_n']}, "
              f"air-normalized {o['air_normalized_dup_hits']}, "
              f"CRAN-name repos {o['cran_name_overlap_repos']}")
        report["composition"] = composition(files, rows)
        print(f"composition: {json.dumps(report['composition'])[:400]}")

    ST.joinpath("measure_report.json").write_text(json.dumps(report, indent=1))
    print("report -> /mnt/h/sepalith/pwc_staging/measure_report.json")


if __name__ == "__main__":
    main()
