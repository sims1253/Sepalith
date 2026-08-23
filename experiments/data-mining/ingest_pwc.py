#!/usr/bin/env python3
"""pwc-archive acquisition — clone/filter/stage the permissively-licensed
R research code linked from the Papers-with-Code archive mirror.

Input:  /tmp/pwc/candidates.jsonl            (pwc_prefilter.py)
        /mnt/h/sepalith/pwc_staging/meta/github_meta.jsonl
Selection rule: GitHub primary language == R (measured via GraphQL; replaces
the scout's ~3.8k heuristic with ground truth — the candidate superset was
the scout's ~7.5k stat-title upper bound + repo-name R markers + cran mirrors).

Per repo (polite: 2-3s gap between repos, GIT_TERMINAL_PROMPT=0):
  git clone --depth 1 --filter=blob:none --no-checkout
  git ls-tree -r HEAD -> select R-relevant paths (ext match case-insensitive):
      .R .Rmd .Rmarkdown .qmd .Rd .Rnw .Rhtml anywhere
      DESCRIPTION/NAMESPACE/LICENSE*/COPYING*/etc (depth<=3)
      C/C++/Fortran under src/ only (CRAN-store shape keeps src/)
      skip renv/library, packrat/libs, .Rcheck, revdep, node_modules
  git show / checkout HEAD -- <paths>  (lazy blob fetch = the real download)
  license decision per the 2026-08-20 ruling (NO NC, NO ND, permissive-only,
  ambiguous=excluded): NC/ND text scan overrides everything; then GitHub spdx
  map; then LICENSE-file text markers; else excluded.
  accepted -> tar-pipe ship to /mnt/h/sepalith/pwc_staging/repos/<owner>__<repo>/
  (local scratch + one tar pipe; /mnt/h per-file round trips are the dominant
   cost — same trick as ingest_bioc.py)

Ledger: every repo gets a line (accepted / license-excluded / no-R-files /
gone / clone-failed / error). Resume: provenance/<owner>__<repo>.json OR a
terminal ledger row (accepted/excluded/no-R-files/gone) marks done; error and
clone-failed rows retry. Inventory shards mirror the bioc staging shape
(source="pwc", normalized=false — air/jarl normalization deferred to merge).
"""
import hashlib, json, os, random, re, shlex, shutil, subprocess, time
from pathlib import Path

ROOT = Path("/mnt/h/sepalith")
STAGING = ROOT / "pwc_staging"
REPOS = STAGING / "repos"
LEDGER = STAGING / "pwc_license_ledger.jsonl"
MANIFEST = STAGING / "manifest.jsonl"
PROV = STAGING / "provenance"
SHARDS = STAGING / "datasets" / "packages"
LOGS = STAGING / "logs"
CAND_F = Path("/tmp/pwc/candidates.jsonl")
GH_META = STAGING / "meta" / "github_meta.jsonl"
WORK = Path("/tmp/pwc/work")

R_EXT_LOW = {".r", ".rmd", ".rmarkdown", ".qmd", ".rd", ".rnw", ".rhtml"}
META_NAMES = {"description", "namespace"}
LIC_BASES = ("license", "licence", "copying", "copyright", "notice")
SRC_EXT_LOW = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".f",
               ".f90", ".f77"}
SKIP_SUB = ("renv/library", "renv/cache", "packrat/libs", "packrat/src",
            ".rcheck/", "/revdep/", "node_modules/", ".git/")
MAX_FILES = 2000
MAX_FILE_B = 2 * 1024 * 1024
PROGRESS_EVERY = 25
TERMINAL = ("accepted", "exclude", "no-R-files", "gone")

# --- license ruling: NO NC, NO ND; permissive-only; ambiguous = excluded ---
_NC_ND_FILE = re.compile(
    r"creative\s+commons\s+attribution[^\n]{0,60}(noncommercial|noderivs)"
    r"|\bcc[-\s]?by[-\s]?(nc|nd)\b"
    r"|creativecommons\.org/licenses/[^\s/'\"]*-(nc|nd)"
    r"|\bnoderivs\b"
    r"|non[-\s]?commercial\s+(use|purposes)\s+only", re.IGNORECASE)
SPDX_MAP = {
    "MIT": "MIT", "X11": "MIT", "ISC": "ISC",
    "Apache-2.0": "Apache", "BSD-2-Clause": "BSD", "BSD-3-Clause": "BSD",
    "BSD-3-Clause-Clear": "BSD", "0BSD": "BSD", "Unlicense": "Unlicense",
    "Zlib": "Zlib", "BSL-1.0": "Boost",
    "GPL-2.0": "GPL", "GPL-3.0": "GPL", "GPL-2.0-only": "GPL",
    "GPL-3.0-only": "GPL", "GPL-2.0-or-later": "GPL",
    "GPL-3.0-or-later": "GPL",
    "LGPL-2.0": "LGPL", "LGPL-2.1": "LGPL", "LGPL-3.0": "LGPL",
    "LGPL-2.1-only": "LGPL", "LGPL-3.0-only": "LGPL",
    "LGPL-2.1-or-later": "LGPL",
    "AGPL-3.0": "AGPL", "AGPL-3.0-only": "AGPL",
    "Artistic-2.0": "Artistic-2.0", "Artistic-1.0": "Artistic-1.0",
    "CC0-1.0": "CC0", "CC-BY-4.0": "CC-BY", "CC-BY-3.0": "CC-BY",
    "MPL-2.0": "MPL-2.0", "MPL-1.1": "MPL-2.0",
    "EPL-1.0": "EPL", "EPL-2.0": "EPL", "CPL-1.0": "CPL",
    "CeCILL-1.1": "CeCILL", "CeCILL-2.0": "CeCILL", "CeCILL-2.1": "CeCILL",
    "CeCILL-B": "CeCILL", "CeCILL-C": "CeCILL",
}
_FLAGGED = {"AGPL": "agpl-redistribution-sensitive", "MPL-2.0":
            "osi-weak-copyleft", "EPL": "osi-weak-copyleft", "CPL":
            "osi-weak-copyleft", "CeCILL": "cecill-family-gpl-compatible"}
_TEXT_MARKERS = [
    ("MIT", "permission is hereby granted, free of charge"),
    ("Apache", "apache license"),
    ("LGPL", "gnu lesser general public license"),
    ("GPL", "gnu general public license"),
    ("AGPL", "gnu affero general public license"),
    ("BSD", "redistribution and use in source and binary forms"),
    ("CC0", "creative commons zero"), ("CC0", "cc0 1.0 universal"),
    ("MPL-2.0", "mozilla public license"),
    ("Unlicense", "free and unencumbered software released into the public domain"),
    ("Zlib", "zlib license"), ("ISC", "isc license"),
    ("CeCILL", "cecill"), ("Artistic-2.0", "artistic license"),
    ("CC-BY", "creative commons attribution"),
]


def classify(spdx, lic_text, desc_license=None):
    """Ruling order: NC/ND text beats everything; then spdx; then the
    DESCRIPTION License field (usethis stub LICENSE files carry no text —
    the declaration is 'MIT + file LICENSE' in DESCRIPTION); then text."""
    if lic_text and _NC_ND_FILE.search(lic_text):
        return "exclude", "NC/ND", "", "nc-or-nd-terms-in-license-file"
    if spdx and spdx.startswith(("CC-BY-NC", "CC-BY-ND")):
        return "exclude", "NC/ND", "", f"spdx:{spdx}"
    if spdx and spdx in SPDX_MAP:
        cls = SPDX_MAP[spdx]
        return "include", cls, _FLAGGED.get(cls, ""), f"spdx:{spdx}"
    if desc_license:
        d, cls, flag, reason = classify_field(desc_license)
        if d == "include":
            return d, cls, flag, f"description-field:{desc_license[:40]}"
        if not lic_text:
            return d, cls, flag, f"description-field:{reason}"
    if lic_text:
        t = lic_text.lower()
        for cls, m in _TEXT_MARKERS:
            if m in t:
                return "include", cls, _FLAGGED.get(cls, ""), \
                    "classified-from-license-file"
        return "exclude", "unclassified", "", "license-file-unclassifiable"
    return "exclude", "unclassified", "", \
        f"no-license-info(spdx={spdx or 'none'},no-license-file," \
        f"desc={desc_license or 'none'})"


_FIELD_PERMISSIVE = [
    ("MIT", lambda s: s.startswith("mit")),
    ("BSD", lambda s: "bsd" in s),
    ("Apache", lambda s: "apache" in s),
    ("Artistic-2.0", lambda s: s.startswith("artistic")),
    ("CC0", lambda s: s.startswith("cc0")),
    ("CC-BY", lambda s: s.startswith("cc by") or s.startswith("cc-by")
     or s.startswith("creative commons attribution")),
    ("LGPL", lambda s: s.startswith("lgpl") or s.startswith("lesser gnu")),
    ("GPL", lambda s: s.startswith("gpl") or s.startswith("gnu general")),
    ("AGPL", lambda s: s.startswith("agpl")),
    ("CeCILL", lambda s: s.startswith("cecill")),
    ("EPL", lambda s: s.startswith("epl")),
    ("CPL", lambda s: s.startswith("cpl")),
    ("MPL-2.0", lambda s: s.startswith("mozilla") or s.startswith("mpl")),
    ("Unlicense", lambda s: "unlicense" in s or "public domain" in s),
]


def classify_field(field):
    """DESCRIPTION 'License:' field classification (from ingest_bioc.py).
    -> (decision, class, flag, reason)."""
    field = (field or "").strip()
    if not field or field.lower() in ("unknown", "what license?",
                                      "what license is it?"):
        return "exclude", "unclassified", "", "empty-or-unknown-field"
    classes = []
    for alt in field.split("|"):
        a = alt.strip().lower()
        if any(m in a for m in ("noncommercial", "noderiv", "no-deriv",
                                "cc by-nc", "by-nd", "| nc")):
            return "exclude", "NC/ND", "", f"nc-nd-alternative:{alt.strip()[:30]}"
        for cls, test in _FIELD_PERMISSIVE:
            if test(a):
                classes.append(cls)
                break
    if classes:
        best = next((c for c, _ in _FIELD_PERMISSIVE for x in classes if x == c),
                    classes[0])
        return "include", best, _FLAGGED.get(best, ""), ""
    return "exclude", "unclassified", "", f"unrecognized-field:{field[:40]}"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def run(cmd, timeout):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    return subprocess.run(cmd, capture_output=True, timeout=timeout,
                          env=env, encoding="utf-8", errors="replace")


def ext_of(name):
    return ("." + name.rsplit(".", 1)[1].lower()) if "." in name else ""


def select_paths(tree_lines):
    """-> (r_files, meta_files, lic_file_or_None, src_files)"""
    r_files, meta_files, lic_files, src_files = [], [], [], []
    for p in tree_lines:
        p = p.strip()
        if not p or p.startswith(".git/") or p.endswith("/"):
            continue
        low = p.lower()
        if any(s in low for s in SKIP_SUB):
            continue
        parts = p.split("/")
        name, depth = parts[-1], len(parts)
        stem = name.rsplit(".", 1)[0].lower()
        ext = ext_of(name)
        if ext in R_EXT_LOW:
            r_files.append(p)
        elif stem in LIC_BASES and depth <= 3:
            lic_files.append(p)
        elif stem in META_NAMES and depth <= 3:
            meta_files.append(p)
        elif low.startswith("src/") and ext in SRC_EXT_LOW:
            src_files.append(p)
    lic_files.sort(key=lambda p: next(
        (i for i, b in enumerate(LIC_BASES)
         if p.rsplit("/", 1)[-1].lower().rsplit(".", 1)[0] == b), 99))
    return r_files, meta_files, (lic_files[0] if lic_files else None), \
        src_files


def ledger(rec):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def process(repo, cand, meta):
    owner, name = repo.split("/", 1)
    key = f"{owner}__{name}"
    wd = WORK / key
    for d in (REPOS, PROV, SHARDS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    base = dict(repo=repo, owner=owner, name=name, ts=now(),
                stars=meta.get("stars"), fork=meta.get("fork"),
                spdx=meta.get("spdx"), branch=meta.get("branch"))
    try:
        if wd.exists():
            shutil.rmtree(wd, ignore_errors=True)
        wd.mkdir(parents=True, exist_ok=True)
        clone_err = None
        for attempt in (1, 2):
            time.sleep(random.uniform(2.0, 3.0))
            p = run(["git", "clone", "--quiet", "--depth", "1",
                     "--filter=blob:none", "--no-checkout",
                     f"https://github.com/{repo}.git", str(wd)], 300)
            if p.returncode == 0:
                break
            clone_err = (p.stderr or "").strip()[-300:]
            time.sleep(120 if ("429" in clone_err or
                               "rate" in clone_err.lower()) else 30)
        else:
            gone = ("not found" in clone_err.lower()
                    or "does not exist" in clone_err.lower())
            ledger({**base, "decision": "gone" if gone else "clone-failed",
                    "reason": clone_err[:150]})
            return {"repo": repo, "status":
                    f"{'gone' if gone else 'clone-failed'}"}

        p = run(["git", "-C", str(wd), "ls-tree", "-r", "--name-only",
                 "HEAD"], 120)
        tree = [l for l in p.stdout.splitlines() if l.strip()]
        p = run(["git", "-C", str(wd), "rev-parse", "HEAD"], 30)
        sha_head = p.stdout.strip()[:12]

        r_files, meta_files, lic_file, src_files = select_paths(tree)
        lic_text = None
        if lic_file:
            p = run(["git", "-C", str(wd), "show", f"HEAD:{lic_file}"], 120)
            if p.returncode == 0 and len(p.stdout) < 1_000_000:
                lic_text = p.stdout
        # DESCRIPTION License field (usethis stub LICENSE files carry the
        # declaration here, not in the LICENSE text)
        desc_license = None
        desc_p = next((m for m in meta_files
                       if m.rsplit("/", 1)[-1].upper() == "DESCRIPTION"), None)
        if desc_p:
            p = run(["git", "-C", str(wd), "show", f"HEAD:{desc_p}"], 120)
            if p.returncode == 0 and len(p.stdout) < 500_000:
                fields, lastk = {}, None
                for line in p.stdout.splitlines():
                    if line.startswith((" ", "\t")) and lastk:
                        fields[lastk] += " " + line.strip()
                    elif ":" in line:
                        k, _, v = line.partition(":")
                        lastk = k.strip()
                        fields[lastk] = v.strip()
                desc_license = fields.get("License")
        if not r_files:
            ledger({**base, "decision": "no-R-files",
                    "reason": "lang R but no R-ext files in tree",
                    "n_tree_files": len(tree), "head": sha_head})
            (PROV / f"{key}.json").write_text(json.dumps(
                dict(repo=repo, included=False, reason="no-R-files",
                     head=sha_head, tree_files=len(tree), ts=now()),
                indent=1))
            return {"repo": repo, "status": "no-R-files"}

        decision, lclass, flag, reason = classify(
            meta.get("spdx"), lic_text, desc_license)
        ledger({**base, "decision": "accepted" if decision == "include"
                else "exclude", "included": decision == "include",
                "license_class": lclass, "license_reason": reason,
                "flag": flag, "license_file": lic_file or "",
                "head": sha_head, "n_R_files": len(r_files)})
        if decision != "include":
            (PROV / f"{key}.json").write_text(json.dumps(
                dict(repo=repo, included=False, reason=reason,
                     license_class=lclass, spdx=meta.get("spdx"),
                     head=sha_head, ts=now()), indent=1))
            return {"repo": repo, "status": f"excluded ({reason})"}

        want = (r_files + meta_files + ([lic_file] if lic_file else [])
                + src_files)[:MAX_FILES]
        psf = wd / ".paths"
        psf.write_text("\n".join(want) + "\n")
        p = run(["git", "-C", str(wd), "checkout", "HEAD",
                 "--pathspec-from-file", str(psf)], 600)
        if p.returncode != 0:
            raise RuntimeError(f"checkout failed: {p.stderr[:200]}")

        stage = wd / "_stage"
        stage.mkdir()
        records = []
        for rel in want:
            f = wd / rel
            if not f.is_file() or f.stat().st_size > MAX_FILE_B:
                continue
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            body = f.read_bytes()
            records.append(dict(
                source="pwc", package=key, version=sha_head,
                license=lclass, upstream=f"https://github.com/{repo}",
                author=owner, path=rel,
                area=rel.split("/")[0] if "/" in rel else ".",
                ext=ext_of(rel), sha256=hashlib.sha256(body).hexdigest(),
                bytes=len(body), n_lines=body.count(b"\n"), normalized=False))

        dest_dir = REPOS / key
        if dest_dir.exists():
            subprocess.run(["rm", "-rf", str(dest_dir)], check=True)
        dest_dir.mkdir(parents=True, exist_ok=True)
        ship = subprocess.run(
            ["bash", "-c",
             f"tar -cf - -C {shlex.quote(str(stage))} . "
             f"| tar -xf - -C {shlex.quote(str(dest_dir))}"], timeout=1800)
        if ship.returncode != 0:
            raise RuntimeError(f"ship failed ({ship.returncode})")

        SHARDS.joinpath(f"{key}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records))
        prov = dict(
            repo=repo, source="pwc", included=True, head=sha_head,
            license=lclass, license_spdx=meta.get("spdx"), flag=flag,
            license_file=lic_file or "", stars=meta.get("stars"),
            fork=meta.get("fork"), archived=meta.get("archived"),
            pushed_at=meta.get("pushed_at"),
            paper_titles=cand.get("paper_titles", []),
            arxiv_ids=cand.get("arxiv_ids", []),
            n_links=cand.get("n_links"), official_links=
            cand.get("official_links"), match_reasons=cand.get("reasons", []),
            n_files=len(records), n_bytes=sum(r["bytes"] for r in records),
            has_DESCRIPTION=any(r["path"].rsplit("/", 1)[-1].lower()
                                == "description" for r in records),
            normalized=False, ingested_at=now())
        (PROV / f"{key}.json").write_text(json.dumps(prov, indent=1))
        with open(MANIFEST, "a") as mf:
            mf.write(json.dumps(dict(
                repo=repo, package=key, version=sha_head, license=lclass,
                license_class=lclass, source="pwc", n_files=len(records),
                n_bytes=sum(r["bytes"] for r in records),
                stars=meta.get("stars"), fork=meta.get("fork"),
                head=sha_head, official_links=cand.get("official_links"),
                n_links=cand.get("n_links"),
                ingested_at=now())) + "\n")
        return {"repo": repo, "status": "ok", "files": len(records),
                "kb": round(sum(r["bytes"] for r in records) / 1e3, 1),
                "license": lclass}
    except Exception as e:
        ledger({**base, "decision": "error", "reason": str(e)[:200]})
        return {"repo": repo, "status": f"error: {str(e)[:120]}"}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main():
    for d in (REPOS, PROV, SHARDS, LOGS, STAGING / "meta"):
        d.mkdir(parents=True, exist_ok=True)
    cands = {c["repo"]: c for c in map(json.loads, open(CAND_F))}
    meta_by_repo, last_ledger = {}, {}
    for l in open(GH_META):
        m = json.loads(l)
        meta_by_repo[m["repo"]] = m
    if LEDGER.exists():
        for l in open(LEDGER):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue  # torn concurrent append
            last_ledger[r["repo"]] = r
    r_lang = sorted(r for r, m in meta_by_repo.items()
                    if m.get("status") == "ok" and m.get("lang") == "R")
    done = {json.loads(p.read_text()).get("repo")
            for p in PROV.glob("*.json")}
    done |= {r for r, v in last_ledger.items()
             if v.get("decision") in TERMINAL}
    todo = [r for r in r_lang if r not in done]
    print(f"candidates={len(cands)} meta={len(meta_by_repo)} "
          f"lang-R={len(r_lang)} done={len(done)} todo={len(todo)}",
          flush=True)

    statef = LOGS / "state.json"
    counts, t0 = {}, time.time()
    for i, repo in enumerate(todo, 1):
        r = process(repo, cands.get(repo, {}), meta_by_repo.get(repo, {}))
        st = r["status"].split(":")[0].split(" ")[0]
        counts[st] = counts.get(st, 0) + 1
        statef.write_text(json.dumps(
            dict(updated=now(), processed=i, total_todo=len(todo),
                 lang_r=len(r_lang), counts=counts, last_repo=repo,
                 elapsed_s=round(time.time() - t0))))
        if i % 5 == 0 or i <= 10:
            print(json.dumps(r), flush=True)
        if i % PROGRESS_EVERY == 0:
            eta = (time.time() - t0) / i * (len(todo) - i)
            line = (f"{now()} {i}/{len(todo)} {json.dumps(counts)} "
                    f"eta_h={eta/3600:.2f}")
            with open(LOGS / "progress.log", "a") as f:
                f.write(line + "\n")
            print(f"PROGRESS {line}", flush=True)
    print(f"RUN COMPLETE {json.dumps(counts)} total_s={time.time()-t0:.0f}",
          flush=True)


if __name__ == "__main__":
    main()
