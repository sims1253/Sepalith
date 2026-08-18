#!/usr/bin/env python3
"""Select ACTIVE R repos (pushed since --active-since) for edit-pair mining.

Sources:
  1. r-universe cran-to-git mapping (local shallow clone) -> package -> github url
  2. download ranking (/mnt/h/sepalith/ranked/*.counts.txt) -> popularity
  3. GitHub GraphQL (via gh, authenticated) -> pushed_at/isFork/isArchived, batched 100/query
  4. known orgs (r-lib, tidyverse, easystats, r-dbi, tidymodels, bioconductor,
     rstudio/posit, cynkra, mlr-org, quarto-dev, rstudio/cheatsheets excluded) via REST org listing

Output: JSON list of {slug, url, package, downloads, pushed_at, default_branch, source}.
Resumable: caches GraphQL lookups in pushed_cache.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

C2G = Path("/mnt/h/sepalith/meta/cran-to-git")
RANKED = Path("/mnt/h/sepalith/ranked/2026-08-15.counts.txt")
OUT = Path("/mnt/h/sepalith/meta/selected_repos.json")
CACHE = Path("/mnt/h/sepalith/meta/pushed_cache.json")
ACTIVE_SINCE = "2026-05-01T00:00:00Z"
MAX_REPOS = 420          # clone target (some will fail / be unusable)
CHECK_TOP = 2600         # check activity for this many top-ranked mapped packages
PER_OWNER_CAP = 15
KNOWN_ORGS = ["r-lib", "tidyverse", "easystats", "r-dbi", "tidymodels", "bioconductor",
              "posit-dev", "cynkra", "mlr-org", "quarto-dev", "insightsengineering",
              "oxford-pharmacoepi", "dieterich-lab"]


def load_mapping():
    pkg2url = {}
    n_skipped = 0
    for f in C2G.glob("*.json"):
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            url = (r.get("url") or "").strip()
            pkg = (r.get("package") or "").strip()
            if not url or not pkg:
                continue
            if not url.startswith("https://github.com/"):
                n_skipped += 1
                continue
            parts = url[len("https://github.com/"):].strip("/").split("/")
            if len(parts) < 2 or any(p.lower() in ("issues", "pull", "tree", "blob", "releases")
                                    for p in parts[2:4]):
                n_skipped += 1
                continue
            slug = f"{parts[0]}/{parts[1]}"
            if parts[0].lower() == "cran":   # cran tarball-import mirrors: version bumps only
                n_skipped += 1
                continue
            if pkg not in pkg2url:           # first universe wins; fine for ranking purposes
                pkg2url[pkg] = slug
    print(f"mapping: {len(pkg2url)} packages, {n_skipped} non-github/invalid skipped")
    return pkg2url


def load_ranking():
    dl = {}
    for line in RANKED.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            dl[parts[1]] = int(parts[0])
    print(f"ranking: {len(dl)} packages")
    return dl


def gh_graphql_batch(slugs):
    """Look up repository metadata for <=100 slugs in one GraphQL request."""
    lines = []
    for i, s in enumerate(slugs):
        owner, name = s.split("/")
        lines.append(
            f'r{i}: repository(owner:"{owner}", name:"{name}") '
            f'{{ nameWithOwner pushedAt isArchived isFork diskUsage '
            f'defaultBranchRef {{ name }} }}')
    query = "query {" + " ".join(lines) + " }"
    for attempt in range(5):
        p = subprocess.run(["gh", "api", "graphql", "-f", f"query={query}"],
                           capture_output=True, text=True)
        if p.returncode == 0:
            try:
                return json.loads(p.stdout)["data"]
            except Exception:
                pass
        sys.stderr.write(f"graphql retry {attempt}: {p.stderr[:200]}\n")
        time.sleep(10 * (attempt + 1))
    return {}


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(c):
    CACHE.write_text(json.dumps(c))


def check_slugs(slugs, cache):
    todo = [s for s in slugs if s not in cache]
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        data = gh_graphql_batch(batch)
        for j, s in enumerate(batch):
            d = data.get(f"r{j}")
            if d is None:
                cache[s] = {"missing": True}
            else:
                cache[s] = {
                    "pushed_at": d.get("pushedAt"),
                    "archived": d.get("isArchived"),
                    "fork": d.get("isFork"),
                    "disk_kb": d.get("diskUsage") or 0,
                    "default_branch": (d.get("defaultBranchRef") or {}).get("name"),
                }
        if i % 500 == 0 and i:
            save_cache(cache)
            print(f"  checked {i}/{len(todo)}", flush=True)
    save_cache(cache)


def org_repos(org, cache):
    out = []
    for page in (1, 2, 3):
        p = subprocess.run(
            ["gh", "api", f"orgs/{org}/repos?sort=pushed&direction=desc&per_page=100&page={page}"],
            capture_output=True, text=True)
        if p.returncode != 0:
            break
        rows = json.loads(p.stdout)
        if not rows:
            break
        for r in rows:
            slug = r["full_name"]
            cache[slug] = {
                "pushed_at": r.get("pushed_at"),
                "archived": r.get("archived"),
                "fork": r.get("fork"),
                "disk_kb": r.get("disk_size") or 0,
                "default_branch": r.get("default_branch"),
                "lang": r.get("language"),
            }
            out.append(slug)
        if len(rows) < 100:
            break
    return out


def main():
    pkg2slug = load_mapping()
    dl = load_ranking()
    cache = load_cache()

    ranked_pkgs = sorted(dl, key=lambda p: -dl[p])
    mapped = [(p, pkg2slug[p]) for p in ranked_pkgs if p in pkg2slug]
    print(f"mapped+ranked: {len(mapped)}")

    slugs_to_check = []
    seen = set()
    for pkg, slug in mapped[:CHECK_TOP]:
        if slug not in seen:
            seen.add(slug)
            slugs_to_check.append((pkg, slug))

    org_slugs = set()
    for org in KNOWN_ORGS:
        got = org_repos(org, cache)
        org_slugs.update(got)
        print(f"org {org}: {len(got)} repos listed", flush=True)
    save_cache(cache)

    print(f"checking activity for {len(slugs_to_check)} ranked slugs + {len(org_slugs)} org slugs")
    check_slugs([s for _, s in slugs_to_check] + sorted(org_slugs), cache)

    def active(slug):
        c = cache.get(slug)
        return (c and not c.get("missing") and not c.get("archived") and not c.get("fork")
                and c.get("pushed_at") and c["pushed_at"] >= ACTIVE_SINCE
                and c.get("disk_kb", 0) < 2_000_000)  # skip gigamonsters

    selected, per_owner = [], {}
    # pass 1: ranked packages (popularity-first)
    for pkg, slug in slugs_to_check:
        if len(selected) >= MAX_REPOS - 60:
            break
        owner = slug.split("/")[0]
        if per_owner.get(owner, 0) >= PER_OWNER_CAP:
            continue
        if active(slug):
            c = cache[slug]
            selected.append(dict(slug=slug, url=f"https://github.com/{slug}", package=pkg,
                                 downloads=dl[pkg], pushed_at=c["pushed_at"],
                                 default_branch=c.get("default_branch"), source="ranked"))
            per_owner[owner] = per_owner.get(owner, 0) + 1
    print(f"after ranked pass: {len(selected)}")

    # pass 2: org repos (diversity + bioc mirrors), fill remaining slots
    for slug in sorted(org_slugs):
        if len(selected) >= MAX_REPOS:
            break
        if any(s["slug"] == slug for s in selected):
            continue
        owner = slug.split("/")[0]
        if per_owner.get(owner, 0) >= PER_OWNER_CAP:
            continue
        if active(slug):
            c = cache[slug]
            selected.append(dict(slug=slug, url=f"https://github.com/{slug}", package=None,
                                 downloads=0, pushed_at=c["pushed_at"],
                                 default_branch=c.get("default_branch"),
                                 source=f"org:{owner}"))
            per_owner[owner] = per_owner.get(owner, 0) + 1

    selected.sort(key=lambda s: -s["downloads"])
    OUT.write_text(json.dumps(selected, indent=1))
    owners = {}
    for s in selected:
        owners[s["slug"].split("/")[0]] = owners.get(s["slug"].split("/")[0], 0) + 1
    print(json.dumps({"selected": len(selected), "owners": len(owners),
                      "top_owners": sorted(owners.items(), key=lambda x: -x[1])[:10],
                      "out": str(OUT)}, indent=1))


if __name__ == "__main__":
    main()
