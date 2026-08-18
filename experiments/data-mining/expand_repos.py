#!/usr/bin/env python3
"""Expansion selection for scale: (a) all active-but-unselected slugs already in
pushed_cache.json, prioritized by recency of push; (b) GitHub search API
language:R pushed:>... (search results carry full repo metadata, no extra lookups).
Excludes squash-mirror 'cran/' owner, forks, archived, huge repos, already-cloned dirs.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

CACHE = Path("/mnt/h/sepalith/meta/pushed_cache.json")
SEL1 = Path("/mnt/h/sepalith/meta/selected_repos.json")
OUT = Path("/mnt/h/sepalith/meta/selected_repos_v2.json")
GITDIR = Path("/mnt/h/sepalith/git")
QUERIES = [
    ("language:R pushed:>2026-07-15 stars:>3", 10),
    ("language:R pushed:>2026-08-01", 5),
]
SLEEP_BETWEEN_PAGES = 3.5  # search rate limit politeness (authenticated: 30/min)


def gh_search(query, pages):
    out = []
    for page in range(1, pages + 1):
        for attempt in range(4):
            p = subprocess.run(
                ["gh", "api", "--method", "GET", "search/repositories",
                 "-f", f"q={query}", "-f", "sort=updated", "-f", "order=desc",
                 "-f", f"per_page=100", "-f", f"page={page}"],
                capture_output=True, text=True)
            if p.returncode == 0:
                data = json.loads(p.stdout)
                out.extend(data.get("items", []))
                break
            sys.stderr.write(f"search retry {attempt} p{page}: {p.stderr[:160]}\n")
            time.sleep(15 * (attempt + 1))
        else:
            break
        if not json.loads(p.stdout).get("items"):
            break
        print(f"  '{query}' page {page}: +{len(out)} cumulative", flush=True)
        time.sleep(SLEEP_BETWEEN_PAGES)
    return out


def usable(meta):
    return (meta and not meta.get("missing") and not meta.get("archived")
            and not meta.get("fork") and meta.get("pushed_at")
            and meta["pushed_at"] >= "2026-05-01"
            and (meta.get("disk_kb") or 0) < 2_000_000
            and meta.get("default_branch"))


def main():
    cache = json.loads(CACHE.read_text())
    already = {s["slug"] for s in json.loads(SEL1.read_text())}
    cloned = {p.name.replace("__", "/") for p in GITDIR.iterdir() if p.is_dir()}

    rows = []
    # (a) cache expansion, most recently pushed first
    for slug, c in sorted(cache.items(), key=lambda kv: kv[1].get("pushed_at") or "",
                          reverse=True):
        if slug in already or slug in cloned or slug.split("/")[0].lower() == "cran":
            continue
        if usable(c):
            rows.append(dict(slug=slug, url=f"https://github.com/{slug}",
                             pushed_at=c["pushed_at"], source="cache",
                             default_branch=c.get("default_branch")))
    print(f"cache expansion: {len(rows)}")

    # (b) search expansion
    seen = {r["slug"] for r in rows} | already | cloned
    n_search = 0
    for q, pages in QUERIES:
        for it in gh_search(q, pages):
            slug = it["full_name"]
            meta = dict(pushed_at=it.get("pushed_at"), archived=it.get("archived"),
                        fork=it.get("fork"), disk_kb=it.get("size") or 0,
                        default_branch=it.get("default_branch"))
            if slug in seen or not usable(meta) or (it.get("language") != "R"):
                continue
            seen.add(slug)
            rows.append(dict(slug=slug, url=it["html_url"], pushed_at=meta["pushed_at"],
                             source="search", default_branch=meta["default_branch"]))
            n_search += 1
    print(f"search expansion: {n_search}")

    rows.sort(key=lambda r: r["pushed_at"], reverse=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(json.dumps({"total": len(rows), "out": str(OUT)}, indent=1))


if __name__ == "__main__":
    main()
