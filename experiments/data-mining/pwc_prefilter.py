#!/usr/bin/env python3
"""pwc-archive prefilter — build the R-candidate repo list from the
links-between-paper-and-code parquet, then bulk-fetch GitHub repo metadata
(primary language + license) via batched GraphQL.

Scout context (docs/research/data-acquisition-scout.md 3c): the pwc-archive
links table has NO language field; the scout's heuristic R-repo estimate was
~3.8k links with a ~7.5k upper bound from stat-flavored titles. This script
reproduces the upper bound (narrow stat-title regex over paper_title ->
7.6k repos) as the candidate SUPERSET, adds repo-name R markers and
cran-owned mirrors, then replaces the guess with MEASURED primary language
from the GitHub GraphQL API (batched 90 repos/query, gh auth = 5000 pts/h,
rate-aware sleeping, 2.5s between queries).

Outputs (resume-safe, one JSON per line):
  /tmp/pwc/candidates.jsonl   repo_path, match reasons, paper linkage (n_links,
                              titles sample, arxiv ids, official flags)
  /mnt/h/sepalith/pwc_staging/meta/github_meta.jsonl
                              per-repo: lang, license spdx, default branch,
                              size, stars, fork/archived flags, status
"""
import json, os, random, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/mnt/h/sepalith")
STAGING = ROOT / "pwc_staging"
META = STAGING / "meta"
CAND_F = Path("/tmp/pwc/candidates.jsonl")
GH_META = META / "github_meta.jsonl"
PARQUET = Path("/tmp/pwc/links.parquet")
BATCH = 90
SLEEP_S = 2.5

# candidate signals (superset; the GraphQL pass turns this into ground truth)
STAT_TITLE = re.compile(
    r"\b(bayes\w*|regression|survival|longitudin\w*|causal inferen\w*|"
    r"econometric\w*|generalized linear|mixed[- ]effects|copula|"
    r"semiparametric|nonparametric regression|spatial statistics|"
    r"statistical inferen\w*|likelihood|posterior|markov chain monte|mcmc)\b",
    re.I)
NAME_R = re.compile(
    r"(^|[-_.])r$|(^|[-_])r([-_]|$)|^r[-_]|[-_]r$|rinr\b|\.r$|rpackage|rrpkg|"
    r"[-_]rinter|[-_]withr")


def build_candidates():
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    gh = df[df["repo_url"].str.startswith("https://github.com/")].copy()
    gh["path"] = (gh["repo_url"].str.replace("https://github.com/", "",
                                             regex=False).str.rstrip("/")
                  .map(lambda p: "/".join(p.split("/")[:2])))
    rows = {}
    for r in gh.itertuples(index=False):
        path = r.path
        owner, name = path.split("/", 1)
        reasons = []
        if STAT_TITLE.search(str(r.paper_title)):
            reasons.append("stat-title")
        if owner.lower() == "cran" or NAME_R.search(name.lower()):
            reasons.append("name-r")
        if not reasons:
            continue
        e = rows.setdefault(path, dict(repo=path, reasons=set(),
                                       n_links=0, official=0, in_paper=0,
                                       titles=[], arxiv=[]))
        e["reasons"] |= set(reasons)
        e["n_links"] += 1
        e["official"] += int(bool(r.is_official))
        e["in_paper"] += int(bool(r.mentioned_in_paper))
        t = str(r.paper_title)
        if t and t not in e["titles"] and len(e["titles"]) < 3:
            e["titles"].append(t)
        a = str(r.paper_arxiv_id)
        if a and a not in e["arxiv"] and len(e["arxiv"]) < 5:
            e["arxiv"].append(a)
    out = []
    for path in sorted(rows):
        e = rows[path]
        out.append(dict(repo=path, reasons=sorted(e["reasons"]),
                        n_links=e["n_links"], official_links=e["official"],
                        mentioned_in_paper=e["in_paper"],
                        paper_titles=e["titles"], arxiv_ids=e["arxiv"]))
    CAND_F.write_text("".join(json.dumps(o) + "\n" for o in out))
    print(f"candidates: {len(out)} unique repos "
          f"(stat-title: {sum('stat-title' in o['reasons'] for o in out)}, "
          f"name-r: {sum('name-r' in o['reasons'] for o in out)}, "
          f"links covered: {sum(o['n_links'] for o in out)})", flush=True)
    return out


QF = ("repository(owner:%s,name:%s){primaryLanguage{name} "
      "licenseInfo{spdxId} defaultBranchRef{name} diskUsage pushedAt "
      "isArchived isFork stargazerCount}")


def gql_batch(batch):
    """Direct GraphQL POST (via curl + gh token): unlike `gh api graphql`,
    per-node failures (deleted/renamed repos) return null nodes alongside
    the good data instead of killing the whole batch."""
    aliases = " ".join(f'r{i}: {QF % (json.dumps(o), json.dumps(n))}'
                       for i, (o, n) in enumerate(batch))
    query = "query{" + aliases + " rateLimit{cost remaining resetAt}}"
    token = subprocess.run(["gh", "auth", "token"], capture_output=True,
                           text=True).stdout.strip()
    import urllib.request
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json",
                 "User-Agent": "sepalith-pwc-prefilter"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            res = json.loads(r.read())
    except Exception as e:
        return None, None, f"http: {e}"
    data = res.get("data") or {}
    rl = data.pop("rateLimit", None)
    recs = []
    for i, (owner, name) in enumerate(batch):
        node = data.get(f"r{i}")
        if node is None:
            recs.append(dict(repo=f"{owner}/{name}", status="gone",
                             note="not-resolvable"))
        else:
            recs.append(dict(
                repo=f"{owner}/{name}", status="ok",
                lang=(node.get("primaryLanguage") or {}).get("name"),
                spdx=(node.get("licenseInfo") or {}).get("spdxId"),
                branch=(node.get("defaultBranchRef") or {}).get("name"),
                kb=node.get("diskUsage"), pushed_at=node.get("pushedAt"),
                archived=node.get("isArchived"), fork=node.get("isFork"),
                stars=node.get("stargazerCount")))
    return recs, rl, None


def main():
    META.mkdir(parents=True, exist_ok=True)
    if CAND_F.exists():
        cands = [json.loads(l) for l in open(CAND_F)]
        print(f"candidates cache: {len(cands)}", flush=True)
    else:
        cands = build_candidates()

    done = set()
    if GH_META.exists():
        for l in open(GH_META):
            done.add(json.loads(l)["repo"])
    todo = [tuple(c["repo"].split("/", 1)) for c in cands
            if c["repo"] not in done]
    print(f"{len(cands)} candidates, {len(done)} fetched, {len(todo)} to go",
          flush=True)

    t0, fetched, err_streak = time.time(), 0, 0
    with open(GH_META, "a") as out:
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            recs, rl, err = gql_batch(batch)
            if err:
                err_streak += 1
                print(f"  batch error ({err_streak}): {err}", flush=True)
                if err_streak >= 5:
                    print("FIVE consecutive batch errors -> aborting "
                          "(resume-safe)", flush=True)
                    return
                time.sleep(60 * err_streak)
                continue
            err_streak = 0
            for r in recs:
                out.write(json.dumps(r) + "\n")
            out.flush()
            fetched += len(recs)
            # rate-aware sleep: never burn the last of the hourly budget
            if rl and rl.get("remaining", 5000) < 300:
                reset = 0
                try:
                    reset = max(0, time.mktime(time.strptime(
                        rl["resetAt"], "%Y-%m-%dT%H:%M:%SZ")) - time.time())
                except Exception:
                    reset = 900
                print(f"  rate low ({rl['remaining']}) -> sleep {reset:.0f}s",
                      flush=True)
                time.sleep(min(reset + 5, 3700))
            else:
                time.sleep(SLEEP_S)
            if (i // BATCH) % 5 == 0 or i + BATCH >= len(todo):
                rate = fetched / max(time.time() - t0, 1) * 3600
                print(f"  {i + len(batch)}/{len(todo)} repos "
                      f"({rate:,.0f}/h effective)", flush=True)
    print("PREFILTER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
