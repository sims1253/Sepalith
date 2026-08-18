#!/usr/bin/env python3
"""Pilot: mine GitHub PR REVIEW data as "instructed edit" pairs.

For merged PRs in R repos (local clones at <git-dir>/<owner>__<repo>):
  reviewer comment (the WHY / instruction)
  + the hunk of the PR's final diff for that file nearest the comment's
    diff_hunk (the edit the author made in response).

Per PR:
  GET /repos/{o}/{r}/pulls?state=closed           (merged only, review_comments>0)
  GET /repos/{o}/{r}/pulls/{n}/comments           (review-thread comments)
  GET /repos/{o}/{r}/pulls/{n}/reviews            (review bodies, context only)
  GET /repos/{o}/{r}/pulls/{n}/files              (final patch per file)

Records follow the scenarios.py/edit-pair JSON shape (see mine_edit_pairs.py):
  {family:"pr_instructed_edit", repo, pr_number, comment_id, path, instruction,
   prefix, region_old, region_new, event_diff:"", license_file, flags}
plus additive provenance (pr_title, dates, match_score, parse_ok, ...).

Transport: `gh api` (authenticated, 5000 req/h) with a disk cache so reruns
are free; still paced (--min-interval) and backs off on rate limits.

Output: <out>/pilot.jsonl + <out>/stats.json (+ <out>/_cache/ for API bodies).
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

DEFAULT_REPOS = [
    "tidyverse/dplyr", "tidyverse/tidyr", "tidyverse/dbplyr", "tidyverse/readxl",
    "tidyverse/tibble", "tidyverse/purrr", "tidyverse/lubridate", "tidyverse/hms",
    "tidyverse/vroom", "tidyverse/duckplyr",
    "r-lib/rlang", "r-lib/cli", "r-lib/testthat", "r-lib/roxygen2",
    "r-lib/pkgdown", "r-lib/processx", "r-lib/callr", "r-lib/pillar",
    "r-lib/cpp11", "r-lib/withr",
    "easystats/insight", "easystats/parameters", "easystats/performance",
    "easystats/effectsize", "easystats/correlation", "easystats/datawizard",
    "easystats/see", "easystats/report", "easystats/modelbased",
    "easystats/bayestestR",
]

MIN_BODY, MAX_BODY = 15, 400
MAX_REGION = 30            # lines; hunk before/after state cap
MAX_PAIRS_PER_PR = 5
MIN_MATCH_SCORE = 0.5      # fraction of comment-hunk before-lines found in matched hunk
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "LICENCE.md", "COPYING")

BOT_AUTHOR = re.compile(r"(github-actions|codecov|\bbot\b|lintr|cran-|rhub|dependabot|renovate)", re.I)
PURE_ACK = re.compile(
    r"^\W*(lgtm|looks good( to me)?|approv(ed|ed!|es)?|thanks?|thank you|nice|cool|done|"
    r"fixed|agree[ds]?|good catch|yes|no|ok(ay)?|oops|whoops|sorry|great|wow|indeed|"
    r"right|true|fair|merged|rebased|updated|pushed)\W*$", re.I)
CI_NOISE = re.compile(
    r"(codecov|github-actions|github actions|workflow run|build failed|ci fail|"
    r"checks? (fail|pass)|coverage (report|decreas|increas)|appveyor|travis|"
    r"error in .*workflow|restart(ing)? (the )?(workflow|ci))", re.I)
INTENT_LEXICON = [
    "fix", "use", "instead", "should", "rename", "replace", "move", "remove",
    "add", "change", "simplify", "avoid", "consider", "suggest", "why", "bug",
    "error", "fail", "name", "argument", "function", "return", "vector", "test",
    "check", "document", "export", "deprecat", "maybe", "prefer", "could",
    "would", "need", "must", "wrong", "missing", "unclear", "typo", "cleaner",
    "simpler", "clearer", "safer", "faster", "consisten", "duplicat", "hardcod",
    "redundant", "helper", "loop", "index", "subset", "mutate", "filter",
    "default", "param", "message", "warning", "stop", "abort", "inform",
    "print", "read", "write", "class", "method", "generic", "s3", "s4", "rcpp",
    "cpp11", "c code", "compile", "assert", "closure", "environ", "global",
    "attribute", "names", "drop", "select", "group", "arrange", "sort", "order",
    "unique", "null", "type", "double", "integer", "character", "logical",
    "string", "length", "dim", "row", "column", "file", "path", "package",
    "library", "depend", "version", "docs", "roxygen", "example", "vignette",
    "news", "testthat", "snapshot", "skip", "lintr", "styler", "spelling",
    "url", "link", "api", "token", "key", "seed", "random", "output", "input",
    "cache", "memory", "copy", "reference", "pattern", "regex", "escape",
    "quote", "brace", "indent", "format", "round", "integerish", "coerce",
    "convert", "cast", "alias", "import", "namespace", "collate", "r6",
    "data.frame", "tibble", "list", "empty", "scalar", "recycl", "group_by",
    "ifelse", "if_else", "case_when", "map", "lapply", "sapply", "vapply",
    "apply", "purrr", "dplyr", "tidyr", "rlang", "cli", "abort", "warn",
]

# ---------------------------------------------------------------- transport --
class GH:
    def __init__(self, cache_dir, min_interval=2.0):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0
        self.calls = 0

    def _key(self, url):
        return self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def get(self, url):
        """GET an api.github.com path (with query string). Returns parsed JSON
        (list or dict), None on 404. Cached aggressively."""
        k = self._key(url)
        if k.exists():
            try:
                data = json.loads(k.read_text())
                return None if data == {"__404__": True} else data
            except Exception:
                pass
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            self._last = time.time()
            self.calls += 1
            try:
                p = subprocess.run(["gh", "api", url], capture_output=True,
                                   text=True, timeout=90)
            except subprocess.TimeoutExpired:
                p = None
            if p is not None and p.returncode == 0:
                data = json.loads(p.stdout)
                k.write_text(json.dumps(data))
                return data
            err = (p.stderr if p else "timeout")
            if "HTTP 404" in err:
                k.write_text('{"__404__": true}')
                return None
            if "rate limit" in err.lower() or "HTTP 403" in err:
                reset = self._reset_at()
                sleep_for = max(5.0, min(reset - time.time() + 2, 3700))
                print(f"    rate-limited; sleeping {sleep_for:.0f}s "
                      f"(reset {reset})", flush=True)
                time.sleep(sleep_for)
                continue
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"gh api failed after retries: {url}: {err[:200]}")

    @staticmethod
    def _reset_at():
        try:
            p = subprocess.run(["gh", "api", "rate_limit", "--jq",
                                ".resources.core.reset"], capture_output=True,
                               text=True, timeout=30)
            return int(p.stdout.strip())
        except Exception:
            return int(time.time()) + 900


def get_all(gh, url):
    """Follow pagination (Link header) up to 5 pages, concatenating arrays."""
    out, page = [], 1
    while page <= 5:
        sep = "&" if "?" in url else "?"
        chunk = gh.get(f"{url}{sep}page={page}")
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < 100 or not isinstance(chunk, list):
            break
        page += 1
    return out


# ------------------------------------------------------------------ hunks ----
def parse_hunk_lines(text):
    """Unified diff text -> [{header, lines:[(tag,content)]}] (tag ' ','+','-','\\')."""
    hunks, cur = [], None
    for line in text.split("\n"):
        if line.startswith("@@"):
            if cur:
                hunks.append(cur)
            cur = {"header": line, "lines": []}
        elif cur is not None:
            tag = line[:1] if line[:1] in (" ", "+", "-", "\\") else " "
            cur["lines"].append((tag, line[1:] if tag != " " else line[1:]))
    if cur:
        hunks.append(cur)
    return hunks


def before_set(hunk):
    return set(c for t, c in hunk["lines"] if t in (" ", "-") and c.strip())


def after_set(hunk):
    return set(c for t, c in hunk["lines"] if t in (" ", "+") and c.strip())


def match_hunk(comment_hunk, file_hunks):
    """Nearest final-patch hunk for a comment's diff_hunk.
    Returns (hunk, score). Primary signal: overlap of before-side (' '/'-')
    line contents; pure-addition comments (new files, <2 before-lines) fall
    back to after-side (' '/'+') overlap."""
    cb, ca = before_set(comment_hunk), after_set(comment_hunk)
    use_after = len(cb) < 2
    base = ca if use_after else cb
    if not base:
        return None, 0.0
    best, best_score = None, 0.0
    for fh in file_hunks:
        fs = after_set(fh) if use_after else before_set(fh)
        score = len(base & fs) / len(base)
        if score > best_score:
            best, best_score = fh, score
    return best, best_score


# -------------------------------------------------------------- instruction --
def instruction_reject_reason(body, author):
    """None if the comment looks like human code-change intent, else a reason."""
    body = (body or "").strip()
    if not (MIN_BODY <= len(body) <= MAX_BODY):
        return "len_out_of_range"
    if author and BOT_AUTHOR.search(author):
        return "bot_author"
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", body)
    if len(words) < 3:
        return "too_few_words"
    if PURE_ACK.match(body):
        return "pure_ack"
    if CI_NOISE.search(body):
        return "ci_noise"
    letters = sum(1 for ch in body if unicodedata.category(ch).startswith("L"))
    latin = sum(1 for ch in body if "LATIN" in unicodedata.name(ch, "").upper())
    if letters and latin / letters < 0.8:
        return "not_english"
    low = body.lower()
    if not any(k in low for k in INTENT_LEXICON):
        return "no_intent_lexicon"
    return None


# ------------------------------------------------------------------ flags ----
def auto_flags(pair):
    f = []
    ro = [l for l in pair["region_old"] if l.strip()]
    rn = [l for l in pair["region_new"] if l.strip()]
    if [l.strip() for l in pair["region_old"]] == [l.strip() for l in pair["region_new"]]:
        f.append("ws-only")
    if ro and rn and \
       all(l.strip().startswith("#") for l in ro) and \
       all(l.strip().startswith("#") for l in rn):
        f.append("comment-only")
    same = sum(1 for a, b in zip(ro, rn) if a == b)
    if ro and rn and same / max(len(ro), len(rn)) > 0.9:
        f.append("mostly-unchanged")
    if not pair["prefix"]:
        f.append("no-context")
    return f


# --------------------------------------------------------------- license -----
def license_at(git_dir):
    try:
        p = subprocess.run(["git", "-C", str(git_dir), "ls-tree", "--name-only",
                            "HEAD"], capture_output=True, text=True, timeout=30)
        names = set(p.stdout.split())
        return next((n for n in LICENSE_NAMES if n in names), "none")
    except Exception:
        return "unknown"


def license_type(git_dir):
    """License TYPE (MIT / Apache-2.0 / GPL-3 / ...) from the repo's
    DESCRIPTION `License:` field, falling back to LICENSE file keywords."""
    desc = git_dir / "DESCRIPTION"
    if desc.exists():
        try:
            m = re.search(r"^License:\s*(.+)$", desc.read_text(errors="replace"),
                          re.MULTILINE | re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if re.search(r"MIT", val, re.I):
                    return "MIT"
                if re.search(r"Apache", val, re.I):
                    return "Apache-2.0"
                if re.search(r"AGPL", val, re.I):
                    return "AGPL-3.0"
                if re.search(r"LGPL", val, re.I):
                    return "LGPL-3.0"
                g = re.search(r"GPL[ -]*(?:\(>=?\s*)?(\d)", val)
                if g:
                    return f"GPL-{g.group(1)}"
                if re.search(r"BSD", val, re.I):
                    return "BSD"
                if re.search(r"CC0", val, re.I):
                    return "CC0-1.0"
                if re.search(r"MPL", val, re.I):
                    return "MPL-2.0"
                return val.split("|")[0].split("+")[0].strip()[:40] or "unknown"
        except Exception:
            pass
    for name in LICENSE_NAMES:
        f = git_dir / name
        if f.exists():
            try:
                head = f.read_text(errors="replace")[:600]
            except Exception:
                continue
            if "Apache License" in head:
                return "Apache-2.0"
            if re.search(r"MIT License|Permission is hereby granted, free of charge",
                         head):
                return "MIT"
            g = re.search(r"GNU (?:Lesser ?)?General Public License.*?Version (\d)",
                          head, re.I | re.S)
            if g:
                pref = "LGPL" if "Lesser" in g.group(0) else "GPL"
                return f"{pref}-{g.group(1)}"
            if "Mozilla Public License" in head:
                return "MPL-2.0"
    return "unknown"


# ------------------------------------------------------------------- mine ----
def mine_repo(gh, slug, git_dir, args, stats):
    prs = []
    for page in range(1, args.pages + 1):
        chunk = gh.get(f"repos/{slug}/pulls?state=closed&sort=created"
                       f"&direction=desc&per_page=100&page={page}")
        if not chunk:
            break
        prs.extend(chunk)
        if len(chunk) < 100:
            break
    stats["prs_listed"] += len(prs)
    merged = [p for p in prs if p.get("merged_at")]
    stats["prs_merged"] += len(merged)

    # The pulls LIST endpoint has no review_comments count, so probe merged
    # PRs newest-first with the comments endpoint; hydrate reviews+files only
    # for PRs that actually have review comments.
    lic = license_at(git_dir)
    lic_type = license_type(git_dir)
    pairs = []
    with_comments_seen = 0
    probed = 0
    for pr in merged:
        if with_comments_seen >= args.max_prs_per_repo or probed >= args.probe_cap:
            break
        probed += 1
        stats["prs_probed"] += 1
        n = pr["number"]
        comments = get_all(gh, f"repos/{slug}/pulls/{n}/comments?per_page=100")
        if not comments:
            continue
        with_comments_seen += 1
        stats["prs_with_review_comments"] += 1
        stats["prs_processed"] += 1
        reviews = get_all(gh, f"repos/{slug}/pulls/{n}/reviews?per_page=100")
        stats["reviews_fetched"] += len(reviews or [])
        files = get_all(gh, f"repos/{slug}/pulls/{n}/files?per_page=100")
        by_path = {}
        for f in files or []:
            if f.get("patch"):
                by_path.setdefault(f["filename"], []).extend(
                    parse_hunk_lines(f["patch"]))
        for c in comments or []:
            stats["comments_seen"] += 1
            body, author = c.get("body", ""), (c.get("user") or {}).get("login", "")
            reason = instruction_reject_reason(body, author)
            if reason:
                stats["comments_rejected"][reason] = \
                    stats["comments_rejected"].get(reason, 0) + 1
                continue
            path = c.get("path", "")
            if not path.endswith(".R"):
                stats["comments_rejected"]["non_r_path"] = \
                    stats["comments_rejected"].get("non_r_path", 0) + 1
                continue
            hunks = by_path.get(path)
            if not hunks:
                stats["comments_rejected"]["no_patch_for_path"] = \
                    stats["comments_rejected"].get("no_patch_for_path", 0) + 1
                continue
            dh = c.get("diff_hunk") or ""
            ch = parse_hunk_lines(dh)
            if not ch:
                stats["comments_rejected"]["no_diff_hunk"] = \
                    stats["comments_rejected"].get("no_diff_hunk", 0) + 1
                continue
            hunk, score = match_hunk(ch[0], hunks)
            if hunk is None or score < MIN_MATCH_SCORE:
                stats["comments_rejected"]["weak_hunk_match"] = \
                    stats["comments_rejected"].get("weak_hunk_match", 0) + 1
                continue
            lines = hunk["lines"]
            region_old = [c2 for t, c2 in lines if t in (" ", "-")]
            region_new = [c2 for t, c2 in lines if t in (" ", "+")]
            prefix = []
            for t, c2 in lines:
                if t != " ":
                    break
                prefix.append(c2)
            if not region_old or region_old == region_new:
                stats["comments_rejected"]["region_not_edit"] = \
                    stats["comments_rejected"].get("region_not_edit", 0) + 1
                continue
            if max(len(region_old), len(region_new)) > MAX_REGION:
                stats["comments_rejected"]["region_too_big"] = \
                    stats["comments_rejected"].get("region_too_big", 0) + 1
                continue
            stats["comments_kept"] += 1
            flags = auto_flags(dict(prefix=prefix, region_old=region_old,
                                    region_new=region_new))
            if author == "Copilot":
                flags.append("ai_reviewer")  # substantive AI review, not junk bot
            if author == (pr.get("user") or {}).get("login"):
                flags.append("by_pr_author")
            if c.get("in_reply_to_id"):
                flags.append("thread-reply")
            if c.get("created_at") and c["created_at"] < (pr.get("merged_at") or ""):
                flags.append("precedes_merge")
            # author changed this area AFTER the comment (approx: the final
            # hunk's before-side no longer equals the comment's hunk side)
            if before_set(ch[0]) != before_set(hunk) or \
               after_set(ch[0]) != after_set(hunk):
                flags.append("hunk_changed_after_comment")
            pairs.append(dict(
                family="pr_instructed_edit", repo=slug, pr_number=n,
                comment_id=c.get("id"), path=path,
                instruction=body.strip(),
                prefix=prefix, region_old=region_old, region_new=region_new,
                event_diff="", license_file=lic, flags=flags,
                source_url=pr.get("html_url") or
                    f"https://github.com/{slug}/pull/{n}",
                license=lic_type,
                # additive provenance / audit
                lang="r", pr_title=pr.get("title", ""),
                comment_created_at=c.get("created_at"),
                merged_at=pr.get("merged_at"),
                comment_author=author,
                comment_url=c.get("html_url"),
                comment_commit_id=c.get("commit_id"),
                match_score=round(score, 3),
                position=c.get("position"),
            ))
            if sum(1 for p in pairs if p["pr_number"] == n) >= MAX_PAIRS_PER_PR:
                break  # enough pairs from this PR
    return pairs


# ------------------------------------------------------------ R validation --
def r_parse_check(pairs, out_dir):
    """Best-effort: does region_new (and region_old) parse as R? One Rscript."""
    tmp = out_dir / "_parse_tmp"
    if tmp.exists():
        for f in tmp.iterdir():
            f.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    meta = {}
    for i, p in enumerate(pairs):
        for side in ("new", "old"):
            f = tmp / f"{i:05d}_{side}.R"
            f.write_text("\n".join(p[f"region_{side}"]) + "\n")
            meta[f.name] = (i, side)
    if not meta:
        return
    script = tmp / "driver.R"
    script.write_text(
        "fs <- list.files('" + str(tmp) + "', pattern='[.]R$', full.names=TRUE)\n"
        "res <- character()\n"
        "for (f in fs) {\n"
        "  ok <- tryCatch({ p <- parse(f); TRUE },\n"
        "                 warning=function(w) TRUE,\n"
        "                 error=function(e) FALSE)\n"
        "  res <- c(res, paste0(basename(f), ' ', ok))\n"
        "}\n"
        "writeLines(res, '" + str(tmp / "results.txt") + "')\n")
    try:
        subprocess.run(["Rscript", str(script)], capture_output=True, timeout=600)
        for line in (tmp / "results.txt").read_text().splitlines():
            name, _, ok = line.rpartition(" ")
            if name in meta:
                i, side = meta[name]
                pairs[i][f"parse_ok_{side}"] = (ok == "TRUE")
    except Exception as e:
        print(f"  R parse check failed: {e}", flush=True)
    finally:
        for f in tmp.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
        try:
            tmp.rmdir()
        except Exception:
            pass


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-dir", default="/mnt/h/sepalith/git")
    ap.add_argument("--out-dir",
                    default="/mnt/h/sepalith/datasets/pr_instructed_v1")
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    ap.add_argument("--pages", type=int, default=3,
                    help="max listing pages per repo")
    ap.add_argument("--max-prs-per-repo", type=int, default=6)
    ap.add_argument("--probe-cap", type=int, default=25,
                    help="max merged PRs to probe for comments per repo")
    ap.add_argument("--min-interval", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N repos")
    args = ap.parse_args()
    if args.limit:
        args.repos = args.repos[:args.limit]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gh = GH(out / "_cache", args.min_interval)

    stats = {
        "repos": [], "prs_listed": 0, "prs_merged": 0, "prs_probed": 0,
        "prs_with_review_comments": 0, "prs_processed": 0,
        "comments_seen": 0, "comments_kept": 0,
        "comments_rejected": {}, "reviews_fetched": 0,
        "api_calls": 0,
    }
    all_pairs = []
    t0 = time.time()
    for i, slug in enumerate(args.repos, 1):
        local = Path(args.git_dir) / slug.replace("/", "__")
        if not local.exists():
            # dir names were lowercased/truncated; prefix match
            cands = [p for p in Path(args.git_dir).iterdir()
                     if p.name.lower().startswith(
                         slug.split("/")[0] + "__" + slug.split("/")[1][:8].lower())]
            local = cands[0] if cands else local
        print(f"[{i}/{len(args.repos)}] {slug}", flush=True)
        try:
            pairs = mine_repo(gh, slug, local, args, stats)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            stats["repos"].append({"repo": slug, "error": str(e)[:200]})
            continue
        all_pairs.extend(pairs)
        stats["repos"].append({"repo": slug, "pairs": len(pairs)})
        print(f"  pairs so far: {len(all_pairs)} "
              f"({time.time()-t0:.0f}s, {gh.calls} api calls)", flush=True)

    r_parse_check(all_pairs, out)
    for p in all_pairs:
        p["parse_ok"] = bool(p.get("parse_ok_new"))  # spec-facing tag
    stats["pairs_emitted"] = len(all_pairs)
    parseable = [p for p in all_pairs if p.get("parse_ok_new")]
    stats["parse_ok_new_rate"] = round(
        len(parseable) / len(all_pairs), 3) if all_pairs else None
    stats["parse_ok_old_rate"] = round(
        sum(1 for p in all_pairs if p.get("parse_ok_old")) / len(all_pairs),
        3) if all_pairs else None
    stats["api_calls"] = gh.calls
    stats["secs"] = round(time.time() - t0, 1)

    with (out / "pilot.jsonl").open("w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "repos"}, indent=2))


if __name__ == "__main__":
    main()
