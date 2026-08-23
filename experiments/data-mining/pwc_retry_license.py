#!/usr/bin/env python3
"""pwc-archive license retry — second classification sweep for repos the
first pass excluded only because GitHub's spdx was null/NOASSERTION and the
LICENSE file was a usethis stub (YEAR/COPYRIGHT HOLDER). For R packages the
license declaration usually lives in DESCRIPTION's 'License:' field — the
same field ingest_cran/ingest_bioc classify from.

Phase 'scan' (light): for every ledger row excluded with
no-license-info*/license-file-unclassifiable, fetch the repo's root
DESCRIPTION via raw.githubusercontent (2s spacing; CDN, not the API), parse
the License field, re-classify with the patched ingest_pwc.classify
(spdx + DESCRIPTION field; NC/ND field alternatives stay excluded per the
ruling). Repos that flip to include are ledgered and queued.

Phase 'flip' (heavy, run AFTER ingest_pwc.py finishes to keep one clone
stream): full clone+stage of flipped repos via ingest_pwc.process()
(provenance/shard/manifest written normally; old excluded provenance
overwritten). Repos whose root DESCRIPTION 404s also go through the clone
path (DESCRIPTION may live in a subdirectory).

Usage: pwc_retry_license.py scan|flip|count
"""
import json, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, "/home/m0hawk/Documents/Sepalith/experiments/data-mining")
import ingest_pwc as ip

RETRY_REASONS = ("no-license-info", "license-file-unclassifiable")
QUEUE = Path("/tmp/pwc/retry_flip.jsonl")
UA = "sepalith-pwc-retry/0.1 (polite single-stream, 2s spacing)"


def _read_jsonl(path):
    """Tolerant JSONL reader: the ingest appends concurrently, so the last
    line can be half-written — skip malformed lines."""
    out = []
    try:
        with open(path) as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    out.append(json.loads(l))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return out


def retry_set():
    last = {}
    for r in _read_jsonl(ip.LEDGER):
        last[r["repo"]] = r
    return [r for r in last.values()
            if r["decision"] == "exclude"
            and (r.get("license_reason") or "").startswith(RETRY_REASONS)]


def meta_map():
    return {r["repo"]: r for r in _read_jsonl(ip.GH_META)}


def fetch_desc(repo, branch):
    for b in (branch, "HEAD", "master", "main"):
        if not b:
            continue
        url = (f"https://raw.githubusercontent.com/{repo}/"
               f"{urllib.parse.quote(b)}/DESCRIPTION")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                t = r.read().decode("utf-8", "replace")
            if "License" in t:
                return t, None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            return None, f"http-{e.code}"
        except Exception as e:
            return None, str(e)[:80]
    return None, "no-root-DESCRIPTION"


def parse_license(desc_text):
    fields, lastk = {}, None
    for line in desc_text.splitlines():
        if line.startswith((" ", "\t")) and lastk:
            fields[lastk] += " " + line.strip()
        elif ":" in line:
            k, _, v = line.partition(":")
            lastk = k.strip()
            fields[lastk] = v.strip()
    return fields.get("License")


def phase_scan():
    import urllib.parse
    rows, metas = retry_set(), meta_map()
    # merge with any earlier scan's queue (multiple scans are fine: rows
    # whose reason no longer prefixes RETRY_REASONS drop out of retry_set)
    prev = set()
    if QUEUE.exists():
        for l in open(QUEUE):
            prev.add(json.loads(l)["repo"])
    print(f"retry candidates: {len(rows)} (prev flips kept: {len(prev)})",
          flush=True)
    flips, still, missing = [], [], []
    with open(QUEUE, "w") as q:
        def emit(repo, m):
            q.write(json.dumps(dict(repo=repo, meta=m)) + "\n")
        for repo in sorted(prev):
            emit(repo, metas.get(repo, {}))
        for i, r in enumerate(rows, 1):
            repo = r["repo"]
            m = metas.get(repo, {})
            time.sleep(2.0)
            desc, err = fetch_desc(repo, m.get("branch"))
            lic_field = parse_license(desc) if desc else None
            decision, lclass, flag, reason = ip.classify(
                m.get("spdx"), None, lic_field)
            ip.ledger({**{k: r.get(k) for k in
                          ("repo", "owner", "name", "stars", "fork", "spdx",
                           "branch")},
                       "ts": ip.now(), "decision":
                       "accepted" if decision == "include" else "exclude",
                       "included": decision == "include",
                       "license_class": lclass, "license_reason":
                       f"retry:{reason}", "flag": flag,
                       "desc_license": lic_field})
            if decision == "include":
                flips.append(repo)
                emit(repo, m)
            elif lic_field:
                still.append((repo, lic_field))
            else:
                missing.append((repo, err))
            if i % 25 == 0:
                print(f"  {i}/{len(rows)} flips={len(flips)} "
                      f"field-excluded={len(still)} no-desc={len(missing)}",
                      flush=True)
    print(f"SCAN DONE flips={len(flips)} field-excluded={len(still)} "
          f"no-desc={len(missing)}", flush=True)
    Path("/tmp/pwc/retry_summary.json").write_text(json.dumps(
        dict(flips=flips, field_excluded=still[:100],
             no_desc=missing[:100], n_flips=len(flips),
             n_field_excluded=len(still), n_no_desc=len(missing)), indent=1))


def phase_flip():
    if not QUEUE.exists():
        print("no queue — run scan first")
        return
    rows = [json.loads(l) for l in open(QUEUE)]
    cands = {c["repo"]: c for c in map(json.loads, open(ip.CAND_F))}
    metas = meta_map()
    print(f"flipping {len(rows)} repos via full clone+stage", flush=True)
    counts, t0 = {}, time.time()
    for i, r in enumerate(rows, 1):
        repo = r["repo"]
        # drop the stale excluded provenance so the accepted one replaces it
        key = repo.replace("/", "__")
        (ip.PROV / f"{key}.json").unlink(missing_ok=True)
        out = ip.process(repo, cands.get(repo, {}),
                         r.get("meta") or metas.get(repo, {}))
        st = out["status"].split(":")[0].split(" ")[0]
        counts[st] = counts.get(st, 0) + 1
        if i % 10 == 0 or i <= 5:
            print(f"  {i}/{len(rows)} {json.dumps(out)}", flush=True)
        ip.LOGS.joinpath("state.json").write_text(json.dumps(
            dict(phase="retry-flip", updated=ip.now(), processed=i,
                 total=len(rows), counts=counts)))
    print(f"FLIP DONE {json.dumps(counts)} "
          f"total_s={time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "count"
    if phase == "scan":
        phase_scan()
    elif phase == "flip":
        phase_flip()
    else:
        rows = retry_set()
        print(f"{len(rows)} repos in retry set")
