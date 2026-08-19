#!/usr/bin/env python3
"""Build a CRAN download ranking from the Posit/RStudio CRAN mirror log.

The cranlogs API caps at 100 packages per request. The mirror's daily log
gives exact counts for every package (~71 MB CSV.gz, ~27k packages), which
ingest_cran.py and select_repos.py consume as
/mnt/h/sepalith/ranked/<YYYY-MM-DD>.counts.txt — one "count package" pair
per line, highest count first.

Usage:
  uv run python experiments/data-mining/download_ranking.py 2026-08-15

Mirror logs appear with a lag of one to two days; a 404 usually means the
day is not published yet, not that the URL is wrong.
"""
import argparse
import csv
import gzip
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path

RANKED = Path("/mnt/h/sepalith/ranked")
URL = "http://cran-logs.rstudio.com/{y}/{y}-{m:02d}-{d:02d}.csv.gz"


def fetch_day(year: int, month: int, day: int) -> Counter:
    url = URL.format(y=year, m=month, d=day)
    req = urllib.request.Request(url, headers={"User-Agent": "sepalith-data-mining/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = gzip.decompress(r.read())
    counts = Counter()
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))):
        pkg = (row.get("package") or "").strip()
        if pkg:
            counts[pkg] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY-MM-DD (mirror logs lag 1-2 days)")
    ap.add_argument("--out-dir", type=Path, default=RANKED)
    args = ap.parse_args()

    year, month, day = (int(x) for x in args.date.split("-"))
    counts = fetch_day(year, month, day)
    if not counts:
        raise SystemExit(f"log for {args.date} published but empty — check the CSV")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.date}.counts.txt"
    with open(out, "w") as fh:
        for pkg, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            fh.write(f"{n} {pkg}\n")
    print(json.dumps({"file": str(out), "packages": len(counts),
                      "top": counts.most_common(5)}))


if __name__ == "__main__":
    main()
