#!/usr/bin/env python3
"""Stack v2 R acquisition — Phase B-2: CC-BY-SA second pass.

Per the updated user ruling (CC-BY-SA ruled IN, ledgered distinctly): fetch the
files excluded by Phase B ONLY because of a cc-by-sa element in an otherwise
permissive-or-benign license set. Sets that ALSO contain NC/ND/unknown
references stay excluded (ambiguity rule unchanged). Pool measured: 513 files
(~2MB) — immaterial at scale, fetched for completeness and ledger clarity.

Reuses Phase B mechanics: same S3 path, same state.db (single writer: run only
when phase B is done), shards appended as shard-sa-*.jsonl, license_class
"cc-by-sa" for the distinct ledger.
"""
import gzip, hashlib, json, sqlite3, sys, time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

STAGE = Path("/mnt/h/sepalith/stack_staging")
META, FILES, LOGS = STAGE / "metadata", STAGE / "files", STAGE / "logs"
LEDGER = STAGE / "license_ledger.jsonl"
LOCAL = Path("/tmp/stack_r_fetch")
DB = LOGS / "state.db"
CONFIGS = ["R", "RDoc", "RMarkdown"]
S3 = "https://softwareheritage.s3.amazonaws.com/content/{}"
UA = "sepalith-stackv2-r-acquisition/0.2 (sa-second-pass)"

PERMISSIVE = {
    "mit", "mit-0", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-4-clause",
    "0bsd", "bsd-2-clause-views", "bsd-3-clause-clear", "unlicense", "isc",
    "zlib", "bsl-1.0", "x11", "ncsa", "ecl-2.0", "postgresql",
    "gpl-1.0-only", "gpl-1.0-or-later", "gpl-2.0-only", "gpl-2.0-or-later",
    "gpl-3.0-only", "gpl-3.0-or-later",
    "lgpl-2.0-only", "lgpl-2.0-or-later", "lgpl-2.1-only", "lgpl-2.1-or-later",
    "lgpl-3.0-only", "lgpl-3.0-or-later",
    "agpl-3.0-only", "agpl-3.0-or-later",
    "cc0-1.0", "cc-by-1.0", "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
    "artistic-1.0", "artistic-1.0-clause8", "artistic-1.0-perl", "artistic-2.0",
    "mpl-1.1", "mpl-2.0", "mpl-2.0-no-copyleft-exception",
    "epl-1.0", "epl-2.0", "cpl-1.0",
    "cecill-1.0", "cecill-1.1", "cecill-2.0", "cecill-2.1", "cecill-b", "cecill-c",
}
PERMISSIVE_PREFIX = ("gpl-", "lgpl-", "agpl-", "bsd-", "apache-", "mit-",
                     "artistic-", "cecill-", "cc-by-", "cc0-")
BENIGN = {
    "licenseref-scancode-warranty-disclaimer",
    "licenseref-scancode-public-domain",
    "licenseref-scancode-public-domain-disclaimer",
    "licenseref-scancode-other-permissive",
    "licenseref-scancode-generic-cla",
}
HARD = ("cc-by-nc", "cc-by-nd", "cc-nc", "gfdl", "json")


def classify_sa_in(dl):
    """include iff every element permissive-or-benign AND at least one SA."""
    has_sa = False
    for lic in dl:
        lid = (lic or "").strip().lower()
        if not lid or lid in BENIGN:
            continue
        if "cc-by-sa" in lid:
            has_sa = True
            continue
        if lid in PERMISSIVE or (lid.startswith(PERMISSIVE_PREFIX)
                                 and not lid.startswith(HARD)):
            continue
        return False
    return has_sa


def main():
    accept = []
    for cfg in CONFIGS:
        pf = pq.ParquetFile(META / f"{cfg}.parquet")
        md = pf.metadata
        for rg in range(md.num_row_groups):
            tbl = pf.read_row_group(rg, columns=[
                "blob_id", "content_id", "path", "detected_licenses", "repo_name",
                "star_events_count", "length_bytes", "src_encoding", "extension",
                "revision_date", "github_id", "is_vendor", "is_generated"])
            cols = [tbl.column(k).to_pylist() for k in
                    ("blob_id", "content_id", "path", "detected_licenses", "repo_name",
                     "star_events_count", "length_bytes", "src_encoding", "extension",
                     "revision_date", "github_id", "is_vendor", "is_generated")]
            for blob, cid, path, dl, repo, stars, lb, enc, ext, rdate, gid, v, g in zip(*cols):
                if not dl or v or g or lb < 100 or lb > 10 * 10**6:
                    continue
                if repo.lower().startswith(("cran/", "bioconductor/")):
                    continue
                if classify_sa_in(dl):
                    accept.append((blob, cid, cfg, path, tuple(dl), repo, stars,
                                   lb, enc, ext, str(rdate)[:10] if rdate else "", gid))
    print(f"SA pool: {len(accept)} files", flush=True)
    con = sqlite3.connect(DB, timeout=120)
    recs = []
    for item in accept:
        blob, cid, cfg, path, dl, repo, stars, lb, enc, ext, rdate, gid = item
        r = con.execute("SELECT status FROM blobs WHERE blob_id=?", (blob,)).fetchone()
        if r:
            continue
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    S3.format(blob), headers={"User-Agent": UA}), timeout=30) as resp:
                data = gzip.decompress(resp.read())
        except Exception as e:
            con.execute("INSERT OR REPLACE INTO blobs VALUES (?,?,?,?)",
                        (blob, "missing", 0, time.strftime("%Y-%m-%dT%H:%M:%S")))
            print(f"missing {blob} {e}", flush=True)
            continue
        try:
            text = data.decode(enc or "utf-8", errors="replace")
        except LookupError:
            text = data.decode("utf-8", errors="replace")
        recs.append(dict(source="stackv2", config=cfg, path=path, repo=repo,
                         blob_id=blob, content_id=cid, licenses=list(dl),
                         license_class="cc-by-sa", star_events_count=stars,
                         extension=ext, revision_date=rdate, github_id=gid,
                         src_encoding=enc, length_bytes_meta=lb, bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest(), content=text))
        con.execute("INSERT OR REPLACE INTO blobs VALUES (?,?,?,?)",
                    (blob, "fetched", len(data), time.strftime("%Y-%m-%dT%H:%M:%S")))
    con.commit()
    if recs:
        p = FILES / "shard-sa-00000.jsonl"
        with open(LOCAL / "shard-sa-00000.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        import shutil
        shutil.copyfile(LOCAL / "shard-sa-00000.jsonl", p)
        (LOCAL / "shard-sa-00000.jsonl").unlink()
    with open(LEDGER, "a") as f:
        f.write(json.dumps(dict(
            type="phase-b2-cc-by-sa", ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
            ruling="CC-BY-SA ruled IN (user directive)",
            pool_size=len(accept), fetched=len(recs),
            bytes=sum(r["bytes"] for r in recs),
            note="sets also containing NC/ND/unknown refs remain excluded")) + "\n")
    print(f"PHASE B2 COMPLETE fetched={len(recs)}", flush=True)


if __name__ == "__main__":
    main()
