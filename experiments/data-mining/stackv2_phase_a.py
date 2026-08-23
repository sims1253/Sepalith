#!/usr/bin/env python3
"""Stack v2 R acquisition — Phase A: metadata parquet download + license histogram.

The Stack v2 on HF is METADATA-ONLY (per-file blob ids, licenses, repo info);
file contents live in the Software Heritage S3 bucket
(https://softwareheritage.s3.amazonaws.com/content/<blob_id>, gzip blobs,
readable anonymously — verified against length_bytes). Phase A downloads the
R-family metadata parquets once, then produces the license histogram +
fetch-budget report that gates Phase B (content fetch).

Layout written (mirrors bioc_staging conventions):
  /mnt/h/sepalith/stack_staging/metadata/<cfg>.parquet       immutable originals
  /mnt/h/sepalith/stack_staging/logs/phase_a_report.json     histogram + budget
  /mnt/h/sepalith/stack_staging/license_ledger.jsonl         append-only decisions

License ruling in force (2026-08-20/21): permissive-only
(MIT/Apache/BSD/GPL-family/CC-BY/CC0 + bioc-precedent Artistic/CeCILL/EPL/CPL/
MPL/AGPL-flagged + unambiguous MIT-equivalents Unlicense/ISC/Zlib/BSL/0BSD/X11/
NCSA/ECL-2.0); NC/ND/SA excluded outright; unknown or ambiguous sets (incl.
LicenseRef-scancode-unknown-license-reference, generic-cla, JSON, OpenSSL...)
-> EXCLUDED and ledgered, never silently dropped. A file is accepted only if
EVERY element of its detected_licenses set is permissive-or-benign; benign
ScanCode markers (warranty/public-domain disclaimers, other-permissive) are
non-license texts and ignored.
"""
import json, shutil, time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "bigcode/the-stack-v2"
REVISION = "e565caa3a78c2423bd374333a472b049eb090e47"
CONFIGS = ["R", "RDoc", "RMarkdown"]
STAGE = Path("/mnt/h/sepalith/stack_staging")
META = STAGE / "metadata"
LOGS = STAGE / "logs"
LOCAL = Path("/tmp/stack_r_fetch")

PERMISSIVE = {  # exact SPDX ids (lowercased) accepted per ruling + bioc precedent
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
# permissive family prefixes for less-common variants (gpl-2.0-plus etc.)
PERMISSIVE_PREFIX = ("gpl-", "lgpl-", "agpl-", "bsd-", "apache-", "mit-",
                     "artistic-", "cecill-", "cc-by-", "cc0-")
BENIGN_REFS = {  # ScanCode markers that are NOT licenses
    "licenseref-scancode-warranty-disclaimer",
    "licenseref-scancode-public-domain",
    "licenseref-scancode-public-domain-disclaimer",
    "licenseref-scancode-other-permissive",
    "licenseref-scancode-generic-cla",  # contributor-agreement text, not a use license
}
HARD_EXCLUDE_PREFIX = ("cc-by-nc", "cc-by-nd", "cc-by-sa", "cc-nc", "gfdl", "json")


def classify(licenses):
    """-> (decision, reason, class_string). Accept iff every element is
    permissive-or-benign; NC/ND/SA and anything unrecognized exclude the file."""
    if not licenses:
        return "exclude", "no-detected-licenses", ""
    classes = set()
    for lic in licenses:
        lid = (lic or "").strip().lower()
        if not lid:
            continue
        if lid in BENIGN_REFS:
            continue
        if lid in PERMISSIVE or (lid.startswith(PERMISSIVE_PREFIX)
                                 and not lid.startswith(HARD_EXCLUDE_PREFIX)):
            classes.add(lid)
            continue
        return "exclude", f"non-permissive:{'|'.join(sorted(licenses))[:120]}", ""
    if not classes:
        return "exclude", f"benign-only:{'|'.join(sorted(licenses))[:120]}", ""
    return "include", "", "+".join(sorted(classes))


def main():
    for d in (META, LOGS, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    report = {"revision": REVISION, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "configs": {}}
    for cfg in CONFIGS:
        dest = META / f"{cfg}.parquet"
        if not (dest.exists() and dest.stat().st_size > 0):
            t0 = time.time()
            p = hf_hub_download(REPO, f"data/{cfg}/train-00000-of-00001.parquet",
                                repo_type="dataset", revision=REVISION)
            tmp = LOCAL / f"{cfg}.parquet"
            shutil.copyfile(p, tmp)          # one bulk copy -> NAS
            shutil.copyfile(tmp, dest)
            tmp.unlink()
            print(f"{cfg}: downloaded+staged {dest.stat().st_size:,} bytes "
                  f"in {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"{cfg}: cached {dest.stat().st_size:,} bytes", flush=True)
        pf = pq.ParquetFile(dest)
        md = pf.metadata
        report["configs"][cfg] = {"rows": md.num_rows, "row_groups": md.num_row_groups,
                                  "parquet_bytes": dest.stat().st_size}

        inc = Counter(); exc = Counter(); lic_hist = Counter()
        bytes_inc = 0; bytes_all = 0; rows = 0; vg = Counter(); stars_hist = Counter()
        for rg in range(md.num_row_groups):
            tbl = pf.read_row_group(rg, columns=[
                "detected_licenses", "license_type", "length_bytes",
                "is_vendor", "is_generated", "star_events_count"])
            for dl, lb, v, g, s in zip(tbl.column("detected_licenses").to_pylist(),
                                       tbl.column("length_bytes").to_pylist(),
                                       tbl.column("is_vendor").to_pylist(),
                                       tbl.column("is_generated").to_pylist(),
                                       tbl.column("star_events_count").to_pylist()):
                rows += 1
                bytes_all += lb
                key = "|".join(sorted(x.lower() for x in dl)) if dl else "<none>"
                lic_hist[key] += 1
                if v: vg["vendor"] += 1
                if g: vg["generated"] += 1
                decision, reason, cls = classify(dl)
                if decision == "include" and not v and not g and lb >= 100:
                    inc[key] += 1
                    bytes_inc += lb
                    stars_hist["0" if s == 0 else ("1-4" if s < 5 else ("5-49" if s < 50 else "50+"))] += 1
                elif v or g:
                    exc["vendor-or-generated"] += 1
                elif lb < 100:
                    exc["tiny-under-100b"] += 1
                else:
                    exc[reason] += 1
            if rg % 10 == 0:
                print(f"  {cfg} rg {rg}/{md.num_row_groups}: rows={rows:,} "
                      f"inc={sum(inc.values()):,} exc={sum(exc.values()):,}", flush=True)
        report["configs"][cfg].update({
            "rows_scanned": rows, "bytes_all": bytes_all,
            "bytes_permissive_nonvendor_ge100b": bytes_inc,
            "files_permissive_nonvendor_ge100b": sum(inc.values()),
            "vendor_generated": dict(vg),
            "star_buckets_included": dict(stars_hist),
            "top_included_license_sets": [{"licenses": k, "files": v}
                                          for k, v in inc.most_common(25)],
            "exclusion_reasons_top": exc.most_common(40),
            "license_histogram_top60": [{"licenses": k, "files": v}
                                        for k, v in lic_hist.most_common(60)],
        })
        (LOGS / "phase_a_report.json").write_text(json.dumps(report, indent=1))
        print(f"{cfg} DONE: accept={sum(inc.values()):,} files, "
              f"{bytes_inc:,} bytes ({bytes_inc/1e9:.2f} GB)", flush=True)
    print("PHASE A COMPLETE", flush=True)


if __name__ == "__main__":
    main()
