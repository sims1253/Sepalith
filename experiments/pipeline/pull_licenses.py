#!/usr/bin/env python3
"""Extract each package's LICENSE file from its local tarball into
provenance/<pkg>.license.txt, so any future public release can carry the
required copyright notices / license texts alongside the shards."""
import json, tarfile
from pathlib import Path

ROOT = Path("/mnt/h/sepalith")
n_ok = n_missing = 0
for prov in sorted((ROOT / "provenance").glob("*.json")):
    meta = json.loads(prov.read_text())
    tb = ROOT / "tarballs" / meta["tarball"]
    if not tb.exists():
        continue
    dest = ROOT / "provenance" / f"{meta['package']}.license.txt"
    if dest.exists():
        n_ok += 1
        continue
    found = None
    with tarfile.open(tb) as tf:
        cand = [m for m in tf.getnames()
                if m.count("/") == 1 and m.rsplit("/", 1)[1].upper() in
                ("LICENSE", "LICENCE", "LICENSE.MD", "LICENCE.MD", "LICENSE.TXT")]
        for m in cand:
            found = m
            break
        if found:
            dest.write_bytes(tf.extractfile(found).read())
            meta["license_file"] = found.rsplit("/", 1)[1]
            prov.write_text(json.dumps(meta, indent=1))
            n_ok += 1
        else:
            (ROOT / "provenance" / f"{meta['package']}.license.txt").write_text(
                f"# no LICENSE file in tarball; declared license: {meta['license']}\n")
            n_missing += 1
print(f"license files extracted: {n_ok}, none-present (declared-only): {n_missing}")
