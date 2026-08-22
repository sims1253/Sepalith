#!/usr/bin/env python3
"""Bioconductor ingestion v0 — polite mirror of release software packages -> SEPARATE STAGING TREE.

Mirrors ingest_cran.py conventions (provenance-from-DESCRIPTION, full-tree
extraction, air format + jarl check --fix, one shard per package) but writes
ONLY to the Bioc staging area. The CRAN store (/mnt/h/sepalith/normalized,
datasets/, provenance/, manifest.jsonl) is NEVER touched — merging is a later,
explicit decision.

Layout under BIOC_STAGING (/mnt/h/sepalith/bioc_staging):
  tarballs/<pkg>_<ver>.tar.gz      immutable originals (kept even for excluded licenses)
  PACKAGES_index.json              release index snapshot (fetched once per run)
  bioc_license_ledger.jsonl        per-package license decision for ALL packages
  manifest.jsonl                   included packages only (merge-ready, source="bioc")
  provenance/<pkg>.json            license/authors/tarball sha/included flag
  datasets/packages/<pkg>.jsonl    one shard per included package
  logs/<pkg>.log                   air/jarl telemetry
  logs/ingest.jsonl                per-package outcome log
  logs/state.json                  resume state (written after every package)
  logs/progress.log                progress line every 25 packages

Normalized trees: /mnt/h/sepalith/normalized_bioc/<pkg>/<ver>/<pkg>/...
(full tarball tree, same shape as the CRAN normalized store; normalization =
air format + jarl check --fix, applied at ingest time exactly like CRAN).

Politeness to bioconductor.org: one request stream, 3-5s sleep between HTTP
fetches, retry with hard backoff on 429/5xx (60s x attempt, max 5), 10-minute
circuit-breaker pause after 3 consecutive failures.

Licensing ruling in force (2026-08-20): NO NC and NO ND data at all; only
clearly permissive items count (GPL/Artistic/MIT/LGPL/BSD/Apache/CC0 count as
permissive for training use; AGPL training-safe but flagged); ambiguous
defaults are EXCLUDED — and recorded as excluded in the ledger.
"""
import gzip, hashlib, json, os, random, re, shlex, shutil, subprocess, sys, tarfile, tempfile, time
import urllib.error, urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("SEPALITH_ROOT", "/mnt/h/sepalith"))
BIOC_STAGING = ROOT / "bioc_staging"
NORM_TREE = ROOT / "normalized_bioc"           # <pkg>/<ver>/<pkg>/...
LEDGER = BIOC_STAGING / "bioc_license_ledger.jsonl"
MANIFEST = BIOC_STAGING / "manifest.jsonl"
INDEX_CACHE = BIOC_STAGING / "PACKAGES_index.json"
PACKAGES_GZ = "https://bioconductor.org/packages/release/bioc/src/contrib/PACKAGES.gz"
MIRROR = "https://bioconductor.org/packages/release/bioc/src/contrib"
MAX_TARBALL = 150 * 1024 * 1024               # skip monsters (recorded, not ingested)
UA = "sepalith-bioc-ingest/0.1 (polite single-stream mirror, retry-with-backoff)"
PROGRESS_EVERY = 25

_last_request = 0.0
_consecutive_fail = 0


def polite_pause():
    """Block until 3-5s have passed since the previous HTTP request."""
    global _last_request
    wait = random.uniform(3.0, 5.0) - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)


def mark_request():
    global _last_request
    _last_request = time.time()


def http_download(url, dest):
    """Polite fetch -> dest. Returns (ok, detail). Hard backoff on 429/5xx."""
    global _consecutive_fail
    for attempt in range(5):
        polite_pause()
        mark_request()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) > MAX_TARBALL:
                return False, f"too-big:{len(data)}"
            Path(dest).write_bytes(data)
            _consecutive_fail = 0
            return True, str(len(data))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, "http-404"
            if attempt == 4:
                break
            if e.code == 429 or e.code >= 500:
                global_sleep = 60 * (attempt + 1)
                print(f"    HTTP {e.code} on {url} -> hard backoff {global_sleep}s "
                      f"(attempt {attempt + 1}/5)", flush=True)
                time.sleep(global_sleep)
            else:
                time.sleep(30 * (attempt + 1))
        except Exception as e:
            if attempt == 4:
                break
            time.sleep(60 * (attempt + 1))
    _consecutive_fail += 1
    if _consecutive_fail >= 3:
        print(f"    3 consecutive download failures -> circuit-breaker pause 600s",
              flush=True)
        time.sleep(600)
        _consecutive_fail = 0
    return False, "exhausted-retries"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_description(text):
    """DESCRIPTION continuation-line parser (same as ingest_cran.py)."""
    fields, last = {}, None
    for line in text.splitlines():
        if line.startswith((" ", "\t")) and last:
            fields[last] += " " + line.strip()
        elif ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            fields[k] = v
            last = k
    return fields


# --- license classification (ruling: permissive-only, no NC/ND, ambiguous=excluded) ---

NC_ND = ("noncommercial", "noderiv", "no-deriv", "cc by-nc", "by-nd", " nc ", "| nc")
# NC/ND detection against LICENSE-file TEXT: ONLY explicit Creative-Commons-
# style designations. Bare-word scans are unusable here — the GPL text itself
# contains "noncommercial" (GPL-2 §3c "noncommercial distribution") and
# "noncommercially" (GPL-3 §7), which are license-clause language, NOT an NC
# restriction on the work.
_NC_ND_FILE = re.compile(
    r"creative\s+commons\s+attribution[^\n]{0,60}(noncommercial|noderivs)"
    r"|\bcc[-\s]?by[-\s]?(nc|nd)\b"
    r"|creativecommons\.org/licenses/[^\s/'\"]*-(nc|nd)"
    r"|\bnoderivs\b"
    r"|non[-\s]?commercial\s+(use|purposes)\s+only",
    re.IGNORECASE)
# order = most-preferred class when several alternatives exist
# (NC/ND alternatives are filtered out before these run, so "cc by" cannot
#  accidentally match "cc by-nc"; EPL/CPL/MPL are OSI-approved weak-copyleft,
#  training-safe, flagged)
_PERMISSIVE = [
    ("MIT",     lambda s: s.startswith("mit")),
    ("BSD",     lambda s: "bsd" in s),
    ("Apache",  lambda s: "apache" in s),
    ("Artistic-2.0", lambda s: s.startswith("artistic")),
    ("CC0",     lambda s: s.startswith("cc0")),
    ("CC-BY",   lambda s: s.startswith("cc by") or s.startswith("cc-by")
                         or s.startswith("creative commons attribution")),
    ("LGPL",    lambda s: s.startswith("lgpl") or s.startswith("lesser gnu")),
    ("GPL",     lambda s: s.startswith("gpl") or s.startswith("gnu general public")),
    ("AGPL",    lambda s: s.startswith("agpl")),
    ("CeCILL",  lambda s: s.startswith("cecill")),
    ("EPL",     lambda s: s.startswith("epl")),
    ("CPL",     lambda s: s.startswith("cpl")),
    ("MPL-2.0", lambda s: s.startswith("mozilla") or s.startswith("mpl")),
]
_FLAGGED = {"AGPL": "agpl-redistribution-sensitive",
            "CeCILL": "cecill-family-gpl-compatible",
            "EPL": "osi-weak-copyleft", "CPL": "osi-weak-copyleft",
            "MPL-2.0": "osi-weak-copyleft"}


def classify_license(field, license_text=None):
    """-> (decision, license_class, flag, reason). decision in include|exclude."""
    field = (field or "").strip()
    if not field or field.lower() in ("unknown", "what license?"):
        return "exclude", "unclassified", "", "empty-or-unknown-license-field"
    alts = [a.strip() for a in field.split("|")]
    classes, needs_file, hard_bad = [], False, False
    for alt in alts:
        a = alt.lower()
        if any(m in a for m in NC_ND):
            hard_bad = True          # NC/ND alternative (only fatal if no permissive alt)
            continue
        if "file license" in a:
            needs_file = True        # remainder may still name a license (e.g. MIT + file)
        matched = None
        for cls, test in _PERMISSIVE:
            if test(a):
                matched = cls
                break
        if matched == "AGPL":
            classes.append("AGPL")   # training-safe, flagged for redistribution
        elif matched:
            classes.append(matched)
    if classes:
        best = next((c for c, _ in _PERMISSIVE for x in classes if x == c), classes[0])
        flag = next((f for c, f in _FLAGGED.items() if c in classes), "")
        return "include", best, flag, ""
    # no alternative directly named a permissive license
    if license_text:
        t = license_text.lower()
        if _NC_ND_FILE.search(license_text):
            return "exclude", "NC/ND", "", "nc-or-nd-terms-in-license-file"
        markers = [("Artistic-2.0", "artistic"), ("MIT", "permission is hereby granted"),
                   ("MIT", "mit license"), ("LGPL", "lesser general public"),
                   ("LGPL", "lgpl"), ("GPL", "gnu general public license"),
                   ("GPL", "gnu affero" if False else "gpl"), ("Apache", "apache license"),
                   ("CeCILL", "cecill"), ("BSD", "bsd"), ("CC0", "cc0")]
        for cls, m in markers:
            if m in t:
                flag = "cecill-family-gpl-compatible" if cls == "CeCILL" else ""
                return "include", cls, flag, "classified-from-license-file"
        return "exclude", "unclassified", "", "license-file-unclassifiable"
    if hard_bad:
        return "exclude", "NC/ND", "", "only-nc-nd-alternatives-in-field"
    if needs_file:
        return "exclude", "unclassified", "", "file-license-no-file-or-empty"
    return "exclude", "unclassified", "", f"unrecognized-license-field:{field[:60]}"


def license_file_scan(text):
    """Ruling safety net: any LICENSE file with NC/ND terms overrides to exclude."""
    if text and _NC_ND_FILE.search(text):
        return "exclude", "NC/ND", "", "nc-or-nd-terms-in-license-file"
    return None


def ledger_write(rec):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def done_pkgs():
    """Resume: packages with a provenance file are fully processed."""
    return {p.stem for p in (BIOC_STAGING / "provenance").glob("*.json")}


def process(pkg, idx, total, idx_meta):
    for sub in ("tarballs", "provenance", "datasets/packages", "logs"):
        (BIOC_STAGING / sub).mkdir(parents=True, exist_ok=True)
    NORM_TREE.mkdir(parents=True, exist_ok=True)
    log = {"package": pkg, "t0": time.time(), "stages": []}
    provp = BIOC_STAGING / "provenance" / f"{pkg}.json"
    ver = idx_meta.get("Version", "UNKNOWN")
    tb = BIOC_STAGING / "tarballs" / f"{pkg}_{ver}.tar.gz"
    base_ledger = dict(package=pkg, version=ver,
                       license_index_field=idx_meta.get("License", ""), ts=now())
    try:
        # 1. local working copy of the tarball: everything below reads THIS file,
        #    so each package costs at most ONE sequential SMB read + one SMB write
        #    (tarball reads over SMB were the dominant per-package cost)
        local_tb = Path(tempfile.gettempdir()) / f"bioc_ingest_{pkg}.tar.gz"
        local_root = None
        if not (tb.exists() and tb.stat().st_size > 0):
            ok, detail = http_download(f"{MIRROR}/{pkg}_{ver}.tar.gz", local_tb)
            if not ok:
                local_tb.unlink(missing_ok=True)
                decision = "no-tarball" if detail == "http-404" else "error"
                ledger_write({**base_ledger, "decision": decision, "reason": detail,
                              "included": False, "license_class": "", "license_field": ""})
                log["error"] = f"download {detail}"
                return {"package": pkg, "status": f"{decision}: {detail}"}
            shutil.copyfile(local_tb, tb)          # immutable original -> NAS mirror
        else:
            shutil.copyfile(tb, local_tb)          # resume path: NAS -> local
        log["stages"].append("download")

        # 2. ONE tarfile scan of the local copy: names + DESCRIPTION + LICENSE
        with tarfile.open(local_tb) as tf:
            names = tf.getnames()
            desc_member = next((m for m in names
                                if m.endswith("/DESCRIPTION") and m.count("/") == 1), None)
            if desc_member is None:
                raise ValueError("no top-level DESCRIPTION in tarball")
            desc = tf.extractfile(desc_member).read().decode("utf-8", errors="replace")
            lic_text = None
            for cand in (f"{pkg}/LICENSE", f"{pkg}/LICENSE.md", f"{pkg}/LICENCE",
                         f"{pkg}/LICENCE.md"):
                if cand in names:
                    m = tf.getmember(cand)
                    if m.isfile() and m.size < 1_000_000:
                        lic_text = tf.extractfile(m).read().decode("utf-8", "replace")
                        break
        d = parse_description(desc)
        ver = d.get("Version", ver)  # DESCRIPTION wins (same convention as CRAN)
        tb_final = BIOC_STAGING / "tarballs" / f"{pkg}_{ver}.tar.gz"
        if ver != idx_meta.get("Version") and tb.name != tb_final.name:
            tb.rename(tb_final)
            tb = tb_final
        tbsha = sha256(local_tb)
        log["stages"].append("provenance")

        # 3. license decision (field + LICENSE-file inspection where present)
        decision, lclass, flag, reason = classify_license(d.get("License", ""), lic_text)
        if decision == "include" and lic_text:
            override = license_file_scan(lic_text)
            if override:  # NC/ND in the actual file beats a permissive field
                decision, lclass, flag, reason = override
        ledger_write({**base_ledger, "decision": decision,
                      "included": decision == "include", "license_class": lclass,
                      "license_field": d.get("License", "UNKNOWN"), "flag": flag,
                      "reason": reason, "tarball_sha256": tbsha,
                      "tarball_bytes": local_tb.stat().st_size})
        log["stages"].append("license")

        if decision == "exclude":
            provp.write_text(json.dumps(dict(
                package=pkg, version=ver, source="bioc", included=False,
                license=d.get("License", "UNKNOWN"), license_class=lclass,
                exclusion_reason=reason, author=d.get("Author", "")[:500],
                maintainer=d.get("Maintainer", "")[:300], upstream=d.get("URL", ""),
                tarball=tb.name, tarball_sha256=tbsha,
                ingested_at=now(), rank=idx), indent=1))
            return {"package": pkg, "status": f"excluded ({reason})",
                    "license": d.get("License", "?")}

        # 4. extract LOCALLY, normalize + inventory locally, then ship the finished
        #    tree to the NAS in ONE bulk tar pipe — SMB per-file round trips were
        #    the dominant per-package cost. The NAS tree is byte-identical to what
        #    extract-to-NAS + air/jarl-on-NAS would produce (CRAN-store shape).
        bad = [m for m in names
               if m.startswith(("/", "..")) or "/../" in m or m == ".."]
        if bad:
            raise ValueError(f"unsafe member paths in tarball: {bad[:3]}")
        local_root = Path(tempfile.mkdtemp(prefix=f"bioc_x_{pkg}_"))
        subprocess.run(["tar", "-xzf", str(local_tb), "-C", str(local_root)],
                       check=True, timeout=600)
        src = next(p for p in local_root.iterdir() if p.is_dir())
        log["stages"].append("extract")

        # 5. normalize: air format + jarl check --fix (identical to CRAN ingest)
        before = {str(f): f.read_bytes() for f in src.rglob("*")
                  if f.is_file() and f.suffix in (".R", ".Rmd", ".qmd")}
        air = subprocess.run(["air", "format", str(src)], capture_output=True,
                             text=True, timeout=300)
        jarl = subprocess.run(["jarl", "check", "--fix", "--allow-no-vcs", str(src)],
                              capture_output=True, text=True, timeout=300)
        changed = sum(1 for p, b in before.items() if Path(p).read_bytes() != b)
        log.update(air_exit=air.returncode, jarl_exit=jarl.returncode,
                   files_seen=len(before), files_changed_by_normalize=changed)
        (BIOC_STAGING / "logs" / f"{pkg}.log").write_text(
            f"AIR STDOUT\n{air.stdout}\nAIR STDERR\n{air.stderr[:4000]}\n"
            f"JARL STDOUT\n{jarl.stdout[:12000]}\nJARL STDERR\n{jarl.stderr[:4000]}")
        log["stages"].append("normalize")

        # 6. inventory shard (same file scope as CRAN: .R/.Rmd/.qmd/.Rd), read locally
        records = []
        for f in sorted(src.rglob("*")):
            if not f.is_file() or f.suffix not in (".R", ".Rmd", ".qmd", ".Rd"):
                continue
            rel = str(f.relative_to(src))
            body = f.read_bytes()
            records.append(dict(
                source="bioc", package=pkg, version=ver,
                license=d.get("License", "UNKNOWN"), upstream=d.get("URL", d.get("BugReports", "")),
                author=d.get("Author", "")[:300], path=rel,
                area=rel.split("/")[0] if "/" in rel else ".", ext=f.suffix,
                sha256=hashlib.sha256(body).hexdigest(), bytes=len(body),
                n_lines=body.count(b"\n"), normalized=True))
        shardp = BIOC_STAGING / "datasets" / "packages" / f"{pkg}.jsonl"
        shardp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        log["stages"].append("inventory")

        # 6.5 ship the normalized tree -> normalized_bioc/<pkg>/<ver>/<pkg>/...
        # (done BEFORE provenance/manifest so a crash mid-ship leaves no
        #  done-marker -> clean retry on resume)
        ndir = NORM_TREE / pkg / ver
        if ndir.exists():
            subprocess.run(["rm", "-rf", str(ndir)], check=True)
        ndir.mkdir(parents=True, exist_ok=True)
        ship = subprocess.run(
            ["bash", "-c",
             f"tar -cf - -C {shlex.quote(str(local_root))} {shlex.quote(src.name)} "
             f"| tar -xf - -C {shlex.quote(str(ndir))}"],
            timeout=1800)
        if ship.returncode != 0:
            raise ValueError(f"tree ship failed (exit {ship.returncode})")
        log["stages"].append("ship")

        # 7. provenance + manifest (Bioc staging manifest only; CRAN manifest untouched)
        prov = dict(package=pkg, version=ver, source="bioc", included=True,
                    license=d.get("License", "UNKNOWN"), license_class=lclass,
                    license_flag=flag, author=d.get("Author", "")[:500],
                    maintainer=d.get("Maintainer", "")[:300],
                    upstream=d.get("URL", ""), bug_reports=d.get("BugReports", ""),
                    published=d.get("Date/Publication", d.get("git_url", "")),
                    tarball=tb.name, tarball_sha256=tbsha,
                    n_files=len(records), n_bytes=sum(r["bytes"] for r in records),
                    files_changed_by_normalize=changed,
                    ingested_at=now(), rank=idx)
        provp.write_text(json.dumps(prov, indent=1))
        with open(MANIFEST, "a") as mf:
            mf.write(json.dumps({k: prov[k] for k in
                                 ("package", "version", "license", "license_class",
                                  "n_files", "n_bytes", "tarball_sha256",
                                  "ingested_at", "rank")} | {"source": "bioc"}) + "\n")
        log["stages"].append("manifest")
        return {"package": pkg, "status": "ok", "files": len(records),
                "mb": round(prov["n_bytes"] / 1e6, 1), "license": prov["license"],
                "normalized_changed": changed}
    except Exception as e:
        log["error"] = str(e)[:300]
        ledger_write({**base_ledger, "decision": "error", "included": False,
                      "license_class": "", "license_field": idx_meta.get("License", ""),
                      "reason": str(e)[:200]})
        return {"package": pkg, "status": f"error: {str(e)[:150]}"}
    finally:
        local_tb.unlink(missing_ok=True)   # local scratch (NAS originals stay)
        if local_root:
            shutil.rmtree(local_root, ignore_errors=True)
        log["elapsed_s"] = round(time.time() - log["t0"], 1)
        with open(BIOC_STAGING / "logs" / "ingest.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main():
    for sub in ("tarballs", "provenance", "datasets/packages", "logs"):
        (BIOC_STAGING / sub).mkdir(parents=True, exist_ok=True)
    NORM_TREE.mkdir(parents=True, exist_ok=True)

    # index: reuse a fresh (<24h) cache, else one polite fetch
    if INDEX_CACHE.exists() and time.time() - INDEX_CACHE.stat().st_mtime < 86400:
        idx = json.loads(INDEX_CACHE.read_text())
        print(f"index cache: {len(idx)} packages", flush=True)
    else:
        polite_pause(); mark_request()
        req = urllib.request.Request(PACKAGES_GZ, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = gzip.decompress(r.read()).decode("utf-8", errors="replace")
        idx, cur = {}, {}
        for line in raw.splitlines():
            if not line.strip():
                if cur.get("Package"):
                    idx[cur["Package"]] = cur
                cur = {}; continue
            if ": " in line and line[0] not in " \t":
                k, v = line.split(": ", 1); cur[k] = v
        if cur.get("Package"):
            idx[cur["Package"]] = cur
        INDEX_CACHE.write_text(json.dumps(idx))
        print(f"index fetched: {len(idx)} packages", flush=True)

    names = sorted(idx)  # deterministic order -> deterministic resume
    already = done_pkgs()
    todo = [n for n in names if n not in already]
    print(f"{len(names)} in index, {len(already)} already done, {len(todo)} to go",
          flush=True)

    statef = BIOC_STAGING / "logs" / "state.json"
    counts = {"ok": 0, "excluded": 0, "error": 0, "no-tarball": 0}
    t0 = time.time()
    for i, pkg in enumerate(todo, 1):
        r = process(pkg, names.index(pkg), len(names), idx[pkg])
        st = r["status"].split(":")[0]
        key = "excluded" if st.startswith("excluded") else st
        counts[key] = counts.get(key, 0) + 1
        state = dict(updated=now(), processed=i, total_todo=len(todo),
                     index_size=len(names), counts=counts,
                     last_package=pkg, elapsed_s=round(time.time() - t0, 1))
        statef.write_text(json.dumps(state))
        print(json.dumps(r), flush=True)
        if i % PROGRESS_EVERY == 0:
            eta = (time.time() - t0) / i * (len(todo) - i)
            with open(BIOC_STAGING / "logs" / "progress.log", "a") as f:
                f.write(f"{now()} {i}/{len(todo)} {json.dumps(counts)} "
                        f"eta_h={eta/3600:.2f}\n")
            print(f"PROGRESS {i}/{len(todo)} {json.dumps(counts)} eta_h={eta/3600:.2f}",
                  flush=True)
    print(f"RUN COMPLETE {json.dumps(counts)} total_s={time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
