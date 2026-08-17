#!/usr/bin/env python3
"""CRAN ingestion v0 — top-downloaded packages -> NAS-backed dataset shards.

Layout under $ROOT (default /mnt/h/sepalith):
  tarballs/<pkg>_<ver>.tar.gz          immutable originals (re-derivation + evidence)
  normalized/<pkg>/<ver>/              post air-format + jarl --fix tree
  provenance/<pkg>.json                license, authors, repo, dates, file inventory
  datasets/packages/<pkg>.jsonl        ONE file per package -> takedown = delete file
  logs/<pkg>.log                       air/jarl telemetry (what normalization changed)

Respect-by-design: nothing leaves the NAS until provenance exists; every record
carries package, version, license, upstream URL, and file sha256.
"""
import gzip, hashlib, io, json, os, subprocess, sys, tarfile, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("SEPALITH_ROOT", "/mnt/h/sepalith"))
MIRROR = "https://cloud.r-project.org/src/contrib"
CRANLOGS = "https://cranlogs.r-pkg.org/top/last-month/{}"
PACKAGES_GZ = "https://cran.r-project.org/src/contrib/PACKAGES.gz"
MAX_TARBALL = 120 * 1024 * 1024   # skip monster packages in v0
SKIP_PKGS = {"BH"}                # header-only/no R content; extraction cost >> value


def http(url, dest=None, binary=False, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sepalith-pipeline/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if dest is not None:
                Path(dest).write_bytes(data)
                return len(data)
            return data if binary else data.decode("utf-8", errors="replace")
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def parse_packages_index():
    """CRAN's authoritative PACKAGES db -> {pkg: {Version, License, ...}}."""
    raw = gzip.decompress(http(PACKAGES_GZ, binary=True)).decode("utf-8", errors="replace")
    db, cur = {}, {}
    for line in raw.splitlines():
        if not line.strip():
            if cur.get("Package"):
                db[cur["Package"]] = cur
            cur = {}
            continue
        if ": " in line and line[0] not in " \t":
            k, v = line.split(": ", 1)
            cur[k] = v
    if cur.get("Package"):
        db[cur["Package"]] = cur
    return db


def top_packages(n):
    """cranlogs caps at 100/call; for larger n use an exact mirror-log ranking."""
    if n <= 100:
        data = json.loads(http(CRANLOGS.format(n)))
        return [d["package"] for d in data["downloads"]]
    ranked = sorted((ROOT / "ranked").glob("*.counts.txt"))[-1]
    pkgs = [l.split()[1] for l in ranked.read_text().splitlines()
            if len(l.split()) == 2][:n]
    print(f"ranking source: {ranked.name} ({len(pkgs)} packages)")
    return pkgs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_from_description(desc_text):
    """Parse the authoritative DESCRIPTION from inside the tarball."""
    fields = {}
    last = None
    for line in desc_text.splitlines():
        if line.startswith(" ") and last:
            fields[last] += " " + line.strip()
        elif ": " in line or ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            fields[k] = v
            last = k
    return fields


def ingest(pkg, idx):
    log = {"package": pkg, "t0": time.time(), "stages": []}
    logf = ROOT / "logs" / f"{pkg}.log"
    provp = ROOT / "provenance" / f"{pkg}.json"
    shardp = ROOT / "datasets" / "packages" / f"{pkg}.jsonl"
    if pkg in SKIP_PKGS:
        return {"package": pkg, "status": "skipped (skip-list)"}
    if provp.exists() and shardp.exists():
        return {"package": pkg, "status": "already-done"}
    try:
        if pkg not in IDX:
            raise KeyError(f"{pkg} not in CRAN index (archived?)")
        ver = IDX[pkg]["Version"]
        url = f"{MIRROR}/{pkg}_{ver}.tar.gz"
        tb = ROOT / "tarballs" / f"{pkg}_{ver}.tar.gz"
        if not tb.exists():
            n = http(url, dest=tb)
            if n > MAX_TARBALL:
                tb.unlink(missing_ok=True)
                raise ValueError(f"tarball too big ({n} bytes)")
        log["stages"].append("download")

        # provenance from the authoritative source: DESCRIPTION inside the tarball
        with tarfile.open(tb) as tf:
            desc_member = next((m for m in tf.getnames()
                                if m.endswith("/DESCRIPTION") and m.count("/") == 1), None)
            desc = tf.extractfile(desc_member).read().decode("utf-8", errors="replace")
        d = provenance_from_description(desc)
        log["stages"].append("provenance")

        # extract to normalized/ (raw stays in the tarball; we keep only normalized trees)
        ndir = ROOT / "normalized" / pkg / ver
        if ndir.exists():
            subprocess.run(["rm", "-rf", str(ndir)], check=True)
        ndir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tb) as tf:
            tf.extractall(ndir, filter="data")
        src = next(p for p in ndir.iterdir() if p.is_dir())
        log["stages"].append("extract")

        # normalize: air format then jarl check --fix, with telemetry
        before = {}
        for f in src.rglob("*"):
            if f.is_file() and f.suffix in (".R", ".Rmd", ".qmd"):
                before[str(f)] = f.read_bytes()
        air = subprocess.run(["air", "format", str(src)], capture_output=True, text=True)
        jarl = subprocess.run(["jarl", "check", "--fix", "--allow-no-vcs", str(src)],
                              capture_output=True, text=True)
        changed = sum(1 for p, b in before.items() if Path(p).read_bytes() != b)
        log.update(air_exit=air.returncode, jarl_exit=jarl.returncode,
                   files_seen=len(before), files_changed_by_normalize=changed)
        logf.parent.mkdir(parents=True, exist_ok=True)
        logf.write_text(f"AIR STDOUT\n{air.stdout}\nAIR STDERR\n{air.stderr[:4000]}\n"
                        f"JARL STDOUT\n{jarl.stdout[:12000]}\nJARL STDERR\n{jarl.stderr[:4000]}")
        log["stages"].append("normalize")

        # inventory: one JSONL record per source file, one shard file per package
        records = []
        for f in sorted(src.rglob("*")):
            if not f.is_file() or f.suffix not in (".R", ".Rmd", ".qmd", ".Rd"):
                continue
            rel = str(f.relative_to(src))
            body = f.read_bytes()
            records.append(dict(
                source="cran", package=pkg, version=ver, license=d.get("License", "UNKNOWN"),
                upstream=d.get("URL", d.get("BugReports", "")), author=d.get("Author", "")[:300],
                path=rel, area=rel.split("/")[0] if "/" in rel else ".", ext=f.suffix,
                sha256=hashlib.sha256(body).hexdigest(), bytes=len(body),
                n_lines=body.count(b"\n"), normalized=True))
        shardp.parent.mkdir(parents=True, exist_ok=True)
        shardp.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        prov = dict(package=pkg, version=ver, license=d.get("License", "UNKNOWN"),
                    license_partial=("file LICENSE" in d.get("License", "")),
                    author=d.get("Author", "")[:500],
                    maintainer=d.get("Maintainer", "")[:300],
                    upstream=d.get("URL", ""), bug_reports=d.get("BugReports", ""),
                    published=d.get("Date/Publication", ""), tarball=tb.name,
                    tarball_sha256=sha256(tb), n_files=len(records),
                    n_bytes=sum(r["bytes"] for r in records),
                    ingested_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    rank=idx)
        provp.write_text(json.dumps(prov, indent=1))
        with open(ROOT / "datasets" / "manifest.jsonl", "a") as mf:
            mf.write(json.dumps({k: prov[k] for k in
                                 ("package", "version", "license", "n_files", "n_bytes",
                                  "tarball_sha256", "ingested_at", "rank")}) + "\n")
        log["stages"].append("inventory")
        return {"package": pkg, "status": "ok", "files": len(records),
                "mb": round(prov["n_bytes"] / 1e6, 1), "normalized_changed": changed,
                "license": prov["license"]}
    except Exception as e:
        log["error"] = str(e)[:300]
        return {"package": pkg, "status": f"error: {str(e)[:200]}"}
    finally:
        log["elapsed_s"] = round(time.time() - log["t0"], 1)
        (ROOT / "logs" / "ingest.jsonl").parent.mkdir(parents=True, exist_ok=True)
        with open(ROOT / "logs" / "ingest.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    for sub in ("tarballs", "normalized", "provenance", "datasets/packages", "logs"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    print(f"fetching CRAN PACKAGES index...", flush=True)
    IDX = parse_packages_index()
    print(f"index: {len(IDX)} packages", flush=True)
    top = top_packages(n)
    print(f"ingesting top {len(top)} by downloads (last month)...", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(lambda t: ingest(t[1], t[0]), list(enumerate(top))):
            results.append(r)
            print(json.dumps(r), flush=True)
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\ndone: {len(ok)}/{len(results)} ok, "
          f"{sum(r['files'] for r in ok)} files, {sum(r['mb'] for r in ok)}MB source")
