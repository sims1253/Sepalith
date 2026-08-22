#!/usr/bin/env python3
"""paper_code_harvester — continuous paper->R-code harvester (Sepalith).

DESIGN (see docs/research/data-acquisition-scout.md for pilot numbers):
The arXiv-LaTeX pilot showed papers embed runnable R in listings/verbatim at
only ~2-5% density with ~100-600 tokens per hit — weak as a VOLUME source but
valid as a permissively-licensed trickle. The high-yield path is arXiv as a
DISCOVERY layer: paper abs pages / PDFs link GitHub repos; R repos are fetched
with a license gate (LICENSE file detection), which is where the tokens are.

Modes (config via --mode, both resume-safe, detached-runnable):
  blocks : polite polling of NEW arXiv submissions in configured categories,
           e-print tarball download, LaTeX code-block extraction, tree-sitter
           parse gate, output to OUTDIR/blocks/<ymd>/*.R + ledger append.
  repos  : for the same papers, scrape the abs page HTML for github.com links,
           gate the repo's detected license against PERMISSIVE set, shallow
           clone / tarball into OUTDIR/repos/<owner>__<name>/, ledger append.

LICENSE RULE (user ruling 2026-08-20): NO NC, NO ND, no ambiguous defaults.
arXiv per-paper license: the Atom API does not expose it; the abs page does
("License (or license to distribute)"). Papers under the default
"non-exclusive license to distribute" are counted EXCLUDED in the ledger
(ambiguous grant) unless a CC-BY/CC0 marker is present on the abs page.
Extracted LaTeX code from excluded papers is recorded but not written to the
training corpus directory (goes to OUTDIR/excluded/ for audit only).

Politeness: >=3.5s between arXiv requests, single-flight, resume-safe (state
file with seen ids), exits cleanly on SIGTERM. Config: category weights.
"""
from __future__ import annotations
import argparse, json, os, re, signal, subprocess, sys, tarfile, time
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "sepalith-harvester/0.1 (research; polite; contact via repo)"}
PERMISSIVE_MARKERS = re.compile(
    r"(CC[ -]?BY(?!-NC)(?!-ND)|CC0|Creative Commons Zero|Apache License|MIT License|GNU (GPL|LGPL|AGPL)|Artistic License)", re.I)
REPO_PERMISSIVE = {"mit", "apache-2.0", "gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0",
                   "agpl-3.0", "bsd-3-clause", "bsd-2-clause", "mpl-2.0", "artistic-2.0", "unlicense", "cc0-1.0"}

DEFAULT_CONFIG = {
    # category: poll weight (requests per cycle roughly proportional)
    "categories": {"stat.AP": 3, "stat.ME": 3, "stat.ML": 2, "econ.EM": 3,
                   "q-bio.GN": 2, "q-bio.QM": 2, "cs.LG": 1, "astro-ph.IM": 1},
    "poll_seconds": 6 * 3600,   # arXiv new-submission cadence; no need to hammer
    "n_per_poll": 50,
    "request_gap": 3.5,
}

class Harvester:
    def __init__(self, outdir, mode, config=None):
        self.out = outdir
        self.mode = mode
        self.cfg = config or DEFAULT_CONFIG
        self.state_path = os.path.join(outdir, "state.json")
        os.makedirs(outdir, exist_ok=True)
        self.state = {"seen": {}}
        if os.path.exists(self.state_path):
            self.state = json.load(open(self.state_path))
        self.ledger = open(os.path.join(outdir, "license_ledger.jsonl"), "a")
        self._stop = False
        signal.signal(signal.SIGTERM, lambda *a: setattr(self, "_stop", True))

    def _get(self, url, dest=None, retries=3):
        for i in range(retries):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if dest:
                    with open(dest, "wb") as f:
                        f.write(data)
                return data
            except Exception:
                if i == retries - 1:
                    return None
                time.sleep(15)
        return None

    def new_ids(self, cat, n):
        url = ("https://export.arxiv.org/api/query?search_query=cat:{}"
               "&sortBy=submittedDate&sortOrder=descending&max_results={}")
        url = url.format(cat, n + 20)  # overfetch, skip seen
        data = self._get(url)
        time.sleep(self.cfg["request_gap"])
        if not data:
            return []
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in ET.fromstring(data).findall("a:entry", ns):
            eid = e.find("a:id", ns).text.split("/abs/")[-1]
            if eid not in self.state["seen"]:
                out.append(eid)
            if len(out) >= n:
                break
        return out

    def paper_license(self, pid):
        """Abs-page license read. Returns 'permissive' | 'excluded-default' | 'unknown'."""
        html = self._get(f"https://arxiv.org/abs/{pid}")
        time.sleep(self.cfg["request_gap"])
        if not html:
            return "unknown"
        html = html.decode("utf-8", "replace")
        m = re.search(r"(?:License|license to distribute)[^<]*</td>\s*<td[^>]*>(.*?)</td>", html, re.S)
        blob = m.group(1) if m else ""
        if re.search(r"CC[ -]?BY(?!-NC)(?!-ND)|CC0|creativecommons", blob, re.I) and not re.search(r"NC|ND", blob):
            return "permissive"
        if "non-exclusive" in blob or not blob.strip():
            return "excluded-default"
        return "unknown"

    def extract_blocks(self, pid, lic):
        src = os.path.join(self.out, "eprint", pid.replace("/", "_") + ".src")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        if not os.path.exists(src):
            if self._get(f"https://arxiv.org/e-print/{pid}", dest=src) is None:
                return []
            time.sleep(self.cfg["request_gap"])
        texs = []
        try:
            with tarfile.open(src) as tf:
                for m in tf.getmembers():
                    if m.name.endswith((".tex", ".Rnw")) and m.size < 2_000_000:
                        texs.append(tf.extractfile(m).read().decode("utf-8", "replace"))
        except Exception:
            return []
        blocks = []
        pat = [(r"<<[^>]*>>=\s*\n(.*?)\n@", "sweave"),
               (r"\\begin\{lstlisting\}(?:\[[^\]]*\])?(.*?)\\end\{lstlisting\}", "lstlisting"),
               (r"\\begin\{(?:verbatim|Verbatim|alltt|minted)\}(?:\[[^\]]*\])?(?:\{[^}]*\})?(.*?)\\end\{(?:verbatim|Verbatim|alltt|minted)\}", "verbatim")]
        for tex in texs:
            for rx, kind in pat:
                for m in re.finditer(rx, tex, re.S):
                    blocks.append((m.group(1), kind))
        return blocks

    def repo_links(self, pid):
        html = self._get(f"https://arxiv.org/abs/{pid}")
        time.sleep(self.cfg["request_gap"])
        if not html:
            return []
        return sorted(set(re.findall(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
                                     html.decode("utf-8", "replace"))))

    def gate_repo(self, owner_repo):
        try:
            meta = subprocess.run(["gh", "api", f"/repos/{owner_repo}"],
                                  capture_output=True, text=True, timeout=60)
            d = json.loads(meta.stdout)
            lic = (d.get("license") or {}).get("key") or "none"
            lang = d.get("language")
            return lic in REPO_PERMISSIVE, lic, lang
        except Exception:
            return False, "error", None

    def fetch_repo(self, owner_repo, dest):
        os.makedirs(dest, exist_ok=True)
        r = subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{owner_repo}.git", dest],
                           capture_output=True, text=True, timeout=600)
        return r.returncode == 0

    def run(self):
        while not self._stop:
            for cat, weight in self.cfg["categories"].items():
                if self._stop:
                    break
                n = self.cfg["n_per_poll"] * weight // max(self.cfg["categories"].values())
                for pid in self.new_ids(cat, max(2, n)):
                    if self._stop:
                        break
                    lic = self.paper_license(pid)
                    rec = {"ts": time.strftime("%FT%T"), "id": pid, "cat": cat, "license": lic}
                    if self.mode in ("blocks", "both") and lic == "permissive":
                        for code, kind in self.extract_blocks(pid, lic):
                            if len(code.strip()) >= 120:
                                d = os.path.join(self.out, "blocks", time.strftime("%Y%m"), cat)
                                os.makedirs(d, exist_ok=True)
                                fn = os.path.join(d, pid.replace("/", "_") + f".{kind}.R")
                                with open(fn, "w") as f:
                                    f.write(f"# source: arxiv:{pid} cat:{cat} env:{kind} license:{lic}\n" + code)
                                rec.setdefault("blocks", []).append(fn)
                    if self.mode in ("repos", "both"):
                        links = self.repo_links(pid)
                        rec["repos"] = []
                        for orp in links[:4]:
                            ok, lic_key, lang = self.gate_repo(orp)
                            if ok and lang == "R":
                                dest = os.path.join(self.out, "repos", orp.replace("/", "__"))
                                if not os.path.exists(os.path.join(dest, ".git")) and self.fetch_repo(orp, dest):
                                    rec["repos"].append({"repo": orp, "license": lic_key})
                    self.ledger.write(json.dumps(rec) + "\n")
                    self.ledger.flush()
                    self.state["seen"][pid] = time.strftime("%FT%T")
                    json.dump(self.state, open(self.state_path, "w"))
            time.sleep(self.cfg["poll_seconds"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/mnt/h/sepalith/datasets/paper_code/")
    ap.add_argument("--mode", choices=["blocks", "repos", "both"], default="both")
    args = ap.parse_args()
    h = Harvester(args.outdir, args.mode)
    h.run()

if __name__ == "__main__":
    main()
