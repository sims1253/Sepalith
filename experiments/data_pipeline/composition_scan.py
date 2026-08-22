#!/usr/bin/env python3
"""Gate 2 supplement: per-extension token composition of the normalized
CRAN corpus, splitting RENDERED output (inst/doc/*.html, rendered .R) from
sources, so the pretraining-corpus cut is exact. Tokens: qwen3.5 tokenizer.
100-package seeded sample, independent of the minhash sample."""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

CORPUS = Path("/mnt/h/sepalith/normalized")
tok = AutoTokenizer.from_pretrained(
    "/home/m0hawk/Documents/Sepalith/experiments/models/qwen3.5-2b-base-text-hf")

TEXT_EXT = {".R", ".r", ".Rd", ".Rmd", ".Rnw", ".qmd", ".md", ".txt",
            ".cpp", ".c", ".cc", ".h", ".hpp", ".sql", ".py", ".sh",
            ".yaml", ".yml", ".toml", ".cfg", ".html", ".css", ".js"}
TEXT_NAMES = {"DESCRIPTION", "NAMESPACE", "LICENSE", "LICENCE", "NEWS",
              "NEWS.md", "CITATION", "AUTHORS", "README", "README.md"}

def bucket(f: Path):
    sfx = f.suffix if f.suffix else f.name
    rendered = "inst/doc" in f.parts or "inst" in f.parts and "doc" in f.parts
    if sfx in {".R", ".r"}:
        return "r_rendered" if rendered else "R"
    if sfx == ".Rd":
        return "Rd"
    if sfx in {".Rmd", ".Rnw", ".qmd"}:
        return "rmd_src" if not rendered else "rmd_rendered"
    if sfx in {".cpp", ".c", ".cc", ".h", ".hpp"}:
        return "C_src" if "src" in f.parts else "C_elsewhere"
    if sfx in {".html"}:
        return "html"
    if sfx in {".md", ".txt"}:
        return "md_txt"
    if sfx in TEXT_NAMES:
        return "meta"
    return "other_text"

packages = sorted(p.name for p in CORPUS.iterdir() if p.is_dir())
sample = sorted(random.Random(43).sample(packages, 100))
agg = defaultdict(lambda: [0, 0])   # bucket -> [tokens, bytes]
per_pkg = []
for pkg in sample:
    pkg_tok = defaultdict(int)
    for f in (CORPUS / pkg).rglob("*"):
        if not f.is_file() or f.is_symlink():
            continue
        sfx = f.suffix
        if sfx not in TEXT_EXT and f.name not in TEXT_NAMES:
            continue
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        text = raw.decode("utf-8", "replace")
        if len(text) < 32:
            continue
        nt = len(tok(text, add_special_tokens=False)["input_ids"])
        b = bucket(f)
        agg[b][0] += nt
        agg[b][1] += len(raw)
        pkg_tok[b] += nt
    per_pkg.append(dict(package=pkg, **{k: v for k, v in pkg_tok.items()
                                        if v}))

N = len(packages)
tot = sum(v[0] for v in agg.values())
out = dict(n_sample=len(sample), extrapolated=dict(
    total_tokens=tot / len(sample) * N,
    by_bucket={k: dict(tokens=v[0] / len(sample) * N,
                       bytes=v[1] / len(sample) * N,
                       share=round(v[0] / tot, 4))
               for k, v in sorted(agg.items(), key=lambda x: -x[1][0])}))
Path("/home/m0hawk/Documents/Sepalith/experiments/eval/"
     "results_minhash_composition.json").write_text(json.dumps(out, indent=1))
with open("/home/m0hawk/Documents/Sepalith/experiments/eval/"
          "results_minhash_composition_perpkg.jsonl", "w") as f:
    for r in per_pkg:
        f.write(json.dumps(r) + "\n")
print(json.dumps(out["extrapolated"], indent=1))
