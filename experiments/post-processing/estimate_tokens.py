#!/usr/bin/env python3
"""Token-budget estimate for the R corpus (Route A sizing decision).

Stratified sample of files per area -> exact token counts via llama-tokenize
(Qwen tokenizer, our serving stack's tokenizer) -> tokens/byte per area ->
scaled by exact per-area bytes from the shards. man/ excluded (roxygen-derived).
"""
import json, random, re, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path("/mnt/h/sepalith")
TOK = "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-tokenize"
GGUF = "/home/m0hawk/Documents/Sepalith/experiments/models/qwen0.5b-q8_0.gguf"
SAMPLE = {"R": 800, "tests": 400, "vignettes": 150, "src": 250, "inst": 100}

records = []
bad = 0
for shard in (ROOT / "datasets" / "packages").glob("*.jsonl"):
    pkg = shard.stem
    for line in shard.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        if r.get("area") in SAMPLE and 500 <= r.get("bytes", 0) <= 2_000_000:
            p = ROOT / "normalized" / pkg / r["version"] / pkg / r["path"]
            if p.exists():
                try:
                    head = p.open("rb").read(1024)
                    if b"\x00" in head:
                        continue
                except Exception:
                    continue
                records.append((r["area"], p, r["bytes"], pkg))
sys.stderr.write(f"candidate files: {len(records)} (bad lines skipped: {bad})\n")


by_area = {}
for a, p, b, pkg in records:
    by_area.setdefault(a, []).append((p, b, pkg))
total_bytes = {a: sum(b for _, b, _ in v) for a, v in by_area.items()}

rng = random.Random(20260817)
sample = []
for a, n in SAMPLE.items():
    pool = by_area.get(a, [])
    sample += [(a,) + t for t in (rng.sample(pool, min(n, len(pool))) if pool else [])]

def count(t):
    area, p, b, pkg = t
    try:
        out = subprocess.run([TOK, "-m", GGUF, "-f", str(p), "--no-escape"],
                             capture_output=True, text=True, timeout=60).stdout
        n = sum(1 for l in out.splitlines() if " -> " in l)  # one line per token
        return (area, b, n, pkg)
    except Exception:
        return (area, b, 0, pkg)

with ThreadPoolExecutor(max_workers=8) as ex:
    counted = list(ex.map(count, sample))

agg = {}
for area, b, tok, _ in counted:
    a = agg.setdefault(area, {"n": 0, "bytes": 0, "tokens": 0})
    a["n"] += 1; a["bytes"] += b; a["tokens"] += tok

print(f"{'area':11s} {'files':>6s} {'tok/byte':>9s} {'GB(sample)':>10s} "
      f"{'est_tokens':>12s}  {'full-GB':>8s}")
grand = 0
for area in sorted(agg):
    a = agg[area]
    tpb = a["tokens"] / max(a["bytes"], 1)
    est = tpb * total_bytes[area]
    grand += est
    print(f"{area:11s} {a['n']:6d} {tpb:9.4f} {a['bytes']/1e9:10.3f} "
          f"{est/1e9:11.3f}B  {total_bytes[area]/1e9:8.3f}")
print(f"\nTOTAL (excl man/, top-{len(set(p for _,_,p in [(x[1],x[2],x[3]) for x in sample]))} pkgs sample): "
      f"~{grand/1e9:.2f}B tokens from {sum(total_bytes.values())/1e9:.2f} GB source")

# extrapolations
n_pkgs = len(list((ROOT / 'provenance').glob('*.json')))
mean_bytes = sum(total_bytes.values()) / n_pkgs
for label, factor in [("full CRAN @ mean 1.0x top-500", 1.0),
                      ("full CRAN @ mean 0.5x (size skew)", 0.5),
                      ("+ all archive versions @2.5x", 1.25)]:
    est_bytes = mean_bytes * 24500 * factor
    tpb = grand / max(sum(total_bytes.values()), 1)
    print(f"{label:36s}: ~{est_bytes/1e9:.2f} GB -> ~{est_bytes*tpb/1e9:.2f}B tokens")
