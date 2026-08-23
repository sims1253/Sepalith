#!/usr/bin/env python3
"""Stack v2 R acquisition — Phase D: measure + final report.

  - 300-file seeded sample of the NET-NEW keep set (from phase C keep ids +
    staged shards), tokenized with the local qwen3.5-2b tokenizer (CPU);
    mean tokens/byte -> net-new token estimate with bootstrap CI.
  - storage accounting (du of staging tree).
  - ledger summary (final counts per license class + exclusion reasons).
  - measure_report.json + merge checklist (analogous to bioc's).
"""
import json, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

STAGE = Path("/mnt/h/sepalith/stack_staging")
FILES, LOGS = STAGE / "files", STAGE / "logs"
QWEN = "/home/m0hawk/Documents/Sepalith/experiments/models/qwen3.5-2b-base-text-hf"
N_SAMPLE = 300
SEED = 20260822


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)

    keep = [json.loads(l) for l in open(STAGE / "datasets" / "stack_keep_ids.jsonl")]
    print(f"keep set: {len(keep):,} files", flush=True)
    keep_blobs = {r["blob_id"] for r in keep}
    rng = random.Random(SEED)
    sample_ids = set(r["blob_id"] for r in rng.sample(keep, min(N_SAMPLE, len(keep))))

    toks = []
    sampled_bytes = 0
    by_cfg_tokens = defaultdict(list)
    chunk_re = re.compile(r"```[\w-]*\{[rR]", re.I)
    rmd_pt = rmd_ct = rmd_pb = rmd_cb = 0
    for shard in sorted(FILES.glob("shard-*.jsonl")):
        with open(shard, encoding="utf-8", errors="replace") as f:
            for line in f:
                if len(sample_ids) == 0:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("blob_id") in sample_ids:
                    sample_ids.discard(rec["blob_id"])
                    n = len(tok.encode(rec.get("content", "")))
                    toks.append((n, rec.get("bytes", 0)))
                    by_cfg_tokens[rec.get("config", "?")].append(
                        (n, rec.get("bytes", 0)))
                    if rec.get("config") == "RMarkdown":
                        parts, in_code, buf = [], False, []
                        for line in rec.get("content", "").splitlines(keepends=True):
                            if not in_code and chunk_re.search(line):
                                in_code = True
                                parts.append(("prose", "".join(buf))); buf = []
                                continue
                            if in_code and line.strip().startswith("```"):
                                in_code = False
                                parts.append(("code", "".join(buf))); buf = []
                                continue
                            buf.append(line)
                        parts.append(("code" if in_code else "prose", "".join(buf)))
                        prose = "".join(t for k, t in parts if k == "prose")
                        code = "".join(t for k, t in parts if k == "code")
                        rmd_pt += len(tok.encode(prose))
                        rmd_ct += len(tok.encode(code))
                        rmd_pb += len(prose.encode("utf-8", "replace"))
                        rmd_cb += len(code.encode("utf-8", "replace"))
            if len(sample_ids) == 0:
                break
    print(f"tokenized {len(toks)} sampled files", flush=True)

    keep_bytes = sum(r["bytes"] for r in keep)
    rates = [n / max(b, 1) for n, b in toks]
    mean_rate = sum(rates) / len(rates)
    # bootstrap CI over per-file rates, applied to total keep bytes
    boot = []
    for _ in range(2000):
        s = rng.choices(rates, k=len(rates))
        boot.append(sum(s) / len(s) * keep_bytes)
    boot.sort()
    est = mean_rate * keep_bytes
    ci = (boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))])

    # ledger summary
    lic = Counter(); excl = Counter(); misc = Counter()
    for line in open(STAGE / "license_ledger.jsonl"):
        rec = json.loads(line)
        t = rec.get("type", "")
        if t == "phase-b-accept-set":
            for e in rec.get("included_license_sets_top25", []):
                lic[e["licenses"]] += e["files"]
            for k, v in rec.get("exclusion_reasons", {}).items():
                excl[k] += v
        elif t == "phase-b2-cc-by-sa":
            misc["cc-by-sa-fetched"] = rec.get("fetched", 0)

    import subprocess
    du = subprocess.run(["du", "-sb", str(STAGE)], capture_output=True, text=True)
    total_staged_bytes = int(du.stdout.split()[0]) if du.stdout else 0

    report = dict(
        generated=now(),
        sample=dict(n=len(toks), seed=SEED, tokenizer="qwen3.5-2b-base-text-hf",
                    mean_tokens_per_byte=round(mean_rate, 4),
                    mean_tokens_per_file=round(sum(n for n, _ in toks) / max(len(toks), 1), 1)),
        net_new=dict(files=len(keep), bytes=keep_bytes,
                     tokens_estimate=int(est),
                     tokens_ci95=[int(ci[0]), int(ci[1])],
                     tokens_ci95_beeg=[round(ci[0] / 1e9, 3), round(ci[1] / 1e9, 3)]),
        storage=dict(staging_tree_bytes=total_staged_bytes,
                     staging_tree_gib=round(total_staged_bytes / 2**30, 2)),
        ledger=dict(included_license_sets_top=dict(lic.most_common(25)),
                    exclusion_reasons=dict(excl.most_common(40)), misc=dict(misc)),
    )
    (STAGE / "measure_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["net_new"], indent=1), flush=True)
    print("PHASE D COMPLETE", flush=True)


if __name__ == "__main__":
    main()
