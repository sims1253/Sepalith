"""Filter inclusionAI/Ling-Coder-SFT parquet shards for rows with 'R' in languages.

CPU-only, nice'd. Writes ling_coder_r.jsonl incrementally (flush every 1000 rows).
"""
import glob
import json
import os
import sys
import time

import pyarrow.parquet as pq

OUT = "/mnt/h/sepalith/datasets/hidden_r_instruction_v1/ling_coder_r.jsonl"
DATASET_URL = "https://huggingface.co/datasets/inclusionAI/Ling-Coder-SFT"
REPO = "inclusionAI/Ling-Coder-SFT"


def main():
    shards = sorted(glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/datasets--inclusionAI--Ling-Coder-SFT/snapshots/*/data/*.parquet")))
    if not shards:
        print("ERROR: no parquet shards found", file=sys.stderr)
        sys.exit(1)
    start_shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    prior_scanned = prior_detected = 0
    if start_shard:
        prior_scanned = start_shard * 200000
        prior_detected = 52860  # detected through end of shard 20 (verified from log)
    print(f"{len(shards)} shards, starting at {start_shard}", flush=True)

    t0 = time.time()
    scanned = prior_scanned
    detected = prior_detected
    lang_counter = {}
    f = open(OUT, "a", encoding="utf-8")
    try:
        for shard in shards[start_shard:]:
            shard_name = os.path.basename(shard)
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=2000, columns=["mid", "messages", "tags", "languages"]):
                mids = batch.column("mid").to_pylist()
                msgs = batch.column("messages").to_pylist()
                tags = batch.column("tags").to_pylist()
                langs = batch.column("languages").to_pylist()
                for i in range(len(mids)):
                    scanned += 1
                    ls = langs[i] or []
                    for l in ls:
                        lang_counter[l] = lang_counter.get(l, 0) + 1
                    if "R" in ls:
                        detected += 1
                        rec = {
                            "mid": mids[i],
                            "messages": [
                                {"role": m["role"], "content": m["content"], "index": m["index"]}
                                for m in (msgs[i] or [])
                            ],
                            "tags": tags[i] or [],
                            "languages": ls,
                            "provenance": {
                                "source_url": DATASET_URL,
                                "license": "apache-2.0",
                                "model": None,
                                "derived_from": {
                                    "dataset": REPO,
                                    "row_id": mids[i],
                                    "shard": shard_name,
                                    "row_idx_in_shard": None,
                                },
                                "harvester": "experiments/hidden_r_harvest/filter_ling.py v1",
                            },
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if detected % 1000 == 0:
                            f.flush()
                            os.fsync(f.fileno())
            print(f"  {shard_name}: cumulative scanned={scanned} detected={detected} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        f.flush()
        os.fsync(f.fileno())
        f.close()

    stats = {
        "source": REPO,
        "scanned": scanned,
        "detected": detected,
        "language_counts_top30": dict(sorted(lang_counter.items(), key=lambda kv: -kv[1])[:30]),
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open("/mnt/h/sepalith/datasets/hidden_r_instruction_v1/ling_stats_tmp.json", "w") as sf:
        json.dump(stats, sf, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "language_counts_top30"}), flush=True)
    print("LANG COUNTS:", json.dumps(stats["language_counts_top30"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
