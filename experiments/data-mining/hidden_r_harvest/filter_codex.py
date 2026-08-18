"""Filter Modotte/CodeX-7M-Non-Thinking parquet shards for hidden R rows.

Two-stage detector (r_detect.py: regex prefilter -> fenced-block confirmation).
CPU-only, nice'd; incremental writes + periodic stats. Provenance per schema.
"""
import glob, json, os, sys, time
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r_detect import stage1, stage2

OUT_DIR = "/mnt/h/sepalith/datasets/hidden_r_instruction_v1"
OUT = f"{OUT_DIR}/codex_r.jsonl"
STATS = f"{OUT_DIR}/codex_stats.json"
REPO = "Modotte/CodeX-7M-Non-Thinking"

shards = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Modotte--CodeX-7M-Non-Thinking/snapshots/*/data/*.parquet")))
print(f"{len(shards)} shards", flush=True)

t0, n_scanned, n_s1, n_detected = time.time(), 0, 0, 0
with open(OUT, "w") as f:
    for si, shard in enumerate(shards):
        try:
            table = pq.read_table(shard, columns=None)
        except Exception as e:
            print(f"shard {si} read error: {e}", flush=True); continue
        cols = table.column_names
        text_cols = [c for c in ("conversations", "messages", "prompt", "instruction",
                                 "input", "output", "response", "question", "answer") if c in cols]
        d = table.to_pylist()
        for i, row in enumerate(d):
            n_scanned += 1
            blob = " ".join(str(row.get(c)) for c in text_cols if row.get(c))
            if not blob or not stage1(blob):
                continue
            n_s1 += 1
            inp = str(row.get("input") or row.get("instruction") or row.get("prompt") or "")
            outp = str(row.get("output") or row.get("response") or row.get("answer") or "")
            ok, _info = stage2(inp, outp)
            if not ok:
                continue
            n_detected += 1
            rec = {k: row.get(k) for k in cols if k in
                   ("id", "conversations", "messages", "prompt", "instruction",
                    "input", "output", "response", "question", "answer", "task_type", "source")}
            rec.update(source_url=f"https://huggingface.co/datasets/{REPO}",
                       dataset_row=f"shard{si}:row{i}", license="apache-2.0",
                       model=None, derived_from=REPO, detection="r_detect.py stage1+stage2")
            f.write(json.dumps(rec) + "\n")
        if si % 10 == 0 or si == len(shards) - 1:
            f.flush()
            stats = dict(scanned=n_scanned, stage1_pass=n_s1, detected=n_detected,
                         shard=f"{si+1}/{len(shards)}", elapsed_s=round(time.time()-t0))
            open(STATS, "w").write(json.dumps(stats))
            print(json.dumps(stats), flush=True)
print("DONE", flush=True)
