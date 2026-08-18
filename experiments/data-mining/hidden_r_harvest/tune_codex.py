"""Run two-stage detector over sample CodeX shards; dump rows for manual inspection."""
import glob
import json
import random
import sys

import pyarrow.parquet as pq

sys.path.insert(0, "/home/m0hawk/Documents/Sepalith/experiments/hidden_r_harvest")
from r_detect import stage1, stage2

shards = sorted(glob.glob(
    "/home/m0hawk/.cache/huggingface/hub/datasets--Modotte--CodeX-7M-Non-Thinking/snapshots/*/data/*.parquet"))
n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 3
# spread across dataset
step = max(1, len(shards) // n_shards)
sample = shards[::step][:n_shards]

random.seed(20260817)
scanned = s1_pass = accepted = 0
accepted_rows = []   # (shard, idx, inp, out, info)
stage2_rejected = [] # (shard, idx, inp, out, info)

for s in sample:
    sname = s.split("/")[-1]
    pf = pq.ParquetFile(s)
    idx = 0
    for batch in pf.iter_batches(batch_size=2000):
        ins = batch.column("input").to_pylist()
        outs = batch.column("output").to_pylist()
        for i, o in zip(ins, outs):
            scanned += 1
            text = (i or "") + "\n" + (o or "")
            if not stage1(text):
                idx += 1
                continue
            s1_pass += 1
            ok, info = stage2(i or "", o or "")
            if ok:
                accepted += 1
                accepted_rows.append((sname, idx, i or "", o or "", info))
            else:
                stage2_rejected.append((sname, idx, i or "", o or "", info))
            idx += 1

print(f"shards={len(sample)} scanned={scanned} s1_pass={s1_pass} ({s1_pass/scanned:.2%}) "
      f"accepted={accepted} ({accepted/scanned:.3%}) rejected_at_s2={len(stage2_rejected)}")

with open("/mnt/h/sepalith/datasets/hidden_r_instruction_v1/_tune_accepted.jsonl", "w") as f:
    for sname, idx, i, o, info in accepted_rows:
        f.write(json.dumps({"shard": sname, "row_idx": idx, "input": i, "output": o, "info": info}) + "\n")
with open("/mnt/h/sepalith/datasets/hidden_r_instruction_v1/_tune_rejected.jsonl", "w") as f:
    for sname, idx, i, o, info in stage2_rejected:
        f.write(json.dumps({"shard": sname, "row_idx": idx, "input": i, "output": o, "info": info}) + "\n")

# print 30 random accepted for manual inspection (trimmed)
sel = random.sample(accepted_rows, min(30, len(accepted_rows)))
for n, (sname, idx, i, o, info) in enumerate(sel):
    print(f"\n===== ACCEPT #{n} [{sname}:{idx}] {info}")
    print("IN :", i[:220].replace("\n", " "))
    print("OUT:", (o or "")[:280].replace("\n", " | "))

print("\n\n########## 10 STAGE-2 REJECTED (false-negative check) ##########")
sel2 = random.sample(stage2_rejected, min(10, len(stage2_rejected)))
for n, (sname, idx, i, o, info) in enumerate(sel2):
    print(f"\n===== REJ #{n} [{sname}:{idx}] {info}")
    print("IN :", i[:180].replace("\n", " "))
    print("OUT:", (o or "")[:240].replace("\n", " | "))
