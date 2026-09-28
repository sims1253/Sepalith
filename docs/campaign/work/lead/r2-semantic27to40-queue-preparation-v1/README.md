# DAT-10 semantic classification queue for replay shards 27–40

This packet prepares a lead-owned, review-only semantic classification launch
for exactly 14 replay shards: 27 through 40. It reuses the byte-pinned
`r2-semantic-queue-root-launch-v3` worker. Intake validates the replay index,
the candidate packet and provenance hashes, the global TRAIN/CPT/license/heldout
bindings, and the exact queued row-ID join for every terminal replay receipt.

`run.py` is plan-only by default. It reports the missing terminal replay
receipts and the bounded command without creating output or launching a
classifier. The explicit `--execute` path re-runs the complete preflight,
rejects any missing or changed replay receipt, and skips a semantic child only
when its source binding and output bytes/hash/row-ID digest are independently
verified. It never passes shards 0–26 to the replay or semantic worker.

The prepared output is `/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards27to40-v1`.
The command is limited to cores 8 and 10, two workers, nice 10, idle I/O, no
CUDA, and 1,500 seconds. No classifier or replay process is launched by this
preparation packet; root reviews and launches after replay shards 27–40 all
have terminal receipts.

Prior semantic preparation evidence is bound separately: receipt
`DAT-10-semantic10952-preparation.json`, output terminal PID 2086119,
10,948 prepared rows, 4 explicit holds, and its independent source-reapply
verification are historical review evidence only.
