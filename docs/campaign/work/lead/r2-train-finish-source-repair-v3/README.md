# DAT-10 TRAIN finish source repair v3

This packet materializes a fresh 15,006-row TRAIN view. It preserves 11,503
rows byte-for-byte and changes all 3,503 newly admitted finish rows by appending
one ASCII `}` byte proven by their pinned finish extractor and source records.
The original 15,006-row pool is unchanged.

Every repaired row binds one of five immutable TRAIN source JSONL files, its
exact source line and line hash, the `braced_expression[1:-1]` extractor rule,
the literal pre-brace source replay, and the unchanged prompt/replacement range.
The target is rebuilt with the pinned protocol and tokenizer. All resulting
sequences remain within 4,096 tokens and all full targets remain within 1,024.

An independent second pass confirmed all 3,503 targets equal the old target
plus `}`, all prompt IDs are unchanged, and all 3,503 raw applied documents
parse under R 4.6.1 without a diagnostic suffix.

The copied context sidecar is byte-identical and remains authoritative for
prompt and replacement geometry. Nested historical target hashes or
`outer_closing_brace_in_label=false` values describe the parent data. Current
target authority is the repaired token-row hash plus `repair-ledger.jsonl`.
Downstream RL buffers must be rebuilt for these 3,503 rows. Their old
`framed_fragment` evidence is obsolete; a new buffer must parse the repaired
raw applied document without a synthetic suffix.

Reproduce into a fresh destination from the plan root:

```sh
nice -n 10 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 docs/campaign/work/lead/r2-train-finish-source-repair-v3/materialize_finish_source_repair.py \
  --output /mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3
```

The command is preparation only and does not authorize training.
