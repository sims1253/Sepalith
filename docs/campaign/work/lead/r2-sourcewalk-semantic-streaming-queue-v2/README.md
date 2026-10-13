# Source-walk semantic streaming queue v2

This packet is a reviewed-code successor to v1. It consumes only committed
independent-replay provenance shards and remains review-only. No queue process
is launched by this packet.

The v1 runner checked exact row-ID closure, but its aggregate child record did
not require the semantic child output hash and its reuse check did not expose a
separate content binding for the source provenance ledger. v2 binds each child
to:

- the source receipt, index shard, provenance-ledger bytes and full row-ID
  digest;
- the candidate packet bytes and packet manifest;
- the queued-ID digest; and
- the semantic-ledger bytes, row count, and output row-ID digest.

The child manifest records both `source_provenance_binding_sha256` and
`output_binding`. A child with the same IDs but changed provenance or semantic
ledger bytes is rejected. The aggregate manifest repeats the source-ledger
and semantic-output hashes for every child before it can be published. A
global terminal status still requires all terminal shards and exact queued-ID
closure; `training_admission` remains false.

Run the synthetic hardening tests without opening campaign payloads:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /usr/bin/python3 -B -m unittest discover -v \
  -s docs/campaign/work/lead/r2-sourcewalk-semantic-streaming-queue-v2/tests
```

The v2 launch command is intentionally omitted from executable commands until
root reviews this packet and provides a fresh output directory. When root
authorizes a future run, use the v3 replay root, the same pinned parser
dependency overlay, and an E-backed output. Keep `--terminal-manifest` absent
until the replay's independently accepted terminal manifest exists.
