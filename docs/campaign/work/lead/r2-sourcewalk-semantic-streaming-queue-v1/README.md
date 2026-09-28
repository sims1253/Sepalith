# Source-walk semantic streaming queue v1

This queue consumes only atomically committed v3 provenance shard receipts. It
validates each receipt's driver, index, global registry, CPT partition, token
rows, candidate packet and ledger identities before invoking an exact frozen
copy of the accepted semantic analyzer v6.

Shard results publish by staging-directory rename and are reusable only when
the complete provenance binding, analyzer/helper identities, semantic ledger
size/hash and exact queued-ID closure still match. Concurrency is explicitly
limited to one or two workers.

The first command in `commands.json` processes shard 5: 2,075 provenance rows,
including 1,878 semantic-queued roxygen rows and 197 provenance holds/no-op
records. The second command discovers the committed shard set at invocation
and reuses already completed shard 5.

Without an accepted global provenance manifest the aggregate status is always
`partial_review_only`, `source_pool_closed=false`, and
`training_admission=false`. A terminal manifest permits `complete_review_only`
only when every terminal shard is included, every terminal receipt still
matches, all queued IDs close exactly, and no duplicate semantic ID exists.

Run tests with temporary files on E:

```bash
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/sourcewalk-streaming-queue-tests \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -B -m unittest discover \
  -s docs/campaign/work/lead/r2-sourcewalk-semantic-streaming-queue-v1/tests -v
```
