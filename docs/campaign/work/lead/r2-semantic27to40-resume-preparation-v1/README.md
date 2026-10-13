# DAT-10 semantic 27--40 timeout resume

The original lead-owned semantic launch reached its fixed 1,500-second
timeout at `2026-09-15T02:33:02.257590Z` with exit `124`.  Its terminal packet
is retained at
`r2-semantic27to40-queue-preparation-v1/terminal.json`.  Eight children are
complete: shards 27--34, 14,758 semantic rows.  Shards 35--40 are the exact
remaining work.  The two interrupted temporary children from 35 and 36 were
moved, without modification, under the E-side output quarantine:

`/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards27to40-v1/quarantine/timeout-20260915T023334Z/`

Run `validate_resume.py` immediately before launch.  It is read-only and
fails if a queue process is live, a completed child changed, an unexpected
child exists, or an unquarantined temporary directory remains.  The printed
command retries only 35--40 with the frozen v3 semantic worker, two CPU
workers on cores 8 and 10, and a 5,400-second timeout.  Existing verified
children remain in the same output directory and are not recomputed.  The
worker is review-only; this packet grants no training admission.

No replay source, sourcewalk receipt, semantic helper, lease, training state,
DEV payload, or final/heldout payload is changed by this packet.
