# DAT-06 sampler preparation

This packet is CPU-only preparation for the parameterized SFT draw sampler.
It consumes admission metadata and never reads prompt text, target text,
token-ID arrays, model weights, final data, or a model-generated target.  The
rows in the tests and the 48,000-draw example are explicitly synthetic
functional metadata; they are not observed editor or corpus coverage.

The implementation is in the next execution source at
`/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`, owner HEAD
`a79d6de38890355ec66d7227c974140160ab314e`.  It adds
`experiments/training/campaign_sampling.py` and its CPU test companion.  The
legacy `campaign_sft_data.py` and trainer files remain untouched.

`validate_admitted_rows` accepts normalized records with stable row, family,
source, package and split identity, semantic no-op status, prompt/target/total
token counts, short/long bucket, natural-long flag, and source kind.  It
rejects held-out/non-train records, duplicate identities, raw prompt/target
fields, contradictory lengths, and prompt/target truncation flags.  If the
caller omits a registry digest, `canonical_token_rows_sha256` hashes only the
normalized metadata; production must pass the independently verified
`campaign_sft_data` token-row digest.

`build_draw_manifest(rows, max_steps, effective_batch, split_id, seed, ...)`
returns the exact schedule interface (`row_ids`, `max_steps`,
`effective_batch`, `split_id`, `token_rows_sha256`) plus draw records and
exposure ledgers.  An explicit `requested_draws`, when supplied, must equal
`max_steps * effective_batch`, so a manifest cannot be mistaken for a valid
optimizer schedule.  The sampler reserves `floor(0.10 * N)` semantic no-op
slots, allocates the remainder using square-root admitted-family counts with
an effective `floor(0.25 * N)` ceiling, and backfills only eligible non-no-op
rows.  Within each family, source round-robin order exhausts unused rows
before a finite next presentation.  Small-pack rows have capacity three;
ordinary rows use the recorded configurable replay ceiling (default eight).
Natural-long rows are limited by the recorded fraction ceiling (default
0.20), with no truncation or padding.  A finite scan stops and returns
`status: "infeasible"`, a global deficit, quota deficits, and reasons when
the requested draw count cannot be filled.

`build_dat06_synthetic_manifest_v2.py --write/--check` creates the complete
synthetic 48k JSON manifest at `DAT-06-48k-draw-manifest-v2.json`, including
every draw record, selected ID, capacity, exposure ledger, and fixed prefix
milestones.  Its output is a replayable functional fixture and is not an
admitted training registry.  The v1 manifest remains beside it unchanged.

The v2 correction applies the family ceiling to no-op and edit rows together,
computes edit quota from capacity left after no-op reservation, persists the
source-rotation cursor across quota/backfill calls, and interleaves category
queues with exact final totals.  The interleaver changes draw order only; it
does not alter row, source, family, token-count, or cap accounting.

The 48,000-draw synthetic example uses 3,000 updates, effective batch 16,
6,000 ordinary rows, replay ceiling eight, 600 semantic no-op rows, five
non-no-op families, and 180 source identities.  It completes with 4,800
no-op draws, 43,200 non-no-op draws, 4,800 natural-long draws, 6,000 distinct
rows, and maximum row exposure eight.  Exact schedule and manifest hashes,
token-count totals, and first/last IDs are recorded in the JSON receipt.

Formal DAT-06 completion remains gated on DAT-05's accepted clean registry.
No real admitted-row profile or primary SFT schedule was available to this
preparation packet, so the receipt does not claim real coverage or a training
decision.
