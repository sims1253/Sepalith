# Noop4100 parse-policy retry preparation

This packet prepares a fresh CPU-only rerender of the complete Noop4100
provider stream using the reviewed parse-unavailable policy. It does not launch
the provider, use CUDA, load a model or optimizer, or modify the previous
output root.

## Why the retry is fresh

The prior root is
`/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/render16`.
Its metadata contains 26 complete shards with 3,502 output rows and failed
shard 0036 with an 8-row partial output. Those terminal records are preserved
under `preserved-old-output/manifest.json`, whose SHA256 is
`12a28dfbd74c1a8576adc5fd798fe4811894e55c84e38246e590088efacf33e1`.

The old complete shards are reference-only. The provider source now has a
parse-unavailable branch, and there is no whole-shard equivalence proof for
every old row. Reusing old complete output could mix provider behaviors, so the
prepared command rerenders all 4,100 provider rows into the fresh root

```text
/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/render16-parse-retry-v1
```

The old root and shard 36 partial are never overwritten, deleted, or silently
quarantined. A malformed or unfinished R snapshot remains a legitimate input;
the new provider emits explicit source-evidence absence and lets the existing
context policy decide whether to support or hold it.

## Pinned inputs and source closure

`inputs/frozen-plan16.json` is the previously reviewed 41-shard plan with SHA
`093fd437f6cc4a364b7273e7936459c836a18a31bc06fd84ec39ec88faacd725`. The
launch gate checks its input root, 41 entries, 4,100 row denominator, every
nonempty shard's input SHA and row count, context size 16,384, and reserve
2,048. There are 31 nonempty shards. Lane 0 runs shards 0012, 0015, 0016,
0018, 0020, 0024, 0025, 0026, 0027, 0028, 0029, 0031, 0034, 0036, and 0037
on core 4 for 2,076 rows. Lane 1 runs shards 0010, 0011, 0013, 0014, 0017,
0019, 0021, 0022, 0023, 0030, 0032, 0033, 0035, 0038, 0039, and 0040 on
core 6 for 2,024 rows.

`source-manifest.json` has SHA
`114b0a410cf15c821ce83c8d569b16d831640f744f3e53d303d849929e503919` and
binds the complete copied TypeScript/R source closure, wrapper, tokenizer
bridge, and run script. The source-import helper emits
`source_import_evidence_unavailable` with the raw source SHA and empty
dependency/evidence fields when R parsing fails. The runtime validates that
record, retains package `NAMESPACE` evidence, and allows existing context
fallback. Missing Rscript, missing paths, helper I/O, invalid JSON, and invalid
namespace evidence remain infrastructure failures.

## Exact launch command

The lead may review and later run:

```bash
bash docs/campaign/work/lead/r2-noop4100-parse-retry-preparation-v1/commands/run_noop4100_all41.sh
```

The command requires the pinned plan SHA, the exact reconstruction input root,
and a fresh retry output root. It runs both explicit lane lists concurrently,
attempts every nonempty shard even if one fails, writes one terminal record per
attempt, and returns aggregate failure when any shard fails. A failed shard is
left visible with its partial output and log; it is not converted into a
provider hold.

## Bounded checks

The copied provider suite passed with:

```bash
python3 -B docs/campaign/work/lead/r2-noop4100-parse-retry-preparation-v1/tests/test_parse_unavailable.py
```

It verified the exact e677 malformed snapshot's structured source-evidence
absence and full-document fallback, missing-Rscript infrastructure failure,
valid-control parity with the existing Noop wrapper, and cursor geometry plus
no-following-function helper parity. It read only the approved e677 and valid
control fixtures. Shell syntax and the 41-shard source/plan gate also pass.

The parse-policy result for e677 is `supported`/`full_document` with
`source_inventory_status: source_import_evidence_unavailable`, an empty
observed dependency list, retained package namespace evidence, and
`selection_target_or_gold_used: false`.

## Admission boundary

The old provider output and new retry output must not be merged by row position.
After the fresh rerender, the lead must verify every new terminal against the
source manifest and input plan, then bind the new provider/runtime hashes into
the downstream data manifest. Any evaluation or serving path for data produced
under this policy must use the same source-import status and fallback behavior;
otherwise train/eval/serving inputs will differ. Actual retry success, timing,
and output counts remain unknown until the lead launches the fresh command.
