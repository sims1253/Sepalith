# DAT-10 source-walk SFT admission review

This packet audits all 86,910 converted source-walk rows without admitting or
materializing a training union. The full-byte token pass is reproducible with:

```sh
ionice -c 3 nice -n 10 taskset -c 0,2 python3 audit_union.py \
  --out /mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-SFT-admission-review-v1
```

The strongest presently reviewable union is 27,821 unique prompt/target pairs:
15,006 repaired accepted rows, 8,595 unique rows from the reviewed roxygen
8,597 view, and 4,220 mechanically unique source-supported no-ops. This is a
review frontier, not admission.

Another 72,052 roxygen rows pass the frozen token/render/geometry checks and
mechanical deduplication, but need full-source semantic-support review. That
review must use R lexical bindings and occurrence-based NSE evidence. A name
missing from a global-name list is not an exclusion.

Every one of the older 10,017 source-walk roxygen IDs is removed when the
reviewed full-context view is used. This prevents the older cropped rows from
being counted alongside the 8,597 reviewed replacements. The 1,420 held IDs
remain held; they do not fall back to their cropped source-walk versions.

`review.json` is the machine-readable result. `mechanical-hold-ledger.jsonl`
names every held row and reason. `family-hold-summary.json` separates no-op and
roxygen effects. The audit recomputes safe-roxygen dedup against the current
repaired 15,006 stream (SHA `3f551c...`); the producer's obsolete `65b2...`
dedup input is not relied upon.
