# DAT-10 terminal union accounting

This packet audits the terminal candidate union from metadata only alongside
the root-owned lossless 16K conversion/cache pipeline. It does not open the 5,375,512,272-byte
`cpt_train.jsonl`, and it does not admit data for training.

The terminal gate is PASS. The union reports **177,190 documents, 332,455
rows, 461,653,440 input tokens, and 460,833,265 payload tokens** across
**8,204 source records**. The source-record decomposition is:

| source | records/groups | documents | rows | input tokens | payload tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| base broader shard | 1 | 14,098 | 27,430 | 38,973,361 | 38,905,169 |
| base global shard | 1 | 11,434 | 18,991 | 24,730,932 | 24,685,393 |
| terminal main suffix | 8,092 | 148,460 | 281,144 | 391,980,391 | 391,285,419 |
| terminal license-alias capture | 108 | 3,124 | 4,713 | 5,703,573 | 5,692,558 |
| `svars` repair candidate | 1 | 53 | 74 | 82,250 | 82,081 |
| `tfprobability` repair candidate | 1 | 21 | 103 | 182,933 | 182,645 |
| **total** | **8,204** | **177,190** | **332,455** | **461,653,440** | **460,833,265** |

The two base manifests expose `supervised_tokens`; the packet derives their
payload count as `supervised_tokens - documents`, which matches the terminal
union convention used by the repaired result manifests. All other component
payload counts come directly from their pinned progress/result manifests.

The 8,092 main groups are the seeded suffix `[775, 8867)`. The earlier 775
package-order units produced 750 prior global-shard groups; those rows are the
two base sources and are not counted again as suffix groups.

The union itself discarded no row: `exclusions.jsonl` is zero bytes,
`manifest.json.counts.exclusions` is `{}`, and every source-manifest record
matches its expected row count, byte count, and recorded SHA-256. This does
not mean the entire source corpus is admitted. Upstream materializers retain
named exclusions and review queues. The main terminal progress records 5,759
producer-level excluded documents: 83 empty, 931 exact prior/held-out
duplicates, 12 within-group duplicates, 4,732 license exclusions, and one
license-review item. The base manifests also have their own named license,
cap, duplicate, empty, and metadata exclusions. These counts remain visible
in `terminal-accounting.json` and are not relabeled as union exclusions.

The six terminal holds are:

1. `Rblpapi`: 19 source-only rows (19 documents, 16,515 payload tokens) stay
   held. The prior review found explicit GPL-3 evidence for the R source but
   also `DESCRIPTION` `FOSS=no` and separately licensed Bloomberg
   headers/binaries; root license review is required.
2. `RivRetrieve`: its `R/data.R` is zero bytes, so the closed exclusion is
   0 documents, 0 rows, and 0 payload tokens.
3. The main repair queue remains recorded for the four known main repair
   groups. Two lanes were recovered into this union (`svars`, 53/74/82,081
   and `tfprobability`, 21/103/182,645) but still need root global-dedup and
   admission review. The other two are the `Rblpapi` and `RivRetrieve` holds.
4. The alias repair queue remains separate. Its two records are `dataquieR`
   (`g-0f3f23606c86a6faf1de`) and `PTXQC`
   (`g-8797e2592d2b0dd69443`); both report
   `tokenizer_empty_or_roundtrip_failure` and emitted no rows. Both source
   paths are statically zero bytes and both records carry the empty SHA-256
   (`e3b0…b855`), so these entries close as no-payload exclusions. They are
   distinct from the separate zero-byte `RivRetrieve` main repair.
5. `svars` is an emitted candidate pending root global dedup and admission.
6. `tfprobability` is an emitted candidate backed by the pinned upstream
   Apache-2.0 metadata recovery, also pending root global dedup and admission.

The known repair identities, alias repair records, and all six hold records
are copied into the accounting artifact with their source and receipt hashes.
The generic alias repair hold in the frozen union is retained as bookkeeping;
this audit records the two named alias entries as closed zero-byte exclusions.
No candidate is silently counted as admitted.

The global base producer has a separate recoverable frontier: 1,999 documents
were excluded by `whole_document_exceeds_remaining_group_cap` across 79
`train_group`/`cpt_train` groups at seeded indices 1–765. The exact paths all
match selected-source metadata and none appears in the terminal union's
177,190-document provenance index, so they are absent from base, main, alias,
and repair outputs by exact source path. The frontier is materialized as
`global-cap-exclusion-frontier.jsonl` with group/package/license/description
metadata. These records have no source SHA-256 in the old exclusion ledger;
future recovery must rehash each source and run content dedup against the
protected and terminal identities before admission. This is a named
recoverable frontier, not a claim that the terminal union covers all eligible
source files.

Duplicate and held-out protection is bound to the terminal v2 builder and its
upstream receipts. The guard set contains the DAT-02 global split, the CPT
partition, 20,090 protected parent SHA-1 identities, and 294 reserved
CPT-validation source identities. Upstream main and alias manifests assert
that non-TRAIN groups were rejected before payload reads. The union builder
then requires `train_group`/`cpt_train`, binds `document_id` to
`source_sha256`, binds `row_id` to `source_sha256:chunk_index`, rejects
reserved/protected identities, rejects duplicate source documents and row
IDs, rejects metadata conflicts, and checks contiguous complete chunks with
native BOS/EOS and vocabulary bounds. The terminal alias snapshot reports
zero recovery-versus-new-main conflicts and six source-level dedup records.

The reproducible metadata-only command is:

```sh
python3 docs/campaign/work/lead/r2-final-union-terminal-accounting-v1/audit_terminal_union.py \
  --union /mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1 \
  --output docs/campaign/work/lead/r2-final-union-terminal-accounting-v1/terminal-accounting.json
```

The resulting artifact is `terminal-accounting.json`. Root must wait for the
active lossless 16K process to finish, then perform the final global-dedup and
training-admission decision. The source union remains a candidate.
