# DAT-10 R2 task mixture notebook acceptance audit

This is a bounded CPU-only independent audit of the already admitted DAT-10 corrected plus short target-only subset. It does not alter the frozen rows, draw schedule, root admission, or the prior 411-row notebook parse.

The frozen 25% no-op schedule was checked against the row geometry and its declared pins. At 250, 500, and 1000 updates it has respectively 4,000, 8,000, and 16,000 draws; no-op draws are exactly 1,000, 2,000, and 4,000; row repetition maxima are 2, 3, and 6 (the ordinary row cap is 8); and policy violation lists are empty. Source-group repetition is reported as coverage information only: maxima are 54, 120, and 265. The sampler policy caps ordinary row replay, not source-group coverage.

The selected 8,115 corrected rows all matched the pinned DAT-05 provenance ledger. Their bounded source inventory is 3,694 existing normalized snapshots, 1,612 exact tarball members, 1,089 authored full `selection_source.document_text` values, and 1,720 builder scenario records whose source line contains the synthetic prefix/region/suffix. The latter representations were not materialized in this stopped audit.

No additional notebook SSH session, R parser, model, server, or generated-code execution was run. The only syntax evidence reused is the accepted prior 411-row parse (`Rscript --vanilla parse` on m0pad, 411/411, exit 0) recorded in the receipt. Therefore this packet supplies schedule and source availability evidence, not a new full corrected-row syntax denominator or a second independent R result.
