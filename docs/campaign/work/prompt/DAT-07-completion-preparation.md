# DAT-07 completion candidate preparation

This bounded pass materialises true-completion candidates for the development
split. It uses only `dev_group` rows in the pinned DAT-07 parent allowlist and
the accepted DAT-03 row audit. The semantic family in the packet is
`finish_block`; `finish_block_compound` and
`finish_block_compound_random` remain out-of-band source variants.

The pass reuses the frozen DAT-04B adapter (`dat04b-completion-adapter-v3`)
and its `VerifiedSourceCache`. Each referenced source JSONL file is streamed
once with 4 MiB buffering, its whole-file SHA256 is checked, and each retained
raw line is checked including its separator. The source constructor is
`finish_block_v5_prefix`: the simulated prediction document is exactly the raw
`prefix`. The target is read only as an out-of-band label. The canonical R
fragment check validates `raw prefix + raw corpus_target + "}"`; the closing
brace is not emitted in the label.

The bounded DAT-03 audit contains 131 matching rows (123 compound and 8
compound-random) from 10 allowlisted groups. Every one passed the R fragment
validator. The frozen adapter converted 67 rows from six groups. Fifty rows
were rejected for `mixed_document_eol_requires_policy`, and fourteen for
`finish_literal_prefix_target_splice_mismatch`; no EOL or target repair was
attempted. One deterministic converted representative per group was emitted,
with common packet aliases `row_ref.file`, `row_ref.line`, and
`row_ref.package_id` for the downstream token audit. The requested 16 distinct
groups therefore remains unmet. The six emitted packets are candidates only;
they are not tokenizer admission or training approval.

The common packet keeps the complete adapter target and source-derived
selection window. It carries `source_derived_simulated_pre_edit` lineage and
`source_builder_window` availability; observed editor coverage is zero. The
prediction context contains no `full_prompt`, `target`, or `model_target`.
The all-six packet file passed the pinned CPU token-audit preparation with six
tokenizer candidates and no rendered collisions. That audit is a downstream
candidate check and does not grant registry admission.
