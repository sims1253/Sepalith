# DAT-04B completion batch preparation

This packet materialises a bounded, reviewable batch from the accepted DAT-03
row audit. It reads only rows satisfying `split=train_group`, an audit family
beginning with `finish_block`, direct row license evidence, and a nonempty
named identity token list. The raw source rows are real train-group rows, but
their prediction windows are source-builder simulations. No observed editor
event trace is claimed (`observed_editor_trace_count=0`).

The batch uses the frozen `dat04b-completion-adapter-v3` constructor. For each
row it first verifies the exact source JSONL line and complete source-file
SHA-256 through `VerifiedSourceCache`. The cache streams each referenced file
once with a 4 MiB binary buffer, retains only requested line bytes including
their original separator, checks file identity before and after streaming, and
rechecks identity for each lookup. A mutation or hash disagreement fails the
batch or excludes the row explicitly; it never falls back to a rescan.

The first 5,000 eligible audit rows form the bounded candidate pool. Output is
round-robin by sorted audit source variant while preserving audit order inside
each variant. The first 512 processed packets are the profile milestone. Since
454 converted packets were available there, the run expanded to all 5,000.
This profile is a conversion-utility signal, not a quality or latency claim.

The source-derived `finish_block_v5_prefix` window uses `raw.prefix` only. A
trailing LF retains its physical empty EOF line and canonical `region_old=[]`;
the nonempty signature line uses the live same-line whole-line replacement
geometry. The source target is out of band. Its exact body fragment is checked
as `raw.prefix + raw.corpus_target + "}"` by the canonical R fragment
validator. The outer brace is used only for that validator and is never added
to `target_body`. Full target whitespace, blank lines, and leading LF are
retained when the adapter can represent the source splice exactly.

## Real run

The real run used DAT-03 audit SHA
`9878a70d5d822f1192ccfc2956346738d668dc38175871556d1af1e16dbccadd`, read
376,830 audit lines, and found 23,817 eligible rows. It selected 5,000 rows
across seven source variants and seven source files. All 5,000 packets have a
stable `row_ref`, normalized family `finish_block`, out-of-band
`source_variant`, package/group identity, full result object, source support,
and selection-source text/hash where converted.

The materialiser produced 4,346 converted packets and 654 explicit
exclusions. The exclusions are 566
`mixed_document_eol_requires_policy` rows and 88
`finish_literal_prefix_target_splice_mismatch` rows. These are retained as
exclusions because silently changing CRLF/mixed-EOL or a literal source splice
would break the lossless contract. No target truncation, model generation,
tokenisation, serving, or bulk admission occurred.

Independent output validation read all 5,000 packets, reproduced the output
SHA, verified every converted selection-source text hash, checked that future
fields are absent from every rendered context, and checked that every
converted packet carries a passed R-fragment check and verified-stream source
support. It found zero invariant errors. The final summary records each of
the seven source paths and full source SHA-256 values.

The synthetic batch test ran the same three-row audit twice and compared the
packet JSONL bytes and deterministic batch ID; both were identical.

The canonical validator was
`/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/cases/validators.py`
with SHA `5bccc747f6be4fc10131246fdf410ee3d2ee3b35cefd418cbda1db1e7f350626`.
The deployed source constructor was
`/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/cases/rules/rules_finish_block.py`
with SHA `6f4fbcb4091d29644e831d1a83cf11caaa25ca95b10b5a6c4ad6f5b4de28142f`.

## Artifacts and limits

- Packet JSONL:
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch.jsonl`
  SHA `42743e70dbde54dd0f4adab42ebd0c2b0a0e7d3a4c4596752ffca272b20adc25`.
- Summary:
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch-summary.json`
  SHA `af389a94ba1318c0b1e30327911a63a867255a691b7b574aa86207a4988740b5`.
- Independent validation log:
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch-validation.log`
  SHA `7a4288ee3219ae79255bb9b1bd7e3f9050a15b11ac334a585898b2f4c55efe58`.
- Regression log:
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-tests.log`
  SHA `733a48c4cf3dc8959b57d07ebfe4bf64a172ff3f3e8a5c4d4b3003e1a3a3b7c1`.

The packet is preparation evidence only. The next selector/token-audit step
must independently inspect source support, token representability, admission
policy, and any observed-editor requirement. Source-derived simulations must
remain distinguishable from editor-event coverage.
