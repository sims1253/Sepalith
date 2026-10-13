# DAT-04A structured scenario/edit-pair adapter

This packet is a CPU-only structural conversion probe for DAT-04. It reads
only the bounded `train_group` references selected from the accepted DAT-03 row
audit. It does not admit rows to a registry, tokenize rows, open final/TU3
data, generate teacher text, or modify a raw input.

The adapter is
`/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training/campaign_admission_structured.py`.
Its interface is `convert_structured(raw, source_ref)`, returning a
`PromptContext` mapping, complete semantic target body, canonical operation
(`replace`, `no_op`, or `delete`), source/provenance metadata, and an explicit
exclusion reason. The frozen PRM-04 protocol remains read-only.

## Source lineage

- `rename_propagation`, `pipe_rewrite`, and `na_rm_propagation` use the exact
  normalized package file at
  `/mnt/h/sepalith/normalized/<package>/<version>/<package>/<path>` as the
  builder-defined before image. The one-hunk `event_diff` is parsed and
  applied line-for-line. The after image is the resulting current prediction
  source. The builders intentionally omit `suffix`; this is recorded as
  `source_builder_omitted_suffix`, not treated as a missing source failure.
- `format_propagation` reads exactly the needed member from
  `/mnt/h/sepalith/tarballs/<package>_<version>.tar.gz` as the raw before
  image. The normalized file is a separate authoritative after reference.
  The event hunk is replayed against raw lines, and a `SequenceMatcher`
  comparison proves that the target raw hunk maps to the corresponding
  normalized target hunk after the event. It does not infer a current file
  from a latest checkout.
- The synthetic insertion families (`comment_drafting`, `comment_insert`,
  `no_op`, and `roxygen_drafting`) can use the exact bounded constructor
  envelope `prefix + region_old + suffix` when a complete file has repeated
  anchors. This is labelled `source_builder_window`; it is not claimed to be
  an observed editor buffer. The legacy `region_old=[""]` insertion sentinel
  becomes canonical `region_old=[]`; a literal user cursor marker is removed
  and its code-point/UTF-16 position is recorded. A raw empty `region_new` in
  the `no_op` family resolves to unchanged `region_old` and operation `no_op`.
- `edit_pairs` uses the verified git parent (`sha^:path`) when that identity is
  present. The bounded candidate converted structurally, but its source has
  no prediction-time evidence for the architectural claim carried by the
  edit. It remains `admission_eligible=false` with
  `pending_prediction_time_evidence` until DAT-05 source validation.
- The sampled `doc_sync` row is the old v1 verbose-description construction.
  It remains structurally auditable but is marked
  `unsupported_v1_doc_sync_description`: the canonical builder diagnosis
  says its exact description is not observable from the prompt and its target
  shape is quarantined. A round-trip conversion does not reopen it.

For rows with an event, the adapter accepts a source only when the exact old
hunk matches a verified before image or the exact new hunk matches a verified
after image. It reconstructs the missing side, reapplies the forward patch,
checks the target region before and after line geometry, and binds the history
range hash to the before image and the replacement range hash to the after
image. It rejects ambiguous, overlapping, unparseable, multi-hunk, or
mixed/lone-CR histories. Whitespace-only replacements stay replacements.
Empty targets on non-`no_op` rows require authoritative deletion evidence.

## Bounded result

The staging bundle contains one deterministic lowest-line `train_group`
reference per requested family: 10 structural conversions from 10 candidates.
Nine operations are replacements and one is `no_op`. Six selections have a
full staged post-history source snapshot; four use a bounded source-builder
window. All ten staged source hashes independently match the selection source
hashes and the PRM-04 replacement range content hashes. The structural result
is still pending source validators (`admission_eligible=false`), with the v1
doc row and edit-pair row carrying their specific quarantine reasons.

This probe demonstrates supported representability across the sampled
families. It does not establish 600–1000 clean final cases, 30 independent
packages, or final/dev coverage. Those counts require the lead’s broader
source-validator and split-selection pass. No final answer, final row, or TU3
row was opened.

## Reproduction

The candidate extraction was one buffered `awk` pass over the accepted
DAT-03 audit, retaining the lowest-line row per family in the non-training
staging file. The bundle command was:

```text
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 experiments/training/campaign_admission_structured.py --bundle \
  --audit /mnt/e/sepalith/campaign-20260915/data-work/DAT-04-structured-candidates-1-lowest.jsonl \
  --source-audit /mnt/e/sepalith/campaign-20260915/data-work/DAT-03-row-audit.jsonl \
  --source-audit-sha256 9878a70d5d822f1192ccfc2956346738d668dc38175871556d1af1e16dbccadd
```

The independent source check is
`python3 docs/campaign/work/data/DAT-04-structured-source-checks.py`; it
reconstructs contexts from staged source bytes and reports only counts,
identities, hashes, and reasons. Focused tests are in
`experiments/training/test_campaign_admission_structured.py` and run without a
model or tokenizer.

The durable non-training artifacts are
`DAT-04-structured-reference-bundle.json`, its staged source directory, and
`DAT-04-structured-source-checks.json` under
`/mnt/e/sepalith/campaign-20260915/data-work`.
