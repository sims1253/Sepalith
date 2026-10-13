# DAT-05 registry preparation

This packet implements the CPU-only admission boundary for PRM-03 token-row
candidates. A `tokenizer_candidate_only` envelope is a structural audit result;
it does not admit a row. The clean registry writer requires a lead-issued
approval manifest whose `manifest_version` is
`sepalith.dat05.approval.v1`, whose status is `approved`, and whose candidate
file entries bind the exact file SHA256, every allowed row ID, split and group
ID, and one or more support-review receipt paths with exact SHA256 values.

The implementation is
`experiments/training/campaign_registry.py`. Its public entry points are:

```python
preflight_candidate_files(candidate_files, approval_manifest=None)
build_registry(candidate_files, approval_manifest, *, registry_path,
               provenance_path, report_path)
```

`candidate_files` is a non-empty list of absolute `{path, sha256}` objects (or
an object containing `candidate_files`). Preflight hashes and parses candidate
rows and returns a denominator report without writing an admitted registry.
With no approval, its status is `preflight_missing_approval`,
`admission_ready` is false, and `would_be_admitted_rows` is zero. The writer
rechecks the file hash after parsing, validates each complete PRM-03 row with
the frozen renderer/tokenizer identity, and writes:

* a bare PRM-03 row JSONL for `campaign_sft_data` and the development evaluator;
* a provenance/exclusion JSONL carrying source file/line/raw-line/source
  hashes, package/group identity, semantic family, source variant, selector
  evidence, target/prompt hashes, tokenizer/renderer identity, approval ID and
  support receipts; and
* a deterministic report with row, split, family, length, exclusion and output
  hashes.

The row's semantic `family` remains authoritative. Backend/source variants are
kept in the provenance index and cannot create separate semantic families.
Only row splits `train` and `dev` and source splits `train_group` and
`dev_group` are accepted. Unknown or duplicate row IDs, candidate hash drift,
missing/incorrect approval, allowlist mismatches, train/dev parent-group or
normalized-package overlap, and exact rendered-prompt cross-split collisions
fail closed. A train prompt with one exact target is kept by stable
`(candidate_file,line,row_id)` order and other copies are recorded as
`duplicate_prompt_target_deduplicated`. A train prompt with conflicting targets
requires an exact lead adjudication listing all conflicting row IDs and the
chosen row. Dev duplicates and conflicting labels are rejected.

The emitted row is never truncated and keeps the full `input_ids`,
`target_body_tokens`, `target_terminal_tokens`, `target_text` and
`target_body_text`. Source-builder target hashes are retained separately from
the framed PRM target hash because they cover different byte strings. No
source discovery, target reconstruction, final/TU3 input or model operation is
performed.

## Bounded real preflight

The only candidate file exercised in this packet is the corrected PRM-04
13-row token audit:

`/mnt/e/sepalith/campaign-20260915/data-work/DAT-04-lead-corrected-13-token-audit/candidate-token-rows.jsonl`

Its declared SHA256 is
`4db63e5c77558648bbe38d9a77079ef15bb0fe867133dbb7973f47090c01e15c`.
All 13 rows pass structural PRM-03, source identity, full-label, tokenizer and
length checks. All remain candidate-only: no lead approval manifest was
supplied, so zero rows are admitted and no registry/index is produced.

The candidate spec and deterministic preflight report are staged at
`/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-candidate-spec.json` and
`/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-registry-preflight.json`.

## Checks and limits

`experiments/training/test_campaign_registry.py` contains six CPU tests for
missing approval and no output, file/allowlist identity failures, each of the
three cross-split collisions, deterministic duplicate/conflict handling with
adjudication, complete EOS/target/source-variant retention, and rejection of
an implicitly admitted status. The test log records 6 tests passed.

The actual approval IDs, row/group allowlists, support-review receipts and
registry output path are lead-owned and were unavailable for this preparation
packet. Therefore this is a registry preflight and implementation gate, not a
claim of admitted training data or model/evaluator readiness. When the lead
supplies the manifest, `preflight_candidate_files(...,
approval_manifest=manifest)` is the dry run; `build_registry` is the only path
that writes the clean registry.
