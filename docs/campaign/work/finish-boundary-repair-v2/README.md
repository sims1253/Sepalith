# DAT-04 finish-boundary repair v2

This directory is an isolated preparation candidate. It does not edit the
production adapter or admit any training, DEV, or final row.

The only accepted materialization contract is:

1. The result is a `finish_block` replacement with the exact
   `finish_block_v5_prefix` provenance, `target_convention=suffix`,
   `target_authority=["corpus_target"]`, and
   `outer_closing_brace_in_label=false`.
2. The selection document is LF text whose SHA-256 matches the selection,
   pre-edit provenance, replacement range, and (where present) `r_fragment`
   prefix. The range is same-line and ends at the UTF-16 end of that hashed
   document. `context.suffix_lines` must be empty.
3. The target body hash matches the emitted lines and ends with its explicit
   final LF. The candidate appends exactly one ASCII `}` byte to that target.

The EOF binding supplies the missing fact that an empty context suffix alone
cannot supply: there are no bytes after the replacement in the document whose
hash was checked. A suffix-bearing, non-LF-target, cross-line, tampered, or
ambiguous case is rejected. There is no caller-supplied suffix override.

Run the bounded CPU driver from the worktree root:

```text
PYTHONDONTWRITEBYTECODE=1 python3 docs/campaign/work/finish-boundary-repair-v2/run_finish_boundary_repair_v2.py --receipt docs/campaign/receipts/DAT-04-finish-boundary-repair-v2-preparation.json
```

The five converted TRAIN windows prove exact framed reconstruction only because
their retained constructor contract identifies the omitted brace; the raw-v3
synthetic fixtures additionally prove equality with complete source bytes.
The candidate remains a production-materialization preparation artifact until
root demonstrates the same document identity and editor application contract
in the live path.
