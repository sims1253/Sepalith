# Expanded TRAIN union audit preparation v1

This packet prepares a review-only audit for 44,507 expected TRAIN rows: 20,191 current rows, 10,682 finalized semantic rows, 9,534 future semantic rows, and 4,100 no-op rows. It never admits or rewrites training data.

`audit_expanded_union.py` verifies pinned manifests and row files, joins token rows to provenance by exact order and row ID, and performs independent prompt checks even when source geometry is unavailable. Missing geometry is recorded per row and blocks only the claim that geometry auditing is complete. It does not exclude otherwise valid rows.

A complete geometry record contains source path/hash, full pre-edit hash, UTF-16 cursor, UTF-16 replacement range, and selected-window hash. Its digest is recomputed. An opaque producer digest or a prompt-derived guess is insufficient. The existing 15,006 context sidecar and reviewed selected-context outputs are named recovery inputs; a follow-up must materialize row-aligned canonical geometry sidecars before admission.

The audit excludes only later exact prompt+target duplicates. It holds every member of same-prompt/different-target and same-geometry/different-target conflicts for review. Same-file/different-cursor rows, same-geometry/same-target rows with different prompts, and repeated `NO_EDIT` targets remain.

The checked-in union template is intentionally partial: semantic9534 and no-op4100 outputs are not bound. Running it after those cohorts are frozen requires a fresh spec revision with exact manifest, token-row, and provenance pins. The command then writes only to a fresh output directory.
