# Expanded-union geometry recovery v2

This fresh preparation fixes the v1 source-identity conflation. It does not run a full cohort or admit training rows.

`source_sha256` now identifies the pinned immutable source snapshot from provenance. `preedit_sha256` separately identifies the current prediction-time editor buffer in `PromptContext.replacement_range.content_sha256`. The original provenance object is copied unchanged into each recovered record. A semantic provider row uses its direct absolute `source_identity.source_path/source_sha256`; an original-15006 row uses nested `source_identity.source_provenance.source_snapshot_path/source_snapshot_sha256`, while its history-applied `after_snapshot_sha256` must equal the replacement range content hash. The relative logical URI in original synthetic contexts is verified as a suffix of the canonical source path; semantic provider file URIs must equal their absolute canonical source path.

A claimed full snapshot must reconstruct exactly under its recorded LF/CRLF policy and must contain both replacement positions at valid UTF-16 boundaries. A partial selected context can establish only the observed zero-width cursor boundary at `len(prefix)`; a nonempty replacement range remains unresolved. No target or token label supplies geometry.

The tests exercise the actual first original and semantic records. Each recovered record is passed to the unchanged `r2-expanded-union-audit-preparation-v2/audit_expanded_union_v1.py::geometry_from_provenance`. Controls reject invalid full-buffer lines/columns, shifted partial boundaries, nonempty unverified partial ranges and pre-edit identity mismatches.

Root may run the full commands only after reviewing these source bytes. Outputs are fresh, review-only ledgers; missing or ambiguous rows remain present as unresolved and are never silently dropped.
