# Source-walk semantic materialization v3

This review-only materializer repairs the v2 editor identity mismatch while retaining bounded, contiguous prompt context.

The reopened normalized source is the authoritative active document. The materializer removes the exact target roxygen block, inserts one physical blank line, and derives the complete pre-edit document. `ReplacementRange.content_sha256` always hashes that complete pre-edit document, and the zero-width replacement range always uses the global source line. Applying the complete target at that range must reconstruct the reopened source byte-for-byte. CRLF sources keep CRLF document identity and application text while the renderer continues to use logical lines.

Prompt selection uses the reviewed production `prm05-source-balanced-v1` algorithm. It selects a contiguous prefix and suffix around the global blank anchor and pins the complete target-function span. A large source may therefore produce a bounded prompt view while its replacement identity remains the complete active document. The materializer records separate full-document and selected-view hashes so they cannot be confused.

The old candidate-local buffer, hash, and range remain mandatory upstream integrity evidence. They are checked and mapped to the global target line, but they are never reused as the production replacement identity.

Rows whose semantic closure requires helper evidence are emitted as named holds (`helper_evidence_production_wiring_unavailable:<names>`). Current production context construction has no reviewed path for typed helper references. Existing candidate typed references are held for the same reason. This packet does not concatenate noncontiguous source spans or claim serving parity for helper-dependent rows.

The actual TRAIN fixture `227626e3a7a234b808a096b1` passes the unmodified TypeScript production selector with a bounded 52-line prefix and 56-line suffix, global line 290, full pre-edit hash `8bb73f454eb7931b1e838bf10e4b2208eb3d9b8914fe8a67b24b7bd22619d559`, and exact source reapplication. Synthetic controls cover stale candidate identity, LF/CRLF, bounded and whole-document selections, helper holds, typed-reference holds, and targets longer than 1,024 tokens.

No corpus materialization or training admission occurred. Root must bind an accepted semantic-v6 shard and exact input hashes before using the command template.
