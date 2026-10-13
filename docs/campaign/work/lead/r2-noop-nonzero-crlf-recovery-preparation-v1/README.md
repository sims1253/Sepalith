# DAT-10 no-op nonzero and CRLF recovery preparation

The frozen v3 no-op preparer reported 4,227 candidates, with 1,115 existing
target-free inputs and 3,112 holds.  Its two broad preparer checks caused
2,501 `after_close_brace` rows with unchanged nonempty selections and 490
uniform-CRLF `blank_between` rows to be held.  The remaining 121 rows are
independent provenance holds and stay separate.

The v4 audit accepts a nonempty no-op only when the recorded `target_body`
equals the recorded `region_old`, the source window is unique in the pinned
raw file, the range bytes agree, and reapplying the unchanged region preserves
the complete source.  It accepts a zero-width blank-between no-op with the
empty target contract.  For `uniform_crlf_to_lf`, it normalizes only a file
whose every newline is CRLF for window and cursor checks, while preserving the
raw source hash.  Mixed-EOL files remain held.

The E-side output is a compact source-backed decision ledger.  It contains no
provider prompt, target prediction, or copied source window.  It reports
2,985 recoverable decisions and six mixed-EOL holds from the 2,991
preparer-overrestricted frontier.  The two representative proofs are
`47a96cc61e9860688ea9114e` (nonzero `after_close_brace`) and
`49da0198cb6f200c0214a9ee` (raw CRLF normalized to the LF packet window).

Run `prepare_recovery_audit.py` only against the pinned TRAIN sourcewalk
inputs.  It does not modify v3, emit provider inputs, access DEV/final data,
or grant training admission.
