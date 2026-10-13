# Semantic byte-hash failure diagnosis

The shard-5 retry did not expose differing source bytes. The exact failing row `25189e2eff136821189db4db` reconstructs the unchanged 35,163-byte LF source with SHA-256 `8bb1ab378599f6a55c45a3f447b7d0a5498e1d821c1c1cb1f1b308b552e1d551`; R hashes those bytes identically.

The row has one exact target-roxygen occurrence followed by `summary.qbrms_p_significance`, but that function name has two top-level definitions in the reopened source. `semantic_scope.R` correctly returned `hold_target_definition_not_unique` with `target_matches=2`. That hold response omitted `parsed_source_sha256`. Python compared missing `None` with the expected hash and raised the misleading `R parsed byte hash mismatch` infrastructure error. Grouping by source path and parse hash could not fix a field omitted by the R hold branch.

The isolated correction adds `parsed_source_sha256` to every R response, including parse errors and semantic holds. Python now requires exact, present hashes and exact unique ID closure for every row submitted to R. It still rejects real mismatches, missing hashes, duplicate IDs, substituted IDs, and lost rows. Rows not submitted because their target occurrence/name is unavailable remain explicit semantic holds in the existing decision path.

The exact one-row end-to-end analyzer now exits zero and emits `hold_semantic_evidence` with reason `hold_target_definition_not_unique`; its R and Python parse hashes are identical. The patch does not turn this row into accepted training data and does not bypass any hash check.

Eight tests cover the original source-variant registration, the duplicate-target hold, non-ASCII UTF-8, uniform CRLF-to-LF reconstruction, two byte variants from one logical origin, parse-error holds, missing/mismatched hashes, and ID substitution. No source or generated R function body was executed; R was used only for parsing and codetools inspection.
