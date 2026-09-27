# Sourcewalk no-op expansion v3

This fresh closure binds the complete 41-shard census: 4,227 exact no-op IDs, of which 4,106 have supported independent provenance and 121 remain explicit provenance holds. It preserves the full-document, target-free prediction contract and the reviewed provider contract.

Distinct row IDs may share `(preedit_sha256, cursor, path)`. That is candidate evidence, not an infrastructure failure: package, workspace, dependency, and later provider context can differ. The preparer retains every row and publishes `duplicate-geometries.jsonl` with every member row ID, workspace root, and absolute source path. Prompt/target deduplication remains a later provider-output review gate. Duplicate row IDs still abort.

Only `RowValidationError`, raised for explicit packet, target, geometry, EOL, or cursor validation failures, becomes a row hold. Source I/O and identity drift, malformed JSON, memory exhaustion, and other infrastructure failures abort and remove the temporary output.

The failed v2 reconstruction is preserved. This packet prepares fresh output under `Sourcewalk-noop-expansion-v3`; it authorizes neither reconstruction nor provider execution nor training admission.
