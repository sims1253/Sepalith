# Sourcewalk no-op expansion v2

This fresh closure binds the complete 41-shard census: 4,227 exact no-op IDs, of which 4,106 have supported independent provenance and 121 remain explicit holds. It preserves the v1 full-document, target-free prediction contract and provider contract.

`candidate_output` first requires the candidate status to equal its pinned ledger status. Only `RowValidationError`, raised for explicit packet/target/geometry/EOL/cursor validation failures, becomes a row hold. Missing or unreadable source files, source identity drift, malformed JSON, memory exhaustion, and other infrastructure failures abort the command and remove the temporary output.

The full reconstruction and provider commands are prepared but not authorized. The separate root/data-expansion replay merge is reused and was not duplicated. Semantic deduplication and training admission remain subsequent root gates.
