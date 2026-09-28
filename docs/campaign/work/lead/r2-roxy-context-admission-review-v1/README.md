# DAT-10 roxygen context admission review

This packet audits the frozen 10,017-row roxygen source-context candidate set. The replay reconstructs each normalized TRAIN before-state in memory, verifies the frozen adapter geometry, parses the selected source definitions, and classifies each frozen unresolved reference. It writes hashes and categories only; it does not serialize source or target text and it does not admit rows to training.

`admission-criteria.json` defines the category and recommendation gates. `sample/` contains the deterministic audit of all 172 formal-mismatch rows, all 392 pending semantic rows, 256 SHA256-ordered unresolved rows, and the four prior full-file contexts above 131,072 tokens. `full/` is the complete 10,017-row census and carries exact recommendation ID ledgers when finished.

Package-qualified names, observed call heads, and references local to a selected function scope are reasonable source references for documentation targets, while free variables and missing same-file definitions remain held for repair/evidence. Documentation tags are checked against target formals after normalizing `\dots` and `\ldots` to `...`. Long contexts and target bodies above 1,024 tokens remain explicitly queued; no target is truncated or excluded by length.
