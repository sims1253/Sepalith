# DAT-10 external namespace holds audit

This review replays all 140 `external_namespace_import_provider_unavailable`
holds from the frozen semantic-763 32K view. It uses only pre-edit TRAIN source,
the already-bound semantic reference inventory, and independently parsed package
`NAMESPACE` files. It never reads a target or gold completion.

The replay identifies 133 **provider-integration candidates**. Every dependency
in those rows has exactly one AST-parsed `importFrom(package, symbol)` origin,
and its call/reference shape is already visible in the pre-edit source inventory.
This evidence names an origin; it does not claim an external implementation,
runtime behavior, complete function signature, or semantic correctness. These
rows remain outside training until the provider is integrated, prediction-time
parity is replayed, token rows are rebuilt, and root admits them.

Seven rows remain held: four still exceed the prediction context policy and
three have ambiguous explicit origins (`%>%` in two rows and `tibble` in one).
Across the original 763-row cohort, integrating the narrow provider would raise
the reviewable context candidates from 616 to 749. Fourteen rows would remain:
the seven above, six pre-existing non-namespace context-policy holds, and one
pre-existing invalid-geometry row.

`namespace_evidence.R` parses directives without evaluating or loading package
code. `namespace_provider.ts` is an isolated adapter for the current
`EvidenceCandidate`/`EvidenceSourceSnapshot` contract. It emits only normalized
origin evidence and fails closed on wildcard imports, multiple explicit origins,
malformed input, unstable files, or invalid snapshot identity. Production still
needs a package-root locator, bounded R helper process, and document/NAMESPACE
change invalidation before this adapter is wired into `extension.ts`.

The complete row decisions and independently parsed namespace evidence are in
`/mnt/e/sepalith/campaign-20260915/data-work/Semantic-external-holds-audit-v1`.
No corpus was admitted or mutated.
