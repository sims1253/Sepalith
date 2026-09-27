# Independent geometry v2 and union audit review

This is a bounded, review-only audit of
`r2-expanded-union-geometry-recovery-preparation-v2` and
`r2-expanded-union-audit-preparation-v2`. It reads only the named source,
manifest, receipt, and first-row fixtures. It does not scan a cohort, modify a
frozen input, use targets or labels, launch a provider, or admit training data.

The geometry producer is supported for the checked surface. Its actual first
original15006 record preserves the immutable source snapshot hash separately
from the current pre-edit hash. Its actual first Semantic10948 record preserves
the direct source identity and pre-edit hash. The existing producer tests pass
5/5, and the existing audit v2 tests pass 4/4.

The integrated gate is rejected pending three fixes or explicit root bindings:

1. `audit_expanded_union_v1.geometry_from_provenance` accepts a relative
   `source_path` when `source_identity` is absent, although the producer rejects
   that path.
2. The audit accepts a `cursor` that differs from
   `replacement_range.start`, although the producer emits and relies on that
   equality.
3. The geometry command writes a separate `geometry.jsonl` output and the audit
   command consumes a separate root-bound provenance pin. A keyed join with
   exact IDs, counts, and order must be recorded before the geometry can affect
   the union audit. Unresolved rows must remain explicit holds.

The reproducible probes are in `tests/test_integration_contract.py`; the exact
hashes and findings are in `findings.json`. Root retains admission authority.
