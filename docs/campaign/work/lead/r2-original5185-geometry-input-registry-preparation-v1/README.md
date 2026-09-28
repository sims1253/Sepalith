# Original 5,185 geometry input registry — preparation v1

This packet locates the four accepted candidate cohorts that account for the 5,185 rows beyond `original15006` in the reviewed 20,191-row pool. It does not admit data or derive geometry from training targets.

The registry keeps three identities separate: immutable scenario-source provenance, the prediction-time full preedit, and the provider-selected context window. The original source hash can legitimately differ from the current preedit hash because history or source transformations may precede prediction. The selected context must bind its replacement content hash to the current preedit and its replacement start to the authoritative prediction cursor.

`candidate616` uses the historical renderer's `full_context` or `bounded_context` according to the candidate provenance mode. `candidate4435` has 4,435 candidates inside a 4,551-row provider domain. Its two input shards plus selected-context file exceed this task's 100 MB read ceiling, so the exact candidate subset join is a required future command and remains explicit rather than claimed here.

The metadata command hashes all candidate provenance (about 8 MB), hashes small manifests and receipts, and inspects one prediction/selection sample per cohort. The full-join command is prepared for a later root-controlled run; it never reads candidate targets.
