# Training data policy

Effective 14 September 2026, by explicit user instruction.

Use all eligible training data for the appropriate CPT, SFT or RL objective. Maintain source inventories and report actual consumed rows and tokens separately from available, scheduled or admitted data.

Time is not a reason to exclude data. Three hours is an estimate, not a training limit; the user explicitly accepts thirty hours if needed. Earlier calendar-based training stops, freezes and delivery targets are superseded. Freeze follows training and development selection. Keep the final evaluation set sealed until that freeze.

Exclude held-out identities, duplicates, leakage, unsupported licenses or provenance, corruption, degenerate examples and contradictory or unverifiable supervision. Record a concrete reason and denominator for every exclusion. Repair recoverable problems.

Keep long examples, mixed-line-ending cases and cases awaiting context or support review in explicit processing queues. A current token limit is not a permanent exclusion. Preserve complete targets and valid geometry when adapting data. Absence from a CPT-specific partition does not invalidate an otherwise admitted global TRAIN group.

Remove arbitrary family, package and row ceilings from coverage decisions. Resource limits may control batches and job duration, but resumable jobs must continue the remaining coverage. Do not treat a scheduled epoch as consumed data.

Preserve the incumbent and b4 rollback. Evaluate intermediate checkpoints and select by development evidence; completing coverage does not imply promoting the last checkpoint. Final evaluation remains separate.

Paid-cloud ceilings remain $60 Anyscale and EUR100 Azure. Longer local training does not authorize a paid-budget increase.

Decision receipt: `receipts/LEAD-all-eligible-data-policy-20260914.json`.
