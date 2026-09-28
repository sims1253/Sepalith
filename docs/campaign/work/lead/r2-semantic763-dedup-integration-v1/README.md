# Semantic-763 dedup integration

This packet evaluates all 763 shard-5 rows previously accepted by the semantic analyzer. It does not admit or append training data.

`integrate.py` streams the authoritative 15,006-row token corpus and its exact context provenance, verifies their pinned hashes and IDs, then verifies both semantic candidate profiles and every target-free prediction input. Candidate token rows are rebuilt by the prior reviewed materializer; this gate re-encodes the complete prompt and target sequence with the pinned tokenizer using the same `encode_special_tokens=True` contract.

Cross-corpus decisions use independent exact keys:

* row ID;
* exact prompt plus target;
* same prompt with a different target;
* exact source hash plus target;
* package, normalized R source path, and target.

The same keys are checked within the selected cohort. Target equality is byte-exact; whitespace and content are not normalized. Identical targets in unrelated source documents remain distinct.

The prediction decision is target-free. A row uses the 16K token row when available. It uses the 32K row only when the reviewed 16K policy cannot fit it. The complete target is joined after selection and is never truncated. External namespace evidence unavailable to serving, unsupported 32K geometry, and the single ambiguous source window remain explicit holds.

The authoritative result is `/mnt/e/sepalith/campaign-20260915/data-work/Semantic763-dedup-integration-v1/final-02`. `attempt-01-superseded` has identical candidate bytes but predates the final explicit within-cohort source/target closure. Neither directory is admitted training state.
