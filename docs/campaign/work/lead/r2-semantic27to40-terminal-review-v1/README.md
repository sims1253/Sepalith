# Semantic shards 27–40 terminal review

This packet independently rehashes and parses the completed resume output. It
does not render predictions or admit rows to training.

Verified result:

* 14 exact shards: 27 through 40.
* Reused: 27 through 34; computed by the resume: 35 through 40.
* 28,753 source provenance rows.
* 24,330 emitted semantic rows with 24,330 unique row IDs.
* 9,535 rows have `semantic_supported_context_closure_root_review_required`.
* 14,795 rows remain `hold_semantic_evidence`.
* 4,423 provenance rows were not emitted into this semantic queue.
* No active temporary/partial artifact was included. Two failed timeout child
  directories for shards 35 and 36 remain under `quarantine/` and were ignored.

The output remains `partial_review_only`, with training admission false and all
three global closure flags false. The next stage is a separately reviewed
provider/render materialization of the 9,535 supported rows, preserving all
hold rows and revalidating target/provenance joins.
