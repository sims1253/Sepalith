# Full-weight post-CPT editing SFT preparation

This packet prepares target-only full-weight editing SFT over the root-reviewed, finish-repaired 15,006-row TRAIN cohort. Every prompt token remains masked. Every target body token, protocol terminal token, and final EOS remains supervised. The materialized maximum sequence is 3,064 tokens and the maximum supervised target including EOS is 933; the runtime interface accepts complete rows up to the prepared 16,384-token bound and has no target cap.

The 18,560-draw schedule is one complete coverage pass plus explicit alignment replays. It retains the reviewed 12-edit + 4-no-op batch contract: all 13,912 edits and all 1,094 no-ops appear before their pool's first replay; 3,546 no-op replays raise no-op exposure to 25%, and 8 edit replays complete the final edit batches. These 3,554 named replays are deliberate class alignment and batch completion, so this horizon is not described as one statistical epoch.

The separate data queue retains 10,017 roxygen context candidates with zero admitted rows. Those rows require root semantic, license, deduplication, context, schedule, and resource admission in a fresh recipe. They are neither mixed into this run nor permanently excluded.

The trainer keeps 381 trainable tensors and 2,516,756,480 trainable parameters with no PEFT. It uses the reviewed Aurora-mix optimizer, exact AddedToken/tokenizer restoration checks, deterministic sequential draw cursor, full optimizer/scheduler/RNG/sampler checkpoints, graceful save-and-stop at an optimizer boundary, and atomic same-filesystem publication on E. Parent, LR, scheduler, checkpoint cadence, and evaluation cadence require root admission.

At nominated checkpoints, the callback writes an evaluation request. Teacher-forced loss is marked diagnostic only. Actual generation over the fixed 75-case development panel is required for edit/no-op quality. Same-model callback evaluation remains disabled until a resource and train/eval/train roundtrip test passes while the full optimizer is allocated. Final data stays sealed.
