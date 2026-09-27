# DAT-07 common adapter review

The frozen v2 propagation rows and local packets were reviewed through the
active `campaign_admission_structured` derived-parent seam. The review read
31 rows and 31 packets, verified the frozen rows SHA256
`3cf0006e9ccb94c461c68869363fdb3ddb9f22ec0ea1ca4fae517872a23f2ebc`, local
packet SHA256
`171ec2287973fbadbdc980de4cf5fdbab59cd8bdce67088e6c2a8a52604444a4`, all 31
per-line hashes, and all 31 parent/group references against the 165-parent
allowlist (SHA256
`ac66fff6c65b4dd41eac7ce952649b4a7dff15237fac9132b148fd3a0243f4c1`). No
final or TU3 input was opened.

For each row the script supplied `family` and
`source=scenario_<family>`, called `_snapshot_ref`, then called
`convert_structured` with `verification=DAT-07-derived-parent`,
`split=dev_group`, and the allowlisted `pkg:*` parent identity. The common
adapter converted 29 rows: rename 8, pipe 8, na.rm 6, and format 7. The
calibrator format row remained excluded with the same
`history_after_prefix_unavailable` reason as the local result. The local
`bplsr` format row converted locally but the common adapter excluded it as
`format_target_not_in_authoritative_after_snapshot`; this is the sole
comparison status difference and must be reviewed as an authoritative-after
lineage mismatch.

For the 29 rows converted by both paths, target bodies, before/after/post
source hashes, history, UTF16 cursor and replacement range, and the rendered
selected-context prompt matched exactly. A fresh full-file tree-sitter R parse
passed for all 29 common outputs. The pinned CPU tokenizer audit found 29
candidates, 2 excluded rows, and 0 rendered collision groups. This is a
candidate review only; it grants no training or final admission.

Artifacts:

- Receipt: `docs/campaign/receipts/DAT-07-common-adapter-review.json`, SHA256
  `ebeb5da2effe961388e92e43012f2fcfe09cb6af6c98daad513a637a69174f13`.
- Review script: `experiments/training/campaign_dev_common_review.py`, SHA256
  `3e1eed3f6e0a44d75764192df9ab768929e90d4a963dad0c9e04855694f0ad73`.
- Common packets:
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-common-adapter-review-packets-v2.jsonl`,
  31 lines, SHA256
  `9aa5bd471db9706fc5c1ae9b70e655328694dd08bb34f490625156e2e460ce72`.
- Token report:
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-common-adapter-token-audit-v2/report.json`,
  SHA256
  `9838bf9fb506644e61c9cdc34f6b8e09d089538adfc1c152262f71953e789fd0`.
