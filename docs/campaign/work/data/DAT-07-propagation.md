# DAT-07 propagation preparation

Status: partial constructor backed candidate panel, pending lead source and
token review. The previously emitted 118-row panel remains frozen as rejected
partial evidence and was not edited by this lane.

The bounded run used only the 165 `dev_group` parents in
`DAT-07-dev-parent-allowlist.json` (allowlist SHA256
`ac66fff6c65b4dd41eac7ce952649b4a7dff15237fac9132b148fd3a0243f4c1`) and the
canonical `scenarios.py` source (observed SHA256
`cccf8ddfff0ae1f64a0113c9612386227df66f9fff4701608c2320bd8eb0250c`). It
used `extract_rename`, `extract_pipe`, `extract_na_rm`, and
`format_pairs_from_lines`; it did not create arbitrary line transforms. Each
converted row has one constructor supplied preceding event, exact before to
after hunk replay, a complete after-event source snapshot, a full post-edit R
parse, and a new `dat07p-*` identity bound to its parent group.

The run began at `2026-09-12T06:31:52Z` and finished at
`2026-09-12T06:33:32Z` (91.168 seconds). It produced 30 converted rows from
25 named parent packages/groups: rename 8, pipe 8, format 8, and na.rm 6.
One additional format candidate was retained in the immutable raw row file as
excluded because its exact after-event prefix geometry was unavailable. No-op,
completion, final, or TU3 rows were generated. The 30 converted rows are one
row per parent group per family at most; the count is a candidate count, not a
claim of 30 independent packages beyond the named package/group counts.

The frozen public structured converter was probed with the honest
`DAT-07-derived-parent` reference and returned its expected
`source_ref_not_DAT03_verified` seam. No DAT-03 identity was copied or
spoofed. The local derived result retains this seam and keeps target text out
of `PromptContext`; root must add an explicit derived-parent source-ref path
before admission.

Artifacts:

- Receipt: `docs/campaign/receipts/DAT-07-propagation.json`, SHA256
  `00358d9e2a55e39a6931c4c70cac2ada48b4c375b7023b6acfe4a1a51a4b5424`.
- Script: `experiments/training/campaign_dev_propagation.py`, SHA256
  `43facf0330300df3a58add00b34e51cb3c3173c029d18cc2f3e176b5b0783aa3`.
- Tests: `experiments/training/test_campaign_dev_propagation.py`, SHA256
  `2bc87a0e9a1a311347edaeb773851a8a4a529e77b213525491f69f72b9f459fd`.
- Immutable derived rows:
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-propagation-v2-derived-rows.jsonl`,
  31 lines, SHA256
  `3cf0006e9ccb94c461c68869363fdb3ddb9f22ec0ea1ca4fae517872a23f2ebc`.
- Common packet file:
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-propagation-v2-packets.jsonl`,
  31 lines, SHA256
  `171ec2287973fbadbdc980de4cf5fdbab59cd8bdce67088e6c2a8a52604444a4`.
- Token audit report:
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-propagation-v2-token-audit/report.json`,
  30 tokenizer candidates, 1 excluded, 0 rendered collision groups, SHA256
  `e363c68a0b50831dd72479f47ef1317371cd77be50524804219683c265f292b2`.

Validation ran with the canonical CPU environment. The four focused tests
passed. The pinned tokenizer audit read the common packet shape and produced
30 candidates; all 8 rename, 8 pipe, 8 format, and 6 na.rm rows fit the 4096
sequence limit in the audit, while 24 fit 2048. This is tokenizer-only
evidence and grants no admission.

The concrete next action is for root to inspect the packet/source hashes,
apply the derived-parent seam in the root integration, and make the separate
scientific and collision decision. The sealed final and TU3 inputs remain
untouched.
