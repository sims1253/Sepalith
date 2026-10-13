# DAT-09 / RL-02 quota coverage v4

This is a CPU-only quota-policy preparation. It preserves the accepted v3 G=4
geometry and source-row accounting while adding a lead-specified floor for
`roxygen_drafting`. No dataset rows were added, no candidate file was opened,
and no model, CUDA, cloud, or RL launch was used.

The v3 candidate and schedule remain preserved at their original paths. The v4
stage is `RL-02-rl-stage-manifest-candidate-v4.json`; its separate frozen
schedule is `source-row-draw-sequence-v4.json`. The explicit lead override is
`RL-02-quota-override-roxygen-floor-v1.json`, SHA256
`0825bd004c54d5cc09449e1654ce3f988d0355c4a0efb6b9b9ac728920b6f586`.
The override is bound by both its file hash and canonical identity hash
`a4a62e20c06b62901bc4aa59c1588b2a16d57dbd484ea549bcd92e6d01f95726`.

The requested finite plan remains 3,000 updates × 8 source prompt draws per
update = 24,000 source draws, 96,000 generated completion rows, and 384,000
TRL dataloader rows. G=4 remains 8 source groups, 32 completions, and 128
rows per optimizer update, with buffer reuse 4. The v4 family exposure is
exactly: finish_block 3,825; format_propagation 6,000; na_rm_propagation 375;
no_op 4,800; pipe_rewrite 2,400; rename_propagation 6,000; and
roxygen_drafting 600. The roxygen floor is 600/24,000 = 2.5%. Exactly 600
finish_block presentations are displaced from v3 (4,425 → 3,825); every other
family exposure, replay cap, and the 25% family ceiling is unchanged. The no-op
coverage remains 20%, above the 10% guard.

The override is checked before row allocation and includes the G=4 geometry,
requested source count, floor, replacement declaration, and complete expected
family ledger. The pool fails closed if finite source capacity cannot meet the
ledger or if any family receives a different count. The v4 schedule is then
written separately and its sequence hash is bound into the stage manifest and
receipt. Repetitions are allowed only in that finite draw sequence; the unique
selected-ID order remains the unchanged lead contract.

The reason for the floor is coverage: the current allocator's validated
additional-family set excludes roxygen, so its prior primary target and final
exposure were zero. Development documentation also records cap failures of
1/8 at 500 steps and 3/8 at 1,000 steps. This change provides modest train
coverage for the supported family without making a quality claim or changing
the reward verifier. A larger future SFT problem that needs roxygen coverage
should reserve it through another explicit lead quota identity.

The source schedule loader and `CampaignRepeatSampler` consume the v4 sequence
before G expansion and TRL buffer reuse. The full 8,440-row binding check passed:
24,000 source draws, 8,365 distinct selected IDs, 384,000 sampler rows, 128
rows in the first update, and exact resume prefix/suffix with source cursor 8.
The existing verifier fixtures still cover all seven families; parser failure
remains distinct from correct no-op, and parser validity does not admit rows.

The v4 stage remains candidate-only and launch-blocked pending root RL
admission, the merged-SFT theta0 manifest and exact merged-weight hash, and the
lead live tokenizer/renderer parent check. The quota change has zero extra GPU
budget and zero additional dataset rows.
