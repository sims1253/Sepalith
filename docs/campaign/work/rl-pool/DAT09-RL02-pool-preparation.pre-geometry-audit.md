# DAT-09 / RL-02 pool preparation

This is a CPU-only, candidate-only preparation artifact. It binds the accepted
RL-02 train/context rows to the original DAT-05 admission evidence and the
approved SFT train-row file, records measured token geometry, and computes a
finite replay plan. It does not modify the global registry, open candidate
file contents, load a model, or authorize an RL launch.

The prepared stage artifact is
`RL-02-rl-stage-manifest-candidate-v1.json` (artifact SHA256
`363d789606779e34b590feac2c20aa07b5b8a9440e762d80115cb6bcdfa6704d`; internal
manifest identity SHA256
`ef27f07f439f613f3ee338cee2e82fe23a5ced8e391b7e85fad753e90516e506`). It has
8,440 selected and eligible train rows in the root lead order. The order file
SHA256 is
`24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d`, its
ordered-ID identity is
`7e994a3e74a642c149908b0737dfd2befed055e336f7b94400ee9773a6a88d2d`, and the
source/geometry row identity is
`d3187195f0ebe401b69df244bbbff0f2a28fb0500185428701f933e31e8eab60`. The
context sidecar SHA256 is
`6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f`.

Family counts are finish_block 2,416; pipe_rewrite 1,683;
format_propagation 1,254; rename_propagation 1,189; no_op 1,140;
roxygen_drafting 675; and na_rm_propagation 83. Evidence is visible for 4,209
rows and local-only for 4,231 rows. All selected rows satisfy the measured
2,048 prompt and 192 completion limits without truncation. Prompt lengths are
120..2,048 (5 at the cap; 3,406 over 1,024), completion lengths are 7..192
(8 at the cap), and combined lengths are 138..2,136 (223 over 2,048).

The readout template requests 3,000 updates × 8 source prompt presentations per
update = 24,000 source draws. Each source draw expands to four generated
completion rows (96,000 completions), and the four TRL buffer-reuse copies yield
384,000 dataloader rows. The policy uses a 10% no-op guard, a 20% no-op target,
a 25% family ceiling, ordinary replay cap 8, and small-pack cap 3. The finite
plan reaches all 24,000 source draws / 3,000 full updates, so its status is
`complete`. Deterministic source-round-robin backfill uses finite eligible rows
only. Source family exposure is finish_block 4,425; format 6,000; na.rm 375;
no_op 4,800; pipe 2,400; rename 6,000; roxygen 0. The no-op target and guard
are both met (4,800 and 2,400 respectively); the family ceiling limits rename
and format to 6,000 and their 1,200-draw primary deficits are explicit.

The frozen source-row draw sequence is
`source-row-draw-sequence-v1.json` (24,000 IDs, 7,765 distinct selected IDs;
artifact SHA256
`95d2f557840d554713ecf01f01117a319a0d4a57c062b27e86631a73653f0d97`; sequence
identity SHA256
`57ec84d2bd359457e49088a0aa3b734a1440757fa7f97bd04e308740312c69c8`). It is
bound to the selected-ID, ordered-ID, row-identity, G=4, eight-groups, and
four-buffer-reuse identities. Repetitions occur only in this finite schedule;
the selected-ID data contract remains unique and ordered.

`campaign_rl_train.py` now verifies this schedule before live trainer
construction and maps its row IDs to the unique selected dataset positions.
`CampaignRepeatSampler` consumes the frozen source sequence first, expands each
source prompt to G candidates, and then applies the four-buffer reuse required
by TRL accumulation. Thus G=4 has 8 source groups, 32 generated completions,
and 128 dataloader rows per optimizer update; the sampler tests also cover the
G=2 geometry and exact resume prefix/suffix.

The current reward verifier fixtures cover all seven families. For every family,
known-correct output receives the exact reward, a wrong output is protocol-valid
but receives zero, semantic no-op behavior is checked, and a missing-terminal
parser failure is separately reported as protocol-invalid. No family is admitted
because it is merely parser-valid; no unsupported family was reported.

The original DAT-05 report remains admitted and hash-bound to the unchanged
11,839-row global registry (11,764 train and 75 dev). The pool excludes 75 global
dev rows and records 3,324 admitted train rows outside this selected RL-02
order. Accepted RL-02 materialization exclusions are retained in the manifest;
selected profile exclusions are zero, and TU3/contradictory exclusions are zero.
Final/dev candidate contents were not opened.

Launch is refused. Root RL admission is false, the merged-SFT theta0 stage
identity and exact merged-weight SHA256 are unresolved, and the live
tokenizer/renderer parent check remains root-owned. The next action is for the
lead to supply the merged-SFT parent manifest and exact merged-weight hash,
then recheck the finite quota decision and perform root admission before any
RL entry or quality claim.
