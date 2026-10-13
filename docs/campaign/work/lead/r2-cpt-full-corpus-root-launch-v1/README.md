# Full-corpus CPT root launch binding v1

This packet binds the complete terminal cache for the current 16K union
snapshot to a fresh destination stage starting from the selected full-state
representative checkpoint at global step 66. It preserves the source model,
hybrid FP32 optimizer state, constant-with-warmup scheduler after its original
two warmup steps, and CPU/CUDA RNG. The new corpus sampler begins at cursor
zero.

The terminal cache contains 183,084 unique rows from 177,190 complete documents
and 10,046 packages. It exposes 461,205,327 input tokens and 461,010,455 causal
loss tokens without truncation. Four named tail replays make 183,088 draws,
11,443 stage updates at effective batch 16, and terminal global step 11,509.
The first mandatory save/evaluation/root decision is stage step 128/global step
194. Only this current snapshot is admitted; the 1,999-document cap frontier
remains pending rehash and content dedup.

An actual CPU preflight rehashed the bound parent and cache, checked the fixed
499-row heldout comparator for document/package isolation, and passed with
initial cursor zero and all 183,088 draws remaining. During that check, the
accepted v2 loader exposed a global-vs-stage cadence bug. V2 is preserved.
Fresh stage-transition v3 changes the check to
`(mandatory_global_step - source_offset) % checkpoint_every == 0`; its runtime
save callback already used this stage-local form. Fourteen tests pass, including
the real Transformers recursive transition and explicit source66 to
stage128/global194 controls.

`guard-launch-command.json` is the exact proposed guard invocation, and
`trainer-command.json` is its child argv. Both are nonexecuting artifacts.
The guard uses 10,800 seconds, a 14 GiB startup floor, a 6 GiB soft floor, and
the reviewed fixed 4 GiB hard floor. Root retains CUDA lease and launch control.
E trainer/archive paths remain the v2 template defaults.
