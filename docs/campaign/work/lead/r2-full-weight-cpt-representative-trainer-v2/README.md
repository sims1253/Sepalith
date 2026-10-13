# Representative complete-document full-weight CPT trainer

This packet prepares an executable diagnostic-cohort trainer. The exact 1,024-document diagnostic cohort is now data-admitted and bound in
`cohort-input.preparation.json`: 1,046 unique 16K rows plus 10 named alignment
replays, 66 updates, and no truncation. Parent, optimizer rates, learning rate,
checkpoint cadence, and mandatory intermediate stop remain unbound. The
template cannot launch; `bind_representative_cpt.py` requires a later root
scientific admission.

Version 2 binds the reviewed `saved_precision.py` helper byte-for-byte. After
`FastLanguageModel.for_training` has completed its preparation and before the
381-parameter inventory or optimizer is created, the trainer reads the
already-admitted parent safetensors header, promotes only tensors saved as F32,
and copies their exact saved values. It requires the admitted parent-specific
F32 tensor and element counts. Both CPT250 and the newly merged cloud1902 candidate have independently
header-derived all-BF16 identities (381 BF16, zero F32 for cloud1902). Neither
parent is selected here. Missing names,
shape changes, nonfinite values, unexpectedly large F32 tensors, count changes,
or helper/audit hash changes fail closed. BF16 tensors are not copied or cast.
The same bound precision identity is part of checkpoint identity, while the
existing full-state resume path and cursor semantics remain unchanged.

The row contract is schema-1 materialized raw-R CPT: manual BOS 0 and EOS 1,
stored labels and attention mask, masked carry tokens, terminal EOS supervised
only on the final document chunk, and exact source spans. Preflight reconstructs
every admitted document from all its chunks and rejects gaps, overlaps,
duplicate chunks, missing terminal chunks, token/label mismatch, package or
document overlap with the fixed 499-row/2048-token causal comparator, or a
schedule that does not expose every unique row before named padding replays.
Natural row lengths through the admitted 8K/16K/32K profile are accepted; the
old 384-row, 8K-only, 24-update LR panel is not embedded.

Training preserves the reviewed 381-tensor/2,516,756,480-parameter full-weight
path, aurora-mix dispatch with FP32 optimizer states, native tokenizer and
AddedToken restoration at runtime/save/evaluation boundaries, sequential exact
draw cursor, finite-gradient/resource telemetry, dense full-state checkpoints,
and same-filesystem seal plus atomic publication on E. It does not write a
duplicate light dense checkpoint.

Root must choose a `mandatory_stop_step` below the full cohort horizon. That
step must align with a full checkpoint, fixed-holdout evaluation, and preserved
milestone. The callback saves and stops at that optimizer boundary. A resume
requires a separate `continuation-admission.template.json` instance binding the
same recipe hash, durable checkpoint manifest hash, path and step. Missing or
changed evidence fails before model loading. Cursor restoration remains
`checkpoint_step * 16`; the resumed process does not stop again at the already
reviewed milestone and must preserve the same schedule and optimizer state.

`root-commands.json` contains exact binder, CPU preflight, initial run, resume,
and external graceful-stop argument arrays. GPU commands still need the root
CUDA lease, host/resource guard, timeout and explicit launch receipt. Bulk
runtime/checkpoint output stays on E.

The CPT250 merged parent is available only as a prepared candidate. The cloud CPT1902 merge is available as a verified but unpromoted candidate.
Parent, LR, checkpoint cadence, and the mandatory gate remain scientific root
decisions. The inherited source mechanics and prior pilot outcomes do not
promote any model-quality result, and this representative diagnostic cohort is
not the full eligible corpus.
