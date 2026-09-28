# Native CPT trainer preparation

This is a fresh runtime source for continuing the exact prefix-extension CPT
stage after checkpoint 194. It does not modify or authorize the active run.

## Preserved training state

The runtime overlay retains the canonical recipe SHA-256
`579269f0f203ae1eed9cdb96af0f535f19c5953e2cf3e19c52615b7545e10f70`
as its scientific identity source. Parent weights/tokenizer, Aurora optimizer,
FP32 state, constant scheduler after warmup, RNG, sampler schedule, global
offset 66, stage cursor, effective batch 16, cadence, and milestones remain in
the checkpoint identity. The fresh runtime source is admitted separately; it
does not forge a checkpoint with a new scientific source identity.

The copied stage-transition cursor helper now also requires
`sampler.cursor == sampler.stage_cursor`. This closes a fail-open metadata case
without changing a valid checkpoint.

## Reusable inputs and resume

`prepare_native_stage_spec.py` deterministically builds a reusable data stage
spec from the canonical bound recipe and small manifests. The prepared spec
contains only the immutable cache, rows, and schedule: 9,330,242,806 bytes
(8.689 GiB). Checkpoint 90 is not copied. The selected resume checkpoint is
verified separately and is also the model bootstrap before Trainer restores its
optimizer/scheduler/RNG state. Later checkpoints never cause the data bundle to
be recopied.

At every launch, `native_launch_attestation.py` independently verifies the
selected full resume checkpoint and the reusable data base. It holds all file
descriptors and advisory locks for the child lifetime. An E checkpoint is
therefore fully verified once on first migration. A later native hot checkpoint
can be resumed without copying the 26.58 GB base again.

The actual front door requires both a root relocation admission and a root
runtime-source migration admission. It proves the runtime overlay has the same
scientific `identity()` as the canonical recipe, then consumes inode-bound
attestations for the staged parent and corpus. Missing, forged, changed, or
same-size-replaced inputs fail before model/CUDA imports.

## Native checkpoint lifecycle

Trainer serialization and checkpoint sealing occur under the native trainer
root. Before Trainer begins a save, the 70 GiB use cap and 70 GiB remaining-free
floor are checked with a conservative 18 GiB next-checkpoint allowance. Native
Trainer retention is one hot checkpoint; durable E retention remains the latest
two plus selected milestones.

After sealing, each payload is copied and hashed in one pass into a hidden E
directory, fsynced, and atomically renamed. The native source remains until
Trainer rotation after a newer durable checkpoint. Resume/root acceptance still
performs full checkpoint verification. An interrupted publication cannot expose
the final E checkpoint name.

## Timing evidence

The runtime emits and fsyncs structured events and flushes each event to stdout:

- `load_bound_start` / `load_bound_end`;
- `resume_validation_end` and `dataset_preflight_end`;
- `model_load_start` / `model_load_end`;
- `trainer_resume_start`;
- `checkpoint_save_requested` and `trainer_serialization_end_seal_start`;
- `native_seal_end_publish_start` and `durable_E_publish_end`;
- `terminal_durable_verify_start` / `terminal_durable_verify_end`.

These events separate source validation, resume validation, model load, Trainer
serialization, seal hashing, cross-filesystem publication, and terminal E
readback. No throughput improvement is claimed until root runs checkpoint-194
under the reviewed guard and compares these durations.

## Required root gates

1. Accept checkpoint 194 with its complete optimizer/scheduler/RNG/sampler
   manifest and exact cursor `(194 - 66) * 16 = 2048`.
2. Review/freeze the reusable base receipt, relocation admission, runtime-source
   migration admission, generated runtime recipe, and continuation admission.
3. Run the actual CPU front door through the attestation wrapper.
4. Confirm native projected use and free-space floor, fresh native output path,
   E archive identity, stable CUDA lease, host memory guard, and deadline.
5. Root alone launches. The first native save is global step 322 unless root
   issues an admitted graceful save/stop at another optimizer boundary.

The initial checkpoint-194 read remains on E in the supplied command. Root may
stage it separately, but that increases native use and must remain in the
capacity accounting until its locked launch ends. The reusable base is never
recopied for later checkpoints.
