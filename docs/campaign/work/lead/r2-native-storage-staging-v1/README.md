# Native storage staging preparation

Status: **prepared and CPU-tested; no bulk copy or training launch performed**.

This packet stages the current checkpoint-90 full state, expanded streaming
cache, source rows, and draw schedule on the native Linux filesystem. The
scientific identities remain their original SHA-256 values. Physical paths are
changed only in a separately admitted runtime overlay.

## Current bound inputs

The source recipe is
`r2-cpt90-prefix-extension-root-v2/bound-recipe.json`, SHA-256
`579269f0f203ae1eed9cdb96af0f535f19c5953e2cf3e19c52615b7545e10f70`.
The prepared stage contains 26,581,176,959 bytes (24.756 GiB):

- checkpoint-90: 17,250,934,153 bytes including its manifest;
- combined cache: 3,923,776,614 bytes including its manifest;
- source JSONL: 5,391,073,868 bytes;
- draw schedule: 15,392,324 bytes.

With two checkpoint-sized native hot saves, projected campaign-native use is
61,083,045,265 bytes (56.888 GiB), below the 70 GiB cap. The capacity command
also requires at least 70 GiB filesystem free after the next save.

## Integrity and publication

`native_stage.py build` opens every source with `O_NOFOLLOW`, hashes bytes while
copying, compares the open source descriptor before and after the copy, fsyncs
the destination, makes files read-only, and atomically renames the completed
bundle. Tree inputs must have an exact manifest inventory; unlisted files and
symlinks are rejected.

`native_stage.py verify-run` independently rehashes native bytes once per
launch, holds every verified file descriptor and a shared advisory lock for the
child lifetime, and passes an inode-bound attestation. A native-aware trainer
can use `native_attestation.require_attested` instead of repeating the same
payload hashes. This is not blind stat trust: each process begins with a full
native-byte verification, and same-size mutations are rejected.

`native_checkpoint_publish.py` copies a sealed native full checkpoint into a
hidden directory on E. It hashes each source payload during the copy, verifies
the exact sealed inventory and optimizer/scheduler/RNG files, fsyncs files and
the staging directory, then atomically renames on E. The final checkpoint name
never exposes a partial copy. It leaves the native source intact; root may
remove it only after independent durable readback/acceptance.

## Required fresh trainer integration

The active trainer and recipe remain unchanged. They cannot consume the runtime
overlay yet for three concrete reasons:

1. `stage_transition_contract.inspect_transition` compares the checkpoint's
   physical path to the old root decision. A fresh source must accept a staged
   path only when `storage_relocation.canonical_paths.source_checkpoint`
   equals that decision path and both locations bind the same campaign manifest.
2. `prefix_extension_contract.verify_source_conservation` follows the source
   path embedded in the cache manifest. A fresh source must use the admitted
   staged rows alias, require its SHA-256 to equal the embedded identity, and
   consume the inherited descriptor attestation. The already reviewed prefix
   relationship may be cached only through a new root-pinned verification
   receipt binding both cache manifests, schedules, source checkpoint state,
   and verifier source hash.
3. `validate_context_and_storage` allows only E trainer/archive roots. Input-only
   staging can keep this output policy. Native hot checkpointing requires a
   fresh lifecycle source with native `output_dir`, the capacity guard before
   each save, and `native_checkpoint_publish.publish` to the existing E archive
   before telemetry says durable. Resume must use the E checkpoint until that
   source has an actual interruption/resume proof.

The runtime overlay changes only physical paths. `identity_view` is compared
before and after relocation and includes parent weights, tokenizer, renderer,
data/cache hashes, source manifest, optimizer policy, and schedule.

## Root sequence

1. Review and hash this frozen packet. Ensure no competing bulk I/O.
2. Run `stage_once` from `root-commands.json`. This is the first and only full
   read of each listed E input during staging.
3. Independently inspect the stage receipt and create a fresh relocation
   admission with the exact bound recipe and stage receipt hashes.
4. Generate the runtime overlay. Do not run it with the current trainer.
5. Copy the current trainer into a fresh packet and implement the three narrow
   integration changes above. Invoke its actual preflight through
   `verify-run`; require the inherited attestation for every staged large file.
6. For native checkpoint output, run the capacity guard before a save, seal on
   native storage, publish to hidden E staging, rename atomically, then perform
   the normal independent E verification before selecting or resuming it.

No I/O benchmark was run because the active trainer and prefix checks were
performing bulk I/O. The earlier 256 MiB C-drive probe is not treated as a
measurement of this native filesystem or a production checkpoint.
