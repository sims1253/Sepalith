# Root staging for LAN automatic A/B

Two private notebook capsules are ready. Root arm directories are unchanged.
No transfer or launch was performed. The accepted merged runtime is unchanged;
only per-arm identity metadata, documentation, guard paths, binding SHA, UUID,
and the required debounce argument differ.

From the PLAN worktree, root can check and transfer with:

```sh
python3 docs/campaign/work/remote-auto-root-staging-v1/transfer_capsules.py
python3 docs/campaign/work/remote-auto-root-staging-v1/transfer_capsules.py --transfer both
```

The first command is local only. The second uses existing SSH authentication to
`m0hawk@192.168.178.40`, verifies the current notebook renderer source hash, and
transfers the two explicit archives. It runs no editor, native process, GPU
operation, port check, or profile operation. Use `--transfer 1500` or
`--transfer 350` to select one arm. Existing identical capsules are verified;
different existing content is rejected without overwrite.

| Arm | Notebook capsule | Instance UUID |
| --- | --- | --- |
| 1500 ms | `remote-auto1500-a-capsule` | `d9d73e96-1510-4aac-8ff8-57340f9c40a2` |
| 350 ms | `remote-auto350-a-capsule` | `f3882149-bf3a-4b02-ba01-ae15fed73165` |

Both capsule paths are beneath
`/home/m0hawk/.local/share/sepalith-campaign-20260915`. Each contains 15 files:
the ten merged capsule files, its original manifest as provenance, the selected
arm binding, unchanged candidate VSIX, derived notebook guard, and complete
per-arm manifest. Files are copied from an explicit list. No certificate,
private key, model, or native binary is included. The private asset key was not
read or hashed.

Each `remote-auto*-plan.json` records the complete guard and editor argv,
archive and capsule hashes, remote run/supervision paths, renderer hash, and
root controller path. The guard keeps the accepted 300-second limit, cleanup
logic, and 2 GiB notebook memory floor. Its only code differences are the arm
paths, UUID, binding SHA, and `--debounce-ms`. The root controller retains its
600-second bound and fresh native process per arm. The planned order is 1500
then 350 as a diagnostic pair, not a promotion sample.

Static verification covers both bindings, UUIDs, model/runtime manifest
equality, VSIX bytes and hash, route, context 4096, output cap 192, one CUDA
bundle, one request slot, and GraphOpt 0. Renderer SHA
`e04cb8ec68016ccf4557690a2c8c908c3534614af1abe4c5c557157548169d78`
comes from the accepted prior notebook guard. This worker did not inspect the
live notebook renderer; the transfer receiver requires that exact fresh source
hash before writing files.

Seven CPU tests pass. They verify complete archive membership and hashes,
per-arm guard/controller alignment, the exact limited guard delta, unchanged
runtime and VSIX bytes, shell-argument round trips, and rejection of wrong arm,
UUID, binding, path traversal, links, duplicate files, extra private-key files,
corrupt payloads, missing files, and oversized archives. No SSH or launch was
used for tests.

Root must complete these launch prerequisites:

1. End corrected RL and verify CUDA ownership is released. Serving cannot
   overlap training.
2. Create and review each root arm's full artifact `manifest.json` and matching
   admission receipt. Both were absent at staging review. Root alone verifies
   the real model/native/library artifacts and keeps its asset key local.
3. Verify notebook capsule transfer receipts, renderer identity, fresh run and
   supervision paths, free owned ports, and the notebook/desktop memory guards.
4. Launch only through the existing root arm controller after admission. It
   owns native identity, gateway, asset service, SSH forwards, and cleanup.
5. Accept actual visible, multiline, and delayed-cancellation controls before
   using A/B visibility metrics. Missing line 2/3 ownership remains blocked;
   sibling view-zone metadata is diagnostic only. Review screenshots, keyboard
   IDs/document versions, frame coverage, gateway/native request evidence, and
   process release separately.

The future root launch commands are:

```sh
python3 docs/campaign/work/lead/remote-auto1500-a/run_remote_editor.py
python3 docs/campaign/work/lead/remote-auto350-a/run_remote_editor.py
```

Do not run them concurrently. Transfer success and process completion do not
admit metrics, establish debounce causality, or promote an arm. Final data was
not accessed and remains under its existing time-and-freeze gates.
