# SFT-11 expanded C recovery packet

This packet resumes the untouched B full checkpoint at step 150 and stops at
the existing mandatory step 250. It preserves the original data, draw order,
effective batch, optimizer schedule, horizon, parent, and checkpoint cadence.

The B source snapshot remains immutable at
`56132b2fd3b21cef88043cff5ea2b1f4db7495a481f3b0d0a7485a5ee3584676`.
The C recovery source has the distinct snapshot identity
`2d659de02afc205588f5c2417e32032a96b677871eb7ef5379085f10cc8de48f`.
It verifies the B manifest at its exact hash, checks every checkpoint file,
requires a full checkpoint, and permits only a source-only identity migration.
All other identity fields must match byte-for-byte. The archived B checkpoint
is not rewritten or resealed.

The same seam permits a later same-source recovery from a full checkpoint such
as C step 200. The step must match `full_every`, remain below a declared future
mandatory stop, and carry the exact recipe-pinned manifest, identity, sampler
schedule, and consumed-draw cursor.

Root must execute the snapshot, verify the returned identity, enqueue the
runner recipe, and launch `run_owned.py`. The runner wrapper explicitly resumes
its fresh paused queue before dispatch. No command in `commands.json` has been
executed except the CPU-only preflight command.
