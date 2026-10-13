# RUN-01 runtime profile implementation

Observed 2026-09-12T19:24:00+02:00. Source-only implementation for root review; no
model, server, CUDA, Vulkan, SSH, cloud, install, or dependency operation was
performed.

The correct source checkout is `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`;
PLAN source files were restored byte-for-byte after an initial worktree-boundary
check. The implementation adds a strict primary b10453 launch profile while
retaining the legacy b4 path.

## Implemented

- `runtime.ts` exports the `cuda` backend and exact primary profile geometry:
  context 4096, batch/ubatch 256/256, parallel 1, generation/batch threads
  6/6, HTTP threads 2, and GPU layers 0 for CPU or 99 for accelerator bundles.
  Vulkan `gpuLayers=99` is the requested setting; the existing measured m0pad
  observation records 43/43 actual offload and is not re-measured here.
- Primary profiles require model SHA equality with the top-level model asset,
  pinned PRM-03 protocol/tokenizer fields, b10453 source commit/tree, and a
  server SHA equal to the executable asset. Every bundled file remains an
  HTTPS, size, and SHA-256 checked asset. Missing launch profile fields reject
  the manifest. Primary cap is 192; no seed requirement was added.
- Explicit backend selection uses `selectBundleForRequest` and fails when the
  requested backend is absent instead of silently selecting CPU. Auto selection
  keeps its existing fallback behavior.
- `buildServerArgs` binds the profile flags in array form. `buildServerEnv`
  prepends the managed runtime directory to `LD_LIBRARY_PATH` on Linux, sets
  `GGML_CUDA_GRAPH_OPT=0`, and removes or sets the Vulkan graph-disable flag
  according to the validated profile.
- `Sidecar.start` classifies every configured manifest before port/path use,
  rejects a primary manifest with a manual `serverPath`, carries the validated
  launch profile/runtime directory/backend from provisioning, and uses the
  exact profile args/environment only for managed primary children. The legacy
  manual and legacy managed route keeps its old `/v1/completions` behavior and
  old defaults.

## Test material

`extensions/vscode-sepalith/scripts/check-runtime-profile.ts` is a model-free
structural test. It validates the local primary fixture shape, model binding,
missing/invalid launch profiles, explicit missing CUDA, exact argv, Linux
loader environment, graph defaults, manual primary rejection, and legacy
acceptance. The reviewable JSON fixture is
`primary-manifest-fixture.json`; it uses dummy HTTPS URLs and one-byte hashes and
is not a release artifact.

The prior PRM-05 test fixture was updated to carry the now-required primary
model SHA and launch profile; no campaign source/client/protocol/request file was
changed.

## Checks

- `npm run build` — PASS (TypeScript compile and esbuild bundle).
- `npm run check-runtime-profile` — PASS.
- `npm run check-runtime` — PASS (58 shared fixtures plus offline integrity,
  cache, override, cancellation, and deadline checks).
- `npm run check-prm05` — PASS (38 assertions).
- `oxlint --deny-warnings scripts/check-runtime-profile.ts src/runtime.ts src/extension.ts` — PASS.
- Full `npm run lint` remains FAIL with 79 errors and 3 warnings in the
  pre-existing campaign/source lint set; the owned runtime, extension, and new
  profile test have no targeted lint findings.

## Remaining gates

No live primary manifest/model, clean-cache dynamic load, host startup, native
readiness, Vulkan/CUDA behavior, GUI placement, or release publication was
performed. The manifest producer must emit the new `modelSha256` and per-bundle
`launchProfile` fields, and root must review/admit the actual SFT500 Q8 model
asset and runtime files before launch. The structural fixture is ready for
root's local preflight; root owns all live launches and acceptance.
