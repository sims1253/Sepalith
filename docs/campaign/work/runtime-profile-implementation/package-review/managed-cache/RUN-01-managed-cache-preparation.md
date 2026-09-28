# RUN-01 managed-cache preparation

Observed: 2026-09-12T20:50:01+02:00. Owner: `anyscale_admission`.

This packet validates the reviewed managed-runtime cache path with the existing local CUDA server and loader-name library copies. It performs file copies, streaming hashes, ELF `readelf -d`, and calls the existing `validateManifest` and cache-hit `install` functions. It does not read or install a model, start a server, query a GPU, import the editor, access the network, or change source/state.

## Inputs and accepted identities

- Reviewed runtime: `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/src/runtime.ts`, SHA-256 `4de415237088984d53eba6b182dba975c12216a516c52f9fecb3a916dc841616`.
- v2 profile receipt: `/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/receipts/RUN-01-runtime-profile-implementation-v2.json`, SHA-256 `323b48fe644b2f0862475f5b951aeafffe27849be2571401f2fb28d3eaee1ae5`.
- Existing fresh VSIX: `vscode-sepalith-primary-profile-0.0.7.vsix`, 38200 bytes, SHA-256 `08604f9a69625a55ad43c65e6ec605325a345a6d304ce107431727dd6922c029`.
- Existing loader-name CUDA bundle: `/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/runtime-profile-implementation/library-closure/packaging/cuda-b10453-alias-bundle`, inventory digest `718a0ed20173b570e50da1fe0de81b28fefe202eb7fda18e815cf1226e033fff`.
- Existing CUDA fragment: `cuda-b10453-alias-manifest-fragment.json`, SHA-256 `c7a5f2fbba853acd40d826080e01d87f142a8f3d4a3b9bdbab0a8a99ce79e2e9`; it has real local sizes/hashes and no URL/model fields, so it is not itself a release manifest.

The bundle has ten regular, non-symlink files: `llama-server` plus `libggml-base.so.0`, `libggml-cpu.so.0`, `libggml-cuda.so.0`, `libggml.so.0`, `libllama-cli-impl.so`, `libllama-common.so.0`, `libllama-server-impl.so`, `libllama.so.0`, and `libmtmd.so.0`. Total copied bytes were 178108840. The server is 17896 bytes, SHA-256 `e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee`; the CUDA backend alias is 155531808 bytes, SHA-256 `3c96a25c15a70fbd568ce77599fe628fdc50c7ce27bc9ebcac297692537e9a78`. The complete individual inventory remains in the input packaging receipt.

## Reproducible probe

The owned preparation probe is [probe_managed_cache.mjs](./probe_managed_cache.mjs), SHA-256 `b4a2f518d612841977ea347c3c65daa45d0e04b865bf6b99a4fbd08091ae732b`. Run from the PLAN root with the existing local Node:

```sh
node --experimental-strip-types docs/campaign/work/runtime-profile-implementation/package-review/managed-cache/probe_managed_cache.mjs
```

The probe temporarily copies the existing alias files into `/tmp`, verifies regular-file status, size and SHA-256, checks every internal CUDA `DT_NEEDED` name with `/usr/bin/readelf`, and removes the temporary directory in a `finally` block. It imports only the reviewed runtime module, whose top-level path contains protocol constants and no model/editor/GPU initialization. The probe replaces `globalThis.fetch` with a throwing function only during cache-hit calls; a fetch would therefore fail the probe rather than silently access the network.

The probe output was:

```text
manifest_actual_cuda=PASS files=10 inventory=718a0ed20173b570e50da1fe0de81b28fefe202eb7fda18e815cf1226e033fff
cache_copy_and_hash=PASS regular_files=10 bytes=178108840
internal_dt_needed=PASS names=libggml-base.so.0,libggml-cpu.so.0,libggml-cuda.so.0,libggml.so.0,libllama-common.so.0,libllama-server-impl.so,libllama.so.0,libmtmd.so.0
runtime_install_cache_hits=PASS assets=10 network=not_called
wrong_library_sha_rejection=PASS
missing_loader_alias_rejection=PASS alias=libggml-base.so.0
managed_cache_static_probe=PASS cleanup=PASS
```

The actual alias bundle was merged into the already reviewed structural primary fixture only for in-memory validation. The fixture's one-byte dummy model was never copied, passed to `install`, or loaded. The wrong-SHA and missing-alias cases call the existing v2 validator and are not new mirror-only test fixtures or product tests.

The direct existing profile check was also run without rebuilding or changing source:

```text
node dist/check-runtime-profile.cjs
runtime primary profile checks passed
```

Node emitted its existing `MODULE_TYPELESS_PACKAGE_JSON` warning while importing the TypeScript runtime; the command exited 0 and made no package change.

## What this proves

The current v2 schema accepts the real CUDA alias bundle when its `requiredLibraries` entries match actual bundle names and hashes. The cache-hit branch of `install()` verified all ten assets without invoking `fetch`. The static ELF graph resolves the internal `.so.0` loader names from the staged files. A wrong required-library SHA and removal of `libggml-base.so.0` from both the bundle and required-library list are rejected as `Invalid primary launch profile`.

This is a static/cache-hit result. It does not prove `install()` can retrieve an asset from a real HTTPS object, that host `libcudart.so.13`, `libcublas.so.13`, `libcuda.so.1`, system libraries, or driver libraries are present on the target, that the managed extension can launch from a clean cache, or that CUDA offload executes. It also does not validate the notebook Vulkan bundle; its target alias closure remains root-owned live evidence.

## Remaining root gates

Root must publish real HTTPS URLs, add the accepted final model/profile and model SHA, materialize the clean managed cache with `serverPath` empty, and run the controlled editor/sidecar startup. Root then owns native protocol/readiness, actual backend/offload, final model admission, GUI activation, and release promotion. The structural fixture, local CUDA fragment, and VSIX are preparation evidence only.
