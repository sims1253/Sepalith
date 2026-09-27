# RUN-01 primary-profile VSIX preparation

Observed: 2026-09-12T20:19:52+02:00. Owner: `anyscale_admission`.

This packet records a fresh review artifact for the current v2 primary runtime profile. It covers packaging and static inspection only. No source, version, lockfile, protocol, client, request, model, server, accelerator, SSH, cloud, GUI, dependency, install, or network operation was performed by this packaging task.

## Artifact

The pre-existing VSIX remains byte-for-byte preserved:

```text
/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/vscode-sepalith-0.0.7.vsix
bytes 36517
sha256 a37097d0d8bcdd08b2b6acf404e48be107f7f9e0c870d4fb4830a4d9ec42f910
```

The new package is written under a fresh filename:

```text
/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/vscode-sepalith-primary-profile-0.0.7.vsix
bytes 38200
sha256 08604f9a69625a55ad43c65e6ec605325a345a6d304ce107431727dd6922c029
```

The package has five archive entries and 142279 uncompressed bytes:

```text
extension.vsixmanifest
[Content_Types].xml
extension/package.json
extension/readme.md
extension/dist/extension.js
```

Packaging used the already-present local cached `vsce` 3.9.2 binary at `/home/m0hawk/.npm/_npx/66fbc91407e86cd3/node_modules/.bin/vsce`, with `npm_config_offline=true`, `--allow-missing-repository --no-dependencies --no-rewrite-relative-links`, and an explicit fresh output filename. The first invocation without `--no-rewrite-relative-links` exited nonzero because the README contains a relative documentation link; the retry passed and did not overwrite either prior VSIX.

## Static package audit

The embedded package hash is `2a9edf2996852a37057008d8255cf61352407d8aecf98a59635e3f0a505b94a2`, with `main` `./dist/extension.js` and version `0.0.7`. The bundled extension contains the primary profile/backend markers `buildServerEnv`, `selectBundleForRequest`, `libggml-cuda.so`, `GGML_CUDA_GRAPH_OPT`, `manifestUrl`, and the managed-primary manual-server rejection. The source validator requires `profile.cudaGraphOptimization === 0`; `buildServerEnv` emits the profile value, so the bundled primary default is graph optimization 0. The profile also carries the validated 4096 context, 256 batch and ubatch, six serving threads, two HTTP threads, parallel 1, requested 99 GPU layers, and 192-token client cap.

The archive contains no `src`, `scripts`, `node_modules`, model (`.gguf`/`.safetensors`), credential/secret, native library, `llama-server`, or lockfile entry. This is an archive-content check only; it does not prove a real managed model manifest, runtime download, clean-cache launch, editor activation, or accelerator execution.

## Validation run

In the extension directory, each requested check was run once with the existing project dependencies:

```text
npm run build                  PASS; TypeScript compile and esbuild bundle
npm run check-runtime-profile PASS;  primary profile/library/backend/manual-bypass/legacy fixtures
npm run check-runtime         PASS; 58 shared manifest and offline runtime fixtures
npm run check-prm05           PASS; 38 assertions
```

The full `npm run lint` was also run. It exits 1 with 79 errors and 3 warnings. The errors are distributed as 11 `campaign_client.ts`, 43 `campaign_protocol.ts`, 4 `campaign_requests.ts`, 6 `campaign_selection.ts`, 3 `context_select.ts`, 5 `history_provider.ts`, 1 `check-campaign-selection.ts`, 1 `check-history.ts`, and 5 `check-prm05.ts`; warnings are 2 in `check-provider-requests.cjs` and 1 in `check-prm05.ts`. The 74 errors outside `check-prm05.ts` are in campaign/context/history/check files absent from the preserved `pre-edit-exec` snapshot. A direct pre-edit-exec lint recorded four `no-unknown-parameters` findings in its old `runtime.ts` and none in its old `extension.ts`; the current targeted runtime/extension files have no findings. No broad lint repair was attempted in this packaging task.

Protected bytes currently hash as follows and were not edited here:

```text
src/campaign_protocol.ts ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824
src/campaign_client.ts   0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333
src/campaign_requests.ts 4fdc6a881b93ae8beb9614c2b210f80fd5b0a2d5592ae02c8174345612a17535
package-lock.json         ef1daabf3f81aea2cdc6e7a00248745d898eeed7caca5df1269f4bc846218880
```

The current packaged source inventory is `src/runtime.ts` `4de415237088984d53eba6b182dba975c12216a516c52f9fecb3a916dc841616`, `src/extension.ts` `4fa6a06d3d3cbf69eda51195dc81b8b793739a1ed0f0763ad3d7d63a475a9727`, `package.json` `2a9edf2996852a37057008d8255cf61352407d8aecf98a59635e3f0a505b94a2`, `.vscodeignore` `3859c88273994dbb6aa0b950e631225c4a07c852737790c5a34b1e118067ae33`, and generated `dist/extension.js` `a04ba4cc54b8c34fd032690de45ae20b8464cb6a395f1de971149aac20fd0386`.

## Gates for root

This is a reviewable package artifact, not a release or promotion. Root still needs to materialize an actual primary HTTPS manifest bound to the accepted final model and tokenizer, provide real hashed executable and sibling backend libraries, leave `serverPath` empty for managed primary startup, and run the clean-cache managed load. Root owns native protocol/readiness, actual editor/GUI placement, CUDA/Vulkan/offload admission, final weights, publication, and spend. The source profile cannot establish accelerator execution. The `vsce` package warning about a missing LICENSE remains a packaging warning; no unrelated project file was added to address it.

Next concrete action: root should verify the VSIX SHA and static inventory, then use the existing real primary-manifest producer and controlled local VS Code profile to test managed installation/launch after the final model and library hashes are available. Keep this fresh filename alongside the old VSIX.
