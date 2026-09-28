# RUN-01/RUN-02 current EXEC editor build preparation

Observed 2026-09-12. This packet prepared the reviewed EXEC `vscode-sepalith` 0.0.7 package for root review. Only project-local dependencies, build outputs, and the VSIX were generated. No source, `package.json`, lockfile, installed editor extension, user setting, GUI, server, model, SSH host, campaign state, or training snapshot was changed.

## Source and dirty-state binding

The EXEC root is `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`, detached at HEAD `a7345e35219ceb624957115c39b869101026a817`. The checkout was already dirty with the reviewed campaign integration. The source hashes below were identical before and after dependency installation, build, checks, and packaging.

| Dirty source file | SHA-256 |
| --- | --- |
| `extensions/vscode-sepalith/package.json` | `3bcaa631b43076bce5f6913bd3ccf252401172698c669ec0b31dc739c80900fa` |
| `extensions/vscode-sepalith/src/extension.ts` | `86367f6615e02e59bed425dea8b33f0c210aaf1c0be6a9c6d91e840111bf5cd9` |
| `extensions/vscode-sepalith/src/runtime.ts` | `3a26fb1677eb97488fa03491ba795ec0cca264db24af8f5f7db983bf46e5bfbf` |
| `extensions/vscode-sepalith/src/campaign_client.ts` | `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333` |
| `extensions/vscode-sepalith/src/campaign_protocol.ts` | `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824` |
| `extensions/vscode-sepalith/src/campaign_requests.ts` | `4fdc6a881b93ae8beb9614c2b210f80fd5b0a2d5592ae02c8174345612a17535` |
| `extensions/vscode-sepalith/src/campaign_selection.ts` | `70a03d86c8d1c2cd0e117edcbd69cfbac193382b20b555e2d0a327e7e006e8e3` |
| `extensions/vscode-sepalith/src/context_select.ts` | `a72edaf733395d7b4839f2ca9ed0c2092e4051359279f6839c93e2d5d04b9d59` |
| `extensions/vscode-sepalith/src/history_provider.ts` | `b2763b7d1b146d398835f4a15eb84e3f0bccfa3bb07b2e9faf5ca9a4eda0a251` |
| `extensions/vscode-sepalith/src/next_edit.ts` | `c184e0d26499290ce3a8f339f2172343ab233b0799fac57a86476ff14e0e2214` |
| `extensions/vscode-sepalith/scripts/check-campaign-protocol.ts` | `f500f65d716c84de474b26f2171d0e293d42ed407d9108532b39465bc092f178` |
| `extensions/vscode-sepalith/scripts/check-campaign-requests.ts` | `67ceacf0177bf15d6bc69a0b21691d2518fa51323dbf265fbf39a586992d7541` |
| `extensions/vscode-sepalith/scripts/check-campaign-selection.ts` | `57d00e1919b4ffe90199ca7d8d9f41b4307ca81c127d942cf6e6a7033bd43a51` |
| `extensions/vscode-sepalith/scripts/check-history.ts` | `8f410e8036002aaa4d5c4256ce7d6bfa7bb2d8c61a6e3c63e3def627c71b568e` |
| `extensions/vscode-sepalith/scripts/check-prm05.ts` | `0a476f17b5096b2f3b07f9645ea8e314d2a544b602c25211a6ec0e29f4127df6` |
| `extensions/vscode-sepalith/scripts/check-provider-requests.cjs` | `b45bdc64391365293b0da26fa112bd790d85f329f28a59286c9635103bf51d14` |

The lockfile remained unchanged at SHA-256 `ef1daabf3f81aea2cdc6e7a00248745d898eeed7caca5df1269f4bc846218880`. The package manifest remained unchanged at SHA-256 `3bcaa631b43076bce5f6913bd3ccf252401172698c669ec0b31dc739c80900fa`.

## Dependency installation

`package-lock.json` is npm lockfileVersion 3. The manifest has no runtime dependencies and pins the development tools `@oxlint/plugins` 1.78.0, `@types/node` ^18.19.0, `@types/vscode` ^1.85.0, `esbuild` ^0.25.0, `oxlint` 1.78.0, and `typescript` ^5.5.0. It does not include `@vscode/vsce`.

The first literal command, `npm ci --ignore-scripts --no-audit --no-fund`, was rejected by the ambient npm 12 `allow-scripts` environment policy before installing anything. It reported `EALLOWSCRIPTS`; the user `.npmrc` allowlist contains `node-pty,msgpackr-extract`, while the lockfile has esbuild’s install script. The successful command was:

```text
env -u npm_config_allow_scripts -u NPM_CONFIG_ALLOW_SCRIPTS npm ci --ignore-scripts --no-audit --no-fund
```

It installed exactly nine lockfile packages locally in `extensions/vscode-sepalith/node_modules` with scripts disabled. `npm ls --depth=0 --json` passed with resolved versions: `@oxlint/plugins` 1.78.0, `@types/node` 18.19.130, `@types/vscode` 1.125.0, `esbuild` 0.25.12, `oxlint` 1.78.0, and `typescript` 5.9.3 (plus their locked platform/type packages). No OS/global package install occurred.

## Checks

| Command | Result |
| --- | --- |
| `npm run build` | **PASS**; TypeScript compile and esbuild bundle; `dist/extension.js` 122922 bytes |
| `npm run check-context` | **PASS**, 46 checks |
| `npm run check-completion` | **PASS** |
| `npm run check-prm05` | **PASS**, 38 assertions |
| `node --experimental-strip-types scripts/check-campaign-protocol.ts` | **PASS**, 14 PRM-04/PRM-03 checks |
| `node --experimental-strip-types scripts/check-campaign-requests.ts` | **PASS**, 5 lifecycle scenarios + 5 timeout assertions |
| `node --experimental-strip-types scripts/check-campaign-selection.ts` | **PASS**, 8 fixture rows; fixture SHA `cae8e84e87a5c3f06479b6d49f156a0616fbff166576c9fe81d9a0ee47a14fa3` |
| `node --experimental-strip-types scripts/check-history.ts` | **PASS**, 51 checks |
| `node scripts/check-provider-requests.cjs` | **PASS**, 9 provider lifecycle scenarios; no server/model |
| `npm run check-runtime` | **PASS**, 58 manifest fixtures plus offline provisioning/integrity/cancellation checks; no executable launched |
| `npm run check-process` | **PASS**, owned-process termination/escalation checks |
| `npm run lint` | **FAIL**, 79 anti-slop errors and 3 warnings in the pre-existing dirty source/helper files; no lint edits were made |

The build and protocol/runtime checks are source/package readiness evidence. They do not establish a VS Code GUI host, model load, server readiness, or editor completion.

## VSIX

The documented `npm run vsix` command reached cached `@vscode/vsce` 3.9.2 but failed because the detached EXEC HEAD prevented repository inference for the README relative link `../../docs/SERVING-PACKAGING.md`. No package file was changed. The equivalent offline packager succeeded with explicit repository bases:

```text
env -u npm_config_allow_scripts -u NPM_CONFIG_ALLOW_SCRIPTS \
  npx --offline --yes @vscode/vsce package --allow-missing-repository \
  --no-gitHubIssueLinking \
  --baseContentUrl https://github.com/sims1253/Sepalith/blob/main/extensions/vscode-sepalith/ \
  --baseImagesUrl https://github.com/sims1253/Sepalith/raw/main/extensions/vscode-sepalith/ \
  --out vscode-sepalith-0.0.7.vsix
```

The packager emitted only the expected missing-license warning. The output is `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/vscode-sepalith-0.0.7.vsix`, 36517 bytes, SHA-256 `a37097d0d8bcdd08b2b6acf404e48be107f7f9e0c870d4fb4830a4d9ec42f910`.

Archive contents are exactly five files:

```text
extension.vsixmanifest
[Content_Types].xml
extension/package.json
extension/readme.md
extension/dist/extension.js
```

The embedded package is `vscode-sepalith` 0.0.7, publisher `sepalith-dev`, main `./dist/extension.js`, activation `onStartupFinished`, and no explicit `extensionKind`. Its package bytes match EXEC `package.json` (`3bcaa...`). The bundled extension is 122922 bytes, SHA-256 `b280cd03df61b19af3ad1348d4d66a2552c2a12c5a4b4f7139690d1fc036ef56`; the archive copy matches that hash. The manifest identity line is `Id=vscode-sepalith Version=0.0.7 Publisher=sepalith-dev` (manifest SHA-256 `19137549b0fa988ef26595a47b0289d2cc2ac67245d298dfc8557733ddf77c14`). `dist/check-*.cjs` is excluded by `.vscodeignore`.

The bundle contains the reviewed native client/protocol strings (`NativeCampaignClient`, `PRM-03`, `r-next-edit`, `NATIVE_EOG_IDS`, and `renderPrompt`). The source identity is `campaign_protocol.ts` SHA-256 `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824` and its renderer is `zeta2-prm03-v1`; the VSIX does not embed a model or server binary.

## Profile and next activation gate

The profile correction is recorded in [RUN-01-editor-build-profile-v2.md](RUN-01-editor-build-profile-v2.md). For adapted PRM-03 weights, root must bind the reviewed model/renderer/tokenizer identity and use the frozen native profile: loaded context 4096, batch/ubatch 256, one slot, RL cap 192, DEV quality cap 512; m0pad CPU reference `-ngl 0 -t 6 -tb 6`, or the reviewed Vulkan profile `-ngl 43 -t 6 -tb 6`. The generic v1 example of context 8192/CPU/8 threads is unaccepted for adapted weights. Current extension spawn arguments do not expose batch/ubatch, so the live host needs a reviewed wrapper/profile or source integration before accepting those flags.

Root’s next check is to install this exact VSIX in the selected WSL or Remote-SSH workspace host, verify `Developer: Show Running Extensions`, provide the approved adapted model/manifest, and run the lead-owned live readiness and editor application smoke. This packet performed no GUI activation, model load, server start, or completion request.

