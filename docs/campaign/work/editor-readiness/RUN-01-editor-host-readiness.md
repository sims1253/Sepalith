# RUN-01/RUN-02 editor-host readiness

Observed 2026-09-12 18:15:41+02:00 by `worker-rl_entry`. This is a bounded, read-only architecture inspection. It did not launch a model or server, activate an extension, build or install a package, change editor settings, enumerate Vulkan, create a tunnel, or touch campaign state.

## Result

The source-backed architecture is reproducible: the VS Code extension host owns one `llama-server` child and `127.0.0.1:18099`, while the editor UI receives inline completions. The current EXEC package has no explicit `extensionKind`; its Node `main` entry and use of filesystem, networking, and child-process APIs support a workspace-host placement by source inference. That inference is not an observation of the active GUI.

The WSL-side host has a running-capable VS Code server layout and an installed Sepalith 0.0.4, but that installation is older than the reviewed EXEC 0.0.7. The authorized AMD notebook `m0pad` has the VS Code CLI, Remote-SSH UI extension, and R extensions, but no `.vscode-server` and no Sepalith installation. No extension-host process was visible on either host at inspection time, and neither checked serving port had a listener. Actual GUI placement and editor acceptance remain open.

## Source identity and intended placement

The reviewed source is the EXEC worktree at `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`, HEAD `a7345e35219ceb624957115c39b869101026a817`.

| File | SHA-256 |
| --- | --- |
| `extensions/vscode-sepalith/package.json` | `3bcaa631b43076bce5f6913bd3ccf252401172698c669ec0b31dc739c80900fa` |
| `extensions/vscode-sepalith/package-lock.json` | `ef1daabf3f81aea2cdc6e7a00248745d898eeed7caca5df1269f4bc846218880` |
| `extensions/vscode-sepalith/tsconfig.json` | `3808f18c63ffdbb05117e33967f5d50bf48ee0032b27f965ea0556791a95be90` |
| `extensions/vscode-sepalith/src/extension.ts` | `86367f6615e02e59bed425dea8b33f0c210aaf1c0be6a9c6d91e840111bf5cd9` |
| `extensions/vscode-sepalith/src/runtime.ts` | `3a26fb1677eb97488fa03491ba795ec0cca264db24af8f5f7db983bf46e5bfbf` |
| `extensions/vscode-sepalith/src/process_lifecycle.ts` | `a621d5c4a0e227b5e5a2cd12191142b6d1c4a779e1329893389efa5c35068e27` |
| `extensions/vscode-sepalith/src/campaign_client.ts` | `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333` |
| `extensions/vscode-sepalith/src/campaign_protocol.ts` | `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824` |
| `extensions/vscode-sepalith/scripts/smoke.ts` | `dff90b22829efc27482d8bf402f0c752a356c1b729557b176fbd5f924f5f4d26` |

The package declares `main: ./dist/extension.js`, `engines.vscode: ^1.85.0`, and `activationEvents: [onStartupFinished]` (`package.json:7-16`). It does not declare `extensionKind` (`rg` found no such key). `extension.ts:15-20` imports VS Code, filesystem, TCP, and `node:child_process`; `Sidecar.start` checks the port, resolves a validated manifest, checks model/server paths, and spawns an array-form child (`extension.ts:295-389`). The child arguments pin alias `sepalith`, temperature 0, loopback host, one parallel slot, configured context and threads, and the selected GPU-layer count (`extension.ts:374-388`). The lifecycle helper terminates only the tracked child (`process_lifecycle.ts:5-31`). Activation registers the R inline provider and commands (`extension.ts:1123-1170`) and starts the sidecar when `sepalith.autoStart` is true (`extension.ts:1225-1232`).

This implies the following placement contract:

* A local workspace puts the extension and its child on that local workspace host. In WSL, this is the Linux VS Code server host, not the Windows UI process.
* A Remote-SSH workspace puts the extension and child on the SSH workspace host. The desktop Remote-SSH extension is a UI component; it must not be treated as the inference owner.
* The UI and extension host communicate through VS Code. The sidecar remains loopback to the extension host. No server-to-notebook tunnel is part of this contract; a tunnel would require a separate measured need and review.

`runtime.ts:107-173` validates the pinned model/profile and `runtime.ts:293-320` installs only size/hash-verified runtime and model assets from a reviewed HTTPS manifest. `README.md:10-17` gives the package path (`npm ci`, build, VSIX, then install) and `README.md:44-59` describes the managed-runtime path. No public managed manifest URL is supplied in the current source.

## Host observations

### Local WSL candidate

The shell host is `DESKTOP-FVJ95BU`, Ubuntu 22.04.5 under WSL2 (`Linux 6.18.33.2-microsoft-standard-WSL2`). The Windows VS Code CLI is `/mnt/c/Program Files/Microsoft VS Code/bin/code`; `code --version` reports `1.137.0`, commit `645f29cc3176500b4b5762ba887cf2a7f0ffdf2c`, x64. Node is v26.6.0, npm 12.0.2, and R is 4.6.1.

`/home/m0hawk/.vscode-server/bin/645f29cc3176500b4b5762ba887cf2a7f0ffdf2c` and `/home/m0hawk/.vscode-server/extensions` are present. The installed extension metadata is:

| Installed package | Version | Main | `extensionKind` | State |
| --- | --- | --- | --- | --- |
| `sepalith-dev.vscode-sepalith-0.0.4` | 0.0.4 | `./dist/extension.js` | absent | `package.json` and `dist/extension.js` present |
| `reditorsupport.r-2.8.8` | 2.8.8 | `./dist/extension` | absent | present; depends on `REditorSupport.r-syntax` |
| `reditorsupport.r-syntax-0.1.4` | 0.1.4 | absent | absent | present |

The installed Sepalith package hash is `be00eead12bb2f084c0ea9d573e5e1384b5ea0541327e39897968f030dce3a85`; its bundled `dist/extension.js` hash is `9614f568963ea821951138a6d8f7ac08c9f0d701e009c688bb7ba1eeb209e79d`. Those artifacts are not the current EXEC package/source identity and cannot close current EXEC activation.

No `code`, Electron, extension-host, or `code-server` process was observed in the Linux process metadata. The visible Node processes belonged to unrelated workspaces/Codex. No listener was observed on 18099 or 18401. A Windows GUI process is outside this WSL process namespace, so the absence here does not prove that no desktop window exists; it does prove that no active Linux extension host was observable through this shell.

The canonical checkout `/home/m0hawk/Documents/Sepalith` has a built `dist/extension.js` (37527 bytes, hash `eabc82770c9de65b5c6a5e467b7868285144de589125c4101ddd851f26049830`), installed `node_modules`, and VSIX files only through 0.0.6. Its package is also version 0.0.7 but differs from the EXEC package in `contributes`, `scripts`, and `devDependencies`; its dist must not be substituted for the reviewed EXEC artifact.

### Authorized AMD notebook `m0hawk@192.168.178.40`

The negotiated SSH target is hostname `m0pad`, CachyOS rolling, kernel `Linux 7.2.3-1-cachyos`, with the previously recorded host-key fingerprint `SHA256:Tn5oE3/sIruL5la/jz4RivzqBO5Wb3dKWt4RD0crq6w`. The target workspace `/home/m0hawk/Documents/Sepalith` exists and resolves to that checkout. Node is v26.8.1, npm 12.0.2, and R is 4.6.1. `/usr/bin/code --version` reports the same VS Code 1.137.0/commit.

The desktop extension directory `~/.vscode/extensions` contains:

| Installed package | Version | Main | `extensionKind` | Placement evidence |
| --- | --- | --- | --- | --- |
| `ms-vscode-remote.remote-ssh` | 0.128.0 | `./out/extension` | `["ui"]` | desktop/UI extension present |
| `ms-vscode-remote.remote-ssh-edit` | 0.87.0 | `./out/extension` | `["ui"]` | desktop/UI extension present |
| `reditorsupport.r` | 2.8.8 | `./dist/extension` | absent | desktop extension present |
| `reditorsupport.r-syntax` | 0.1.4 | absent | absent | desktop extension present |
| `sepalith-dev.vscode-sepalith` | — | — | — | not found |

`~/.vscode-server`, its `bin`, and its `extensions` directories are absent. No code, Electron, Node, or extension-host process was observed, and no listener was observed on 18099 or 18401. Thus this SSH shell establishes target OS/workspace availability and desktop Remote-SSH metadata, but not a connected Remote-SSH workspace or extension-host placement.

The reviewed b10453 CPU server built in the earlier runtime packet is present at `/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server`, 16000 bytes, SHA-256 `e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6`. No approved model was present at the checked canonical packaging path, and no serving process was started. The target canonical extension checkout has no `node_modules` or `dist`; its package hash is `f86262ee5910b023323d3c9b0b8b088f82037fd938b674f283454ad05bc249f9`, and its `extension.ts` is the older `368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285`, so it is not the current EXEC source identity.

## Dependencies and canonical assets

The reviewed EXEC package has `package-lock.json` but no `node_modules`, `dist/extension.js`, or 0.0.7 VSIX. `node`, `npm`, and `npx` are available locally; `tsc`, `esbuild`, `oxlint`, and `vsce` are not on the local PATH. The same four build commands are absent on m0pad, although Node/npm are available. This is a preparation gap, not a build failure: no install or compile was attempted under this packet.

The canonical checkout contains usable-looking old build artifacts, but its package differs from EXEC and its VSIX inventory stops at 0.0.6. The correct current artifact must be produced from the reviewed EXEC source after the lead selects the workspace host. The runtime binary alone does not establish model identity or extension readiness.

## Minimal activation and application check

After lead admission, use one chosen workspace host and keep the source identity above fixed.

1. For a local WSL run, open the repository in a WSL VS Code window. For m0pad, open the desktop VS Code window and use `Remote-SSH: Connect to Host...` for `m0hawk@192.168.178.40`; the target must show `/home/m0hawk/Documents/Sepalith` (or a separately reviewed EXEC checkout) as the remote workspace. Do not use an SSH tunnel for the loopback sidecar.
2. In the chosen workspace-host checkout, run the reviewed package commands from `extensions/vscode-sepalith`: `npm ci`, `npm run build`, and `npx @vscode/vsce package --allow-missing-repository`. Install the resulting 0.0.7 VSIX from the connected VS Code window, confirming that the install location says WSL or `SSH: m0pad` as appropriate. Record the VSIX and bundled `dist/extension.js` hashes.
3. Before allowing automatic startup, set `sepalith.autoStart` to `false`, choose the reviewed manual b10453 `serverPath` and approved model path, set `port` to 18099, `threads` to 8 or the reviewed CPU limit, `contextSize` to 8192, `backend` to `cpu`, and `gpuLayers` to 0. A managed run instead requires the reviewed HTTPS `manifestUrl`, an empty `serverPath`, and a profile-matching model override. The current source supplies no public manifest URL.
4. Run `Sepalith: Start server` from the connected window. In the `Sepalith` Output channel, record manifest/model/server identity, readiness, and errors. Independently record that exactly one workspace extension-host child owns the listener; readiness requires a real completion request, not only `/health` (`extension.ts:322-339`, `430-441`). Stop with `Sepalith: Stop server` and record the tracked-child exit and closed port.
5. For the edit application check, open a controlled `.R` document in that same workspace and use `Sepalith: Suggest now` (or the registered R inline keybinding) at a known cursor. Capture the rendered request identity and returned `InlineCompletionItem`; verify the ghost text appears in the R editor, then apply it and compare the resulting document to the extension's accepted edit. Exercise one no-op and one cancellation/stale edit while preserving request IDs and the host/port evidence. `Sepalith: Show Logs` and `Sepalith: Dump Stats` provide the extension-side record.
6. For RUN-02, run the frozen renderer/context fixture checks from the same reviewed EXEC package before the live editor request. Retain fixture IDs, prompt/token identities, cache/fresh mode, stale-response decision, and applied document bytes. A GUI activation alone does not close the RUN-02 acceptance until the training/serving fixture IDs and stale/cache results are recorded.

The GUI proof that closes placement is `Developer: Show Running Extensions` in the connected window, showing `vscode-sepalith` under WSL or `SSH: m0pad`, paired with process metadata from that same workspace host. The proof must use the reviewed 0.0.7 VSIX. An installed package or source directory without a running extension host is only availability evidence.

## Acceptance disposition

| Acceptance item | Disposition from this packet |
| --- | --- |
| Reproducible host-placement and launch profile | **Source-backed / partial**: child arguments, port, profile gate, readiness probe, and tracked shutdown are pinned; active GUI placement is not observed. |
| One explicit server/extension owner and fallback | **Source-backed / partial**: the extension owns its child and leaves a responding external server untouched; no live owner or listener was present to verify. |
| No unreviewed server-to-notebook tunnel | **Closed from facts**: no tunnel was created or assumed; the source uses workspace-host loopback. |
| Target OS/workspace availability | **Observed** for WSL Ubuntu and m0pad CachyOS paths; connected editor workspace remains unobserved. |
| Current EXEC extension build/install | **Pending**: dependencies, dist, and 0.0.7 VSIX are absent in the reviewed checkout. Existing 0.0.4/old canonical artifacts do not match. |
| Live sidecar health/completion and lifecycle | **Pending**: no server or model was loaded, no listener was present, and no GUI activation was permitted. |
| RUN-02 fixture identity, cache/fresh correctness, stale rejection | **Pending**: this inspection did not run the serving fixture or a live editor request. |
| Editor inline suggestion and edit application | **Pending**: requires a real connected GUI window and a lead-owned launch. |

The earlier RUN-01 serving architecture, target-host intake, and runtime-preparation receipts remain valid as source/build evidence. This packet supersedes their lack of current host metadata only for the two hosts inspected here; it does not promote a GUI observation that was not made.

