# RUN-04 editor integration preparation

This packet prepares a real VS Code extension-host check for the Sepalith
0.0.7 primary-profile VSIX. It does not claim that VS Code, the remote editor
host, a native server, or a model has been launched. The harness uses a
temporary workspace and synthetic R files only. It records editor events and
loopback requests in a result file chosen by the launch command.

## Pinned inputs

The source is the accepted execution snapshot at
`/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith`.
The pinned package facts are:

| input | identity | value |
| --- | --- | --- |
| `package.json` | SHA-256 `2a9edf2996852a37057008d8255cf61352407d8aecf98a59635e3f0a505b94a2` | version `0.0.7`, engine `^1.85.0` |
| `package-lock.json` | SHA-256 `ef1daabf3f81aea2cdc6e7a00248745d898eeed7caca5df1269f4bc846218880` | lockfile v3 |
| accepted VSIX | SHA-256 `08604f9a69625a55ad43c65e6ec605325a345a6d304ce107431727dd6922c029` | `vscode-sepalith-primary-profile-0.0.7.vsix` |

The extension source used by that VSIX was inspected at these hashes:

| file | SHA-256 |
| --- | --- |
| `src/extension.ts` | `4fa6a06d3d3cbf69eda51195dc81b8b793739a1ed0f0763ad3d7d63a475a9727` |
| `src/runtime.ts` | `4de415237088984d53eba6b182dba975c12216a516c52f9fecb3a916dc841616` |
| `src/campaign_protocol.ts` | `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824` |
| `src/campaign_client.ts` | `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69c9c1333` |
| `src/campaign_requests.ts` | `4fdc6a881b93ae8beb9614c2b210f80fd5b0a2d5592ae02c8174345612a17535` |

## What the harness does

`extension-host-harness/extension.js` is loaded by the actual VS Code
extension host. It imports the built-in `vscode` API and Node's built-in
`http`, `fs`, `path`, `crypto`, and `os` modules. It has no npm dependencies
and does not use a mocked VS Code implementation.

Before calling `sepalith.startServer`, it starts a loopback HTTP server on a
non-primary port (default `18119`). The server implements only the legacy
OpenAI-compatible `GET /health` and `POST /v1/completions` mechanics. Its
responses are deterministic synthetic text and its result file stores only
synthetic prompt heads/tails and hashes. It recognizes the extension's
completion probe (`prompt: "x"`, `max_tokens: 1`) separately from suggestions.
The server delays the next actual suggestion during the stale test and records
request abort/close/write state.

The harness then activates the installed `sepalith-dev.vscode-sepalith`
extension and calls the package's registered `sepalith.startServer` command.
It observes the external-sidecar completion probe before opening two temporary
R buffers. The event sequence is:

1. Open `run04-a.R`, focus its cursor, and insert text with
   `TextEditor.edit`. The harness records open, active-editor, selection, and
   text-change events. The extension's synchronous `onDidChangeTextDocument`
   path is the source-backed trigger for its `HistoryProvider` capture.
2. Set one synthetic warning through a diagnostic collection and wait for
   `languages.onDidChangeDiagnostics`.
3. Call `sepalith.suggest` and verify the legacy prompt contains
   `run04-a.R` and the `<|user_cursor|>` anchor.
4. Open/focus `run04-b.R`, call `sepalith.suggest`, and verify the request
   changes to the new workspace-relative file.
5. Move the B cursor so the request key differs from the cached B completion,
   delay the next request, issue a suggestion, edit the active buffer while it
   is in flight, and issue a fresh suggestion. Verify the stale request is
   aborted or its response closes before a write, and that the fresh prompt
   includes the edited buffer. The source also performs a document identity
   check before publishing a response; the provider is private, so the
   harness records that final UI non-publication needs a real editor review.

The harness does not invoke an acceptance command. `extension.ts` exposes
`sepalith.applyNextEdit` for an identity-bound multiline workspace edit, but it
does not expose a test-only completion acceptance hook. The built-in command
name is recorded for manual review when available; the harness never invents
or invokes one.

## Isolated launch

Run this command from a host with a real VS Code display or from the authorized
Remote-SSH editor host. Root owns the launch. It installs the exact accepted
VSIX into a disposable extension directory, creates a disposable user-data
directory and workspace, and writes no user settings or files:

```sh
set -eu
RUN_DIR="$(mktemp -d /tmp/sepalith-run04.XXXXXX)"
mkdir -p "$RUN_DIR/workspace/.vscode" "$RUN_DIR/profile" "$RUN_DIR/extensions"
cat >"$RUN_DIR/workspace/.vscode/settings.json" <<'JSON'
{
  "sepalith.autoStart": false,
  "sepalith.manifestUrl": "",
  "sepalith.port": 18119,
  "sepalith.debounceMs": 0,
  "sepalith.requestTimeoutMs": 3000,
  "sepalith.scopeContext": false
}
JSON
code --user-data-dir "$RUN_DIR/profile" \
  --extensions-dir "$RUN_DIR/extensions" \
  --install-extension "/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/vscode-sepalith-primary-profile-0.0.7.vsix" \
  --force
SEPALITH_HARNESS_PORT=18119 \
SEPALITH_HARNESS_RESULT="$RUN_DIR/result.json" \
code --new-window \
  --user-data-dir "$RUN_DIR/profile" \
  --extensions-dir "$RUN_DIR/extensions" \
  --extensionDevelopmentPath "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/editor-integration-preparation/extension-host-harness" \
  --disable-gpu --disable-updates --skip-welcome-page --disable-workspace-trust \
  "$RUN_DIR/workspace"
```

The command must be run as a complete sequence. The first `code` invocation
only installs the accepted VSIX into the isolated directory; the second starts
the actual extension host with the harness. The harness leaves the window open
for inspection. Set `SEPALITH_HARNESS_CLOSE=1` only when the result file has
been captured and the root operator wants automatic window close.

The executable requirements are a VS Code CLI version at least `1.85.0` with
the `--user-data-dir`, `--extensions-dir`, `--install-extension`,
`--extensionDevelopmentPath`, and `--new-window` options, plus the Node runtime
bundled into the VS Code extension host. The harness has no package manager
step and no external runtime dependency. A real display is required for the
editor event check; an authorized Remote-SSH VS Code window is also valid.

The local worker did not run this command. No native server, model, CUDA, SSH,
or editor host was launched while preparing it. The repository contains no
`@vscode/test-electron` dependency; that is intentional because a mocked or
headless substitute would not close the requested real editor-event gate.

## Source-backed boundaries

The package contributes these commands: `sepalith.startServer`,
`sepalith.stopServer`, `sepalith.suggest`, `sepalith.copyPrompt`,
`sepalith.showLogs`, `sepalith.clearRequests`, `sepalith.applyNextEdit`,
`sepalith.dumpStats`, and `sepalith.refreshRuntime`. The harness uses only
`startServer` and `suggest`.

The legacy route uses `POST /v1/completions` with `max_tokens: 320` and the
extension's fixed stop list. Its `context_build` prompt contains the cursor
anchor. This controlled route does not prove PRM-03 prompt bytes, tokenizer
EOS handling, native latency, or model quality.

The primary route is materially different. `Sidecar.start` validates a final
manifest before checking an occupied port and refuses to attach a primary
manifest to an unidentified external server. Therefore the loopback server
cannot be paired with the primary profile. A primary run needs the final
validated HTTPS release manifest, its managed runtime/model assets, a free
port, and a real native server. The same two-buffer event sequence can be
reused after those inputs are supplied; this packet does not fabricate them.

Two current source seams are explicit in the result and receipt:

* `selectPromptContext` always constructs `diagnostics: []`, and the
  extension registers no diagnostic listener. The harness can prove the editor
  diagnostic event, but it cannot claim diagnostic bytes reached a primary
  prompt.
* `HistoryProvider` is captured synchronously before provider invalidation on a
  text-change event. The legacy prompt intentionally has no history section;
  primary history serialization must be checked in the managed live path.
  Private provider state and acceptance callbacks are not exposed, so stale
  request transport evidence and source identity guards still require a real
  editor-host observation for final publication/acceptance.

## Preparation result

The harness package and JavaScript passed static syntax/JSON checks during
preparation. The actual editor host, controlled lifecycle, event sequence,
stale cancellation, and primary managed path remain pending root execution.
The receipt is intentionally `prepared_static_gui_pending`; it does not turn a
launch command into a test result.
