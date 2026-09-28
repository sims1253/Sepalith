# RUN-04 editor trigger diagnosis

This packet is a read-only diagnosis of the retained notebook runs
`primary-editor-managed-c` and `primary-editor-managed-d`. It prepares a
harness candidate; it does not alter the accepted VSIX, the EXEC source, the
managed launcher, or the notebook, and it does not claim editor admission.

## Finding

Run d reached the same managed prerequisites as run c and then stopped at the
first manual suggestion boundary. Its extension activation, managed
application settings, native health, R language assignment, active editor,
and cursor selection all passed. The harness then failed after its 65-second
clipboard poll with `last=(no request yet)`. It had no event for command
dispatch, built-in inline-trigger dispatch, provider invocation, sidecar
request, or ghost visibility. Therefore d distinguishes “no prompt was
published through `copyPrompt`” from a proven provider or model failure, but
does not identify which private asynchronous step was absent.

The current EXEC extension explains why the boundary is opaque. At
`src/extension.ts:1199-1201`, `triggerInlineSuggestion()` calls
`void vscode.commands.executeCommand("editor.action.inlineSuggest.trigger")`.
At `:1235-1242`, `sepalith.suggest` clears debounce state and calls that
function without returning or awaiting it. The provider is registered at
`:1267`. The harness’s `copyPrompt` command is also fire-and-forget at
`:1246-1249`; clipboard polling is the only product-side publication it can
observe. `vscode.commands.getCommands(true)` can prove command registration,
but VS Code exposes no supported enumeration of inline providers or ghost
buffers.

This leaves two plausible classes of explanation for d: the editor was not in
the stable/focused state required by the built-in command at the instant the
command was dispatched, or the command ran but provider/sidecar work did not
publish `lastPrompt`. The retained events do not separate those classes. A
readiness barrier and command-surface record are warranted; a harness-only
retry cannot prove provider execution.

## Retained evidence

The evidence is copied under the existing lead packet and was only read.

| Run | Evidence | Result |
| --- | --- | --- |
| c | `docs/campaign/work/lead/notebook-editor-evidence/c-run-host-result.json` (`e5d8b993609741e2bbe34fb27643942782f58bffe3456376144c7871fd941342`) | Activated in 8 ms; health `ok` at 6030 ms; active/selection at 6070/6130 ms; first prompt at 6409 ms, SHA `15b89ccb7f434360f12c3a1dab41d301ddcd45adbdc166a8fe54d2d7b1aca458`, 505 chars. Later typing check failed because the captured hash was unchanged. |
| c | `docs/campaign/work/lead/notebook-editor-evidence/c-run-run-result.json` (`ff42a90654134e0d77158b41cdf9bc02c4c9844ebe08fe1b8547bdbd92cb8186`) | Managed launcher host exited code 0, ten manifest assets; direct native probe passed with 132 tokenized IDs and 13 generated tokens. |
| d | `docs/campaign/work/lead/notebook-editor-evidence/d-run-host-result.json` (`1a1ebc09a78cfdd4d6996259ff00e2ac8bf84794c20f0558e86906675bc7b4f9`) | Activated in 45 ms; health `ok` at 6070 ms; active/selection at 6128/6214 ms; no prompt event; failed with `primary prompt was not published to copyPrompt ... last=(no request yet)`. |
| d | `docs/campaign/work/lead/notebook-editor-evidence/d-run-run-result.json` (`05784fe4b26151dc0fef6bd1dc42849ca995af5b81d78a8a43342c69f45e2e59`) | Managed launcher host exited code 0, ten manifest assets, same application settings (`serverPath: ""`, Vulkan, port 18403, editor timeout 5000 ms). |

The unchanged accepted harness and launcher inputs were
`primary-editor-harness/extension.js` SHA
`2bd395ee201357e97f4f62c6abb378a7312aca5a56860406159e22e86870cfe3` and
`run_primary_managed_route.mjs` SHA
`dbf94fc191ba84d9e38c01880cdbedfab0108a2a3257a6438fc7cb1204be0a91`.
The EXEC source read for this diagnosis was
`extensions/vscode-sepalith/src/extension.ts` SHA
`798a53049f44751b030bd190351d655d6e586c8fbaa958eaed9c0a37dc305d53`.

## Candidate harness change

`extension.trigger-observability.js` is a copy of the accepted harness with a
small diagnostic change:

1. After the existing active-editor and selection events, it checks the actual
   `window.activeTextEditor`, URI, document version, cursor, and optional
   `window.state.focused` value. The state must remain unchanged for 300 ms.
2. Before each manual trigger it records the command-surface result for both
   `sepalith.suggest` and `editor.action.inlineSuggest.trigger`, plus the
   sidecar-ready observation. It records `suggest_command_invoked` and
   `suggest_command_returned`; the latter is explicitly marked
   `outer_command_only` because the current extension returns before its
   delegated built-in command completes.
3. It observes `copyPrompt` for 5000 ms, recording only attempt, elapsed time,
   prompt length, and SHA. It never puts prompt text or its prefix into a
   timeout error.
4. It retries the same `sepalith.suggest` command once, after 250 ms, only if
   native readiness, both command registrations, active editor identity,
   document version, cursor, and focus are still stable. The retry event
   carries the exact reason. If any precondition changes, the retry is
   suppressed and the run fails with its bounded observation evidence.
5. A second missing prompt fails after the second bounded poll. It does not
   invoke the built-in command directly, so the retry remains an observation
   of the product route rather than a bypass.

The candidate intentionally leaves provider invocation, sidecar request
arrival, inline ghost contents, and real acceptance as open gates. Closing
those gates needs a supported product telemetry seam or root-side correlation
with extension output/server logs; a clipboard check cannot establish them.

## Validation and launch shape

`node --check` passes for the candidate. The CPU-only
`test_trigger_observability.cjs` loads the actual candidate with a minimal
mock VS Code event surface and exercises its identity predicates. It passes
for a stable owner and suppresses retry for cursor movement, document-version
change, cross-document focus, and an unfocused window. It also checks the
one-retry/five-second bounds, event markers, and absence of raw prompt-prefix
logging. No VS Code host, sidecar, model, network, SSH, CUDA, or GUI process
was started by this preparation.

If root chooses to run the candidate, it should replace only the harness path
in a fresh disposable run root and preserve the existing managed launcher,
application settings, accepted VSIX, process-scoped CA, and `--wait`. The
result must report a bounded `suggest_command_returned` event even when its
`result_kind` is `undefined`, and must keep a final no-prompt result as a
diagnostic failure. A prompt obtained on the retry is evidence that the
bounded retry recovered prompt publication; the precise cause remains
unproven. It is not proof that the first command dispatched a provider
request or that ghost acceptance works.
