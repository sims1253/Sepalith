# RUN-03/RUN-04 b4 editor preparation

This packet prepares the first editor-host check for the protected b4 CPU fallback. It uses the accepted Sepalith VSIX, a root-owned notebook b4 server, and a real VS Code extension host. The worker did not start VS Code, the server, a model, Xvfb, SSH, or a network service. The harness creates only synthetic R buffers inside a disposable workspace.

## Frozen identities

The source closure is RUN-01 snapshot `41a509aaa6b2205add579e12db9afbdad8d4c40155baf798d4f263009989ebfc`. The accepted source and package files are in EXEC:

| item | path | SHA-256 |
| --- | --- | --- |
| VSIX used for the editor host | `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/vscode-sepalith-primary-acceptance-v2-0.0.7.vsix` | `f415801f6bbd88319c89164fa5bb9d29bd40c60f179a03196c3d5fec1103d276` |
| `src/extension.ts` | EXEC extension source | `798a53049f44751b030bd190351d655d6e586c8fbaa958eaed9c0a37dc305d53` |
| `src/context_build.ts` | EXEC extension source | `3bc148464dde243d6d4edca49ba2a3536de541f035afb1c5a68d756e36ef1047` |
| `src/runtime.ts` | EXEC extension source | `4de415237088984d53eba6b182dba975c12216a516c52f9fecb3a916dc841616` |
| `package.json` | EXEC extension package | `2a9edf2996852a37057008d8255cf61352407d8aecf98a59635e3f0a505b94a2` |
| `run_eval.py` `render_zeta2` | EXEC canonical legacy renderer | `7fc6d4d796856ef3697365a462a8a1f5b0a9876d1f2e55cb88325a6ff7ef493d` |
| PRE-08 fallback ledger | PLAN receipt | `36d6a619f3c8fd00a66ea4940f933275cb5edef2347b1187c2a5ef97588122ae` |

The notebook artifact is the transferred protected model `/home/m0hawk/.local/share/sepalith-campaign-20260915/models/b4/packaging_b4-Q8_0.gguf`, 2,012,011,904 bytes, SHA-256 `e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d`. The measured notebook runtime is `/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server`, SHA-256 `e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6`, build 10453 commit `3cb7ffb1a`, with runtime identity receipt `docs/campaign/work/lead/notebook-b4-cpu-runtime-identity.json` (SHA-256 `0569e1707d79bb3f8f5f39a9a9e63d1fcb1f6f5cf613e8a5f07bbae8ddb1968e`). The PRE-08 local fallback binary hash `123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804` is a different build identity and must not be substituted for the notebook runtime.

The exact measured notebook server profile is CPU-only: `-t 6 -tb 6 --threads-http 2 --parallel 1 -c 8192 -b 256 -ub 256 -ngl 0 -lv 4`. The extension request contract is fixed by the accepted `extension.ts`: `POST /v1/completions`, `max_tokens=320`, `temperature=0`, `stream=false`, and the seven source stop strings. Case labels do not alter those request fields.

## What selects the b4 route

The accepted extension has no separate `b4` profile selector. The source-backed selection is:

```text
manifestUrl="" + serverPath=<root b4 runtime> + modelPath=<root b4 model>
        + port occupied by the root server
        -> startServer completion probe
        -> sidecar state "external"
        -> nativeClient() is null
        -> legacy buildPrompt/postCompletion path
```

The harness sets `backend=cpu`, `gpuLayers=0`, `threads=6`, `contextSize=8192`, `autoStart=false`, `debounceMs=0`, `requestTimeoutMs=5000`, and `scopeContext=false`. `scopeContext=false` is required for the plain legacy editor context. An occupied port is required: the extension refuses to attach an unidentified external process to a primary manifest, while the empty-manifest path intentionally probes and adopts a compatible external server.

The exact six/batch/thread profile is therefore supplied by the root-owned server command. The extension's manually configured managed-spawn branch supplies only `-t 6`, `-ngl 0`, `-c 8192`, and `--parallel 1`; it does not supply the measured `-tb`, `--threads-http`, `-b`, `-ub`, or `-lv` flags. Do not call that branch profile-equivalent for b4. Start the full-profile server in the root terminal and let the extension adopt it as external.

## Renderer boundary

The b4 quality and notebook clients use the protected `run_eval.render_zeta2` implementation. The editor's legacy branch calls `context_build.buildScopedPrompt`/`renderPrompt`. With scope disabled, both have the same suffix, filename, merge-marker, and cursor-marker skeleton. They are not byte-equivalent in the current accepted source: `run_eval.render_zeta2` always emits an `<[fim-prefix]><filename>edit_history` section before the file name, while the extension legacy path has no `event_diff` input and emits `<[fim-prefix]><filename>{path}` directly. The editor harness records the captured prompt and marks canonical byte equivalence pending. It does not insert the PRM-03 renderer or claim that the b4 client prompt and editor prompt are interchangeable.

The extension's `parsePrediction` remains the accepted parser source. The harness does not copy the parser or pretend that a prompt capture includes native stop metadata; root should correlate the editor result with the root server log and the already accepted b4 endpoint records.

## Harness and launch files

`extension.js` is a real VS Code extension-host harness. It uses the built-in `vscode` API and Node built-ins only. It performs these events in one run:

1. Preflight `/health`, `/props`, and `/v1/models`, including exact model path and `n_ctx=8192` checks.
2. Activate the accepted VSIX and call `sepalith.startServer`; with the root server already answering, this exercises the external-sidecar adoption path.
3. Open three synthetic R buffers and record open, active-editor, selection, and text-change events.
4. Type in a replacement buffer, trigger the legacy route, capture the actual extension prompt, and leave a bounded operator window for `Alt+.` and `Tab` acceptance.
5. Trigger a separate complete-code no-op applicability probe with the same request policy and verify that the document is not mutated automatically. No label is sent to the extension, and the visible ghost/no-op decision remains a root review item.
6. Trigger a cancellation candidate, edit the active document while the request is in flight, trigger a fresh request, and verify that the deliberate user edit remains the document content. The result records timestamps and prompt hashes for correlation with the server log; stale transport nonpublication is pending because provider handles are private.
7. Call `sepalith.stopServer` and verify that the root-owned external sidecar still answers. Root then stops the server owner and verifies the port is free.

`run_b4_editor_cycle.mjs` verifies the VSIX and runtime hashes, checks the model size and optionally streams its full hash with `--verify-model`, writes isolated settings, installs the VSIX, and starts the host. It never starts a server. It refuses to reuse a run directory and retains all output. `test_b4_editor_preparation.py` is a static, model-free check.

The result is written to `<run-root>/host-result.json`; the launcher writes `preflight.json`, `settings.json`, `launch.json`, VSIX install logs, VS Code host logs, and `run-result.json`. The expected harness status is `b4_legacy_editor_route_exercised_acceptance_pending`. A successful process exit alone does not close the editor acceptance gate.

## Root launch commands

Root must verify that the chosen loopback port has no competing owner before starting a fresh sidecar. The following command is the exact notebook profile used by the accepted b4 CPU cycle. Redirect stdout/stderr into the root-owned run directory or supervisor so the process identity and request timing can be reviewed:

```sh
set -euo pipefail
PORT=18403
BIN=/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server
MODEL=/home/m0hawk/.local/share/sepalith-campaign-20260915/models/b4/packaging_b4-Q8_0.gguf
RUNTIME_SHA=e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6
MODEL_SHA=e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d
test -x "$BIN"
test -f "$MODEL"
test "$(sha256sum "$BIN" | cut -d ' ' -f1)" = "$RUNTIME_SHA"
test "$(stat -c %s "$MODEL")" = 2012011904
test "$(sha256sum "$MODEL" | cut -d ' ' -f1)" = "$MODEL_SHA"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  -t 6 -tb 6 --threads-http 2 --parallel 1 -c 8192 \
  -b 256 -ub 256 -ngl 0 -lv 4
```

The `run_b4_editor_cycle.mjs` command below performs the editor-host launch against that answering server. Pick a fresh `<run-id>` each time. For the first run, call it `fresh_external` after the root server has just started; for a second run, leave the same server running and call it `existing_external`. The latter proves that the extension can adopt an already-running b4 sidecar without taking ownership.

```sh
set -euo pipefail
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
EXEC=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912
RUN_ROOT=/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/b4-editor-<run-id>
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
/usr/bin/timeout --foreground --signal=TERM --kill-after=15s 720s \
  /usr/bin/node "$PLAN/docs/campaign/work/b4-editor-preparation/run_b4_editor_cycle.mjs" \
  --code /usr/bin/code \
  --vsix "$EXEC/extensions/vscode-sepalith/vscode-sepalith-primary-acceptance-v2-0.0.7.vsix" \
  --model /home/m0hawk/.local/share/sepalith-campaign-20260915/models/b4/packaging_b4-Q8_0.gguf \
  --runtime /home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server \
  --port 18403 --mode fresh_external --run-root "$RUN_ROOT" \
  --manual-trigger 0 --manual-wait-ms 15000 --close 1 --verify-model
```

Use `--mode existing_external` and a different run root for the second editor run. The root-owned server must remain running between these two commands. The first run's harness calls `stopServer`, but the accepted extension leaves an external server alive; root should verify `/health` before the second run and record that the same server PID/runtime/model identity was retained.

For genuine manual trigger and Tab evidence, run the same launcher from a visible editor session instead of Xvfb, set `--manual-trigger 1 --close 0`, press `Alt+.` when `manual_trigger_wait_started` appears, and press `Tab` once when the inline ghost is visible. The harness records the before/after document hashes and `onDidChangeTextDocument` event. Do not use `sepalith.applyNextEdit` for this inline acceptance. An Xvfb invocation is suitable for mechanical lifecycle/cancellation checks, but it cannot establish a human-visible Tab acceptance unless the root adds an explicit, separately recorded input driver.

The smallest acceptance evidence is:

| check | required evidence | current status |
| --- | --- | --- |
| b4 identity | model size/SHA, runtime SHA/dependencies, `/props.model_path`, `n_ctx=8192`, root PID ownership | endpoint identity is accepted; editor rerun pending |
| fresh sidecar editor route | root fresh server launch + extension external adoption + prompt/request result | pending |
| existing sidecar editor route | second isolated profile, same answering sidecar, no extension-owned stop | pending |
| manual trigger/Tab | visible `Alt+.` then `Tab`, document version/hash delta, post-edit R parse if desired | pending |
| no-op applicability | same request policy, no label in prompt/settings, visible no-op ghost review with no acceptance | pending |
| cancellation | edit during request, fresh prompt hash, no stale document mutation, server/extension abort correlation | pending |
| cleanup | root server owner stopped, no listener, all root children gone | prior endpoint cycle cleanup passed; editor rerun pending |

The prior root-owned b4 CPU endpoint cycle is useful setup evidence: four synthetic requests completed at 1306.563, 1299.526, 1048.432, and 1020.631 ms, with model SHA `e343...`, notebook runtime SHA `e68d...`, and server PIDs 165078/165079 later absent. It exercised the legacy client only. It did not exercise a VS Code editor, manual acceptance, cancellation, or no-op applicability, so this packet does not upgrade those gates.

## Static validation and limits

Run the model-free checks from PLAN:

```sh
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B docs/campaign/work/b4-editor-preparation/test_b4_editor_preparation.py
node --check docs/campaign/work/b4-editor-preparation/extension.js
node --check docs/campaign/work/b4-editor-preparation/run_b4_editor_cycle.mjs
```

These checks passed during preparation. They prove file syntax, accepted source/artifact pins, explicit legacy settings, and that the launcher does not create or manage a server. They do not prove the notebook model load, actual editor latency, prompt byte parity, stale ghost nonpublication, no-op quality, or manual acceptance. The packet retains those as root live gates.
