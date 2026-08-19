# Sepalith VS Code extension — v0 spec ("prove the plumbing")

Goal: a minimal, testable VS Code extension that serves next-edit suggestions
for R files from a local llama.cpp sidecar running a Sepalith GGUF. v0 is
about END-TO-END PLUMBING, not suggestion quality — the current models are
weak and garbage suggestions are acceptable. Do not gold-plate.

Owner note: implement exactly this spec. Where it is silent, choose the
simplest thing that works and write a one-line comment. TypeScript,
no frameworks beyond @types/vscode.

## Non-goals (v0)

- No edit-history ring buffer, no multi-region edits, no related-file
  context, no PSM format (no model is trained on it yet).
- No automatic sidecar download; the user provides the GGUF path.
- No telemetry, no marketplace publishing.

## Configuration (contributes.configuration, prefix `sepalith.`)

- `modelPath` (string): absolute path to a GGUF. Default:
  `/home/m0hawk/Documents/Sepalith/experiments/models/abl_dropout-Q8_0.gguf`.
- `serverPath` (string): absolute path to the llama-server binary. Default:
  `/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server`.
- `port` (number, default 18099).
- `threads` (number, default 8, max 8 — more is slower on this machine).
- `contextSize` (number, default 8192 — some prompts need ~5.6k tokens).
- `autoStart` (boolean, default true): start the sidecar on activation.
- `debounceMs` (number, default 1500): idle time before an automatic
  suggestion attempt. 0 disables auto-trigger (manual only).

## Commands

- `sepalith.startServer` / `sepalith.stopServer`: sidecar lifecycle.
- `sepalith.suggest` (keybinding `Alt+\`): request a suggestion at the
  cursor NOW.

## Sidecar contract (hard rules — each earned by an incident)

1. Spawn: `<serverPath> -m <modelPath> --port <port> --host 127.0.0.1 -c
   <contextSize> --parallel 1 -t <threads> -ngl 0`. CPU-only on purpose:
   the GPU is shared with training; GPU serving misbehaves under load.
2. Readiness: `/health` returns ok DURING model load — probe
   `POST /v1/completions` with `{"prompt":"x","max_tokens":1}` until HTTP
   200 (poll every 3 s, give up after 180 s, surface the error).
3. Shutdown: kill the TRACKED child pid only. NEVER `pkill` by a name
   pattern (incident history: it kills unrelated jobs).
4. If the port is already answering, treat the server as external: use it,
   do not spawn, do not kill on deactivate.

## Prompt render (must match experiments/eval/run_eval.py `render_zeta2`)

For the active cursor in an R document, build ONE prompt (plain text):

```
<[fim-suffix]>
<suffix lines: everything below the cursor's line>
<[fim-prefix]><filename><relative file path>
<prefix lines: everything above the cursor's line>
<<<<<<< CURRENT
<|user_cursor|>
=======
<[fim-middle]>
```

- The region is EMPTY in v0 (cursor marker alone) — that is the
  "signature" style the finish-block models were trained on.
- Truncate the prefix from its START (keep the tail) so prefix+suffix
  stays under ~6000 characters; log when truncating.
- Request: `POST /v1/completions` body
  `{"prompt": <prompt>, "max_tokens": 640, "temperature": 0,
  "stop": [">>>>>>> UPDATED"], "stream": false}`.
- Parse: everything before the first `>>>>>>>` is the predicted region;
  strip `<|user_cursor|>` and any leading blank lines. The FIRST line of
  the parse replaces/completes the cursor's line; remaining lines insert
  after it. Render as a single ghost-text inline completion at the cursor.

## UX

- `registerInlineCompletionItemProvider` for language `r` only.
- Auto-trigger: debounce on document change; manual `Alt+\` bypasses the
  debounce. Only one request in flight; cancel the previous (AbortController
  or ignore-by-request-id — simplest correct option is fine).
- Status bar item (left): `Sepalith: <state>` where state is one of
  `starting… | ready (model name) | external server | stopped | error`.
  On hover/tooltip: last latency + tokens.
- Output channel `Sepalith` logging: every request's latency, prompt char
  count, truncation events, sidecar stdout/stderr tails (ring-buffer last
  50 lines into the channel on error).
- Errors surface as status-bar state, NOT modal popups.

## Build and hand-off

- `npm init` a minimal package.json (engines.vscode `^1.85.0`,
  `main: ./dist/extension.js`), bundle with esbuild (`npx esbuild
  src/extension.ts --bundle --outfile=dist/extension.js --platform=node
  --external:vscode`), devDeps: typescript, @types/vscode, esbuild.
- Package with `npx @vscode/vsce package` (if vsce needs a publisher, use
  `--allow-missing-repository` and publisher `sepalith-dev`; do NOT
  publish anywhere).
- README.md at the extension root: install-by-copy
  (`code --install-extension vscode-sepalith-0.0.1.vsix`), the two
  commands, the config table, and a "expect bad suggestions" warning.
- The user tests on WSL/Remote: keep everything path-absolute, no shell
  tricks in spawn (pass args as an array).

## Acceptance (verify before reporting done)

1. `npm run compile` (tsc --noEmit) and the esbuild bundle succeed.
2. `vsce package` produces a .vsix.
3. A smoke script `scripts/smoke.ts` (run with `npx tsx`) that: starts the
   sidecar against the configured model, waits for readiness per rule 2,
   sends one rendered prompt for a tiny synthetic R buffer, prints the
   parsed suggestion and latency. Zero-score is fine; crashes are not.
4. Extension activates in the Extension Development Host without the
   sidecar erroring (manual check is acceptable if headless fails).
5. `git commit` the extension in the Sepalith repo with a message that
   starts `vscode extension:`.

## Environment constraints (shared machine — obey)

- Jobs `ingest_cran`, a detached python runner, and eval llama-servers may
  be running: DO NOT kill, restart, or `pkill` anything you did not start.
- Stay ≤8 CPU threads total.
- Ports 18085-18087 may be in use; the configured default is 18099.
