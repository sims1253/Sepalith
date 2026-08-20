# Sepalith (VS Code extension)

Minimal next-edit-suggestion extension for R files, backed by a local
`llama-server` sidecar running a Sepalith GGUF. v0 proves the end-to-end
plumbing (sidecar → zeta2 prompt → ghost text).

> **Expect bad suggestions.** The current models are weak. Garbage
> suggestions are acceptable at this stage; crashes are not.

## Install (by copy)

    npm install
    npm run build
    npx @vscode/vsce package --allow-missing-repository
    code --install-extension vscode-sepalith-0.0.1.vsix

Everything is path-absolute, so this works unchanged on WSL / Remote.

## Commands

| Command | Title | Keybinding |
| --- | --- | --- |
| `sepalith.startServer` | Sepalith: Start server | — |
| `sepalith.stopServer` | Sepalith: Stop server | — |
| `sepalith.suggest` | Sepalith: Suggest now | `Alt+\` (in R files) |

`Alt+\` requests a suggestion at the cursor immediately, bypassing the
debounce. Automatic suggestions fire after `debounceMs` of idle editing in R
files.

## Configuration (`sepalith.*`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `modelPath` | `/home/m0hawk/Documents/Sepalith/experiments/models/abl_dropout-Q8_0.gguf` | Absolute path to the GGUF. |
| `serverPath` | `/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server` | Absolute path to the llama-server binary. |
| `port` | `18099` | Sidecar port. |
| `threads` | `8` (max 8) | CPU threads for the sidecar; more is slower on this machine. |
| `contextSize` | `8192` | Context window; some prompts need ~5.6k tokens. |
| `autoStart` | `true` | Start the sidecar on activation. |
| `debounceMs` | `1500` | Idle time before an automatic suggestion attempt; `0` disables auto-trigger (manual only). |
| `scopeContext` | `true` | Scope-aware prompt context: pin the enclosing function and add a top-level file outline (LSP document symbols, brace-scan fallback when no R language server is running). Off = plain prompt. |

## Behaviour notes

- The sidecar is CPU-only (`-ngl 0`) on purpose: the GPU is shared with
  training and GPU serving misbehaves under load.
- Only the exact child process this extension spawned is ever killed. If the
  configured port is already answering, that server is treated as external:
  it is used as-is and never touched on deactivate.
- State lives in the status bar on the left (`Sepalith: ready (model)`,
  `starting…`, `external server`, `stopped`, `error`); hover it for the last
  request latency and token count. Errors surface there, never as popups.
- Per-request logs (latency, prompt char count, prefix truncation, sidecar
  stdout/stderr tails on error) go to the `Sepalith` output channel.

## Smoke test

    npx tsx scripts/smoke.ts

Starts the sidecar against the configured model, waits for readiness, sends
one rendered prompt for a tiny synthetic R buffer, and prints the parsed
suggestion and latency.

## Context unit tests

    npm run check-context

Exercises the pure scope-context helpers in `src/context_build.ts` (outline
formatting and caps, the brace-scan enclosing-function fallback, pin/suffix
truncation, v0.0.6 prompt parity, and the stable-prefix property) under
plain node — no vscode import.
