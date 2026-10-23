# Sepalith (VS Code extension)

Next-edit suggestions for R files, served by a local `llama-server` sidecar
running the Sepalith arm A GGUF (SFT-v2-r4-lr1e-5, checkpoint 536). The
extension renders the PRM03 prompt contract `zeta2-prm03-v1` that the model was
trained on, byte-identical to `sepalith.campaign_protocol.render_prompt`, and
parses the answer with the same rules as `parse_output`.

## Install (by copy)

    npm ci
    npm run build
    npx @vscode/vsce package --allow-missing-repository \
      --baseContentUrl https://github.com/sims1253/Sepalith/blob/main/extensions/vscode-sepalith
    code --install-extension vscode-sepalith-0.1.0.vsix

Set `sepalith.serverPath` to a llama.cpp `llama-server` (validated: b10453).
With `modelPath` empty, the first start downloads the arm A Q8_0 GGUF
(2.7 GB, sha256-pinned) into the shared Sepalith cache.

## Commands

| Command | Title | Keybinding |
| --- | --- | --- |
| `sepalith.startServer` | Sepalith: Start server | — |
| `sepalith.stopServer` | Sepalith: Stop server | — |
| `sepalith.suggest` | Sepalith: Suggest now | `Alt+\` / `Alt+.` (in R files) |
| `sepalith.acceptEdit` | Sepalith: Apply previewed edit | `Alt+Enter` (while a preview is shown) |
| `sepalith.dismissEdit` | Sepalith: Dismiss previewed edit | `Escape` (while a preview is shown) |

`Alt+\` requests a suggestion at the cursor immediately, bypassing the
debounce. Automatic suggestions fire after `debounceMs` of idle editing in R
files.

## Configuration (`sepalith.*`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `modelPath` | empty | Absolute path to a GGUF. Empty: the arm A Q8_0 GGUF from `scholzmx/sepalith-2b-edit-sft` (folder `arm-a`), downloaded once and verified. Other qualified tiers of the same model: `arm-a-imat-Q6_K.gguf` (smallest tier that passed the C09 quant gate) and the Q4_K_M-mixA-e4 tier (quant autoresearch); neither is published yet, so set the local path. |
| `serverPath` | empty | Absolute path to the llama-server binary. |
| `port` | `18099` | Sidecar port. |
| `threads` | `8` (max 8) | CPU threads for the sidecar (`-t` and `-tb`). |
| `contextSize` | `16384` | Server context. A request needs its prompt plus the 1,024-token cap. |
| `autoStart` | `true` | Start the sidecar on activation. |
| `debounceMs` | `1500` | Idle time before an automatic suggestion attempt; `0` disables auto-trigger (manual only). |
| `warmup` | `true` | Prefill-as-you-settle: ~200 ms after an edit, cursor move, file switch or accepted suggestion, send the next request's exact prompt token IDs with `n_predict: 0` so the suggestion request hits a warm prompt cache (measured pooled edit TTFT 690.9 → 74.3 ms on the target laptop). Off = every request pays its own prefill. |
| `allowNoEdit` | `true` | The model may answer `[NO_EDIT]`, which shows nothing. Off: a no-op answer is replaced by the model's best edit (the no-op's first token, id 80, is banned for the first step only). A probability threshold on the no-op was measured to change nothing for this model, so there is only this switch. |
| `speculative` | `false` | N-gram speculative decoding (`--spec-type ngram-simple`, n 2, m 4): about 1.4x faster warm suggestions, but not token-identical to plain greedy decoding (it failed the lossless gate). |
| `postAcceptCooldown` | `true` | After accepting, suppress the next auto-trigger until your next keystroke. |
| `debugMode` | `false` | Log the full rendered prompt and raw completion for every request. |

## How a suggestion is made

1. **Context** (`src/prm03_context.ts`). The editable region is the cursor
   line. A non-blank line is a one-line region with `<|user_cursor|>` at the
   cursor. A blank line is an insertion point: an empty region with no cursor
   marker (an auto-indented blank line renders the same way, and its whitespace
   is replaced by the edit). Prefix and suffix lines come from the
   `prm05-source-balanced-v1` window (6,000 UTF-16 units, alternating nearest
   lines). The edit history is the most recent committed line-level edit in the
   same file, spelled like the TRAIN `event_diff` (`User edited "R/f.R":` and a
   diff hunk); an edit is committed when the cursor leaves its lines, so the
   line being typed never appears as history. No outline, diagnostics or
   retrieval records are sent: arm A's TRAIN rows have none. These rules match
   the TRAIN distribution: all 15,006 rows have the cursor on the region's first
   line, 7,940 have an empty region and 6,167 a one-line region.
2. **Tokens.** The prompt is tokenized by the server with `parse_special:
   false` and no BOS, and sent as token IDs with one manual BOS 0, which is
   exactly the training tokenization (`hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0`).
3. **Request** (`src/completion.ts`). `/completion` with the quant gate's
   sampling: temperature 0, `top_k` 1 as the only sampler, no penalties, no
   string stops, seed 3407, cap 1,024, `cache_prompt: true`.
4. **Answer.** Valid only when generation ends with EOS 1 (130073 or a cap hit
   means no suggestion), carries no control tokens, and the text has exactly one
   `>>>>>>> UPDATED` line with nothing after it. `[NO_EDIT]`, a body equal to
   the region, or an empty body at an insertion point is a no-op: nothing is
   shown. Any malformed answer shows nothing. The body replaces the region
   exactly; whitespace is not trimmed (a leading blank line is part of about
   15% of TRAIN `finish_block` targets), only CR before LF is dropped.
5. **Display.** If the new text only adds characters at the cursor, it is ghost
   text (Tab accepts). Other changes (renames, rewrites, deletions) are shown as
   a preview after the line, with the full diff on hover; `Alt+Enter` applies it
   after checking that the document is unchanged.

## Sidecar flags

`src/server_args.ts`: `--alias sepalith --host 127.0.0.1 -c 16384 --parallel 1
-t N -tb N -ngl G --cache-ram 2048 --temp 0 --seed 3407 --no-context-shift
--no-webui`. These are the quant gate's flags (`gguf_dev_eval.py`) plus the
prefill confirmation's `--cache-ram 2048`, context and alias. The gate's
`--no-cache-prompt` is not used because the warm-up relies on the prompt cache.
Flash attention, batch sizes and KV types stay at llama.cpp defaults, as in both
measurements.

## Managed runtimes

Set `sepalith.manifestUrl` to a reviewed HTTPS release manifest and leave
`serverPath` empty. The extension validates the declared model/prompt profile,
installs hashed runtime/model files outside the extension, and reuses verified
cached assets on offline restarts. No public release URL is supplied yet.

A managed `modelPath` override must match the manifest's model hash and size.
For a different development model, set both `serverPath` and `modelPath`
explicitly and verify its prompt compatibility yourself.

Use **Sepalith: Refresh runtime manifest** to fetch a new manifest. Ordinary
starts use the cached version. A failed refresh retains the previous manifest;
stop and start the server after a successful refresh to apply it. Backend choices
are `auto`, `cpu`, `vulkan` and `metal`; unavailable choices may use a compatible
CPU bundle. Asset downloads have deadlines and verify both size and SHA-256.

## Offline development checks

From the repository root, with npm dependencies installed:

```sh
python3 scripts/check_product.py
```

This checks TypeScript, PRM03 parity, manifests, cached provisioning and
owned-process termination using fake assets and tiny Node processes. It launches
no model server. Actual editor acceptance and native-platform release testing
remain separate. See [serving and packaging](../../docs/SERVING-PACKAGING.md).

## Behaviour notes

- Manual server configuration defaults to CPU (`gpuLayers: 0`). Managed
  runtimes select a supported backend; set `backend: "cpu"` to require CPU.
  Reserve a suitable resource window when sharing a machine with training.
- Only the exact child process this extension spawned is ever killed. If the
  configured port is already answering, that server is treated as external:
  it is used as-is and never touched on deactivate.
- State lives in the status bar on the left (`Sepalith: ready (model)`,
  `starting…`, `external server`, `stopped`, `error`); hover it for the last
  request latency and token count. Errors surface there, never as popups.
- Per-request logs (latency, prompt char count, prefix truncation, sidecar
  stdout/stderr tails on error) go to the `Sepalith` output channel.

## PRM03 parity checks

    npm run check-prm03          # golden fixtures: render, parse, selection, history, editor rebuild
    npm run check-completion     # output classification: edit, no-op, malformed, cap hit, EOS 130073
    npm run check-prm03-train    # local: all 15,006 TRAIN rows (+ --census FILE: 120,048 census outputs)
    npm run e2e-cpu -- --cases 8 --out /tmp/e2e.json   # local: real llama-server on CPU, TRAIN cases

`tests/fixtures/prm03-golden.json` is built from TRAIN data only (DAT10 v4
sidecar and rows, and C12 census outputs on TRAIN prompts) by
`tests/fixtures/build_prm03_fixtures.py`. Rebuild it after changing
`campaign_protocol.py` or `campaign_selection.py`; the fixture records both
source hashes. `tests/e2e_crosscheck.py` re-decodes the e2e token IDs with the
HF tokenizer and re-classifies them in Python.

## Warm-up unit tests

    npm run check-warmup

Exercises the warm-up scheduler in `src/warmup.ts` (debounce, at most one
warm-up in flight, newest-state supersede, the disable setting, and that a
warm-up body equals the real request body except `n_predict: 0`) under
plain node.

## Editor integration checks

See [editor acceptance](../../docs/EDITOR-ACCEPTANCE.md) for the isolated extension-host tests and the remaining native release checks.
