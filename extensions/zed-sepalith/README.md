# Sepalith in Zed

Zed's built-in OpenAI-compatible edit prediction provider supports `zeta2`, the
format used by the current Sepalith models. Use the explicit format; inferring
from a Qwen model name would select the wrong prompt format.

1. Download `sepalith.sh` for Linux/macOS from the reviewed Sepalith GitHub
   release. It detects the OS and CPU, probes Vulkan on Linux, and uses Metal on
   macOS. `SEPALITH_BACKEND=cpu` selects CPU on Linux. On Windows, download the
   launcher matching your CPU and choose CPU or Vulkan with a working driver.
2. Run the launcher in a terminal. It downloads the pinned runtime from GitHub
   and the GGUF from Hugging Face, checks size and SHA-256, then starts the local
   server. Later starts reuse verified files. Keep the terminal open; Ctrl-C stops
   the server. The launchers use the OS shell and curl, with no Node/Python runtime.
3. Merge [settings.json](settings.json) into Zed's user settings. It enables
   automatic predictions for R only, with a 1.5-second debounce. The endpoint
   stays on `127.0.0.1:18099`. If VS Code already owns a compatible server there,
   use it instead of launching a second process.
4. In an R file, run `editor: show edit prediction`; accept with Zed's prediction
   keybinding. No cloud API key is required by the local server.

Release launchers are generated with:

```sh
python3 scripts/packaging/zed_launchers.py release.json out/zed
```

The release manifest must already contain final GitHub and HF URLs and hashes.
Upload the generated launchers beside `release.json`. No release has been
published yet. `SEPALITH_HOME` optionally changes the cache root. The default
is `$HOME/.local/share/sepalith` on Linux/macOS or `%LOCALAPPDATA%/Sepalith` on
Windows. To recover from a GPU startup failure, use the CPU launcher; on macOS,
change `-ngl 99` to `-ngl 0` to use the same binary on CPU.

This integration uses Zed's own context selection and edit rendering. Its
`zeta2` prompt includes a rewrite region and can differ from VS Code's current
cursor-only prompt. API compatibility does not establish equal editing quality.
The server HTTP smoke passed locally; Zed UI acceptance and per-editor quality
still require validation in Zed.

Sources checked 2026-09-06:
[edit prediction configuration](https://zed.dev/docs/ai/edit-prediction),
[OpenAI-compatible request implementation](https://github.com/zed-industries/zed/blob/main/crates/edit_prediction/src/open_ai_compatible.rs).
