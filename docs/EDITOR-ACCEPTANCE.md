# Editor acceptance before release

VS Code now has tests inside a real extension host with a local fake HTTP server.
They cover stable activation, replacement ranges, cache reuse, cancellation,
stale documents, failed-request retry, actual inline accept/undo, acceptance
cooldown, file switching, external-server preservation and reconnection.

```sh
npm --prefix extensions/vscode-sepalith ci
xvfb-run -a npm --prefix extensions/vscode-sepalith run test:editor
```

Linux uses Xvfb. On a desktop, omit `xvfb-run -a`. The default host is VS Code
1.104.3. Set `SEPALITH_VSCODE_VERSION=1.85.2` to test the supported minimum.
The launcher creates isolated user settings and extensions directories, disables
GPU rendering, and downloads only the named editor version. CI tests both versions.
The offline `scripts/check_product.py` suite remains independent of editor downloads.

The first host run exposed experimental completion callbacks that prevented stable
activation. Full acceptance now uses the stable completion-item command. Counters
report generated offers and full accepts. They do not measure displayed ghost text
or partial acceptance. Cancelling a request aborts HTTP work; changing the document
or active editor invalidates pending and cached suggestions. Failed requests can
be retried with the same prompt.

## Native release matrix

These tests establish editor behavior with known responses. They do not establish
suggestion quality or native runtime compatibility. Before release, record:

| Target | Required evidence |
|---|---|
| VS Code on Windows, Linux and macOS | Install the packaged extension, provision a pinned runtime/model, accept and undo an edit, restart offline, verify shutdown ownership |
| Positron | Repeat the R editing checks in the named Positron release |
| Zed | Capture the actual request and proposed rewrite region, then accept, reject and undo an edit |
| GPU backends | Native startup and failure/fallback checks for each shipped backend |

Use this R fixture in every editor:

```r
# Summarize finite observations without changing the caller's values.
mean_finite <- function(values) {
  values <- values[is.finite(values)]
  answer <-
}
```

Place the cursor after `answer <-`. Save the exact document, cursor, editor version,
runtime/model identities, HTTP request and response in the private artifact store.
The summary should describe the proposed range and resulting document. Repeat with
a delayed response while typing and while switching files. Accept an edit, undo it,
and confirm the original bytes return. Stop only the server owned by the extension.

Zed uses a different context and rewrite region from VS Code. Compare each recorded
request with that editor's intended contract; do not require equal prompts or infer
equal edit quality from a successful HTTP response. Zed and Positron UI checks remain
pending. No public release or model selection follows from these tests.

The harness follows the
[VS Code extension-host testing interface](https://code.visualstudio.com/api/working-with-extensions/testing-extension).
