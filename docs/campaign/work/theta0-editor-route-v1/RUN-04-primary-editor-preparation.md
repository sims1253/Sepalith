# RUN-04 selected theta0 Q8_0 primary managed editor route preparation

This packet prepares the root-owned live primary editor gate. It does not
launch VS Code, a native server, a model, SSH, CUDA, or a GUI.

`build_primary_manifest.mjs` reads only the known notebook paths under
`/home/m0hawk/.local/share/sepalith-campaign-20260915`. It checks the pinned
selected theta0 step1000 Q8_0 model size and hashes the ten small Vulkan runtime files against the
b10453 inventory. It emits a manifest only when every path and identity check
passes. The model is an explicit `modelPath` override, so the extension’s
`provision()` performs its full model SHA check without downloading another
multi-gigabyte copy. The local model URL is intentionally served as a 409
response by the launcher to make a missing override fail closed.

The manifest has one Linux x64 Vulkan bundle for selected theta0 Q8_0, the loader-facing regular names
required by `runtime.ts`, and the admitted profile: context 4096, batch and
ubatch 256, parallel 1, six worker threads, HTTP threads 2, `-ngl 99`, CUDA
graph 0, Vulkan graph enabled, and source commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` with tree
`f9a9f82f92eb23b6dbc05494e542ddb1f907a0c4`.

`run_primary_managed_route.mjs` is the bounded root launcher. It creates a
one-day self-signed certificate in the owned run directory, serves the real
manifest and small runtime siblings over `https://127.0.0.1`, and passes that
certificate through `NODE_EXTRA_CA_CERTS` to the VS Code process. This is a
process-scoped trust path; it does not modify system or user trust. The
launcher installs the accepted 0.0.7 VSIX into disposable `user-data` and
`extensions` directories, writes the application-scoped manifest URL, model
override, empty `sepalith.serverPath`, Vulkan backend, native port, and
five-second editor timeout to `user-data/User/settings.json`, and does not
write workspace settings. The harness reads those settings back through the
VS Code configuration API. It caps the host at 600 seconds, passes `--wait`
so the owned CLI remains attached to the window, records bounded
stdout/stderr, records every manifest request, stops the owned sidecar through
the extension command, closes HTTPS, and removes only transient directories
created below a fresh exclusive run root. SIGINT, SIGTERM, and timeout kill
the owned process group, including native descendants; run-root logs and
compact evidence remain.

The harness creates two synthetic R files and uses the built-in VS Code API
for opening, language assignment, cursor movement, typing, history-triggering
changes, file switching, and a concurrent edit during a suggestion. It reads
the actual prompt copied by the active PRM-03 provider, records its hash and
section geometry, then reuses that exact string for the native `/tokenize` and
integer-ID `/completion` routes with BOS 0, `n_predict=192`,
`cache_prompt=true`, and `return_tokens=true`. Readiness accepts only health
`status: "ok"`; `loading` is recorded and retried, while `error` fails the
harness. The document check is explicitly a document-integrity smoke: an
inline ghost is not document content, so stale transport cancellation and
ghost nonpublication remain pending for root log/UI correlation. The harness
introspects inline acceptance commands and invokes `sepalith.applyNextEdit`;
unavailable or unobserved UI surfaces remain pending instead of becoming a
synthetic pass. The synthetic sequence contains no campaign data or user
files.

The CPU backend comparison remains separate and is not used as evidence
for this route. A successful primary result requires the exact manifest, the
managed child, native protocol responses, and an actual VS Code display. A
native response or a GUI event alone does not establish model quality or
promotion.

## Root command

After reviewing `preflight.json` on the notebook, root may run the following
shape with a newly chosen, nonexistent run directory. The launcher refuses an
existing `--run-root` and never cleans a prior run:

```sh
node /path/to/theta0-editor-route-v1/run_primary_managed_route.mjs \
  --timeout-ms 600000 \
  --code /usr/bin/code \
  --vsix /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/vscode-sepalith-renderer-parity-v3-0.0.7.vsix \
  --target-root /home/m0hawk/.local/share/sepalith-campaign-20260915 \
  --candidate Q8_0 \
  --port 18403 \
  --run-root /home/m0hawk/.local/share/sepalith-campaign-20260915/runs/primary-editor-managed-<fresh-run-id>
```

The default native port is 18403. If `--port` is supplied, the same value is
written to application settings and passed to the harness; the manifest
server continues to use its own ephemeral HTTPS port. The VS Code editor
request deadline is 5000 ms. The direct native protocol check has a separate
60000 ms diagnostic deadline and does not measure five-second editor latency.

The `--vsix` path must pass the accepted SHA in the receipt. If the notebook
does not have that worktree path, root must provide a byte-identical copy and
the launcher will verify it before any GUI process starts. The target host
must also have the real Vulkan bundle and selected model at the pinned paths;
missing or mismatched inputs produce exit 75 and no manifest.

## Live gates

- Accepted VSIX SHA and all ten runtime asset checks pass on the launch host.
- All ten Vulkan assets and the selected model pass path, size, and hash
  checks; no example or dummy URL is accepted.
- `manifest-requests.json` records the local HTTPS manifest and asset fetches
  through `NODE_EXTRA_CA_CERTS`, and managed provisioning copies only the
  verified small bundle assets.
- `host-result.json` records the values read from application settings and
  `vscode-host.stderr.log`/extension diagnostics show the actual managed
  native command, selected backend, and health transition. The launcher’s
  `launch.json` is only the requested host argv, not proof that the sidecar
  started.
- `serverPath` remains empty, the sidecar is owned by the extension, and the
  managed child reaches readiness on the selected Vulkan backend.
- Native `/tokenize` and integer-ID `/completion` return bounded evidence
  with the pinned 4096/192 geometry.
- Real editor events and the document-integrity smoke are reviewed from
  `host-result.json`; stale cancellation/ghost nonpublication and any
  unobserved acceptance/apply behavior remain open.
