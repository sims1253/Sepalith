# RUN-01 acceptance-command identity preparation

Observed 2026-09-12T21:32Z. This is an isolated CPU-only preparation against
the exact current EXEC source. EXEC, its tests, snapshots, live state, model
files, and bundles were not edited. No network, SSH, server, model, or GPU
process was started.

## Patch

The base is current `extensions/vscode-sepalith/src/extension.ts`, SHA-256
`e2ff2f24665fd3eca2aaa25c9777dea10005d2117f7ab93c9ae10e466a3d0f69`. The
paired current provider fixture is
`extensions/vscode-sepalith/scripts/check-provider-requests.cjs`, SHA-256
`5b9e1ea6b808fb5685f5dda79508cc219ea78c9f5367e98bc57d7f46a50c73cb`.

`extension.ts.patch` is a compact unified diff. The proposed copy imports the
existing framework-free `offsetAt` and adds a bounded
`Map<string, InlineAcceptanceBinding>` to the provider. When an inline item is
returned, the map retains only:

- the item document URI;
- `document.version + 1`, the expected version after the one inline edit;
- the SHA-256 of the predicted complete post-edit buffer; and
- the current sidecar `runtimeGeneration`.

The predicted buffer is hashed in memory from the existing document and exact
replacement range/text. It is never placed in command arguments or logs. The
item command receives one opaque per-provider ID. The registered command looks
up and consumes that ID, resolves the open document by URI, and requires all
four bindings to match before incrementing acceptance or touching the global
debounce. A missing, replayed, closed-document, changed-version/hash, or
runtime-stale ID is discarded with a short rate-limited diagnostic. The map is
bounded at 128 entries and URI bindings are removed when a document closes.

This preserves the stable `InlineCompletionItem.command` after-insertion path
and the existing `postAcceptCooldown` behavior. The current extension's
`InlineCompletionItem.command` API remains the stable contract documented in
the [official VS Code API reference](https://code.visualstudio.com/api/references/vscode-api#InlineCompletionItem).
Partial acceptance remains unsupported/unverified; the patch does not claim to
solve that limitation.

One integration detail is required in the existing EXEC mock: its
`workspace.textDocuments` must contain the document whose item is accepted, and
its command helper must forward `item.command.arguments`. The production VS
Code workspace already supplies that document list.

## Checks

`test-acceptance-identity.cjs` bundles the proposed copy with the current EXEC
modules and a mocked editor. It passed all focused checks:

1. a real predicted full insertion matches URI/version/hash, records one
   acceptance, cancels only the insertion debounce, and permits the next edit;
2. a delayed old command after a newer edit does not cancel the fresh debounce
   or increment acceptance;
3. a command whose URI is absent from the open-document list is ignored;
4. a same-URI changed buffer fails the hash/version binding; and
5. a sidecar runtime transition fails the runtime-generation binding.

The test also asserts that command arguments do not contain the complete
fixture document text. Output:

```
acceptance-identity: valid full accept, stale edit, wrong URI/hash, and runtime invalidation passed; no full text in command args
```

A read-only TypeScript compile of the proposed copy, with symlinked current EXEC
modules in `ts-src/`, passed with `tsc --noEmit --strict --skipLibCheck`.

## Review limits

The expected version is conservative (`pre-edit version + 1`) and the complete
post-edit hash is authoritative. An editor behavior that applies an inline item
without one document-version increment will be rejected rather than counted.
The focused test models the documented after-insertion command ordering; it does
not establish live notebook partial-acceptance dispatch. Root must apply the
patch to the unchanged EXEC source, update the existing mock to pass command
arguments and expose open documents, rebuild, and re-run the live activation
and acceptance gates. No quality or promotion claim is made.
