# Expanded provider extension integration preparation v1

This is an isolated, opt-in future extension candidate. It copies the accepted E750 cap-flex extension source and combines it with the frozen expanded-provider parity closure and abortable-helper v2. It does not modify or install the current E750 extension, VSIX, b4 profile, editor, runtime, or model.

`sepalith.expandedProvider` defaults to `false`. The false branch retains the copied E750 primary selector and all matched runtime profile limits. When explicitly enabled on a file inside a workspace, `ExpandedProviderRoute` uses the managed `NativeCampaignClient` as the tokenizer, takes `contextSize` and `maxOutputTokens` from that matched client, and sets the full-document policy generation reserve to the same output cap. Its selected `PromptContext` goes directly to `client.complete` and then `planNextEdit`; the extension does not rebuild the range.

The existing `SuggestionRequests` timeout remains the sole complete request deadline. It starts before scope resolution and aborts the same request lease used by namespace helpers, provider tokenization, and model transport. The route receives the remaining absolute deadline and caps the two serial R helpers to at most 2,000 ms within it. It does not start another five-second window. A test with a 120 ms complete request spends 70 ms before model work and proves the model receives the same abort at the original boundary.

The extension packages only the two exact SHA-pinned parse helpers. The copied helper implementation adds the root-review fixes: it checks cancellation again after asynchronous realpath/hash verification and immediately after listener registration. A FIFO-controlled test aborts during helper hashing and proves no child was spawned; a registration-race test proves the newly spawned owned PID is removed with no delayed marker. The editor-shaped shared-request test proves timeout abort reaches and cleans an actual R helper.

The namespace cache key starts from a stable `realpath@sha256` identity. The coordinator recomputes it before selection and after provider tokenization, while document URI/version/content, cursor request generation, runtime generation, and restart state are also checked before model invocation and again before emitting an edit.

Two authorized TRAIN fixtures pass the composed route with exact mode, replacement range, content hash, prompt token count, and prompt SHA. No DEV data is included or used for training. TypeScript validation and an isolated esbuild bundle pass using the existing pinned local extension toolchain. No VSIX was built because root review and a target/profile binding are still required.

This packet is not release admission. A fresh root-reviewed extension snapshot must bind an expanded target, re-run live editor cancellation/invalidation and latency checks, and preserve the disabled default until those gates pass.
