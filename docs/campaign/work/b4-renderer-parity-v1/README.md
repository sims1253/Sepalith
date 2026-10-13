# RUN-03 b4 renderer parity v1

This packet repairs the legacy, scope-off prompt assembly in the execution
extension. The pinned reference is
`experiments/eval/run_eval.py:render_zeta2` (SHA256
`7fc6d4d796856ef3697365a462a8a1f5b0a9876d1f2e55cb88325a6ff7ef493d`). The
legacy route has no history provider, but Zeta-2 still reserves the empty
history pseudo-file:

```text
<[fim-suffix]>
... suffix lines ...
<[fim-prefix]><filename>edit_history
<filename>R/file.R
... prefix lines ...
<<<<<<< CURRENT
... region with <|user_cursor|> ...
=======
<[fim-middle]>
```

`src/context_build.ts` now emits that exact empty-history layout. Its function
signatures are unchanged. `scripts/check-context.ts` invokes the actual pinned
Python renderer for focused fixtures, so the parity checks do not duplicate a
Python expectation in TypeScript. The checks cover an empty region, a cursor
after a supplementary Unicode character at its real UTF-16 offset, a
multi-line suffix, and explicit empty history.

Scope-aware rendering keeps the same empty history slot and inserts the
extension-only outline after the real file pseudo-file and prefix. This is a
deliberate extension adaptation: `render_zeta2` has no outline slot, so parity
is claimed only for scope-off/no-history prompts. The primary PRM-03 path uses
`context_select.ts` and `campaign_protocol.ts`; neither it nor `extension.ts`
was edited.

## Validation

From the execution extension directory:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON scripts/check-context.ts
npx --no-install tsc --noEmit --pretty false
```

Both pass (`check-context: OK (51 checks)`). `git diff --check` also passes.
No model, server, VS Code, Xvfb, VSIX, GPU, SSH, network, or campaign-state
operation was performed.

The accepted RUN-01 source snapshot remains identity
`41a509aaa6b2205add579e12db9afbdad8d4c40155baf798d4f263009989ebfc`, but its
source bytes are intentionally changed in the two files listed in the
receipt. The previously accepted VSIX was not rebuilt; root must review the
diff and rebuild the isolated editor artifact before any live b4 check.
