# RUN-01 acceptance-command identity v2 preparation

Observed 2026-09-12T21:41:33Z. The first identity candidate is preserved in
this directory (`extension.ts.proposed`, `extension.ts.patch`, and the v1
receipt copy). This v2 review again used only isolated owned files; current
EXEC and its provider tests remain byte-identical to the requested bases.

## Cross-document and newer-action fix

The v2 patch retains the first candidate's opaque command ID and URI,
expected post-edit version, complete post-edit SHA-256, runtime-generation
binding, one-shot consumption, 128-entry bound, and close cleanup. It changes
the cooldown side effect in two ways:

1. The provider increments a private action epoch on every invalidation and
   when a new provider action begins.
2. The global debounce records its owner URI, post-event document version, and
   the provider epoch at which it was scheduled.

A valid acceptance still increments acceptance telemetry when its retained
complete-buffer identity matches. It clears a debounce only if that timer is
still owned by the same URI/version and epoch. It no longer calls the global
`provider.invalidate()` from the acceptance command: the accepted document's
`onDidChangeTextDocument` event already performed that invalidation. Therefore a
command from A can count an actual A insertion while leaving a newer B timer or
request alive. A newer same-document cursor request advances the epoch and
likewise protects its timer from a delayed old command. Stale URI/version/hash
or runtime bindings are discarded before either counting or side effects.

The timer owner is cleared whenever the timer is cleared or fires. A manual
suggestion command also clears the owner. Disabling debounce clears any prior
timer before returning. The patch does not alter partial-acceptance semantics;
that remains unsupported/unverified.

## Focused evidence

`test-acceptance-identity-v2.cjs` bundles `extension.ts.v2.proposed` with the
current EXEC modules and a mocked editor. It passed seven focused scenarios:

- valid full acceptance cancels its own insertion debounce and permits the next
  edit;
- delayed acceptance after a newer edit cannot cancel a fresh timer or count;
- a valid A acceptance cannot cancel B's timer or an in-flight B request after B
  edits and starts that request, while A acceptance counting remains possible;
- a same-document cursor request protects newer work from the old command;
- wrong URI, changed same-URI hash/version, and runtime transition are rejected;
- command arguments contain no complete document text.

Output:

```
acceptance-identity: valid full accept, stale edit, cross-document/cursor ownership, wrong URI/hash, and runtime invalidation passed; no full text in command args
```

A strict no-emit TypeScript compile of the proposed source passed using
read-only symlinked current EXEC modules in `ts-src-v2/`.

## Integration and limits

The unified diff is based directly on current EXEC extension SHA-256
`e2ff2f24665fd3eca2aaa25c9777dea10005d2117f7ab93c9ae10e466a3d0f69`. Root
must integrate v2 rather than v1, update the existing mock to pass command
arguments and expose open documents, then rebuild and run the live notebook
acceptance checks. The exact `document.version + 1` contract is intentionally
conservative; a runtime that applies an inline item without one version
increment will be ignored. The v2 mock proves event ordering and ownership,
not live notebook partial acceptance. No quality or promotion claim is made.
