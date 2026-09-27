# RUN-01 stable VS Code API independent review

Observed 2026-09-12T21:21Z. This is a CPU-only, read-only review of the current
RUN-01 candidate. I did not edit EXEC, its tests, snapshots, the model stack, or
live state; I did not start a server, model, GPU job, SSH session, or network
request. Root owns packaging and the real notebook launch.

## Inputs and compatibility

The reviewed EXEC files were:

- `extensions/vscode-sepalith/src/extension.ts`, current SHA-256
  `e2ff2f24665fd3eca2aaa25c9777dea10005d2117f7ab93c9ae10e466a3d0f69`, compared
  with preserved pre-change `extension.ts.before`,
  `4fa6a06d3d3cbf69eda51195dc81b8b793739a1ed0f0763ad3d7d63a475a9727`.
- `extensions/vscode-sepalith/scripts/check-provider-requests.cjs`, current
  SHA-256
  `5b9e1ea6b808fb5685f5dda79508cc219ea78c9f5367e98bc57d7f46a50c73cb`, compared
  with preserved `check-provider-requests.cjs.before`,
  `b45bdc64391365293b0da26fa112bd790d85f329f28a59286c9635103bf51d14`.
- `extensions/vscode-sepalith/node_modules/@types/vscode/index.d.ts`, version
  1.125.0, SHA-256
  `b38554611e1421314f60a3854dda5f00ad70819a44bdde4ed94e2776d49277cc`.

The extension declares the stable engine range `^1.85.0`. The local stable API
declaration defines `InlineCompletionItem.command?: Command` and documents it as
executed *after* inserting the completion (lines 5331-5366). The same contract
is in the [official VS Code API reference](https://code.visualstudio.com/api/references/vscode-api#InlineCompletionItem).
Therefore assigning `item.command` is compatible with the stable provider API;
the removed `handleDidShowCompletionItem` and
`handleDidPartiallyAcceptCompletionItem` members were correctly absent from the
provider. The command is registered as `sepalith.didAcceptInline` before the
inline provider is registered, and is attached in both primary and legacy item
construction paths.

The candidate VSIX recorded by the root receipt was
`vscode-sepalith-primary-stable-api-0.0.7.vsix`, SHA-256
`dc45d0d843f479c8ba3f4814f4eb4dd4ed71e5bc75a776fc04c7f30e5987623f` (38,189
bytes). The preserved accepted profile VSIX remains
`vscode-sepalith-primary-profile-0.0.7.vsix`, SHA-256
`08604f9a69625a55ad43c65e6ec605325a345a6d304ce107431727dd6922c029` (38,200
bytes). Root must rebind the review to any later rebuild hash.

## Behavior review

For a normal full insertion with `postAcceptCooldown=true`, VS Code first applies
the completion edit and then invokes the item command. The document-change
handler captures history, invalidates the provider, and schedules the normal
debounce. `didAcceptCompletionItem` then clears that pending debounce and
invalidates again. The next genuine document edit gets a fresh debounce. This is
the intended cooldown sequence and avoids swallowing the next user edit.

The current checker uses the actual bundled provider in a mocked editor. Its
stable-registration assertions confirm both proposed hooks are absent. Its
acceptance scenario verifies the command name, fires the insertion change before
running the command, observes no insertion-triggered suggestion, then fires a
second change and observes exactly one trigger. The run passed:

```
npm --prefix /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith run compile
# tsc --noEmit: PASS
node /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/scripts/check-provider-requests.cjs
# 10 actual provider lifecycle scenarios passed; no server or model started
```

The same suite still passes the late transport response, LSP scope, restart,
timeout, and configuration invalidation scenarios. `postAcceptCooldown=false`
leaves the document-change invalidation/debounce behavior in place; the command
only records acceptance in that mode.

The no-op path is source-correct: primary planning returns `[]` for
`plan.mode === "none"` before constructing an `InlineCompletionItem`, and the
legacy path returns `[]` when there are no output lines. No command can therefore
be attached to a no-op. The current checker does not have a dedicated no-op
regression case, so this is a static finding rather than a complete behavioral
gate.

## Open findings

1. **Medium, stale-command race.** `didAcceptCompletionItem()` receives no item,
   request, document, or generation identity. A delayed command from an old item
   can increment the global acceptance count, clear the single global
   `debounceTimer`, and invalidate the current provider request. The existing
   late-response tests do not exercise this command race. The narrow fix is to
   bind an opaque item/request generation to the command (or make the command
   verify the current accepted item/document before clearing or invalidating),
   then add an old-item-after-new-item mock case. Keep the live gate open until
   this is either fixed or explicitly accepted as a low-frequency limitation.

2. **Medium, partial acceptance is unverified.** The stable API supplies an
   after-insertion command but no stable item-specific partial-accept callback
   equivalent to the removed proposed hook. The mock manually invokes the
   command; it does not prove how notebook UI partial acceptance dispatches it,
   nor distinguish partial from full acceptance. Treat the counter as
   “acceptance command fired after insertion” until a live UI check establishes
   the observed semantics. Do not claim partial-accept telemetry is complete.

3. **Low, generated-counter naming.** `SepalithProvider.shown` increments when an
   item is constructed and returned, not when VS Code actually renders ghost
   text. The UI and dump-stats payload now say `generated`, which is accurate,
   but the counter name and comments still say `shown`/“ghost text rendered”.
   Rename or clarify before relying on this as displayed-ghost telemetry.

4. **Low, no-op test coverage.** The no-op source path is safe, but the current
   10-scenario checker lacks an explicit empty-plan case. Add one if the release
   checklist requires executable no-op evidence.

The normal full-accept cooldown is therefore source-compatible and mocked-pass,
while stale-command identity, real notebook activation, and full/partial UI
semantics remain live gates. This review makes no quality or promotion claim.
