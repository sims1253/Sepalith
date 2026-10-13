# RUN-04 exact-prompt prewarm validation harness

This packet is an independent, CPU-only adversarial harness for a future
prewarm scheduler. It contains a TypeScript adapter contract, a deterministic
clock and transport, a small known-good reference used only as a fixture, and
fault injection tests. It does not change the VS Code extension or implement a
production scheduler.

The future module must expose an adapter equivalent to:

```ts
type SchedulerFactory = (
  dependencies: ExactPrewarmSchedulerDependencies,
) => ExactPrewarmScheduler;

interface ExactPrewarmScheduler {
  scheduleWarm(snapshot: ExactPromptSnapshot): void;
  requestForeground(snapshot: ExactPromptSnapshot): Promise<NativeSlotResponse>;
  dispose(): void;
}
```

`ExactPromptSnapshot` is supplied by the existing provider boundary. It carries
the complete rendered PRM-03 text, the tokenizer output before the manual BOS,
and the complete URI/version/content-hash/cursor/runtime identity. The native
request must contain the unchanged text and exactly
`[0, ...encodedPromptTokenIds]`, with `maxOutputTokens: 1` and
`cachePrompt: true`. The scheduler owns one transport slot. A same-key
foreground subscriber may join a warm request; a different foreground key
cancels warm work and waits for the old slot to settle before launching.

Run the fixture reference and all adversarial checks with:

```sh
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/opus10-validation
NODE_NO_WARNINGS=1 node --experimental-strip-types test_exact_prewarm_scheduler.ts
```

The expected result is nine correct-reference lifecycle groups and eight of
eight injected faulty behaviors detected. Type-checking uses the pinned
extension toolchain without emitting files:

```sh
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/opus10-validation
PYTHONDONTWRITEBYTECODE=1 /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/node_modules/.bin/tsc --noEmit --strict --target ES2022 --module NodeNext --moduleResolution NodeNext --allowImportingTsExtensions --skipLibCheck --typeRoots /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/node_modules/@types exact_prewarm_contract.ts test_exact_prewarm_scheduler.ts
```

The tests exercise rapid typing, foreground takeover during warm work,
same-key subscribers, no-op suppression, stale responses after URI/version/
cursor/runtime changes, exact prompt and BOS preservation, one-slot overlap and
foreground starvation, timer and abort cleanup, disposal, and post-disposal
requests. Each bad reference is expected to violate an observable assertion;
the harness reports detection rather than treating the fault as a pass.

The clock and transport are mocks. They establish ordering and ownership
invariants, but they do not measure native prefill latency, prove llama.cpp
cache behavior, or validate the production provider. A future Opus response
must provide a thin adapter around its module and rerun these cases against the
actual implementation before any integration or latency claim.
