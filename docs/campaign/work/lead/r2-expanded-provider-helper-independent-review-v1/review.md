# Expanded-provider abortable-helper independent review

Verdict: the helper is a useful isolated component, but the future extension integration is not ready to bind yet. The exact two TRAIN fixtures preserve the namespace/source-evidence classification under a 2,000 ms shared budget. Context-selection parity remains untested in the new helper packet, and two process-control conditions need either a narrow source fix or an explicitly narrower ownership claim.

## Verified

- The source and artifact manifests match every listed byte. The predecessor serving-parity source manifest is pinned as `b76c998e…`.
- `RequestHelperBudget` starts before package discovery and is shared by the serial NAMESPACE and source helpers. The independent 2,000 ms replay completed e677 in about 712 ms and the valid control in about 714 ms on this host. These are observations, not a five-second editor latency acceptance.
- An editor abort or version change reaches the owned helper signal. PID/start-tick checking prevents a recycled leader PID from receiving TERM/KILL. Output buffers remain bounded at 64 KiB.
- Exact e677 remains `source_import_evidence_unavailable`; missing executable, nonzero exit, malformed helper JSON and invalid inventory stay infrastructure/unresolved paths. The valid control remains supported. The pinned R helpers parse fixed bytes and contain no `system`, `system2`, `pipe`, shell, package-load, or source execution path.

## Findings

1. **Integration blocker — the packets have no composed selection test.** `test_train_parity.ts` invokes only `resolvePredictionNamespaceEvidence`; it never composes `abortableNamespaceResolver` with `ExpandedPrimaryContextCoordinator`, `providerSelector`, tokenization, and `selectPredictionContext`. It therefore proves evidence classification, not full-document/complete-span mode, replacement range, or prompt SHA. The future extension needs an actual editor-shaped composed test for e677 and the valid control before building a VSIX.

2. **Boundedness defect under signaling errors.** `requestStop()` launches `killOwnedGroup()` with `void` and does not retain or await that promise. Concurrent abort/budget/overflow events can start multiple cleanup calls. If `process.kill` fails with anything other than ESRCH, the detached promise can reject while `runOwnedHelper` remains blocked on `exit`. Use one memoized stop promise, race/await it with exit, convert cleanup errors to a typed terminal failure, and retain an outer hard settlement bound.

3. **The group ownership claim is broader than the implementation.** Cleanup first requires the leader `ChildProcess` to remain alive. If the group leader exits after spawning a descendant, `sameOwnedProcess` returns false and no group signal is sent. The independent fixture observed a descendant write its delayed marker after `runOwnedHelper` had settled. The two pinned R helpers do not spawn descendants, so this does not falsify their present fixture execution. Before claiming a general owned-process-group primitive, either keep an owning wrapper leader alive until the group is empty or narrow the API to audited single-process helpers and reject/verify descendants.

4. **Cache-identity production wiring remains absent.** The adapter compares an observed `<real namespace path>@sha256:<digest>` only after helper completion, but the active extension does not yet provide that expected identity or its invalidation source. A future integration must bind how the NAMESPACE identity is obtained and refreshed without using stale cache metadata.

The key candidate TypeScript files remain mode 0644. Their manifests bind bytes, but they are not filesystem-immutable. Root should copy reviewed bytes into a fresh integration snapshot, add the fixes/tests above, then freeze that snapshot. Do not edit or replace the active E750 selector.
