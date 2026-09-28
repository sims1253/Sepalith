The supplied evidence establishes one visible inline Q8-route suggestion and
three actual unsaved commits. It does not establish valid multiline output
or in-flight production cancellation. Root retains final acceptance.

The capsule analyzer replay and 92 independent assertions agree on 2467
saved frames, zero reported dropped frames, 59 harness events, two
deterministic controls, and four production-route cases. All six case windows
have a ghost-free baseline, focused visible frames, coverage through their
endpoint, and maximum frame gaps below 100 ms. The largest case gap is
35.1 ms. The observer eventually reports a CDP `Runtime.evaluate` timeout,
but its final saved frame is 4914 ms after the final case ends. The error
limits whole-run observer completion; it does not erase the verified saved
case windows.

| Case | Observation | Limit |
| --- | --- | --- |
| Deterministic visible control | 114 sampled frames; 94 with the sentinel. First rendered interval 221–239 ms. Screenshot confirms it. | Plaintext test provider; not Q8 evidence. |
| Deterministic cancelled control | Provider call starts, receives cancellation, then returns the stale sentinel anyway. Zero visible sentinel frames during the 1801 ms post-edit window. | Instrumentation/VS Code control only. |
| Q8 inline | 318 sampled frames; 226 with ghost text `1`. First rendered interval 1426–1443 ms after the outer command. Screenshot confirms the ghost. | One synthetic observation on an already-ready managed route. |
| Q8 stale | Zero ghost frames over a 5203 ms post-edit window. The recorded invalidation is 45 ms after trigger. | No request-start/cancel/response ledger establishes an in-flight production request at the edit. |
| Q8 multiline 1 | Actual commit adds six lines. No ghost appears under the configured observer selectors. | Exact buffer fails R parse; no positive multiline observer control. |
| Q8 multiline 2 | Actual commit adds five lines. No ghost appears under the configured observer selectors. | Exact buffer fails R parse; no positive multiline observer control. |

`ghost-01.png` visibly shows the deterministic sentinel in its plaintext
document. `ghost-02.png` visibly shows an italic gray `1` in `primary-accept.R`
after `result <- value +`. The second screenshot was requested 1455 ms and
completed 1512 ms after the Q8 case trigger. This supplies a later visibility
witness. The 1426–1443 ms interval brackets the first positive DOM animation
frame; it is not a direct compositor-presentation timestamp. All clocks here
are from the notebook OS. These are outer-command-to-render observations,
not native/provider latency, tunnel measurements, or a representative p95.
The run uses Xvfb at 1280×800; no physical user-display timing was measured.

The three exact `after_text` values match their stored SHA-256 values and
their text-change records. Independent reconstruction preserves the original
prefix and suffix and verifies each inserted text hash, character count,
document version change, and line-count delta. The unsaved buffers were
copied byte-for-byte to `q8-inline.after.R`, `q8-multiline-1.after.R`, and
`q8-multiline-2.after.R` for the parser handoff.

Root's newly supplied `root-r-parse.json` binds its results to those same
hashes. The inline buffer passes. Both multiline buffers fail with
`unexpected end of input`: each lacks the outer function's closing brace.
The result is **1/3 parse successes overall and 0/2 multiline parse
successes**. No repair or R execution was performed. The multiline commits
prove that `editor.action.inlineSuggest.commit` can insert multiple lines;
they do not prove valid complete R, semantic usefulness, or explicit
multiline-range `WorkspaceEdit` acceptance. The old replacement ranges are
same-line ranges, which the pinned product routes through inline completion.

All 11 capsule file hashes match the manifest. The local VSIX hash matches
the preflight's accepted identity:
`a56d2e442ae7abf266ef5f508d25443d00acecb9a78f92787b7ac332e3e2836d`.
Its package version `0.0.7` matches the activation event. Source inspection
shows that the deterministic provider is restricted to plaintext control
files and disposed before the R cases. The R cases invoke the production
`sepalith.suggest` command, and the pinned VSIX registers its R provider.

Settings, launch metadata, and preflight agree on the managed Vulkan route,
4096 context, 5000 ms request deadline, and selected theta0 Q8 path. The
declared model digest is
`22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559`.
Preflight checked model size only and explicitly says it did not read model
content. The pinned VSIX contains a checksum guard for the model override,
but this local evidence set does not include the native load log or `/props`.
This review therefore binds the configured production route and source
identity; it does not independently attest loaded model bytes. No worker
model access occurred.

The target renderer's recorded Linux file hash agrees between ready and
observer receipts. The capsule's earlier Windows renderer inspection is a
different source artifact. The target renderer file itself was not supplied
locally, so its hash was checked for receipt consistency only. The accepted
screenshots and controls support the observed selectors. No claim of a
complete transitive source closure follows.

Production cancellation remains pending. The saved event stream contains no
production request ledger. The pinned VSIX creates an output channel named
`Sepalith`; request/response lines do not include request IDs or timestamps.
Its richer `RequestLog` stays in memory, uses `toLocaleTimeString()`, and
updates an aborted tree entry without appending a cancellation line. Root
can limit any log collection to the owned run's `user-data/logs/` subtree;
the exact timestamped child filename is absent from these local inputs.
Recovered untimed channel lines may still be insufficient to order the
45 ms edit against a request. See `logger-source-review.json` for source
locations and the exact owned log root.

The saved host exits with code zero, and the guard receipt reports no
survivors after 57.735 seconds. This is process-cleanup evidence, not blanket
experimental acceptance or a current live process audit. No state, leases,
network, GPU, native server, model weights, or final content were accessed or
changed by this review.

Replay locally from this directory:

```sh
node replay_analyzer.mjs
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -B review.py
```
