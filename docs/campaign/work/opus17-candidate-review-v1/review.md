# Opus17 source candidate review

Decision: do not admit the submission as written. The standalone scheduler is
sound under its supplied fake-clock cases, but the integration has a real manual
trigger bug and fails TypeScript checks. A minimal repaired private candidate is
available for root review. No default-delay promotion or runtime launch occurred.
Only the extracted final answer and supplied packet were read. The raw external
stream was not accessed.

## Blocking submission defects

1. The integration uses `vscode.InlineCompletionTriggerKind.Explicit`, which does
   not exist. The installed stable declaration defines `Invoke = 0` and
   `Automatic = 1`. With `editorAutomaticRequests=false`, actual Invoke requests
   are therefore treated as automatic and suppressed. The actual integrated
   provider under a mocked editor reproduced zero request starts and zero items
   for an Invoke call. This defeats both manual suggest and timer-triggered
   requests in the proposed experiment arm.
2. `c.get("editorAutomaticRequests", true) !== false` infers literal true under
   the repository types and produces TS2367. The test's default imports from
   node:test and node:assert/strict produce two TS1259 errors with the unchanged
   repository tsconfig. These are four distinct compiler diagnostics observed
   in the initial preparation phases, including the nonexistent enum member.
3. The submitted `git apply --recount` patch does not apply to the accepted remote
   source. Its last context also predates the remote disconnect comment. After
   updating that comment only, all 15 old hunks matched exactly once, but the
   approximate patch still failed git apply. The review applied the 15 unique
   exact contexts privately and regenerated an exact diff. The regenerated
   diff passes git apply --check against another private accepted baseline.

The private review variant changes only the enum to Invoke, uses get<boolean>,
uses named/namespace test imports, and adapts the one remote comment context.
No runtime profile, transport, tokenizer, prompt, model, shared-request deadline,
or stale-publication implementation was altered. The extracted original module,
test and integration diff remain intact beside the repaired variant.

## CPU evidence and limits

Ten final commands passed on one CPU, plus the independent patch check:

* 17 supplied scheduler fake-clock tests: all pass on the original module.
* 12 actual integrated extension-hook scenarios: correct default1500 and fixed350
  timing, superseding edits, immediate manual trigger, switch/close cancellation,
  post-accept cooldown on/off, and automatic suppression after the enum repair.
  Three scenarios explicitly reproduce two remaining behavioral limits below;
  a successful test run does not turn those limits into product acceptance.
* One submitted-variant Invoke case reproduces the manual suppression defect.
* The existing 10 provider lifecycle scenarios and 38 PRM-05 assertions pass.
  The existing provider mock needed the real new event registration and enum,
  and an active document for the new timer fire gate. Scheduler, acceptance and provider methods are actual private source;
  editor and transport boundaries are simulated. Runtime-profile checks also pass.
* Typecheck and extension bundle pass for the repaired private variant.

Remaining limitations reproduced in the actual integrated handlers:

* An empty save event still arms the timer. With no cached result, the later
  explicit provider call enters the native-client completion method (mocked
  at the transport boundary). Thus the proposed experiment's
  zero-inference no-op invariant is not implemented. This behavior also existed
  before the candidate; it must not be presented as a new fix or a clean invariant.
* Changing debounce from350 to0 while its timer is pending does not cancel that
  timer. It still fires at350. The configuration callback invalidates requests
  but does not cancel scheduler work. This is a remaining live-setting limit.

The standalone cooldown test also correctly documents that a late acceptance
callback cannot retract a timer that already fired. CPU tests cannot establish
that real VS Code always delivers acceptance before a shorter timer expires.
With editorAutomaticRequests=false, VS Code may re-query automatically and hide
a visible explicit item after receiving an empty result; actual editor behavior
requires CDP observation. No editor, gateway, network, model or GPU ran here.

## Default and fixed350 assessment

The package default remains1500ms and editorAutomaticRequests defaults true.
There is no adaptive algorithm; the module is a fixed-delay scheduler. Its
normalization now clamps positive delays to50..60000ms and substitutes1500 for
NaN/infinite/non-number values. It also adds switch/close cancellation and an
active-document/version fire gate. Therefore the default configuration values
are unchanged, but saying all default behavior is unchanged is too broad.

Fixed350 is defensible as an experimental setting. It is not supported as a new
shipped default by the supplied measurements. The accepted remote extension
already supports sepalith.debounceMs=350; a configuration-only A/B avoids adding
this scheduler patch before its extra semantics are needed. Keep the1500 default
until root explicitly admits a measured result.

The source confirms that the existing provider serves editor Automatic calls
independently of Sepalith's timer. Their actual delay/frequency is not measured
here. First diagnose that route: if VS Code already produces results before the
1500 timer, reducing this timer may alter waste rather than first-visible latency.
The supplied toy nine-edit script fires the timer2 times at1500,4 at350,3 at500
and5 at200. Those are scheduler wakeups, not measured physical inferences or GPU
cost; sharing and cache behavior must be observed separately.

Root admission should use the same selected Q8/model/runtime/renderer, interleave
1500 and350 with editorAutomaticRequests=true first, and retain post-accept
cooldown. Use real CDP ghost/screenshot evidence for last-edit to first-visible
latency, gateway/backend completion IDs for physical dispatch/cancellation counts,
and actual edit/no-op output review. Provider invocation counters count callers,
not native inference; shared callers can join one request. Subtracting returned
items from completion starts can therefore give misleading waste counts.
The proposed `timing ... items` timestamp is provider-return timing, not actual
visibility. The accepted CDP observer supplies the missing renderer observation.

Root must decide whether to admit a configuration-only350 A/B, request the small
remaining scheduler fixes, or retain1500. If the repaired scheduler is used, root
must review its opt-in automatic suppression separately and observe its effects
in a real editor. This review does not promote either path.

## Files and reproduction

`auto_trigger.ts`, `check-auto-trigger.ts` and `integration.diff` are exact
extracted blocks. `submitted-extension.ts` retains the uniquely applied source
before the two integration repairs. `reviewed-candidate.diff` is the exact
applicable repaired candidate; `extension/` contains its source and bundle.
`inputs.json`, `patch-context.json`, `patch-check.json`, `initial-cpu-results.json`
and `cpu-validation.json` retain identities and actual CPU evidence. Dependencies
are used read-only through the accepted extension's node_modules symlink.
Run `python3 run_cpu_checks.py` for the final CPU suite. No VSIX was produced.
