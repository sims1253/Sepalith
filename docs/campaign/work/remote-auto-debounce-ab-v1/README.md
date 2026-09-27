# Automatic typing LAN A/B preparation

Prepared for root review after the GPU training pilot. Nothing was launched.

BLOCKED FOR LIVE ADMISSION: root subsequently observed all three multiline
sentinels in a screenshot while the inherited frame classifier captured only
the first. The separate worker owns work/remote-renderer-viewzone-repair-v1.
Root must merge that reviewed observer delta and rerun positive controls before
using this A/B. Zero matching spans do not establish absence of visible ghost.
The analyzer marks visibility metrics inadmissible and blocks stale-absence
claims when either positive control fails. CDP timeout repair is also owned by
that worker. This capsule owns only A/B input capture and measurement changes.
This capsule uses the existing accepted remote extension and its root-supplied
Q8 binding/VSIX; it contains no Opus17 patch and no production source changes.

## What the source establishes

`debounceMs` controls Sepalith's single explicit timer after document events.
It does not gate every provider invocation. The accepted provider forwards
InlineCompletionContext but its implementation ignores trigger kind, so VS Code's
own Automatic requests can run before that timer. Manual suggest clears the timer
and triggers immediately, but the actual A/B cases never invoke manual suggest.
The source has no exact live trigger-kind/request-ID telemetry. A lower configured
delay must not be credited for an earlier result that came from another route.

The only setting difference between the two isolated profiles is debounceMs1500
versus350. The same accepted settings remain: automatic editor inline suggestions
on, postAcceptCooldown true, matched remote CUDA, context4096/output192, timeout5s,
scope off and synthetic debug logging on. `editorAutomaticRequests` is not added
or changed. Prompt/model/tokenizer/client code is identical. Controls stay on
plaintext files and never count as selected-Q8 evidence.

## Root launch

Use the existing root-owned 300-second process/memory guard after the training
pilot is finished and runtime ownership is available. Establish the same exclusive
SSH loopback forward on notebook127.0.0.1:18403. Run each arm separately with a
fresh run root, profile, workspace and extension installation. Supply the actual
notebook renderer hash as in the accepted remote editor capsule:

```sh
xvfb-run -a --server-args='-screen 0 1280x800x24' node run_remote_editor.mjs \
  --code /usr/bin/code \
  --vsix /CAPSULE/candidate.vsix \
  --vsix-sha256 b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1 \
  --binding /CAPSULE/binding.json \
  --binding-sha256 26cff670485b2384758af0c478e78aa040e5f1f36d10e3c753257011dd55ce94 \
  --instance-id 223bdcf2-c958-4477-93e2-72a80c769152 \
  --renderer-source /usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js \
  --renderer-sha256 NOTEBOOK_RENDERER_SHA256 \
  --run-root /EXISTING-PARENT/fresh-auto-1500-a \
  --debounce-ms 1500 --debug-port 19403 --timeout-ms 180000
```

Repeat with a new run root and --debounce-ms350. The parser rejects other arms.
The run root must not exist; its parent must exist. Use an ABBA or BAAB order
across fresh runs to expose drift. One pair is a pilot, not a promotion sample.
Runtime/model caches are not reset by fresh editor profiles; report warm/cold and
order effects rather than describing these as fresh native prefill trials.

VSIX, binding and renderer hashes are checked before launch. No local native
provisioning, model hashing, TLS asset server or SSH child is introduced. Root owns
runtime/forward cleanup; the launcher touches only its owned editor children.
All synthetic buffers, profiles and evidence remain for root review.

## Actual input and sequences

The accepted visible, multiline and cancelled-result controls run first. The
observer's ghost selector, geometry, ancestor visibility, occlusion, focus and
frame-coverage checks are unchanged. The multiline control still requires actual
insertion and view-zone sentinel evidence.

Four actual cases then run automatically:

1. Typing bursts at120ms plus300/600ms pauses, ending in an incomplete R expression
   followed by a6.5s observation window. Short pauses expose competing Automatic
   calls; the final pause gives either explicit timer time to run.
2. Type an incomplete expression, wait up to3.5s for a new currently dispatched
   completion, then type another character immediately. A missed window remains
   pending. Wait6.5s to observe cancellation and possible successor work.
3. Type, then switch after100ms to a plaintext document. The old R source must not
   produce visible ghost text on the new active editor. Observe6.5s.
4. Type, pause, attempt normal inline commit, pause2s, then save and observe6.5s.
   Record whether an item actually committed, postaccept work and save/no-op work;
   do not assume the accepted timer suppresses every empty save.

Each case starts after6.6s of drain time, hides any pre-typing item, and requires
an idle gateway snapshot. File names and input text match across profiles. Input
uses CDP Input.dispatchKeyEvent keyDown/keyUp, not document text replacement.
The renderer records actual captured keydown epoch and monotonic timestamps with
unique input IDs; the harness also verifies the editor document version changed.
These are trusted synthetic browser keyboard events, not physical keyboard/OS
latency measurements. The initial file and control setup are excluded from the
case metrics. No manual production request command runs during actual cases.

## Evidence and analysis

Run `node analyze_auto.mjs /RUN_ROOT`. The result always starts not_admitted.
It includes control results, gateway dispatch-count deltas, observed completion IDs,
per-key frame coverage, new-ghost latency intervals and stale/switch observations.
Renderer keydown and frame monotonic times share one renderer clock. Already
visible ghosts before a key are explicitly separated from newly observed ghosts.
A ghost before the current timer deadline is flagged as competing-route evidence.
It is not an exact reconstruction of which production callback started the work.

GET /sepalith/observation is the implemented identity-tagged endpoint. Sequential
20ms polling records phase/sequence/ID with notebook send/receive times. A separate
5ms bounded poll gates cancellation. Sequence deltas count completion HTTP dispatch
under an exclusive gateway even when a fast request's ID is missed. Polling does
not itself invoke native completion. Counts are not model-token counts or GPU cost.
Root must retain backend_dispatched/request_cancelled/release ledger rows and native
metrics to evaluate waste and burden.

For a uniquely observed request, the analyzer supplies a tentative dispatch-to-DOM
visible interval bounded by local sequential polls. It never subtracts desktop UTC
or monotonic time from notebook time. That is a temporal candidate, not proven
request-to-publication identity: the accepted extension does not expose the request
ID associated with an inline item. Exact attribution needs root ledger/output
correlation or separately reviewed instrumentation. Extra Automatic callers,
shared requests, caches and successor work can confound naive matching.

After invalidation, any visible ghost may belong to successor work. No stale
publication claim follows from text alone. Observed absence requires contiguous
focused frames for at least5s. Native task cancellation/release still needs native
evidence; HTTP dispatch means bytes written, not task admission. Provider return
or unchanged document text never serves as visible-ghost proof.

The observer retains the accepted10000-frame buffer and12-screenshot cap; root
must check dropped samples and whether the relevant cases have screenshots.
Gateway observations have a10000-row cap and report drops. DOM frames precede
compositor presentation, so screenshots are separate visible evidence. Raw code
and buffers are synthetic only. Invalid/missing output is not latency success.

## Root admission gate

First review positive controls, fresh-profile/config/binding identity, actual
keyboard events and frame coverage. Reject or repeat runs with identity failures,
missed input, dropped relevant observations, overlapping gateway users or unresolved
stale publication. Record valid no-op outputs separately from failed requests and
from absent visibility. Compare the same final pauses, using successful and failed
trial denominators; never silently discard failed/empty cases to improve latency.

Only after multiple counterbalanced pairs should root consider a larger measured
sample for promotion. Evaluate last-key-to-visible p50/p95 with interval/coverage
limits, completions and cancelled/wasted work per100 edits, GPU milliseconds,
extension CPU, postaccept/save behavior and actual output usefulness. A delay that
saves apparent time by producing more stale or invalid output does not pass.
Any proposed latency/waste thresholds need root admission; no fixed threshold or
new default is silently imposed by this packet. Preserve1500 until root accepts
real evidence. If Automatic calls dominate both arms, report that debounce was not
shown causal and keep the result diagnostic.

CPU reproduction: syntax-check the mjs/js files, then node check_auto_preparation.mjs.
These tests validate config isolation, source seams and synthetic analyzer rejection
rules. They do not establish that a live editor accepted the input or showed a ghost.
