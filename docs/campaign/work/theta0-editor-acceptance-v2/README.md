# RUN-04 editor acceptance v2 preparation

This capsule adds a renderer observer to the accepted theta0 Q8 managed route.
The worker ran CPU checks only. Root must review and lease the notebook before
launch. The v1 capsule and production source remain unchanged.

`primary-route-spec.json`, the manifest builder, and `primary.vsix` are exact
copies of the accepted capsule. They retain theta0 step1000 Q8_0, renderer
`zeta2-prm03-v1`, renderer-parity-v3-0.0.7, Vulkan b10453, context 4096,
batch/ubatch 256, and the five-second editor request timeout.

The copied launcher starts a loopback CDP observer beside the actual VS Code
host. It uses Node's global `WebSocket`; check that this exists on the notebook.
It records target renderer source SHA-256, ghost text nodes, their rectangles,
focus, occlusion, and one timestamp per animation frame. It captures screenshots
after transitions to visible ghost text. The selectors come from installed
VS Code 1.137.0 source; `source-observability.json` records the inspection.
The target source must contain those selector tokens. This compatibility check
does not guarantee that a later renderer uses identical markup.

The harness first runs two labeled plaintext provider controls: one visible
suggestion and one delayed result returned despite cancellation. These controls
test the observer and VS Code behavior. They do not call or evaluate Q8.
Four separate R cases use the unchanged production provider and selected Q8:
inline acceptance, a concurrent edit, and two bounded multiline insertion
attempts. No completion text is injected into the Q8 route.

Root launch command, after copying this directory to the notebook and checking
that native port 18403 and renderer port 19403 are free:

```sh
xvfb-run -a --server-args='-screen 0 1280x800x24' node /home/m0hawk/.local/share/sepalith-campaign-20260915/theta0-editor-acceptance-v2/run_primary_managed_route.mjs --code /usr/bin/code --vsix /home/m0hawk/.local/share/sepalith-campaign-20260915/theta0-editor-acceptance-v2/primary.vsix --target-root /home/m0hawk/.local/share/sepalith-campaign-20260915 --candidate Q8_0 --port 18403 --debug-port 19403 --renderer-source /usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js --run-root /home/m0hawk/.local/share/sepalith-campaign-20260915/runs/theta0-editor-acceptance-v2-a --timeout-ms 180000
node /home/m0hawk/.local/share/sepalith-campaign-20260915/theta0-editor-acceptance-v2/analyze_renderer.mjs /home/m0hawk/.local/share/sepalith-campaign-20260915/runs/theta0-editor-acceptance-v2-a
```

Use the lead's process supervisor around this command. It must retain evidence
and verify that all owned processes and both ports are released. Use a fresh
run directory. Resolve the actual target renderer source path if packaging
differs. The worker did not run these commands.

CPU validation from this directory:

```sh
node tests/test_analysis.mjs
node --check observe_renderer.mjs
node --check analyze_renderer.mjs
node --check run_primary_managed_route.mjs
node --check primary-editor-harness/extension.js
node --check primary-editor-harness/acceptance-v2.js
```

Review `renderer-frames.jsonl`, `harness-events.jsonl`, screenshots,
`renderer-observer.json`, `renderer-result.json`, `renderer-analysis.json`, and
`acceptance-v2-buffers.json`. Parse the exact recorded accepted R buffers;
the harness does not save them to disk. Preserve the ordinary managed-route
inventory, manifest, source/VSIX pins, and provider/native logs too.

Acceptance limits:

- Animation-frame callbacks observe rendered DOM before compositor presentation.
  The analyzer reports an interval to the first rendered ghost. Inspect the
  associated screenshot to establish actual visible content, and use its
  recorded capture times for a conservative first-visible bound. A 5.2-second
  quiet wait is never a latency measurement.
- Missing baseline, focus loss, or frame gaps above 100 ms leave timing open.
  Screenshots occur after the detected transition and may show a later state.
- Actual Q8 cancellation remains open until root proves that a production
  request was in flight before the edit and identifies any successor request.
  The existing Request Log view exposes request numbers, outcomes, prompts,
  and raw results (`extension.ts:538–633`). A bounded follow-up is to capture
  that real view before/after the stale case and correlate its unique
  `primary-stale.R` prompt with native task/cancellation logs. Open the view
  before the observation window; clipboard commands can perturb suggestions.
  If those logs cannot establish ordering, a reviewed request-event hook is
  still required. Empty ghost frames cannot substitute for that evidence.
- A multiline generated body can be committed inline. Two real Q8 attempts may
  still produce no applicable multiline output; that remains pending. The
  WorkspaceEdit command has no request entry point for a multiline old range:
  `selectPromptContext` fixes the old range to one line, and
  `makeApplicationPlan` routes only by that old-range geometry. Testing that
  separate command requires a future reviewed explicit-selection request path.

The analysis never declares campaign acceptance, quality, representative p95,
or final release readiness.
