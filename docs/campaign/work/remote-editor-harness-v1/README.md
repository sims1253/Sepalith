# Remote notebook editor capsule

Prepared for a root-owned launch only. No SSH, editor, native process, network
request or model read ran during preparation. Root supplies its existing
300-second process/memory guard and reaps only owned editor descendants. This
capsule never owns or terminates the desktop runtime, gateway or SSH forward.

The launcher installs the supplied candidate VSIX into fresh user-data and
extensions directories and opens one disposable workspace. It does not serve a
TLS manifest, provision a native server, download assets or hash a model. It
checks only supplied VSIX, binding and renderer source hashes, then compares
public runtime identity with the binding. The extension independently performs
its full validated-manifest, live identity and /props checks.

Application settings: remote binding supplied by root; port18403; backend CUDA;
context4096; request timeout5000ms; debounce0; scope off; raw debug on for these
synthetic documents; autoStart false; postAcceptCooldown true. The output cap192
comes from the validated primary model profile, not an invented editor setting.
The harness invokes sepalith.startServer and waits for the actual status-bar DOM
to report Sepalith ready, plus checks runtime identity and native /props.

Root launch shape, under its 300-second guard (all paths supplied by root):

```sh
xvfb-run -a --server-args='-screen 0 1280x800x24' node run_remote_editor.mjs \
  --code /usr/bin/code \
  --vsix /CAPSULE/candidate.vsix \
  --vsix-sha256 b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1 \
  --binding /CAPSULE/binding.json \
  --binding-sha256 26cff670485b2384758af0c478e78aa040e5f1f36d10e3c753257011dd55ce94 \
  --instance-id 223bdcf2-c958-4477-93e2-72a80c769152 \
  --run-root /EXISTING-PARENT/fresh-remote-editor-run \
  --renderer-source /usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js \
  --renderer-sha256 NOTEBOOK_RENDERER_SHA256 \
  --debug-port 19403 --timeout-ms 180000
```

The run root must not exist and its parent must exist. Port18403 must already be
the root SSH loopback forward; CDP19403 must be free. The launcher retains logs,
settings, profile, buffers, frames and screenshots. Process stdout/stderr are
capped at4MiB each; CDP frames have the accepted10000-frame bound and screenshots
are capped at12. Root checks descendants after exit; a launcher exit is not
experiment acceptance. The renderer hash must be the actual notebook file, not
the Windows source-inspection hash in this packet.

The existing ghost selector and all geometry, ancestor visibility, occlusion,
focus and contiguous-frame coverage checks are retained. Installed source
inspection confirms additional ghost lines render as suggest-preview-text /
view-line under editor view zones. New metadata marks ghost elements inside
view zones. A separate plaintext deterministic multiline provider emits three
sentinel lines and its actual commit must insert exactly those lines. The
analyzer also requires the second and third sentinels in visible view-zone
frames. This proves observer/insertion mechanics only; it is never Q8 evidence.

Actual model cases are q8-inline, q8-stale and two multiline R function fixtures.
The original four fixtures are retained; no optional rich long-source case was
added in this timebox. The Q8 model SHA is22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559;
CUDA server SHA is in root-supplied-identities.json. Model bytes were never read.

For q8-stale, snapshot GET /sepalith/observation, trigger the actual product
command, then poll every5ms for at most1500ms. Require an exact instance header,
a newer dispatchSequence, phase dispatched and non-null completionId before the
real edit. The endpoint is implemented by the reviewed gateway; it reports
schema1, completionId, phase, utc and dispatchSequence. Dispatch means HTTP bytes
written to the backend socket, not native admission. Record the causal response
and notebook timestamps. If a response wins the race, cancellation remains
pending. Root correlates the recorded completionId with backend_dispatched,
request_cancelled and release rows. Transport release does not prove native task
release, and visible successor work requires separate identity review.

No-item commits and responses that fail to add multiple lines are marked fail,
not silently relabeled as passes. Multiline braces/marker defects also fail.
Other semantics remain root review: balanced braces alone do not prove valid R
or useful output. Complete before/after synthetic buffers are retained. The
same-line source range legitimately permits multiline inline insertText;
explicit WorkspaceEdit acceptance is still outside this route.

After the run use `node analyze_renderer.mjs /RUN_ROOT`, inspect the screenshots,
and compare actual buffers and gateway/native evidence. DOM animation frames
precede compositor presentation; the first-visible interval is outer command to
first observed rendered ghost, not native latency or a representative p95.

CPU reproduction: node --check each mjs/js file and node check_preparation.mjs.
All six recorded commands passed, including15 pure guard/analyzer assertions.
The analyzer fixtures are synthetic and make no visible-editor claim. Source
pins preserve the accepted v2 capsule and private extension identities. The
packet adds no production edits or campaign state changes.
