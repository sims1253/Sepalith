# RUN-04 renderer view-zone repair

The retained `renderer-frames.jsonl` has one visible selector root for the
multiline control. Its screenshot shows all three sentinel lines, while the
old collector records only `OBSERVER_MULTI_FIRST` and has no line-level
view-zone metadata. The old stream therefore cannot support a line 2/3
presence or absence claim for actual Q8 multiline suggestions.

This packet leaves the accepted observer untouched and supplies:

- `observe_renderer-viewzone-repair.patch`, a reviewed patch for the collector;
  it records visible sibling `.view-line` text in the nearest explicit
  view-zone or view-lines container, with per-line computed-style, viewport and
  `elementFromPoint` occlusion geometry;
- `observe_renderer-viewzone-repair-on-ab.patch`, the same collector and
  shutdown delta rebased onto `remote-auto-debounce-ab-v1/observe_renderer.mjs`,
  preserving its keyboard-input capture and `Input.dispatchKeyEvent` service;
- `analyze_renderer_viewzone.mjs`, which consumes the enriched shape and marks
  old Q8 multiline frames `indeterminate_view_zone_metadata_missing`;
- `check_viewzone_repair.mjs`, a CPU-only retained-evidence and synthetic
  enriched-frame test;
- `renderer-analysis-viewzone-repair.json`, generated from the retained run.

The collector also catches a bounded `Runtime.evaluate` timeout after the
already-collected stream, records `stop_reason`, performs a short best-effort
collector stop, closes the WebSocket, and returns a graceful timeout status.
The timeout is an observer transport problem; it does not change model or
editor acceptance.

Run the checks without an editor, server, SSH, GPU or model:

```sh
node analyze_renderer_viewzone.mjs ../remote-editor-a-independent-review/remote
node check_viewzone_repair.mjs

# If root admits the keyboard-input observer capsule, apply this matching
# delta from the PLAN root with patch -p0 in a temporary/reviewed copy:
patch docs/campaign/work/remote-auto-debounce-ab-v1/observe_renderer.mjs \
  < observe_renderer-viewzone-repair-on-ab.patch
```

Because the retained frame stream predates the repaired collector, a fresh
bounded editor capture is required before classifying actual Q8 multiline
ghost lines. Root owns applying or admitting the patch.
