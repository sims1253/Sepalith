# Actual automatic 1500 ms renderer review

The multiline observer control fails, so A/B visibility metrics remain blocked.
The visible control and delayed-cancellation control pass. The notebook guard
ended at 2026-09-13T14:24:18.559404Z after 85.321 seconds, child exit 0, with no
owned survivors. This establishes terminal cleanup, not experiment acceptance.

The unchanged analyzer reviewed 4,446 captured frames, all focused and visible,
with zero dropped frames. Four gaps over 100 ms occurred before the first
control; the maximum was 350.6 ms. Control windows and the 13 automatic input
windows meet the analyzer's bounded coverage rule. The last recorded frame was
44 ms after the last case ended. The observer then returned its graceful
Runtime.evaluate-timeout status. That tail status does not establish coverage
beyond the last recorded frame.

Screenshot `remote/ghost-02.png` visibly contains all three multiline sentinels.
The old measured stream records only FIRST as classified ghost text in 95
frames. Seven bounded diagnostic frames locate SECOND and THIRD under sibling
`.view-zones`, zone ID `d1`, at x=418, y=111 and y=130, height 19. Their line
containers are 1,000,000 pixels wide. The old center-point calculation therefore
clamps x to 1279 on a 1280-pixel viewport, outside the editor, and reports these
visible text lines occluded. Diagnostic rows remain unscored; ordinary editor
code was not promoted to ghost evidence. The actual multiline insertion check
passed independently, with the exact three-sentinel buffer.

The exact notebook renderer source was read at its pinned SHA-256
`e04cb8ec68016ccf4557690a2c8c908c3534614af1abe4c5c557157548169d78`.
Bounded excerpts show its ghost renderer creates a `.suggest-preview-text`
wrapper with direct `.view-line` children and width 1,000,000 px. Its view-zone
infrastructure assigns `monaco-view-zone`. This supports a narrow continuation
selector and text-range measurement; it does not justify scoring arbitrary
sibling zones or ordinary `.view-lines`.

The private `observe_renderer-v2.patch` adds only
`.view-zones [monaco-view-zone] .suggest-preview-text > .view-line` continuation
roots. Each line must have that specific ghost wrapper and zone ancestry.
`Range.getClientRects()` on text nodes supplies text rectangles; visibility,
ancestor styles, viewport inclusion and occlusion checks apply to each rectangle.
A 128-rectangle bound fails closed. Ordinary document lines do not qualify.
The CDP/input-dispatch/500 ms shutdown source is unchanged. The 11-file private
`observer-v2-capsule` plus manifest is ready for root review and fresh staging.
It has not been deployed. Root must derive fresh run identities/guards and
recapture the controls before admitting metrics; compare arms under the same
reviewed observer version.

CPU tests exercise the recorded million-pixel line-container shape with
synthetic 176.4-pixel text-range fixtures, confirm hit points remain inside the
editor, reject occluded continuation and ordinary document text, preserve focus
and input IDs, and verify shutdown source identity. The candidate uses the real
browser Range API, but these CPU fixtures do not claim that actual browser text
ranges have already been recaptured.

All 13 requested CDP keys have unique matching trusted/focused renderer events,
matching characters, and actual document-version increases. These are synthetic
renderer keyboard events, not physical OS input. The four case input counts are
6, 3, 2 and 2. There are 1,530 gateway observation rows, zero observation errors
and zero drops; case dispatch-sequence deltas are 5, 2, 0 and 3. These are HTTP
dispatch observations. They do not establish native task admission, release,
exact request-to-item attribution, or debounce causality. Root owns that review.
The analyzer's blocked latency fields are retained as diagnostics and are not
admitted metrics or a representative p95.

The actual remote capsule's 14 manifest-listed files match the staged manifest
`da5a795cac7b3da01f9f0d949f5bb1173cf751cdddc1c8fdf72dda8a18684c8e`,
including observer `11f7298f68b0c8748cf9fd8891513b35b4111ddf4f265e3b81663dc369a61feb`.
Run evidence reports the expected 1500 ms arm, UUID, binding/VSIX hashes, renderer
source, CUDA Q8 identity, context 4096/output 192 and GraphOpt 0. This reviewer
read no model weights and performed no launch, remote write or resource action.
Only 31 named completed evidence files and the exact capsule/source identities
were read through SSH. Their remote/local hashes match.
