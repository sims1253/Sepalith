# RUN-04 independent remote editor review

This directory is a read-only review snapshot for `RUN-04-remote-editor-a`.
`review_remote_editor.py` checks the retained desktop gateway/native evidence,
remote harness events, exact synthetic R buffers, renderer frames/screenshots,
causal cancellation ledger, process start-ticks and owned ports. It does not
launch a server, editor, native process, GPU job or model.

The review keeps deterministic plaintext controls separate from selected-Q8
model cases. The controls show visible single-line and multiline ghost text;
the canceled control does not publish its sentinel. The selected-Q8 inline
case inserts one character and parses after insertion. Both selected-Q8
multiline cases change the buffer but fail `parse(text=...)` with an unclosed
function. The stale case observes gateway dispatch before the edit and then a
requester disconnect/backend cancellation; the gateway does not claim native
task admission or release.

Run the CPU-only review from this directory:

```sh
python3 review_remote_editor.py
```

The command regenerates `input-manifest.json`, `review-report.json`, the exact
accepted-after R buffers, and the campaign receipt. It invokes R only as
`parse(text=readLines(...))`; it never evaluates the buffers. The three PNGs
are retained for visual review, and `remote/renderer-analysis.json` is the
bounded frame analysis from the supplied analyzer.
