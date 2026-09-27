# Merged remote automatic debounce capsule

Ready for root transfer and live control review. No editor, network, native
process, GPU, or model was launched during preparation. Root must accept the
actual visible, multiline, and delayed-cancellation controls before using any
A/B visibility metric. Host completion alone is insufficient.

`capsule-manifest.json` lists exactly ten transfer files and their SHA-256
hashes. Transfer those files with the same relative paths, plus that manifest.
The candidate VSIX and binding remain separate root-owned immutable inputs.
Their supplied identities are unchanged in `root-supplied-identities.json`.
No production model, runtime, transport, tokenizer, renderer contract, extension
code, or settings changed. The only arm setting difference remains
`sepalith.debounceMs`: 1500 or 350.

The observer applies the accepted A/B-specific repair
`a83f22e4d9fe803e822d5f28a86c086b2b3f0ea278ac34c8849757d0d8065d04`.
It preserves keyboard capture and CDP keyDown/keyUp dispatch, records geometry
and occlusion only for lines inside a specific ghost-ancestor view zone, and saves a bounded graceful timeout result
when a frame-drain Runtime.evaluate call times out. Such a timeout still leaves
coverage incomplete; it is not an accepted measurement.

The analyzer now consumes that line-level shape. Multiline control success
requires contiguous focused visible frames and both continuation sentinels in
lines with finite positive rectangles, explicit visibility, viewport inclusion,
and no occlusion. Raw selector text alone cannot satisfy it. The A/B analyzer
also requires the delayed control's start, cancellation, resolution, and
observed nonpublication. These changes repair integration and admission checks.

CPU validation passed ten commands: eight syntax checks and two test programs.
The original 21 A/B assertions pass. Added tests exercise the actual injected
collector against synthetic DOM geometry; reject missing metadata, hidden or
unfocused frames, zero-width and occluded lines; preserve captured input IDs;
and exercise real observer control flow with a fake WebSocket through keyboard
dispatch, retained evidence, Runtime.evaluate timeout, and socket close.
No network socket is opened by these tests. `cpu-validation.json` retains output.

Root may use this command after admitting the owned gateway, SSH forward,
renderer identity, editor process guard, and exclusive runtime window. The
uppercase paths below are root-supplied absolute paths. The run root must be
fresh and its parent must already exist. Keep the existing root VSIX and
binding files unchanged.

```sh
xvfb-run -a --server-args='-screen 0 1280x800x24' node /CAPSULE/run_remote_editor.mjs \
  --code /usr/bin/code \
  --vsix /ROOT-IMMUTABLE-INPUTS/candidate.vsix \
  --vsix-sha256 b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1 \
  --binding /ROOT-IMMUTABLE-INPUTS/binding.json \
  --binding-sha256 26cff670485b2384758af0c478e78aa040e5f1f36d10e3c753257011dd55ce94 \
  --instance-id 223bdcf2-c958-4477-93e2-72a80c769152 \
  --renderer-source /usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js \
  --renderer-sha256 ROOT_VERIFIED_NOTEBOOK_RENDERER_SHA256 \
  --run-root /EXISTING-PARENT/fresh-auto-1500-a \
  --debounce-ms 1500 --debug-port 19403 --timeout-ms 180000

node /CAPSULE/analyze_auto.mjs /EXISTING-PARENT/fresh-auto-1500-a
```

Use a separate fresh run root for 350 ms. Preserve the existing root 300-second
guard and counterbalance runs; one pair cannot establish a default change or
representative p95. The launcher has not been run by this worker.

Admission remains pending on the first repaired actual editor capture. Check
line 2/3 rectangles, focus, occlusion, and screenshots together. In particular,
ordinary `.view-lines` and the shared `.view-zones` root never classify as
ghost lines. The same isolated editor exposes at most two diagnostic sibling
view-zone roots at most every 250 ms, twelve lines per root and 512 characters per line, with
geometry and zone IDs. These rows are explicitly excluded from classification.
Root must inspect that diagnostic evidence and the screenshot to establish
actual ownership; missing line 2/3 mapping keeps metrics blocked.
Verify keyboard input IDs, trusted captured events, actual document versions,
profile/config/VSIX/binding identities, dropped frames/inputs, screenshot
coverage, and complete observation windows. A missed cancellation window stays
pending. A timeout or failed control cannot prove ghost absence.

Gateway dispatch means HTTP bytes written, not native task admission or release.
Exact request-to-item attribution still needs root ledger/output correlation.
VS Code Automatic requests may bypass the explicit debounce timer, so early
activity does not establish causality. Root retains native burden/cancellation
review, process release, and final admission. No final data was accessed.
