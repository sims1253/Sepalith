# Semantic namespace provider integration v1

This isolated production snapshot adds prediction-time package-root discovery,
bounded NAMESPACE parsing, current-buffer import discovery, exact evidence
injection, and freshness checks to the reviewed full-document policy. It does
not modify the active extension or admit training rows.

`namespace_evidence.R` reads each NAMESPACE file once into a bounded byte
snapshot, hashes those bytes, and parses text derived from that same snapshot.
Uniform CRLF is recorded and converted to LF for R parsing; mixed or lone CR,
unstable files, malformed directives, conditional directives, wildcard-only
origins, and multiple explicit origins fail closed. No package is loaded and no
package function is called.

`source_imports.R` receives the exact unsaved pre-edit buffer as base64 plus its
SHA-256. It identifies the first top-level function after the blank roxygen
anchor and uses `codetools::findGlobals` on constructed function closures. The
function bodies are inspected but never called. It follows unique same-file
helper functions and intersects observed references with explicit NAMESPACE
imports. The target/gold documentation is not an input.

`namespace_runtime.ts` locates the nearest package root containing both
DESCRIPTION and NAMESPACE without escaping the workspace. Each request uses
bounded helper processes and owned temporary files. `full_document_policy.ts`
injects typed required evidence before tokenization, checks the document after
each async boundary, and uses one tokenizer call when the full document fits.
The copied `extension.ts` passes the unsaved buffer, document version, workspace
root, and shipped helper paths through this path.

The final target-free replay resolved all 133 prior provider candidates. It
matched the reviewed dependency set exactly in all 133 rows, selected 132 full
documents and one complete-span context, and produced 279 namespace evidence
records from 123 NAMESPACE files. After targets were joined, all 133 remained
new against the exact 15,006-row corpus plus the prior 616 candidates. Complete
targets were re-encoded and reapplied to the original source; no target was
truncated.

The reviewed candidate output is
`/mnt/e/sepalith/campaign-20260915/data-work/Semantic-provider-integration-v1/final-01`.
The rendered target-free stream is pinned separately as
`/mnt/e/sepalith/campaign-20260915/data-work/Semantic-provider-integration-v1-rendered-final.jsonl`.
The seven rows excluded by the preceding namespace audit remain excluded and
are not part of this 133-row denominator. Root review and production wiring are
required before corpus admission.
