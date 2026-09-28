# Semantic serving geometry parity v1

This packet audits the actual VS Code serving path against semantic materializer v2 and contains one isolated TypeScript adapter patch. It does not modify the production extension or the frozen materializer.

## Current path

`extension.ts:843` calls `selectPromptContext`. `context_select.ts:109-140` selects the active full editor line as the replacement range and a bounded source view as prompt prefix/suffix. `history_provider.ts:610-637` correctly binds the range to the full unsaved document SHA, version, URI, and global line coordinates. `campaign_protocol.ts:497-510` checks that identity immediately before producing an inline or workspace edit.

A blank editor line therefore produces the same zero-width operation as semantic roxygen insertion. If the entire pre-edit source is retained, materializer v2 full-source expansion and current serving match exactly. Actual TRAIN row `227626e3a7a234b808a096b1` passed all prefix, suffix, range, EOL, and hash comparisons at global line 290 with pre-edit SHA `8bb73f454eb7931b1e838bf10e4b2208eb3d9b8914fe8a67b24b7bd22619d559`.

A bounded prompt deliberately does not pretend its cropped view is the active document. It keeps the full document hash and global range. Materializer v2 candidate mode instead requires the cropped prompt view to equal the replacement identity and uses coordinates local to that view. The same actual row has full source SHA `75b80a9d...`, 49,523 bytes, and target line 290, while its candidate view has SHA `966066a9...`, 1,682 bytes, and line 30. The editor stale gate would reject that candidate context against the real document. Candidate mode is blocked from training admission until the contract is corrected end to end.

## Typed helper evidence adapter

Current `context_select.ts:129` always emits `selected_references: []`, although the reviewed freshness and budget logic already exists in `campaign_selection.ts:420-523`. The isolated patch adds `SemanticContextOptions`:

- `requiredSourceSpan`: one contiguous global source interval containing the edit line and target function;
- `requiredEvidence`: exact source-backed helper excerpts in source order;
- `currentEvidenceSources`: captured identities, versions, and hashes for freshness checks;
- `evidenceBudgetUtf16Units`: an explicit budget that fails if required evidence does not fit.

The patched pure selector maps fresh included helpers to existing typed `EvidenceRecord` values. It rejects stale, incomplete, over-budget evidence and a required source span that does not contain the edit. Existing callers are unchanged when `semantic` is absent.

The integration fixture runs document DTO -> source/evidence selector -> `PromptContext` validation -> renderer -> output parser -> application planner -> full-document range application. LF and CRLF apply exactly. It also verifies stale document identity and stale helper identity failures. The notebook independently replayed the 25-check fixture on two assigned CPU cores.

## Admission boundary

The adapter proves that the existing DTO and renderer can represent correct semantic contexts. Production still lacks a provider that computes reviewed semantic source spans and helper evidence from the current unsaved buffer. `extension.ts` does not pass `semantic` options. The next implementation must add that provider and use the same selection contract in materialization; prompt equality by itself remains insufficient.

No model, generated R execution, GPU, cloud job, sealed development data, final data, production source edit, or corpus materialization was used.
