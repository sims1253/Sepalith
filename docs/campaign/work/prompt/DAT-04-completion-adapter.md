# DAT-04B completion and case-source adapter

This packet owns a lossless, fail-closed adapter for completion rows. It is
separate from the worker-data structured scenario adapter. The implementation
is in
`/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training/campaign_admission_completion.py`.

The public API is:

```python
convert_completion(raw, source_ref) -> {
    "status": "converted" | "excluded",
    "context": PromptContext.to_dict() | None,
    "target_body": list[str],
    "operation": "no_op" | "replace" | "delete" | None,
    "provenance": dict,
    "reason": str,                 # only on exclusion
}

source_ref_from_audit(audit_row, document_path, uri, document_version) -> dict
```

Converted results also expose `selection_source` at the top level, copied
from provenance. It contains `availability`, lineage, content SHA256,
selected start/end lines, document version and the durable source JSONL path,
line and hashes. This lets an admission caller verify the prediction-time
window without treating it as prompt evidence or target text.

`source_ref` verifies an admitted `train_group` or `dev_group` row by JSONL
file, one-based line, raw-line SHA256, full source-file SHA256, split and row
identity. It also requires a URI and nonnegative document version. The normal
path accepts a UTF-8 `pre_edit_observed` document path or text and records its
full content hash. Mixed EOL and lone CR are rejected. LF is the wire format;
CRLF is retained as the document EOL identity and no global trimming occurs.
The selected range stores code-point and UTF-16 geometry and the exact source
content hash.

The accepted bounded source-derived path is explicit and has no observed
editor claim. `document_role=source_derived_simulated_pre_edit` may use only
these constructors:

* `finish_block_v5_prefix`: `raw.prefix` is the observed source string; its
  final logical line is the current line (possibly an empty EOF line) and
  preceding lines are the prefix. The target and `full_prompt` are never read
  for document construction.
* `scenario_lines_lf`: the document is the LF join of raw `prefix`,
  `region_old` and `suffix`, with no hidden target inserted. The constructor
  checks that the complete constructed window hash and line sequence match
  the document used for selection.

The provenance labels this lineage `source_derived_simulated_pre_edit`, uses
`selection_source.availability=source_builder_window`, and includes the
constructed source text, synthetic URI/version, content SHA256, selected line
span and durable JSONL identity. An observed snapshot instead uses
`availability=full_snapshot`. The source constructor is replayable from
prediction-time row fields and never from a target or teacher instruction.

`extensions/vscode-sepalith/src/context_select.ts:96-145` is the deployed
selection reference. `selectPromptContext` selects the complete current line,
with a same-line replacement range and cursor code-point/UTF-16 position. The
adapter follows that geometry for both finish and scenario suffix rows:

* Earlier observed lines remain in `context.prefix`.
* For a nonempty current line, `context.region_old` is the complete line,
  including all spaces. For a finish prefix ending in LF, the physical final
  empty logical line is selected and represented canonically as
  `region_old=[]` with cursor `(-1, null, null)`.
* The cursor is inside a nonempty current line and the replacement range starts
  at character zero and ends at its UTF-16 length. The EOF empty line has a
  zero-width same-line range at its true final line.
* Scenario suffix targets are combined with the typed line. The default
  `completion_join=next_line` inserts one LF after the text before the cursor,
  then the continuation and any observed tail. `same_line` is an explicit
  opt-in. The current implementation rejects an unverified same-line tail.
  Finish targets use a different literal splice: the replacement body is
  `current_line + raw corpus_target` with no unconditional LF insertion or
  leading/trailing trimming.
* The old zero-width suffix insertion route is retained only as
  `unsupported_route_candidate`; it is never the primary training context.
  Multiline inserted text is representable in the target body. A multiline
  range still belongs to an explicit `WorkspaceEdit` caller and is outside
  this automatic-inline adapter.

`experiments/post-processing/assemble_sft_v5.py:242-281` and
`experiments/synthetic-data/cases/rules/rules_finish_block.py:273-325,530-652`
supply the finish source-builder convention. The prefix is byte-contiguous
with the corpus target: the family gate reconstructs the body from
`prefix + target + '}'`, while the outer brace is intentionally outside the
label. A signature prefix ends on `{` and a mid-body prefix can end in LF;
the latter has a real empty EOF line. The adapter preserves that line and
splices the literal target bytes without unconditional LF insertion or
leading/trailing trimming. Later blank lines, leading or trailing spaces and
terminal LFs remain unchanged. A target that is only an empty line remains an
actual newline replacement; `[]` is the canonical empty continuation. A
nonempty region with `[]` is `delete`, while a full copy or empty continuation
is `no_op`. The noncanonical `['']` label is excluded.

Target fields are separated from context. `region_new` and `corpus_target`
are source-authoritative and must agree byte-for-byte after line decoding;
`model_target` and generic `target` are quarantined alternatives and cannot
override a source value. Every selected target is serialized through the
frozen parser and parsed back, preserving exact spaces, blank lines, Unicode,
marker-like quoted text and literal EOS text. Full marker lines,
`[NO_EDIT]` body lines and other reserved framing lines are rejected. The
adapter records conflicts and ignored future fields, but no target or
`full_prompt` value enters `PromptContext`.

The adapter supplies empty history, diagnostics and retrieval because no
bounded provider is part of this packet. It does not claim editor execution,
`WorkspaceEdit` behavior, serving latency, tokenizer parity, or model quality.
Rows whose source identity, complete target, before-image geometry or
representability is ambiguous are excluded with a stable reason.

## Bounded review and next gate

`/mnt/e/sepalith/campaign-20260915/data-work/DAT-04-completion-reviewed-examples-v3.json`
contains three verified `train_group` finish rows from authored, compound
random and compound sources. Each converts structurally using the explicit
finish-prefix constructor, with current-line ranges and deterministic source
window hashes. They are source-builder simulations, not observed editor
coverage and not final registry admissions. Four previously reviewed scenario
rows remain excluded because their normalized source is not an observed
before-image (one ambiguous blank anchor and three hidden completion spans).
This bounded result does not establish absence across other roots.

The follow-on admission worker must independently verify the audit/prediction
boundary and tokenizer gate, then pass only `status == "converted"` rows to
the frozen token-row builder. It must preserve the source-derived lineage
label until an observed editor snapshot replaces it.
