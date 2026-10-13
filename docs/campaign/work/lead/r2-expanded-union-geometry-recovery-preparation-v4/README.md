# Expanded-union geometry recovery v4

This review-only packet corrects the cursor and source-binding losses found in
v3. `context.cursor` is relative to `region_old`: `region_line_index` selects
that region line, while `code_point_column` and `utf16_column` describe its
cursor. The producer maps the region line to the absolute line supplied by the
replacement range or `selection_geometry.context_range`; it never uses the
length of a bounded prompt prefix as the absolute line. The first original row
therefore produces `cursor={line:42,character:25}` for the nonzero range
`42:0` through `42:43`.

`selection_geometry.availability=full_snapshot` describes source availability.
It does not prove that the captured prefix and suffix are a complete document.
V4 records `context_window_complete` and verifies the recomposed preedit hash
only when span evidence covers the complete source line range. Truncated
windows retain their source hash and typed provenance without being rejected
for an expected window-hash difference.

Source locators are typed and preserved. Absolute paths are `file` locators;
`builder:synthetic/...` paths are `synthetic` locators and must carry the
simulation marker. Nested file locators join the package plus URI-relative
path, while synthetic locators retain their exact builder path. Missing or
untyped locators remain unresolved; no absolute path or full-buffer evidence is
invented. The only cursor sentinel is the exact triple
`region_line_index=-1`, `code_point_column=null`, `utf16_column=null`. It is
accepted only for a zero-width replacement and maps to that anchored range
start. Missing or partly populated cursor fields remain unresolved.

`validate_geometry_consumer_v4.py` checks the typed source-identity join,
source-kind marker, producer anchor evidence, geometry digest, and cursor
inside the replacement range. It permits a cursor that differs from range
start when the replacement is nonzero. The frozen audit-v2 packet and the v3
packet remain unchanged; GEOM-AUDIT-002's old `cursor == range.start` rule is
superseded by the evidence-join rule here. Relative-path and explicit
geometry-provenance findings remain admission gates.

The tests use exact original synthetic and truncated full-source rows, an
exact semantic truncated sentinel, the original nonzero cursor, and small
in-memory controls. They cover typed locator preservation, source availability
versus window completeness, absolute cursor anchoring, Unicode UTF-16
conversion, sentinel boundaries, cursor/range bounds, producer column
mismatch, consumer evidence joins, and unresolved untyped locators. No cohort
scan, model, target, label, CUDA, training launch, or frozen-input mutation is
performed.

Root may run the two cohort commands in `commands.json` only after reviewing
the source and pins. Their outputs are fresh review-only ledgers; unresolved
rows remain explicit and are never silently dropped. Geometry output must be
joined to exact union provenance rows before any admission decision.
