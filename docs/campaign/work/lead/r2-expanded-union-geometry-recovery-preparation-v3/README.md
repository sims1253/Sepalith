# Expanded-union geometry recovery v3

This review-only packet corrects the cursor loss in v2. `context.cursor` is a
relative position: `region_line_index` addresses `region_old`, while
`code_point_column` and `utf16_column` describe the same cursor on that line.
For a full snapshot, v3 maps the line to `len(prefix) + region_line_index`,
checks the code-point/UTF-16 conversion, checks the absolute cursor against the
full buffer and replacement range, and preserves the absolute UTF-16 cursor in
`source_cursor_geometry`. The actual first original row therefore produces
`cursor={line:42,character:25}` for the nonzero range `42:0` through `42:43`.

The only cursor sentinel is the exact triple
`region_line_index=-1`, `code_point_column=null`, `utf16_column=null`. It is
accepted only when the replacement range is zero-width and at the captured
prefix boundary. Missing or partly populated cursor fields remain unresolved;
the replacement range is never used to guess an explicit cursor.

`validate_geometry_consumer_v3.py` is the corresponding consumer guard. It
requires an absolute source path and source-identity join, checks the cursor
inside the range and against the producer evidence, and permits a cursor that
differs from range start when the replacement is nonzero. The existing audit
v2 packet is frozen; its old GEOM-AUDIT-002 recommendation to require
`cursor == replacement_range.start` is superseded by this packet. The relative
path and explicit geometry-provenance binding findings remain open for the
union admission review.

The tests use one exact original row, one exact semantic sentinel row, and
small in-memory controls. They cover the nonzero original range, Unicode
code-point to UTF-16 conversion, valid zero-width sentinel, nonzero sentinel
rejection, cursor/range bounds, producer column mismatch, consumer evidence
joins, and unchanged audit-v1 geometry compatibility. No cohort scan, model,
target, label, CUDA, training launch, or frozen-input mutation is performed.

Root may run the two commands in `commands.json` only after reviewing the
source and pins. Their outputs are fresh review-only ledgers; unresolved rows
remain explicit and are never silently dropped. Geometry output must be joined
to the exact union provenance rows before any admission decision.
