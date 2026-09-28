# Final CPT corpus launch inputs

This packet prepares the downstream data path after the terminal all-source union
builder. It does not read the moving union payload, run the 16K rechunker, build
the cache, admit data, or launch training.

The current root guard is the v2 union builder writing
`/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1`. Its terminal
metadata must pass `verify_terminal_union.py` before `cpt_train.jsonl` is
opened. The gate requires all 8,092 main groups, alias capture of all 8,092
main groups, and the builder's explicit TRAIN, heldout, no-truncation, and no
retokenization flags.

After the gate, the reviewed closure rechunker consumes the union's generated
`input-manifest.json` and writes one lossless context-16,384 stream. It
reassembles every complete 2,048-token document, retains all source payload
and carry metadata, and emits `document-provenance.jsonl`. Its result manifest
is the hash source for the next step.

`make_one_pass_schedule.py` reads the verified 16K row stream and creates a
seeded permutation with seed 3407. If the stream has `R` unique rows, it emits
`(-R) % 16` named replays at the tail. Therefore draws are `R + replay_count`
and updates are `ceil(R / 16)`. The first `R` draws contain every row exactly
once, so no prefix is repeated before complete unique coverage. The generated
schedule binds the exact row-stream SHA256 and records stage-local cursor zero
with global optimizer-step offset 66.

The reviewed full-corpus streaming cache then validates every row, document
boundary, carry token, BOS/EOS/label contract, and schedule hash before
materializing bounded-memory int32 streams and a SQLite identity index on E.
Its manifest must bind all row, document, package, token, draw, replay, and
cache-file counts. The stage-transition binder should consume those bindings
with `destination_sampler=fresh_cursor_zero`; the first destination update is
stage step zero and global step 66, while later dataset-managed resumes use the
stage-local draw cursor.

All downstream commands and source hashes are in `commands.json`. The two
small synthetic tests verify that the terminal gate does not open payload bytes
and that the schedule/cache path has minimal tail replay arithmetic.
