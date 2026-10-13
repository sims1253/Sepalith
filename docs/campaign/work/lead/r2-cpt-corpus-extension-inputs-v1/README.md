# CPT corpus extension inputs

This packet prepares a reviewed, append-only extension to the frozen 16,384
token CPT corpus. It does not change the accepted corpus, its cache, its
schedule, or the checkpoint. The recovery source has 1,998 complete documents
and is reblocked with the pinned lossless rechunker. The producer omitted two
source-range aliases required by that rechunker; `prepare_extension.py`
creates a derived 2,048-token stream by adding only those aliases.

The combined row stream is the byte concatenation of the existing stream and
the reblocked recovery stream. The fresh cache is built from that stream with
the reviewed `cpt_streaming_cache.py` source. The existing cache remains
unchanged. The combined schedule keeps the old schedule's 183,084 unique row
IDs in their exact order, removes only the old four-row alignment tail, appends
all 2,234 recovery rows once, and adds the ten rows required to align the new
total to an effective batch of 16. The preparation binding records the
verified step-82 cursor 256 and global optimizer offset 66. The cache itself
has no checkpoint identity. Before a production transition, root must bind the
exact later full-state checkpoint that supplied the runtime cursor and offset,
and verify that its cursor is before the old alignment tail. The trainer can
then resume at that supplied dataset position while retaining complete
documents in the new cache.

The prefix verifier expects `coverage.unique_rows`, `coverage.draws`, and
`coverage.updates`. `prepare_extension.py finalize-cache-compatibility` adds
these equivalent fields to the schedule and updates the fresh cache manifest's
schedule pin without rewriting cache payloads. This is needed when the cache
builder was started from the pre-compatibility schedule.

The schedule builder requires all of these values from an explicit checkpoint
binding: checkpoint step and ID, source schedule SHA256, source draw cursor,
global optimizer-step offset, and a full checkpoint manifest. It rejects a
cursor after the old alignment tail because removing an already-consumed
alignment draw would change history. The checked preparation input is step 82,
cursor 256, offset 66, with the root-verified full-state checkpoint; that input
is historical evidence until root binds the later runtime checkpoint.

The output remains a candidate for root data and stage admission. The packet
does not decide whether the recovery rows are admitted and does not launch
training. Root must independently verify prompt/target or source admission,
heldout and duplicate registries, checkpoint state, and the trainer's
consumed-prefix contract before using the fresh cache.
