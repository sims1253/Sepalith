# Source-walk semantic materialization v1

This review-only template converts root-accepted semantic-v6 roxygen closures
into the pinned PRM03 editing-SFT protocol. It processes one provenance shard
at a time so source, packet, and semantic identities remain exact and memory is
bounded.

Every run requires exact hashes for the semantic child manifest and ledger,
provenance ledger, candidate packet, and a root-created accepted-ID file. The
semantic manifest must bind the reviewed v6 analyzer plus both R helpers. Each
accepted ID must join the independently replayed TRAIN provenance and the
candidate packet's row, group, package, source line, raw-line hash, normalized
source path/hash, and `train_group` split.

For every row the materializer reopens and hashes the normalized source, reruns
the v6 semantic decision, and requires the recorded closure to match. It removes
the one exact target-roxygen occurrence from model-visible context. It retains
the complete target function and every recursively required helper function,
in original source order, with an exact hash for every selected span. Internal
comments in selected functions remain byte-identical. Existing protocol
history, diagnostics, retrieval, cursor, and replacement-range evidence remain
bound through the source packet's `PromptContext`.

The full target is serialized unchanged. The pinned native tokenizer and
campaign protocol build the training row, and the independently reviewed strict
validator checks BOS 0, EOS 1, token ranges, terminal tokens, counts, and prompt
boundary geometry. Context and target length never filter a row: all sequence
buckets are reported, targets over 1,024 tokens remain complete, and every
failure is written to `holds.jsonl` with the accepted denominator conserved.

No full semantic replay, materialization, or training admission is part of this
packet. Root must first freeze accepted semantic-v6 ledgers and fill the hashes
in `commands.json`, then review each shard output and an exact global merge.
