# DAT-10 roxygen full-context recovery

This packet replays the exact 5,942 rows held by the prior roxygen context
review for free-reference evidence. It is a source-only recovery preparation
packet. It does not admit rows to TRAIN, rewrite source or target payloads, or
replace the prior review packet.

The replay calls the current admission classifier for the held rows. The
classifier includes nearest-function lexical scope and R `$`/`@` member-label
handling. The historical `full-v3` result remains pinned as a comparison
input. The local line-span speed override only replaces an equivalent
quadratic helper at runtime; it does not change source parsing or prior
artifacts.

Rows are recoverable by scope correction when the current classifier no longer
finds a free reference. A full-file recovery is separately marked only if the
full source is within 131,072 tokenizer tokens and contains a matching
top-level definition. Ordinary R or imported/package references are marked for
root API review. Residual non-call free references remain held for source or
semantic evidence. Full-file fallback never silently crops a source or target.

The frozen shard outputs cover the complete held denominator as the disjoint
union of `materialization-shard-0000` and `materialization-shard-0001`.
`materialization-v1` is retained as superseded preliminary output because it
used the pre-member-label replay result.
