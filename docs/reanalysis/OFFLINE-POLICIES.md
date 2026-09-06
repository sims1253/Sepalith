# Offline policy specification

These policies are retrospective and fixed before inspecting their row-level
wins and losses in this reanalysis. Historical aggregate scores were already
known. No threshold sweep or test-set tuning is authorized by this specification.

B8b selector: use B4 unless the visible user edit changes whitespace only, the
B4 output changes the current region's lexical tokens, and B8b preserves those
tokens. Strings and comments are indivisible tokens. This requires both saved
outputs and the prompt; it cannot use target text, validator scores, family labels
or future outcomes. It costs two generated outputs. Report exact oracle union
separately. Group packages by SHA256(package): first byte < 102 is development,
otherwise evaluation. Do not change the rule after reading evaluation outcomes.

LOC1 hybrid: equal-weight reciprocal rank fusion with constant 60, Muninn-small
and historical BM25 full rankings, ties by original candidate index. Also report
Muninn+BM25 as a separately named sensitivity analysis, not a selected winner.
Use the same candidate corpus and return at most k unique candidates (k=5,10)
as each standalone baseline. Split by SHA256(repo), first byte < 102 for
development. No parameter fitting. Report hit@k, mean per-query multi-gold
recall@k and all-gold hit@k. This matches retrieval output count, not compute or
prompt token count; combining systems has extra cost. Cached embeddings may be
multiplied but no model can be loaded or run. Require metric parity with saved
results before reporting reconstructed rankings. Missing/mismatched caches must
fail explicitly, without inference or selective row deletion.
