# DAT-10 provider infrastructure diagnostic

This packet contains three bounded full-renderer cases. It extracted only the approved TRAIN rows: Semantic9535 shard 0027 row `b15446c8d06df1c77e9caac6`, Semantic9535 shard 0028 row `b777001bc9e4d6462cb2a8c1`, and Noop4100 shard 0036 row `e677ee6a8da38436f4bdb6b6`. Extraction stopped at source lines 372, 372, and 9. No other input rows were retained or rendered.

Each case used a fresh copied `render_shard.ts`, its exact TypeScript module closure, the original tokenizer bridge and tokenizer JSON, and a copied `namespace_runtime.ts` that records helper phase, message, exit code, signal, killed flag, stdout, and stderr. Commands were CPU-only, pinned to cores 8 or 10, nice/ionice priority, and capped at 120 seconds. No campaign output root was touched.

## Results

| case | renderer result | output | diagnostic |
| --- | --- | --- | --- |
| Semantic9535 / 0027 | exit 0; one supported row | 1 row | no infrastructure error |
| Semantic9535 / 0028 | exit 0; one supported row | 1 row | no infrastructure error |
| Noop4100 / 0036 | exit 1 | zero rows | source helper exit 1 |

The Noop case reproduces the failure in about 0.93 seconds. Its rich log begins:

```text
phase: source_helper
message: Error in parse(text = text, keep.source = TRUE, encoding = "UTF-8"):
<text>:219:30: unexpected '>'
218:   if (lower.tail) {
219:     if (length(q) > distMode >
Execution halted
code: 1
signal: null
killed: false
```

The namespace helper completed first. The source helper then calls base R `parse()` on the row's pre-edit snapshot and exits nonzero on the malformed expression. The frozen Noop packet explicitly keeps parse failures fatal; its `no_following_function` change handles only fully parsed source. This is a source-snapshot syntax failure, not a helper timeout, missing namespace, tokenizer failure, or resource race. The two Semantic rows pass through the same full renderer and bridge in isolation.

The existing Noop output root remains untouched. Its 3,502 valid rows across 26 shards remain the usable artifact. The failed shard 0036 must not be treated as complete, and the zero-byte diagnostic output is not a valid replacement.

## Minimal fix and decision boundary

The minimal safe fix for observability is the copied runtime change in this packet: preserve the child error fields and log them before returning the existing unresolved result. The controller should also replace its bare `assert all(c.returncode == 0 for c in children)` with an explicit exception containing each lane exit code; otherwise simultaneous child exits produce an empty `error` field.

The minimal behavior change that would let this exact Noop row continue is a policy change: catch R parse errors in `source_imports.R`, emit a structured parse-error status, have `namespace_runtime.ts` return a non-infrastructure policy reason, and let `render_shard.ts` write a hold row. That changes hold accounting and must be bound into a new source manifest. It is not applied here. Under the frozen fatal-parse policy, quarantine or repair this exact malformed source snapshot before any rerun; do not blind-rerun the full 4,100-row stream.

Because one of the three cases fails, no full failed Semantic shard 0027 rich-logging run was prepared or launched. The copied closures and exact case commands remain available for root review.

