# RUN-05 lower source-budget candidates

This packet replays the six existing synthetic RUN-04 long-panel events with
the exact EXEC production `selectPromptContext`/`selectPromptSource` path and
the existing PRM-03 renderer. It uses the complete source document, actual
history records, diagnostics, scope pins/outlines, cursor positions, old
regions, and replacement-range identities already captured by the input
fixture. It does not fabricate a prefix or alter document geometry.

Each event is materialized at both source budgets: 3,000 and 1,500 UTF-16
units. Required scope spans fit at both budgets. The native profile remains
4,096 context tokens, 192 output tokens, parallel 1, no server launch, and
greedy stop semantics. These are latency/quality tradeoff candidates only; the
production default remains 6,000 UTF-16 units and this packet makes no quality
claim.

Run CPU-only checks:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types build-budget-candidates.ts --check
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python check-budget-candidates.py --out token-report.json
```

The local tokenizer uses the pinned MiniCPM5 tokenizer with
`add_special_tokens=False`, `split_special_tokens=True`; manual BOS 0 is
counted once outside tokenization. All 12 candidate prompts fit 4,096 plus
192 output tokens. Token counts are in `token-report.json`.

After root selects an already-running endpoint, replay one budget through the
existing long-panel `runNative` client:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-budget-candidates.ts \
  --fixture long-context-budget-candidates.json --server http://127.0.0.1:PORT \
  --budget 3000 --out run05-long-budget3000.json
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-budget-candidates.ts \
  --fixture long-context-budget-candidates.json --server http://127.0.0.1:PORT \
  --budget 1500 --out run05-long-budget1500.json
```

The wrapper only filters the selected six events and delegates transport,
tokenization, completion, stop parsing, identity checks, and timing to
`long-transition-panel.ts`'s existing `runNative`. It never launches a server.

