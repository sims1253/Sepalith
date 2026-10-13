# RUN-05 context stress fixture v1

This packet is a CPU-only preparation artifact for native context-boundary
replays. `stress-context-fixture.ts` reads the complete synthetic RUN-04
rainfall source fixture, appends 120 complete synthetic `stress_window_*` R
functions, and uses the pinned EXEC `context_select.ts` and PRM-03
`campaign_protocol.ts` renderer to build three source-backed cases. The full
document, source line selections, complete-buffer identity, prompt text, and
prompt hashes are in `context-stress-fixture.json`.

The selector budgets are 5,750, 11,000, and 23,000 UTF-16 units. Only the first
case is near the 2K profile and it has a 1,830-token prompt including manual BOS
plus 192 output tokens (2,022 total). The 4K case is 3,384 prompt tokens with
output (3,576 total), and the 8K case is 6,963 prompt tokens with output
(7,155 total). The 4K case explicitly overflows the 2K profile; the 8K case
explicitly overflows both 2K and 4K. Enlarged source budgets are exploratory
stress controls. The production selector default remains 6,000 UTF-16 units,
and no quality profile is proposed by this fixture.

Run the deterministic renderer/geometry checks and local tokenizer checks from
this directory:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types stress-context-fixture.ts --check
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python check-stress-fixture.py --out token-report.json
```

Root may replay against an already-running endpoint with the no-launch CLI:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-stress-fixture.ts \
  --fixture context-stress-fixture.json --server http://127.0.0.1:PORT \
  --case near-2k --context 2048 --out run05-near-2k-2048.json
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-stress-fixture.ts \
  --fixture context-stress-fixture.json --server http://127.0.0.1:PORT \
  --case near-4k --context 4096 --out run05-near-4k-4096.json
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-stress-fixture.ts \
  --fixture context-stress-fixture.json --server http://127.0.0.1:PORT \
  --case near-8k --context 8192 --out run05-near-8k-8192.json
```

The replay CLI sends pinned no-special `/tokenize` requests, inserts manual BOS
ID 0 exactly once, and sends greedy `/completion` only when prompt plus 192
tokens fits the requested profile. It reports an explicit `contextOverflowRejected`
control without dispatching completion when a case is selected against 2K or 4K.
It never launches a server and does not read model weights.

