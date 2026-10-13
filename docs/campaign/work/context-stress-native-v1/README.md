# RUN-05 native stress wrapper

This wrapper consumes the immutable `context-stress-fixture-v1` fixture and
delegates native replay to the robust
`work/lead/theta0-cuda-context-b/replay-context-trace.ts` exports:
`runNative`, `profileFor`, `runContextProfileChecks`, and its source-pin
verification. The robust client retains `/props` validation, profile deadlines,
full raw token responses, token IDs, stop fields, metrics, and denominator
separation. This wrapper never launches a server.

CPU-only validation:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-context-stress.ts --check
```

Expected profile denominators are explicit in `manifest.json` and the trace
output: at 2,048, only `near-2k` fits and the other two are
`context_budget` overflow controls; at 4,096, `near-2k` and `near-4k` fit and
`near-8k` overflows; at 8,192, all three fit. Use `--diagnostic` with the 2K
profile when the diagnostic deadline is required.

Root replay commands after endpoint launch:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-context-stress.ts \
  --fixture /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/context-stress-fixture-v1/context-stress-fixture.json \
  --server http://127.0.0.1:PORT --context 2048 --diagnostic --out run05-stress-2048.json
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-context-stress.ts \
  --fixture /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/context-stress-fixture-v1/context-stress-fixture.json \
  --server http://127.0.0.1:PORT --context 4096 --out run05-stress-4096.json
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-context-stress.ts \
  --fixture /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/context-stress-fixture-v1/context-stress-fixture.json \
  --server http://127.0.0.1:PORT --context 8192 --out run05-stress-8192.json
```

