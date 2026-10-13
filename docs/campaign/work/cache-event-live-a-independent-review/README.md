# RUN-04 cache-event-live-a independent review

This packet independently reviews the completed `RUN-04-cache-event-live-a` event capture with CPU-only JSON/JSONL/hash/geometry checks. The copied remote evidence contains only the explicitly selected run and supervision JSON, JSONL, logs, PNGs, and three synthetic control buffers; no user-data, authentication material, profile cache, or private key was copied.

Run the review with:

```text
python3 /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/cache-event-live-a-independent-review/review_cache_event_live.py /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
```

The review passes all seven actual event rows. Each source snapshot is bound to its recorded URI, version, content hash, replacement range, UTF-16 cursor, source-backed prefix/region/suffix, prompt hash/size, request-key hash, and complete instrumented provider trace. The run has one trace process, 107 trace rows, zero dropped records, `scopeContext=true` as the private probe, and diagnostics remain unsupported.

The renderer evidence passes all three deterministic controls. The visible control has 94 matching frames with a first rendered interval of 220–238 ms. The multiline control has 94 matching frames with verified first/second/third view-zone text, focus, in-viewport geometry, and no occlusion; its first interval is 219–240 ms. The stale control has no post-invalidation ghost frame and records the delayed provider start/cancel/resolve sequence. The observer retained 768 frames with zero drops and stopped after a CDP `Runtime.evaluate` timeout during stop; the run still has a graceful-stop record. The four PNGs are retained and hashed separately. PNGs do not prove model ghost causality.

There are twelve direct transport joins (six event cases, each with `/tokenize` and `/completion`) using exact gateway request ID, instance ID, route, body hash, byte count, HTTP 200, and release evidence. The unchanged repeat has no transport and is covered by the separately hashed root exact cache-hit link to the prior baseline EOS completion. The retained analyzer remains unchanged with `full_request_binding=false`; this packet reports the root link alongside it rather than rewriting that result.

The gateway ledger has 43 backend dispatch/release pairs: 6 `/tokenize`, 6 `/completion`, and 31 automatic `/props` calls. No request is cancelled, no backend is killed, and all responses complete. The gateway explicitly marks native task acceptance and release as unproven for every backend row. The native log separately records six processed/released tasks, prompt-cache states and timings, all `truncated=0`, and clean slot idleness; no native task ID is assigned to a gateway request by ordering or clock subtraction. Native and gateway timing values therefore remain separate ledgers.

`review-report.json` contains the full bounded row, transport, cache, gateway, renderer, native, timing, and input-hash data. `artifact-hash-manifest.json` records all 32 remote source/destination hashes with every match.
