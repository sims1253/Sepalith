# Abortable expanded-provider helper preparation v2

This isolated source candidate addresses the independent review of v1. It does not edit or activate the current E750 selector, b4 profile, extension, VSIX, editor, or service.

The production namespace path is deliberately narrow. `namespace_runtime.ts` invokes only the two reviewed R parse helpers whose exact SHA-256 values are compiled into the source. Both helpers are statically checked for process-spawning primitives and execute under `/usr/bin/Rscript --vanilla` with an empty user profile/environment. An alternate helper path cannot pass the compiled hash gate. This packet makes no general descendant-process cleanup claim.

`owned_helper.ts` captures the spawned leader PID and Linux start tick, shares one monotonic request budget across both serial helpers, bounds output, and uses a memoized cleanup promise. Abort, timeout, or output overflow sends TERM only to the still-matching leader, waits, then sends KILL only to that same owner. All result paths await cleanup. A separate hard-settlement timer is cleared after normal exit, and timeout/abort tests prove the owned PID is absent while an unrelated process remains alive. A descendant-spawning fixture is rejected by the hash gate before spawn.

`produceNamespacePathIdentity` hashes one stable, bounded byte snapshot and emits `realpath@sha256:<digest>`. The coordinator requires a fresh identity before selection and after asynchronous provider/tokenizer work, so a NAMESPACE mutation cannot reuse cached evidence or emit a suggestion. Document version and content checks remain active.

The composed test runs the adapter, coordinator, provider selector, and pinned tokenizer bridge for two authorized TRAIN fixtures. It preserves exact provider mode, replacement range, document identity, prompt token count, and prompt SHA. Generated R and package code are never executed.

This is a review candidate, not an integrated extension or release admission. Integration must wire the identity producer into the real extension request lifecycle and repeat cancellation, invalidation, TRAIN parity, editor latency, and packaging tests in a fresh production snapshot.
