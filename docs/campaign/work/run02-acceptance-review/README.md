# RUN-02 serving contract acceptance review

This is a bounded CPU-only audit of the RUN-02 brief. It reads the six named
receipts, rehashes the explicit RUN-01 source snapshot closure and its small
named evidence artifacts, and runs the pure/mock checks from the accepted EXEC
source. It does not load a model, start a server, use a GPU, connect over SSH,
or enumerate the workspace.

The exact brief requires three things: serving and training fixture IDs must
agree; stale or out-of-date responses must be rejected; and cache reuse must
preserve correctness without reusing old-checkpoint state. The preparation
receipt is deliberately partial because the named receipts do not publish a
single serving/training fixture-ID manifest or a live old-checkpoint swap.

The brief's research source was also checked: `docs/research/72h-prompt-training-contract.md`
is 13,860 bytes with SHA-256
`9e13ce23792e98d638597f93a1368d2b95a2563fc55166b49f5c9ea6472e5241`.

## Source and evidence closure

`verify_run02_acceptance.py` rehashed all 17 files listed by
`RUN-01-acceptance-source-snapshot.json` under
`/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith`.
Every current file and every copied snapshot file matched the receipt's byte
count and SHA-256. The source snapshot identity is
`41a509aaa6b2205add579e12db9afbdad8d4c40155baf798d4f263009989ebfc`, with
source Git head `a7345e35219ceb624957115c39b869101026a817`.

The script also rehashed the five RUN-01 identity artifacts, six RUN-04 full
acceptance artifacts, the PRM-05 validation log, and the RL-01 theta0 gate.
All named artifact hashes matched. It did not read the PRM-04 model or any
model weights. The machine-readable results are in
`source-closure.json` and `artifact-closure.json`.

## Criterion map

| RUN-02 criterion | Result | Evidence and limit |
| --- | --- | --- |
| Serving and training fixture IDs agree | **Partial** | `campaign-protocol` (14 assertions), PRM-05 (38), context (46), history (51), and RL-01 (20 native prompt identities; 6 source-authoritative TRAIN rows) all pass. The receipts do not contain one shared ID-to-prompt-SHA manifest for the serving fixtures and training rows, so full ID equality is unproved. |
| Stale or out-of-date responses are rejected | **Mechanical pass; live stale-ghost pending** | `campaign_protocol.ts:478-510` binds URI/version/content SHA before application; `next_edit.ts:23-59` requires the current request lease; `extension.ts:833-840` and `999-1004` recheck lease, runtime generation, restart state, cancellation, and document identity after awaited work; `history_provider.ts:478-507` rejects stale/desynchronized replay. Mock lifecycle and acceptance identity tests pass, and RUN-04 has one real full inline acceptance. A real stale response from a live sidecar and partial/stale ghost nonpublication remain unobserved. |
| Cache reuse preserves correctness and checkpoint freshness | **Partial** | Primary requests use integer prompts and `cache_prompt:true` (`campaign_client.ts:157-175`) and key provider reuse by rendered prompt, replacement identity, and cursor (`extension.ts:849-858`). Edit/runtime/config invalidation clears in-flight work and cached items (`extension.ts:269-299`, `710-718`); late transports are lease-checked. Runtime manifests are revalidated on every use and installed assets are hash-partitioned (`runtime.ts:400-455`). No named run tags an old-checkpoint response and then proves a newly loaded checkpoint cannot receive that old response. |

The primary native and legacy routes are distinct. The primary route uses
`/completion`, manual BOS, `cache_prompt:true`, and protocol EOS/terminal
validation. The legacy fallback uses `/v1/completions`, `max_tokens:320`, and
the fixed seven-marker stop list at `extension.ts:163-177`. A cache result is
only reused for the same key on either route; the receipts should not be read
as evidence that the two server APIs have identical cache semantics.

## CPU checks

The ten targeted commands are recorded in `targeted-tests.json`; all returned
zero:

- protocol: 14 checks;
- PRM-05 integration: 38 assertions;
- shared request lifecycle: 5 scenarios plus 5 timeout assertions;
- provider lifecycle: 10 mocked scenarios;
- acceptance identity: stale edit, ownership and runtime invalidation cases;
- context: 46 checks;
- history: 51 checks;
- runtime manifests/cache/cancellation: 58 fixtures and offline checks;
- primary runtime profile: pass.

The runtime TypeScript check was bundled to the owned review directory so the
EXEC `dist` tree was not changed.

## Smallest remaining closure test

Using the accepted source snapshot, run one controlled response whose payload
is tagged with the old runtime/checkpoint generation, invalidate or restart
the managed sidecar, and assert that the late old response is rejected and
cannot populate `lastItems`. In the same bounded run, publish a frozen
ID-to-prompt-SHA manifest for the serving fixtures and training rows and
compare it byte-for-byte. These two checks close the live gaps without loading
weights or changing the renderer.

The requested final output in the campaign brief is
`docs/campaign/receipts/RUN-02-serving-contract-fixture.json`; this worker
produces the preparation receipt
`RUN-02-serving-contract-fixture-preparation.json` and leaves final lead
acceptance to root.
