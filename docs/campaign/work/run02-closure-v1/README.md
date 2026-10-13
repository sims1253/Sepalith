# RUN-02 mechanical closure artifacts

This packet closes the two mechanical RUN-02 gaps identified by the prior
read-only audit. It owns only this directory and
`RUN-02-serving-contract-fixture-closure.json`.

The shared fixture is the existing four-row native-probe TRAIN fixture in
`docs/campaign/work/serving-readiness/native-probe-train-fixture.jsonl`.
`build_shared_prompt_manifest.py` streams the existing RL-02 context sidecar,
constructs `PromptContext` through the execution Python package, and calls the
actual Python `sepalith.campaign_protocol.render_prompt`. It then calls the
current extension `campaign_protocol.ts::renderPrompt` through
`render_shared_fixture.ts`. No context is reconstructed from target text.

All four IDs and exact UTF-8 prompt SHA-256 values match both renderers and
the accepted native-probe manifest:

| ID | Family | Prompt tokens | Prompt SHA-256 |
| --- | --- | ---: | --- |
| `0003ea6fc6a4d0b3b354efba` | `no_op` | 345 | `5c08ad690b01d6cf03f9ec54de44bcbcb4c12d552cd521b31f942b86ac8d1f8a` |
| `000a6aaa48aee4291dbcb0cb` | `rename_propagation` | 886 | `4837faf50d9fe4b5d98814fd4903b4b3257d52a5adb2e55edf0b365b894a1f02` |
| `000d2bf789b999f887d12308` | `format_propagation` | 1216 | `80f454949d462a9ad97916d54b3c829c054597a0b90ea6d1fec7b4b58438d320` |
| `00ef45e53aea030d3465f860` | `na_rm_propagation` | 1902 | `8574f7d139161502ab3e1c63248f42f93be12d2dbadfe3d05b16458ec9bc689a` |

Each row is `split=train`, starts with exactly one manual BOS `0`, has
`target_start = prompt_token_count + 1`, has the stored terminal-token suffix
before the separately appended canonical EOS `1`, and has one EOS at sequence
end. The accepted PRM-04 receipt independently records the 4/4 joint-boundary
and terminal-suffix proofs. The context sidecar labels remain target/reward
free.

The production provider test bundles the accepted current `extension.ts` and
runs its real `provideInlineCompletionItems` path behind the existing mocked
VS Code boundary. A controlled `NativeCampaignClient.complete` seam injects an
old response that remains pending. The test transitions sidecar state
`ready -> stopped -> ready`, which increments the production runtime generation
and invalidates the provider. A fresh generation publishes first; only then is
the old response released. The old request returns zero items, and the later
cache hit still returns `new-generation + 1`. This catches both stale
publication and stale cache overwrite. Process, HTTP, server, model and GPU
operations are forbidden by the harness.

Results and exact hashes:

- shared manifest: `shared-id-prompt-manifest.json`, SHA-256
  `431e1bba94f992826308e3701d9e2a48014693128893e66ca0f1a441cddec304`;
- delayed-generation result: `delayed-generation-result.json`, SHA-256
  `e6b8b37cb7e7955bc1f5283d917b64df29c7f14d8159b80aa1ef97a8df8bd4b5`;
- Python builder and TS renderer source pins are recorded in the shared
  manifest; the provider source is `extension.ts` SHA
  `798a53049f44751b030bd190351d655d6e586c8fbaa958eaed9c0a37dc305d53`.

RUN-04 still owns the live visual stale-ghost observation. This packet proves
the deterministic provider/runtime-generation logic gate and does not claim a
real VS Code ghost display.
