# RUN-04 prepared native adapter

This is an isolated, opt-in integration seam for the accepted Opus iteration-2 exact prewarm scheduler. It copies the reviewed execution `campaign_client.ts`, `campaign_protocol.ts`, and `campaign_requests.ts` into `src/`, and copies the accepted scheduler/contract without changing the production extension or campaign state.

`prepareExactPrompt(context, identity, promptText, encodedPromptTokenIds, noOp)` validates the context identity and exact renderer output once, then freezes the context, prompt, IDs, and snapshot together. `NativeExactAdapter` binds that packet to the scheduler. The scheduler still adds exactly one BOS ID. `NativeCampaignClient.completePrepared()` sends those BOS-inclusive IDs directly to `/completion`; it does not call `/tokenize` or render again.

Warm requests are fixed at `n_predict: 1`, `cache_prompt: true`. Their raw token/stop metadata is validated and published as `evidenceOnly`; the cap-1 result is never parsed as PRM-03 and never returned to an editor. An identical foreground waits for the warm transport to settle, discards its token, and sends a real foreground request with the fixed 192-token cap. Foreground keeps the canonical EOS, native-control, vocabulary, prompt metadata, truncation, and `parseOutput` checks.

The deterministic client and scheduler checks run with:

```sh
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHONDONTWRITEBYTECODE=1 npx --yes tsx --test docs/campaign/work/prewarm-native-adapter-v1/tests/*.test.ts
```

The synthetic harness is executable without a server. `--on` exercises the complete warm-then-foreground transport shape; without `--on` it reports the disabled path and zero calls:

```sh
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHONDONTWRITEBYTECODE=1 npx --yes tsx docs/campaign/work/prewarm-native-adapter-v1/harness/native_transport_harness.ts --on
```

The harness transport is deliberately in-memory. Root owns any real localhost 18403 launch and may substitute its supervised `JsonPostTransport` after reviewing this seam.
