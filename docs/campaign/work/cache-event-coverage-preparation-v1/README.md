This packet supports bounded host cache/context correctness for the selected private extension. It does not complete RUN-04. The current product has no diagnostics-refresh subscription and always supplies an empty diagnostics array to PRM-03. An editor recapture cannot close that capability gap without a separately reviewed product change.

The CPU driver bundles the unchanged production entry and its 12 imported modules. It appends test access only, using the existing provider-test approach. The editor, LSP and model-client boundaries are synthetic. The actual provider, symbol cache, result cache, HistoryProvider, source selection, renderer and stale-result guards execute. The only completion is a deterministic no-edit response; there is no model, tokenizer or network call.

There are 20 event rows and 131 passing assertions, plus three named negative controls. The scope-off sequence matches the LAN arms. The scope-on sequence is a separate probe of an existing feature; it does not authorize a live settings change. Empty LSP responses exercise the real brace-scan fallback and its cached negative symbol result.

| Event | Observed host behavior | Remaining limit |
|---|---|---|
| Unchanged repeat | Reuses completed result; scope-on reuses symbol result but rebuilds scope/context | Synthetic boundary timing only |
| Cursor-only | Same document identity; cursor-marked prompt changes; symbol result reused | Real editor dispatch/publication not measured here |
| History change | Insert/revert restores exact source bytes; two genuine observed edits remain in prompt history; version change refreshes symbols | History-only editor trace and native prefix effects remain unmeasured |
| Diagnostics refresh | No production subscriber; diagnostics absent; same request result reused | Unsupported by current product |
| Anchor move | Long source window moves from line 42 to 212; new function source and application range replace old distant spans; symbol result reused | Actual editor/model context-budget and prefix costs remain unmeasured |
| R-to-R file switch | Identical text in another URI gets its own path and empty history; previous URI result cannot be reused | Actual file-switch dispatch/publication remains unmeasured |
| Close/reopen | Clears per-URI history and symbol result | Synthetic editor boundary |

The negative controls change content without increasing version, return a stale LSP response after an edit, and return a stale completion after newer work. The real provider rejects stale cache reuse/publication. A target-only canary is never supplied to source/context/client inputs and is absent from every rendered prompt. History text enters prompts only after a legitimate synthetic source edit.

`cpu-trace.json` retains actual rendered prompt bytes and SHA-256, document URI/version/full-buffer SHA, cursor, actual request-cache key, context captured at the client boundary, history audit and independently recomputed selection spans. A cached row links to the preceding identical context and the actual cached request key. Host-work counters distinguish LSP calls, scope rebuild calls, result hits and document reads. Selection-audit reads are separate. Millisecond values are local CPU fixture observations, not notebook editor latency or native prefix-cache timing.

`source-closure.json` records all 13 resolved production module origins. Eleven match canonical EXEC source; the selected remote connection and extension entry differ, as expected for this private candidate. The selected packaged extension bundle matches the retained VSIX. `entry-test-access.patch` contains the exact appended test-access suffix. Accepted product, provider, gateway and fixtures were not edited.

Root can replay the frozen CPU candidate into a fresh root-owned output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/cache-event-coverage-preparation-v1/run_cpu.py --output-root /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/cache-event-coverage-replay-a
```

The runner refuses an existing output directory or changed source pins, uses one CPU, disables CUDA, and has a 180-second ceiling. Its replay into this packet's `replay-check` directory passed in 0.717 seconds including process exit. No editor launch is staged: full live event/timing admission is blocked by diagnostics support and absent production host-cache timing/context telemetry. `live-prerequisites.json` lists the exact gates for root review; another broad latency sweep would not resolve them.
