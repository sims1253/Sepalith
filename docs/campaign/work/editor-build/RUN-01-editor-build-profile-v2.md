# RUN-01 editor profile addendum v2

This addendum preserves the earlier editor-host readiness note as v1 and corrects its generic activation example for the reviewed campaign model. It is a profile clarification only; no source, package metadata, editor setting, server, model, or GUI state was changed here.

The adapted model must be admitted against the PRM-03 identity `zeta2-prm03-v1`, schema `sepalith.prompt.prm03.v1`, and the exact tokenizer/protocol hashes carried by the reviewed manifest. The b4 legacy model and its defaults are a separate fallback identity and cannot be substituted for adapted PRM-03 weights.

The frozen native campaign profile is:

| Field | Value |
| --- | --- |
| loaded context | `4096` |
| server batch / ubatch | `256 / 256` |
| parallel slots | `1` |
| RL policy generation cap | `192` new tokens |
| 75-case DEV evaluator quality cap | `512` tokens |
| m0pad CPU reference | `-ngl 0`, `-t 6`, `-tb 6` |
| m0pad Vulkan profile | `-ngl 43`, `-t 6`, `-tb 6` |

The rollout prompt/completion geometry remains the separately frozen `2048 + 192`; loading context 4096 retains the longest DEV evaluator cases and does not expand the RL policy cap. The current extension source exposes `contextSize`, `threads`, `gpuLayers`, and `parallel=1` through its child arguments, but it does not expose batch/ubatch settings. Therefore a live adapted-model editor check must use a reviewed server wrapper/profile that binds `-c 4096 -b 256 -ub 256` (and the selected CPU/Vulkan `-t/-tb/-ngl` values), or receive a separately reviewed source integration. The package artifact alone does not prove those runtime flags.

The v1 generic example of `contextSize=8192`, CPU backend, and 8 threads is retained as an unaccepted development fallback. It must not be applied to adapted PRM-03 weights or used as campaign acceptance evidence. Root must supply the exact adapted model/manifest identity, select the host backend, and record the runtime command/hash before any GUI activation. This addendum does not claim GUI placement or live editor acceptance.

Evidence: `extensions/vscode-sepalith/src/campaign_protocol.ts:9-25`; `docs/campaign/receipts/RUN-06-prefill-options.json` (`ctx=4096`, `batch=256`, `ubatch=256`, `threads=6`, `ngl=0`); `docs/campaign/receipts/RUN-09-q8-vulkan-profile-admission.json` (`b256ub256ctx4096ngl99t6tb6lv4`) and the root-reviewed profile correction (`ngl=43` for the m0pad Vulkan device).

