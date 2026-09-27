The initial diagnosis supports a new TRAIN-grounded R competence stage before another targeted RL attempt. It does not support resuming the rejected RL identity or merely adding unchanged corrective SFT steps. The selected theta0 Q8 remains the fallback; all new training requires root review and resource admission.

The actual native192 panel has26/43 exact edits and25/32 strict no-ops. Fourteen of17 failed edits are finish/documentation cases. Five finish targets fit192 including terminal and EOS, but none succeeds. All three protocol-valid finish outputs fail Tree-sitter parsing after application to their exact source-bound document; the other three finish outputs cap. Six corrected gold documents parse. This is stronger evidence of incomplete output than protocol failure alone, while parsing still cannot establish useful semantics.

Context and capability must be separated. All six finish cases have empty history/suffix/scope, no selector omissions or overflow, and five have only132–161 prompt tokens. Some exact original bodies are underdetermined by this context. One prefix explicitly specifies exact factorial and arbitrary precision yet receives repeated factor conversions, establishing a concrete instruction/semantic miss. None of these DEV cases or targets is eligible for new training.

Two documentation references and one finish reference require392,404 and248 tokens including EOS, exceeding192. The all-case denominator remains75; report this reachability stratum without changing labels. Every actual capped case had a reachable target, so the cap does not explain the six observed runaways. Documentation exact mismatch also conflates paraphrase, invented facts and true content errors;0/8 exact alone is insufficient to label all outputs semantically wrong.

The tested old reward gives exact equality +0.2 matching whole-line F1 after hard token/protocol gates. It has no full-document R parse or behavioral check. In the fresh5 pilot only8/40 candidate groups had reward variance; all32 no-op candidates were already exact and supplied zero advantage, while non-exact finishes supplied about half of absolute advantage. Broader sampling and a TRAIN-only reward discriminator should be validated before another GRPO update. Parse-only gains in the corrective SFT arms did not translate into exact or no-op gains.

A first root decision can select an explicit TRAIN-only curriculum: inferable short source-grounded R infilling, local continuation and constrained transformations, intact braces and terminal budget, preserved conservative edit/no-op examples, and hard no-ops mined from untouched TRAIN sources. Assess whether source examples reveal enough information to determine their target. Keep separate diagnostic TRAIN samples and original split/provenance; do not derive labels from DEV misses. Run one bounded SFT milestone, inspect fresh candidate reward variance by family, then decide whether a changed targeted RL pilot is justified. Use native Q8 PRM03 context4096/output192 for promotion; HF512 evaluations are separately labeled and not output-equivalent.

Exact CPU replay:

```sh
python3 -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-development-diagnosis-v1/audit_native_dev.py
python3 -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-development-diagnosis-v1/audit_train_metadata.py
```

Both scripts restrict affinity to two available CPU cores and import no model framework. DEV replay checks75 unique IDs, exact prompt rendering/hashes, canonical target-region hashes, token counts, protocol and quality, five aggregate fields, six gold document parses and two parser controls. TRAIN replay streams11526 rows, validates token geometry and unique IDs, confirms zero DEV ID/package overlap, and hashes only the117.9MB TRAIN file. Prior group split evidence remains separately attributed because group IDs are absent from the token-row schema.

External review input is opus-review-packet-v2.md. Its de-identified DEV observations are diagnostic only. Revision1 remains preserved with the documented one-token EOS counting correction. No data, reward source, training recipe, campaign state, model, final row, native process or resource lease was modified or launched.

Notebook work can proceed independently of desktop training:

1. Existing selected-theta0 Vulkan latency controller: work/lead/theta0-notebook-long-a/run_notebook_long.py and notebook_long_server.py, with pinned work/lead/theta0-long-context-a/replay-long-context.ts. It uses selected Q8 SHA22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559, context4096, batch/ubatch256, six notebook threads, ngl99, six synthetic events and five-second requests. Root-only command below. The desktop client is CPU Node; inference stays on the notebook. Root must release the daily reverse tunnel that owns notebook127.0.0.1:18403 and take the notebook lease. No desktop CUDA is needed. Record actual notebook backend and timings; results include SSH transit and do not measure editor visibility. Server140s, client65s, remote outer180s; proposed whole root-command guard360s covers readiness and bounded evidence retrieval.

```sh
timeout --signal=TERM --kill-after=20s 360s python3 -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/theta0-notebook-long-a/run_notebook_long.py --candidate Q8_0 --context 4096 --run-name RUN-R2-notebook-theta0-vulkan-a
```

2. Existing selected-theta0 CPU quality comparator: work/theta0-notebook-cpu-dev-preparation/run_theta0_notebook_cpu_dev_v2.py and remote_notebook_theta0_cpu_server_v2.py. It uses CPU ngl0,6threads,batch/ubatch256,context4096 and corrected75 DEV. Its historical output cap is512, so it is an independent CPU quality diagnostic, not the requested delivery192 latency comparison. Root-only historical command shape:

```sh
timeout --signal=TERM --kill-after=20s 1240s python3 -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/theta0-notebook-cpu-dev-preparation/run_theta0_notebook_cpu_dev_v2.py --run-root /home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-R2-notebook-theta0-cpu-a
```

There is no reviewed drop-in selected-theta0 CPU192 latency controller identified in this bounded review. A matched CPU/Vulkan latency A/B needs a small separately reviewed CPU launcher rebind plus the same192 client/fixtures; changing ngl alone in a receipt is not evidence. Do not use the older primary500 Q6 latency pair as theta0 evidence. Notebook-local model and small binary/library hashes are checked by the existing root launchers at launch, not read by this worker. Root should verify fresh run names, staged server bytes and owned-process cleanup; this packet does not admit either launch.
