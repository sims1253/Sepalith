Root cut this fresh5 candidate. Do not resume, export or promote it. The complete paired DEV result provides no exact-match gain, and finish protocol behavior worsened.

| Corrected DEV metric | Matched theta0 BF16 | Fresh5 BF16 |
|---|---:|---:|
| Exact edits | 26/43 | 26/43 |
| Correct strict no-ops | 25/32 | 25/32 |
| False suggestions | 5 | 5 |
| Protocol valid | 68/75 | 67/75 |
| Responses at 512-token cap | 7/75 | 8/75 |
| Exact finish cases | 0/6 | 0/6 |
| Protocol-valid finish cases | 4/6 | 1/6 |
| Capped finish cases | 2/6 | 5/6 |

All 75 cases use the same corrected target authority and retained output at the same 512-token diagnostic cap. Prompt token counts match for 75/75. Production classification reproduces stored protocol and predicted-no-op flags for 150/150 classifications. There are zero paired exact improvements and zero paired exact regressions; 63 raw outputs are unchanged and 12 changed. The retained baseline is SFT-primary-3000-a/evaluations/cases-step-1000.json. This comparison does not equate the 512-token diagnostic cap with the deployed 192-token cap.

Four previously uncapped finish cases now cap: e623a61b5a4c066358a477f2, d11581e9cfa4e3971aa1466e, 157517ba47dbab157f7c361a and f43de3e77f2d92ed7b223464. Case 4f08633513b5c525240d2540 becomes uncapped at 93 tokens but remains inexact. Case 04834fef4fe59742f13677a9 remains capped and inexact. Two roxygen cases become protocol valid but remain inexact: dat07-existing-2838b904f1e44b7440e6da9f and dat07-existing-81ddc4c6ddd226b04dcf0910. Exact before/after fields and all changed IDs are in paired-case-changes.json.

The run completed five optimizer updates and 160 candidates from the planned 40 source groups. Every group has four candidates; generated token hashes, reward/output hashes, prompt identity, step/group indices and source order match the planned first 40 draws. All five gradient records are finite and nonzero. Each reports gradients for all 588 trainable tensors; the first has 294 nonzero tensors and the next four have 588. Observed BF16 load capacity is 4096, with rank 16, alpha 16, 294 adapter attachments and 25,116,672 trainable parameters. The load audit reports unchanged vocabulary and 8,246 prompt parity checks. These are audit records, not a worker read of weights.

Eight of 40 reward groups have nonzero variance. Finish has 24 candidates across six groups: zero exact, four groups with variance, mean reward 0.0217508101, and ten missing canonical EOS. Its 2,749 generated tokens are 36.36% of the total 7,561. Finish accounts for 50.40% of reconstructed absolute advantages and 52.45% of the absolute-advantage-times-token proxy. No-op has 32/32 exact candidates, reward 1.2 throughout all eight groups, zero group variance and zero reconstructed advantage; its 384 tokens are 5.08% of generated tokens. Zero no-op advantage does not protect no-op behavior from changes to shared parameters.

Advantages are reconstructed from the logged four rewards using sample standard deviation plus 0.0001, as in the pinned trainer. No actual advantage tensors were logged, and Python double arithmetic does not reproduce Torch float32 rounding. The token-weighted values are diagnostic proxies, not measured gradient shares. Reward and gradient evidence establishes that the training path ran; it does not establish quality improvement. The new corrected-target NLL is 1.1953210115 across 3,450 target tokens and is recorded only as a standalone value. Old uncorrected-target NLL is not compared.

Optimizer step five finished at 14:10:11.688958 UTC. The host guard completed at 14:21:17.213341 UTC with child exit 0 and elapsed 1,152.775052509 seconds. The interval after step five was 665.524383 seconds, within the 750-second reserve. Root reports outer elapsed 1,154.867555240 seconds and independently verified seven owned IDs absent at 14:22:22 UTC. Root owns resource release and budget accounting.

The final bounded telemetry tail no longer contains the five optimizer rows because a large final identity line follows them. The preserved 14:12:29 snapshot contains all five rows; the final snapshot contains terminal and complete generation, reward and gradient evidence. This review never parsed the large identity lines. No further model, weight or test action occurred after root's cut instruction. The worker used one CPU, retained only bounded allowed evidence, and made no production, campaign state or lease changes. No final evaluation data was accessed.
