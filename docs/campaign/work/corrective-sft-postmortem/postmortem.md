# SFT-06 corrective trajectory postmortem

Observed at: 2026-09-13T04:03:56Z

## Decision

Cut the corrected SFT-06 trajectory. Do not spend the remaining optional corrective-training budget and do not restart the same recipe. Preserve the selected SFT1000 checkpoint and the accepted serving/fallback work for the parent campaign decision.

The corrected labels are a plausible data repair, but the saved trajectory does not show a materially better training intervention. At step 50, exact edits were 23/43 versus 26/43 for the selected SFT1000 baseline, strict no-op correctness was 23/32 versus 25/32, and false suggestions were 8 versus 5. At step 100, the corresponding corrected results were 25/43, 22/32, and 8; this remains below baseline on all three behavior gates. The corrected run improved the cap count only in the sense recorded by the diagnostic, but cap feasibility is not a semantic-quality gain.

Finish behavior does not change the decision. Structural post-document parsing was 4/6 at step 50 and 2/6 at step 100, while exact finish matches were 0/6 at both points. The independent finish review records semantic drift/incomplete setup and confirms that the serving cap can make some exact finishes impossible; this is a feasibility limitation, not evidence of learned correctness.

## What the objective actually measured

The pinned SFT source uses full-text labels. The collator masks only padding positions with -100; it does not mask the prompt. The evaluator's prompt denominator is `target_start - 1` and its target denominator is `len(input_ids) - target_start`, so the target denominator includes the target body and terminal EOS. On the fixed 75-case panel these denominators are 91,180 prompt tokens and 3,450 target tokens for every compared readout. The corrected schedule exposure is 3,498,885 prompt tokens and 218,482 target tokens over 3,200 draws.

Therefore the saved prompt/target NLLs are full-text-loss diagnostics with fixed, comparable denominators; they are not evidence that a target-only objective was trained. Corrected step-50 NLLs were 1.054596 prompt and 1.182568 target, and step-100 NLLs were 1.063092 and 1.185800. The selected SFT1000 baseline was 1.049481 and 1.175485. Both corrected readouts are higher on both denominators.

## Data and trajectory facts

The materialized correction has 4,051 changed rows, 7,475 unchanged rows, and 238 excluded rows. All 4,051 changed rows retain their prompt IDs; 3,313 have equal target-token length and 738 gain exactly one target token. No changed row loses target tokens. The 1,140 semantic no-op rows remain present, and the schedule contains 320 no-op presentations out of 3,200 (10%). The evaluation still shows 8 false suggestions at both corrected checkpoints, versus 5 for the baseline.

Both corrected stages use a fresh LoRA on merged SFT1000 theta0 with rank 32, alpha 64, learning rate 2e-4, 3% cosine warmup, AdamW fused, seed 3407, and full-text loss. The step-50 to step-100 stage resumes the saved optimizer, scheduler, and RNG state from the corrected checkpoint; checkpoint and state integrity passed. There is no saved evidence of an optimizer-restore defect. Independent replay equivalence was not run, so that remains a limitation rather than a positive claim.

The six finish target body-plus-terminal lengths are 136, 164, 247, 46, 73, and 154 tokens. The recorded step-50 generated lengths are 89, 40, 512, 173, 508, and 512. In particular, the 247-token target cannot be exact under a 192-token cap. This report records that serving constraint and does not reopen renderer or output semantics.

## Hypotheses versus evidence

A target-only loss, changed loss weighting, finish oversampling, or a lower-strength LoRA/LR could be reasonable future hypotheses. None was tested by SFT-06: the run used the full-text collator and the fixed corrected schedule above. The observed regression on edits/no-ops and unchanged exact finish score supplies no basis to allocate another run to any one of those hypotheses now.

No new bounded training proposal is admitted. No source delta, data reweighting, or displaced run budget is authorized by this postmortem. The 1,775.576667824993 seconds of unused correction ceiling remains unspent.

## Future preflight, only if separately re-opened

No CPU preflight remains necessary to close SFT-06 under the cut decision. If a future owner proposes a new intervention, its CPU preflight must first prove the actual `labels == -100` mask and prompt/target denominator totals on synthetic rows and corrected samples; pin a new objective, source, data, and schedule identity; exclude the 238 rejected rows and document the 711 absent packet rows; verify fresh optimizer initialization; and run an isolated interruption/resume equivalence check before any model run. The 75-case panel must retain the baseline exact-edit/no-op gates and report finish exactness separately from cap feasibility.

## Evidence limits

This is a saved-artifact review. It does not load model tensors, generate new outputs, or establish independent replay equivalence. The recommendation is about allocation of further SFT-06 corrective work; the parent agent retains the campaign-level launch and promotion decision.
