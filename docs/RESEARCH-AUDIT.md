**Sepalith research audit — first pass, 2026-09-06**

> **2026-09-06 retrospective correction:** The completed [saved-result reanalysis](reanalysis/README.md) separates doc_sync meaning and preservation from canonical reconstruction, reconstructs B8b paired selection, and reports LOC1 any-gold hit, multi-gold recall and parser defects separately. Historical scores and claims below are retained as recorded.

Purpose: identify flawed comparisons, overstated conclusions and useful work to preserve before rebuilding the repository. The user permits replacing every part of the implementation and wants both research and product development supported.

Three reviewers used `gpt-6-astra` at low reasoning effort: production recipe; data/evaluation/RL; alternative architectures. The primary agent reviewed retrieval, serving and benchmark design, and checked the key findings against local source files. This was a read-only audit of documents, representative code and small saved summaries. No experiments, tests, model calls, large corpus scans, queue changes or process controls were performed. External papers and model training cutoffs were not independently verified.

Some evidence links point to local research notes or result artifacts excluded
from Git. This report preserves the finding and its scope; a fresh clone does
not contain the complete underlying experimental record. Reproduce or retrieve
the cited artifact before treating a proposed reanalysis as verified.

**Main conclusion**

Implementation status: the first implementation batch corrected the V1d
ties-half calculation, saved summary and report, with regression tests. It also
corrected ML1's design document to require a compatible AR control and a revised
budget. The live queue was not edited; its older ML1 summary still needs to be
reconciled by the active queue manager before any dispatch. Findings below retain
the original audited claims so the reasons for the corrections remain visible.

Several operational decisions remain reasonable, especially retaining the current LoRA recipe and declining to adopt unsuccessful experimental arms. Their scientific interpretations are often broader than their evidence. The most valuable recovery work initially requires inspecting existing outputs and repairing comparisons, rather than repeating training.

Do not interpret this audit as evidence that the rejected methods will win. It establishes specific defects and missing comparisons. Preserve negative results with their actual scope.

**1. Repair ML1's control identity before using its planned comparison. Confirmed, high confidence.**

The [looped-model plan](research/2026-09-06-looped-matryoshka-plan.md) at lines 73 and 98–105 names a 32,768-vocabulary trunk and a banked autoregressive D1 control, claiming identical data/trunk and FLOP matching. The [D1 launch script](../scripts/run_d1_chain.sh) explicitly trains masked diffusion (`poc_diff.train_md`) and evaluates `--md-ckpt`. Its model defaults use a different vocabulary; the [D-grid results](../experiments/training/poc_diff/RESULTS.md) report MD span accuracy.

This control does not establish the proposed causal-language-model comparison. Equal token×loop counts also do not prove equal compute across different heads, vocabularies and objectives. The loop idea remains viable as a hypothesis; its comparison specification needs repair.

Next: reconcile checkpoint, tokenizer, objective, data and evaluation identities. Find a genuinely compatible saved AR control or revise the experiment's budget. Do not silently substitute another checkpoint. The queue itself has not been edited by this audit.

**2. Repair V1d preference accounting. Confirmed, high confidence.**

[V1D_RESULTS.md](../experiments/eval/V1D_RESULTS.md), lines 37–40, says order flips count as ties and receive half credit. [pairwise_pref.py](../experiments/eval/pairwise_pref.py), lines 523–526, includes flips in the denominator but gives half credit only to ordinary ties.

The v7/rl_v2c table gives 17 v7 wins, 43 RL wins, 73 ties and 15 flips over 148 points. Under the documented convention, RL's ties-half rate is `(43 + (73 + 15)/2)/148 = 0.5878`, not the reported 0.639. The [saved summary](../experiments/eval/results_pairwise_pref_analysis.json) reports v7's implemented rate as 0.3615; its complement is approximately the reported RL rate, but complementing is invalid when flips receive no half credit. The corresponding documented-convention rate for v8_2 is 0.8311, versus reported 0.8041.

The direction of consistent preferences still favors the same models. The effect-size accounting needs correction, and repeated points within trajectories warrant grouped uncertainty estimates. Ties also do not by themselves mean both completions are unusable: the judge can tie equally desirable outputs.

Next: reconcile saved calls and summary counts, explicitly define ties/flips/abstentions, and recompute uncertainty by trajectory. Keep this useful instrument; do not propagate its current effect sizes into product decisions.

**3. B9 did not test the originally planned parameter-selection mechanism. Confirmed, high confidence.**

The [bakeoff plan](research/2026-08-31-base-bakeoff-plan.md), lines 234–241, proposes selecting modules using parameter-gradient importance. [selekt_data.py](../experiments/training/selekt_data.py), lines 26–40, explicitly substitutes token-space loss filtering while retaining all trainable modules. This substitution was documented before the run; it is not evidence of an undisclosed change.

The observed 47.1% versus 76.5% exact score strongly rejects that token-filtering recipe. It does not reject parameter-space selection or all masking. Fewer contributing token losses need not imply sparse parameter updates. The broader closure in [bakeoff results](../experiments/training/base_bakeoff/RESULTS.md), around line 565, should be narrowed.

Next: inspect saved selection masks by family and token role. Keep plain SFT as incumbent. Mark the original module-selection intervention as untested, not failed.

**4. doc_sync may contain recoverable capability hidden by its target contract. Confirmed scope problem; benefit from rescoring remains unknown.**

[TU2 results](../experiments/synthetic-data/TU2_RESULTS.md) report close paraphrases scoring zero. The data/evaluation reviewer traced canonical target wording to deterministic constructors in [scenarios.py](../experiments/synthetic-data/scenarios.py), including hardcoded descriptions and a name-based description grammar. This is not a clean test of general documentation quality.

Repeated zero scores on the same 15 examples across several models do not independently prove a construction-only cause. Training all parameters does not turn those 15 tasks into a high-powered capacity experiment. Likewise, TU1's direction-consistency rule cannot evaluate a sufficiency contrast in families with no insufficient examples.

Next: manually inspect all 15 prompts, targets, raw responses and validator reasons. Separate parameter coverage, placement and factual correctness from canonical wording. Preserve the structural validators; decide whether semantic documentation or exact reconstruction is the intended task before changing its metric. Do not retroactively relabel semantic rescoring as the original exact-match result.

**5. TU2 and D3 change more than their headline causal explanations suggest. High confidence from reports; no new causal effect estimated.**

[TU2](../experiments/synthetic-data/TU2_RESULTS.md) has a useful same-prompt teacher-target comparison, but its exact-match result does not establish that alternate targets are semantically worse. Its volume/epoch decomposition also changes prompt selection and family proportions. [D3 preparation](../experiments/training/poc_diff/D3_PREP.md), lines 80–84, changes median prompt/span lengths from 316/38 to 234/289, alongside mix changes. D3 therefore does not isolate unique-data breadth.

Both implementations lost on their chosen benchmark. The claims that teacher targets are generally harmful or that breadth loses to repetition are not separately identified. D2's improvement with more budget on the same pool remains useful evidence.

Next: semantically rescore saved TU2 responses; compare D3's existing preparation summaries for task geometry and loss-bearing token distribution. Any later breadth experiment should hold rendering and task distribution fixed.

**6. B8b has a small positive obscured by its 'adds nothing' summary. Confirmed observation; low confidence in general benefit.**

[B8b results](../experiments/training/base_bakeoff/RESULTS.md), lines 410–433, show format exactness increasing from 52.2% to 56.7%, with 9 wins and 6 losses. Overall validity and exactness decline slightly, and the paired comparisons do not establish a benefit. Retaining the incumbent is reasonable.

However, '0.0pp on every measured axis' is inaccurate. Equal aggregate scores do not mean equal predictions; similar losses do not establish that SFT annihilated the midtraining weight changes. The 'within .001 at EVERY checkpoint' statement also disagrees with the first listed loss pair.

Next: inspect the 15 discordant format examples and compute paired effect intervals from saved rows. Retain midtraining as an unproven transfer hypothesis, not a proven no-effect mechanism.

**7. PFT1 supports LoRA as the default, with a narrower conclusion than 'adapter ceiling disproved.' High confidence on scope; resume confound unconfirmed.**

[PFT1](research/2026-09-06-lora-vs-fullft.md) loses decisively under its preregistered adoption criteria. One full-FT learning rate, seed and optimization setup does not identify a representation ceiling. Its changed-microbatch checkpoint resume coincides with a reported loss discontinuity; this is a reason to inspect state restoration, not proof that resuming caused the outcome.

The retention result is also useful: both LoRA and FT improve general-R BPB over the base, but FT improves it less. That is an incumbent-relative regression, not R forgetting relative to the base. FT approximately preserves base performance on the control while LoRA worsens it. The control consists of only 78 license texts, so calling it broad general-text retention would overstate its reach.

Next: inspect saved optimizer/scheduler/trainer state and distinguish base-relative retention from incumbent-relative performance. Preserve retention as a separate objective without changing the operational LoRA default on this evidence.

**8. Several architecture gates cannot support the conclusions attached to them. Confirmed, high confidence.**

- [DDOT's gate](../experiments/training/poc_ddot/RESULTS.md), lines 35–37, requires strictly beating a fixed-window position-MSE baseline of zero; its squared-error metric cannot do so. Its poor text metrics independently support no-adopt, but length-MAE improves from about 158 to 58. Keep the length-control observation and correct the impossible criterion.
- [POC-DIFF](../experiments/training/poc_diff/RESULTS.md), lines 31–35, claims best-of-8 superiority over an unmeasured AR sampling result, reasoning from zero greedy accuracy. That inference is invalid. Its actual MD capability remains demonstrated. The document already discloses oracle length, larger training budget and cacheless AR; those prevent treating it as a resolved product-paradigm comparison.
- [X5-S0](../experiments/training/poc_diff/X5_S0_RESULTS.md) finds a useful residual/correctness association. Its earliest stable-tail step is computed using future trajectory information, so the reported 7.8% NFE saving is retrospective headroom, not a deployable stop policy. Preserve residual telemetry; test a policy using only observations available at decision time. No conclusion is drawn here about unfinished X5-S1.

**9. H1 and LOC1 leave substantial context-selection work open. High confidence on scope; product value unproven.**

[H1](../experiments/harness_search/H1_RESULTS.md) tests short contexts on which several truncation/context knobs are inert. Failure to improve that battery does not exhaust the prompt/context space. Keeping the shipped defaults is still justified by those results.

[LOC1](../experiments/data-mining/LOC1_S0_RESULTS.md) finds useful retrieval performance, with neural-only hits as well as lexical-only hits. That complementarity is worth examining before paying for another training recipe. But its queries are commit messages, candidates come from the child commit, and some query text names the function. Matching query inputs across methods does not remove those biases from an absolute product-usefulness gate.

Additionally, [rank_metrics](../experiments/data-mining/loc1_run_eval.py), lines 109–116, measures whether any gold function is retrieved: hit@k. The report labels this recall@k despite multi-gold examples. This does not measure coverage of all functions needed for a cross-file change.

Next: inspect saved neural/lexical disagreements and evaluate a held-out hybrid ranking if cached scores permit it. Keep parent-state, natural-query and multi-target coverage checks distinct from this initial localization probe. No claim of hybrid improvement is established yet.

**10. Public benchmark and quantization plans need narrower guarantees. High confidence on missing support.**

The [benchmark proposal](research/2026-09-06-r-edit-benchmark-proposal.md), lines 158–183, assumes a new future package carve makes packages never previously present in training, and claims post-cutoff status across released models. A future exclusion rule cannot establish past absence. Check source lineage and model-specific cutoff evidence before promising either property. This audit does not establish actual contamination.

[P12](research/2026-09-04-p12-roofline-results.md), lines 182–190, discards Q6 at every tier based on throughput/size before comparative quality has been measured. Q5 has slower prefill in that same table. Q6 could remain relevant if lower-bit arms lose quality; it is not demonstrated to be dominated on the full product objective. Preserve the measured speed results and treat the three-arm choice as budget prioritization.

**Reviewer leads retained for a second pass**

- The optimizer reviewer found an omitted update-norm normalization in the tested MuonH variant and an untested uniform-control + constant-tail + averaging combination. Verify the implementation receipts before treating those mechanisms as closed.
- O1's data-selection no-adopt appears sound, while 'count > curation' is a broader causal claim than its comparisons establish.
- Per-family package splits and repeated use of the 255-row battery support a development benchmark, not automatically a globally source-disjoint final test. Inspect provenance manifests to establish actual overlap; no leakage is proven here.
- The [E1 result report](../experiments/training/rl/results/E1_RESULTS.md) was located during final review. Its setup states 1,600 completions per arm, while the mechanism section calls 656 random-arm format completions 20.5%; 656/1,600 is 41%. This is a confirmed inconsistency in the report, but the correct count and denominator require trajectory reconstruction. It does not establish that the primary group-weighted comparison is wrong. Keep that reconstruction in the recovery backlog.
- Sparse telemetry limits tail-stability claims in the optimizer/stability experiments. Review sample counts before retaining p99.9 claims.

**Recovery order and rebuild implications**

1. Repair the ML1 comparison specification and V1d accounting before they guide further decisions. Use an isolated change and respect the active queue manager's ownership.
2. Recover evidence without retraining: doc_sync inspection, TU2 semantic rescoring, B8b discordant cases, D3 distribution audit, LOC1 disagreement analysis and PFT1 resume-state inspection. Some require artifacts not inspected in this pass; schedule any substantial computation outside the quiet benchmark window.
3. Build experiment records that link planned intervention, executed configuration, deviations, metric definitions, uncertainty and narrowly scoped verdicts. Include explicit labels for untested, not adopted, underpowered, invalid comparison and superseded.
4. Have the runner bind every attempt to code/data/model identities and feasible gates. Check that a baseline is the intended model and that the success criterion is mathematically attainable before allocating compute.
5. Separate development selection from final product evaluation. Preserve raw responses so changing a scorer does not require rerunning a model. Version every scorer and retain original results beside exploratory rescoring.
6. Design the destination workflows using this evidence. Existing scripts, directories and coordination protocols may all be replaced. Compatibility serves artifact preservation and meaningful comparison, not preservation of the current architecture.

This is a targeted first pass, not an exhaustive review of every experiment, external paper, checkpoint or dataset. Findings do not authorize new compute spending or change parked user decisions. The next implementation batch should repair the evidence records and establish the queue/run contracts that prevent these errors from recurring.
