Revision2: corrects inclusive target lengths by adding the separate EOS token; includes independently replayed75-case quality and applied-R syntax evidence. Revision1 remains retained.

TASK RL-R2-DIAG: independent review of broader R training followed by targeted RL.

This packet contains authorized development diagnosis. It is not training data or a final evaluation. Do not turn any DEV example, target, identifier, or paraphrase into a training example. Recommend derivations from the independently split TRAIN sources only. Root will review and launch; no launch or promotion is implied. The previously delivered theta0 Q8 model remains the fallback.

Decision requested: choose the smallest scientifically defensible broader R SFT/data curriculum and subsequent reward change that could improve semantic completion and conservative editing. Explain why it differs from the two failed corrective SFT attempts and the failed five-update corrected GRPO pilot. Rank alternatives, state falsifiers, and propose a bounded first experiment. Do not merely recommend more unchanged steps.

Observed delivered model: selected theta0 step1000, Q8_0, native llama.cpp b10453 CUDA, graph optimization disabled, batch/ubatch256, PRM03 renderer, context4096, output cap192 including terminal tokens, greedy temperature0. This is the actual delivery route. The new native DEV run completed all75 requests with matching HF/native prompt tokenization. Targets were used out of band for scoring, never sent in the completion request.

Actual corrected DEV75, single fixed panel, no final data:

| Family | Cases | Exact edits or strict correct no-op | Valid protocol | Capped |
|---|---:|---:|---:|---:|
| Rename propagation | 8 | 8 | 8 | 0 |
| Pipe rewrite | 8 | 8 | 8 | 0 |
| Format propagation | 7 | 5 | 7 | 0 |
| na.rm propagation | 6 | 5 | 6 | 0 |
| Roxygen drafting | 8 | 0 | 7 | 1 |
| Finish block | 6 | 0 | 3 | 3 |
| No-op | 32 | 25 | 30 | 2 |

Totals:26/43 exact edits,25/32 strict no-ops,5 protocol-valid unwanted suggestions,69/75 valid protocol,6/75 capped. The two capped no-op failures are additional to the5 false suggestions. Of17 failed edit cases,14 are documentation or block completion. Three exact references exceed192 tokens: two documentation targets392/404 tokens and one finish target248, including terminal. These remain failed cases; do not change the DEV denominator or truncate gold. The other five finish references fit192 but none is exact. An independent CPU application/Tree-sitter check finds0/3 parsable full R documents among the protocol-valid finish outputs; all6 corrected reference documents parse. The3 capped finish outputs are not applied. This is a syntax check, not execution or semantic proof. All six actual capped responses had reachable gold targets: runaway generation/terminal behavior is a real separate problem.

Context audit: all6 finish rows have empty history, scope and suffix. Five have only132–161 prompt tokens; the sixth has380. All6 report no selector overflow or omissions. Missing behavioral context is therefore a source-construction/conditioning hypothesis, not observed runtime context truncation. Exact original implementation recovery can be underdetermined even when the target is syntactically correct.

Small de-identified DEV observations, diagnostic only:

1. A completion named for a factorial-like operation produces repeated `is.factor`/`as.factor` conversions, while the reference performs arbitrary-precision multiplication. The47-token reference fits the cap; the model emits161 tokens and valid edit protocol. This is a semantic/intent error, not just a missing terminal. The visible prefix explicitly specifies exact factorial and arbitrary-precision output, supporting a real semantic instruction failure. Exact recovery of every unseen implementation remains a separate ambiguity.
2. A short documented clustering wrapper has a74-token reference. The model instead invents dependency checks and validation, reaching192 without a complete terminal. A different finish response adds a long numbered commentary about repeated sampling until the same cap.
3. Documentation output claims a matrix return where the reference describes a vector. Another output invents an author attribution. Other documentation outputs are plausible shorter paraphrases but still fail exact match. Therefore0/8 exact is neither proof that all8 are semantically worthless nor evidence that they are correct. The current scorer does not adjudicate documentation truth.
4. On one no-op the model adds `na.rm = TRUE`; on another it rewrites a function header/body. Five valid unwanted edits and two runaway edits show that protocol correctness does not enforce restraint.
5. One format failure retains a compact conditional rather than propagating braces; another adds a space after `function`. One na.rm case adds the requested option but also substitutes the wrong existing variable. These are local consistency problems distinct from body synthesis.

Current reward, exact source inspected (frozen be406557 source, campaign_rl_train.py lines936-1103):
- Reject invalid token IDs, missing/noncanonical EOS, any early control token, exceeded cap, or invalid PRM03 edit protocol: reward0.
- Otherwise apply parsed edit semantics to obtain replacement lines (a parsed no-op uses old region), compare to TRAIN expected replacement lines.
- Reward = exact line-list equality +0.2*line_F1. line_F1 uses difflib matching whole lines, rstrips lines for shaping and removes trailing blank lines. Exact success typically1.2.
- This reward does not parse the full applied R document, evaluate R behavior, check data-flow or documentation truth, distinguish useful semantic alternatives, or specifically penalize a false edit below a generic zero. Boilerplate-line overlap can produce tiny positive rewards without correct behavior. A protocol-valid edit that copies the old region is normalized to no-op by the protocol parser.
- No changes to these semantics are admitted here. Any proposed extra R semantic reward must have source-backed, deterministic TRAIN-only tests; arbitrary package code execution is not an acceptable quick verification shortcut. Parse safety alone is not semantic correctness.

TRAIN evidence available, split independent of DEV:
- Corrected TRAIN11526 unique rows: finish4051, pipe2027, format1719, rename1578, no-op1140, documentation874, na.rm137. The correction restores missing outer braces from TRAIN source;238 contradictory/unsupported original rows were excluded. It does not invent new implementations.
- No-op rows are9.89% of examples but only1.21% of canonical target-token mass. Full-text training can dilute this further with prompt loss.
-1251 TRAIN prompts exceed2048 tokens (maximum3000);2031 canonical targets including EOS exceed192 (maximum933). Long-output training is not automatically suitable for192-token delivery.
- The corrected old-RL eligible intersection has8246 rows, excludes194 contradictory old IDs, and changes2222 targets. Its family counts: finish2222,pipe1683,format1254,rename1189,no-op1140,documentation675,na.rm83. These fit the old2048-prompt/192-target RL geometry. No TRAIN/DEV ID, group or package overlap was found by the existing audit.
- Old RL used a fixed24000-draw interleaved schedule. First five updates:40 source prompts/160 completions (four candidates each),10 format,10 rename,8 no-op,6 finish,4 pipe,1 documentation,1 na.rm. First25 updates contain200 prompts:50 format,50 rename,40 no-op,32 finish,20 pipe,5 documentation,3 na.rm. A schedule dominated by already solved transformations may not build missing R competence.

What already failed (all separate identities, none promoted):
- Corrective full-text SFT on the restored-brace data, fresh LoRA on merged theta0: step50 gave23/43 edits,23/32 no-ops,8FP,3caps,4/6 parsable finish documents but0/6 exact. Step100 gave25/43 edits,22/32 no-ops,8FP,9caps,2/6 parsable finish documents and0/6 exact. Parsing gains did not establish useful semantics.
- Corrective target-only SFT,25 updates/400 source cursor:24/43 edits,23/32 no-ops,8FP,4caps,0/6 exact finish;3/3 applicable finish buffers parsed. One parsed output was252 tokens and would not fit delivery192.
- Fresh corrected GRPO, five finite nonzero optimizer updates,40 prompts/160 outputs: only8/40 groups had nonzero reward variance. Rename40/40 candidates, no-op32/32, pipe16/16 and na.rm4/4 were all exact with zero within-group variance. Finish24 candidates had0 exact and10 missing EOS; documentation4 candidates had0 exact and3 missing EOS. Finish had36.36% generated-token share and50.40% absolute advantage share; no-op had5.08% generated tokens and zero advantage. Small proxy differences among wrong finishes supplied much of the learning signal.
- GRPO corrected DEV comparison at HF cap512 retained26/43 edits,25/32 no-ops,5FP,0/6 exact finish, but finish protocol4→1 and caps2→5. No paired exact gain.
- These old diagnostics used HF cap512 and sometimes an unmerged adapter baseline. They are not output-equivalent to delivered native Q8 cap192. A standalone merged HF diagnostic later gave26/24/6 while delivered native Q8 gives26/25/5. Preserve exact model/merge/quantization/runtime identities when evaluating new training.

Proposed hypotheses to critique, not predetermined decisions:
A. Broader TRAIN-only R competence curriculum: source-grounded short function/body infilling, local continuation and constrained transformations with intact syntactic boundaries and enough visible context. Use existing TRAIN source/provenance, package/group split and source-hash checks. Separate genuinely inferable examples from arbitrary recovery of an unseen original function body. Keep PRM03, explicit no-op, and conservative edit examples in the mixture; do not replace everything with generic whole-file next-token training.
B. Prepare hard no-ops from untouched TRAIN source where the current model actually makes false edits, plus paired minimal requested edits from that same TRAIN domain. Balance at the source-prompt/loss level as well as token mass. Preserve an untouched TRAIN-only diagnostic subset for judging the mining, without reusing DEV as targets.
C. Reserve full output/terminal budget in training targets. Generate eligible short-span examples from TRAIN syntax/source boundaries; do not silently truncate a long desired function or alter DEV. Report both all-case quality and cap-reachable strata.
D. Before new RL, run a small TRAIN candidate diagnostic: coverage, exact/parse/semantic checks, valid terminals and within-group reward variance by family. Only proceed if the proposed reward separates demonstrably better candidates, not merely longer or boilerplate-rich wrong code. R parse checks can be a gate; stronger behavior checks need independently justified TRAIN tests and isolation. Penalizing all changes can collapse edit recall, so retain exact edit/no-op floors.
E. Evaluate one bounded SFT checkpoint before RL. Then conduct a small fresh RL pilot with explicit changed data/reward/schedule/source identity, no resume from rejected RL. Keep fallback untouched. Judge actual delivered native192 quality before promotion; HF512 remains diagnostic.

Return requested (compact): ranked root decisions; concrete TRAIN derivation/reward pseudocode and manifest fields; tests that reject reward gaming/ambiguous supervision; bounded pilot and stop criteria; tradeoff between generic R corpus adaptation and PRM03 task grounding; uncertainties that require measurement. No unseen job claims. Do not request or use final rows. Root reconciles remaining local runtime and external quota before launch.
