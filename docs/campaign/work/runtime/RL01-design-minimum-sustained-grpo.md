# RL-01/RL-03 minimum sustained GRPO design

Status: design verified from frozen source and installed TRL 0.24.0. This packet makes no model, framework, CUDA, server, cloud, data-registry, or training launch. It does not implement the future trainer.

The smallest safe sustained run has one explicit PRM03 train manifest, one future merged-SFT parent, one fixed candidate count (G=2 or G=4), one deterministic prompt sampler, one fixed-ID generation adapter, and the existing checkpoint/control/evaluation callbacks. The first run should measure the mechanism and resume path before adding a scheduler treatment, vLLM, multi-GPU grouping, or multiple policy iterations.

## Frozen identities and constraints

The execution source observed at the receipt timestamp is:

- packages/sepalith/src/sepalith/campaign_protocol.py, SHA256 5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156.
- experiments/training/campaign_rl_profile.py, SHA256 c84f64bc26a7f7ebf7e211e52ef89581d53696464f3fd7d14cd6c4afc0b3d5e9.
- experiments/training/campaign_checkpoint.py, SHA256 64202f4e5b88c94a74794442be3a4955f07d195079c9d64bbcff2850dbadd0ba.
- experiments/training/campaign_control.py, SHA256 9a5b0614f134ecf6b804d2f0d5e37a4fecf2b1dc53589b92f17580678f9dbdac.
- experiments/training/campaign_eval.py, SHA256 e378fb6e3e2b29854f81f8cae2663e5e6326f71d05168b025633129e1fa5060f.
- experiments/training/rl_smoke.py, SHA256 986ff5a3eba7b8718103b51118ad3e817b62a924544d7771bfc4775debe7bc03.
- experiments/training/test_rl_smoke_el.py, SHA256 fcf589ccec9b9726f78d7a269fae59f23e6fa5cbb3d1136a409711622a9adf62.

The pinned parent available for the mechanism profile is MiniCPM5-2B Midtrain revision 8dc5f6055b90fe4b9422340810b270b9569f37f3, weights SHA256 38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad. It is a base/profile input. It is not an admitted RL parent. A sustained run must name a future merged-SFT manifest and merged-weight hash before loading or attaching an RL adapter.

PRM03 is sepalith.prompt.prm03.v1, renderer zeta2-prm03-v1, with tokenization policy hf_split_special_tokens_true_native_no_bos_no_parse_special_manual_bos0_terminal_eos1_final_lf_v1. Vocabulary size is 130560; BOS is 0; canonical EOS and PAD are 1; native EOG IDs are [1, 130073]. valid_generation_tokens requires integer in-range IDs, a final canonical 1, and no native CONTROL ID before that terminal. ID 130073 stops native generation but is noncanonical for this campaign and must receive zero reward. A CONTROL ID before terminal is also invalid. EOS/PAD sharing ID 1 requires positional masks.

The current RL profile fixes rank 16, alpha 16, seven target modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), 294 expected attachments, and 25,116,672 trainable parameters. It bounds prompts at 2048 IDs, completions at 192 IDs, and managed context at 2240 IDs. Candidate counts 2 and 4 are profile candidates; no optimum or quality claim follows. The profile uses stored input_ids[:target_start], including manual BOS 0, rather than re-tokenizing prompt text. Its policy batch is prompt_count multiplied by candidate_count; generation is currently issued per prompt and exact rollouts are recomputed together.

The installed venv is TRL 0.24.0. Its observed source hashes are:

- trl/trainer/grpo_config.py: e71900f3a0590cceb1ad67f07e9845fc3b517eae85f3d16300a32a4b652a00bb.
- trl/trainer/grpo_trainer.py: bb6905182acf2ec426f1dc9ffc37e5210d58ee47d2764f7cf12dbb90a4a28954.
- trl/trainer/utils.py: 7096881df3fd78411be857e79b82f7c0eb8f3b5d9b0771de06275bacc9af98e3.
- trl/data_utils.py: 48add872228a9b6d923483be8cb4dcb62e7c0e53253604c743c176b6e1dd9841.
- trl-0.24.0.dist-info/METADATA: 351c5b78dcea3e968b39ebadac53d728aa4e7310e113fff55626af5bf6eec333.

## Required future input contract

Keep the canonical PRM03 training row unchanged and validate every row with validate_training_row. The current row contains exact stored IDs and labels but no PromptContext; therefore full-copy no-op parsing needs a second, explicit context manifest. Require these three hashed inputs:

1. A candidate/admitted train JSONL containing canonical rows. Each row must have split=train, a unique ID, input_ids beginning with one manual BOS, target_start, target_operation, target_body_text, renderer/tokenizer policy identities, and the proven terminal suffix.
2. A context JSONL keyed by row ID. It must contain only the pre-edit PromptContext fields. Verify render_prompt(context) equals row.prompt_text, context.region_old, replacement range, document identity, and source hash. Reject target, future-edit, or unverified provider fields in this manifest.
3. An ordered selected-ID JSONL. Resolve exactly those IDs, reject duplicates/missing rows, and retain file order. No discovery of dev/final rows and no call to a candidate-admission routine is allowed inside the trainer.

Normalize each row to a TRL-compatible envelope:
prompt = {"text": row["prompt_text"], "ids": row["input_ids"][:row["target_start"]]}.
The envelope is carried through the identity data collator; the fixed-ID generator consumes prompt["ids"] and never sends prompt["text"] through tokenizer encoding. Row metadata (context, target_operation, target_body_text, family, package, ID) remains out-of-band for the reward. The context-manifest SHA, candidate-file SHA, selected-ID SHA, ordered-ID hash, row count, and split are part of the data identity.

The model-visible prompt must be pre-edit PRM03 text only. The target and expected region are reward labels; they must not be appended to the prompt or used to select the generated answer. Prompt overflow is a failed input, not a truncation opportunity. A future SFT/row loader may perform the same canonical build_training_row checks, but the GRPO path must not silently re-tokenize or rebuild rows from text.

## Narrow trainer seam

Add a future experiments/training/campaign_rl_train.py containing a recipe loader, CampaignGRPOTrainer, CampaignRepeatSampler, and CampaignPRM03Reward. Keep the implementation small and pin it to the observed TRL source hash.

TRL reuse is appropriate after IDs are fixed:

- _generate_and_score_completions already pads prompt IDs left and completion IDs right, calls reward functions with row metadata, gathers rewards, groups by num_generations, computes advantages, and prepares the tensors consumed by _compute_loss.
- _get_per_token_logps_and_entropies selectively computes completion logits and is preferable to materializing full prompt logits.
- _compute_loss supplies clipping and optional KL machinery. The first arm should set beta=0, so no reference model is silently introduced.
- campaign_checkpoint.checkpoint_callback, campaign_control.control_callback, and campaign_eval.development_evaluator provide the persistence, control, and dev-panel seams described below.

Override the generation path because stock TRL is incompatible with PRM03. GRPOTrainer._generate_single_turn calls the processing class with text, max_length=self.max_prompt_length, left padding, truncation, and add_special_tokens=False (grpo_trainer.py around lines 1270-1280). It also sets a scalar self.eos_token_id and masks at the first occurrence of that one ID (around lines 1300-1305). That would lose manual-ID provenance, permit left truncation, and mishandle EOG 130073.

CampaignGRPOTrainer._generate_single_turn should:

1. Accept the prompt envelopes from the identity data collator and assert each stored ID list starts with BOS 0, is in range, and is at most 2048 IDs.
2. Left-pad those exact lists for the generation batch with attention ones. Do not call tokenizer encode, apply_chat_template, or any text truncation. prompt["text"] is for logs only.
3. Call model.generate with max_new_tokens=192, sampling settings recorded in the recipe, eos_token_id=[1, 130073], pad_token_id=1, and num_return_sequences=1 because TRL’s sampler already repeats each prompt G times. Keep use_vllm=False for the first arm.
4. Strip the exact prompt, retain the first EOG ID, and account generated IDs through the current trim_generated_sequence/account_generation_records logic. Do not use a substring stop. Keep noncanonical EOG and CONTROL-containing outputs so the reward can classify and count them; do not convert them into a valid EOS.
5. Return prompt and completion ID lists in the shape expected by the base _generate method. The base method may decode for logs, but the reward must decode the raw completion IDs with skip_special_tokens=False and must exclude the final terminal ID from body text.

If the pinned TRL minor version changes the private return shape, copy only the small surrounding adapter or override _generate_and_score_completions; do not copy the entire trainer without a source-hash review. A CPU fixture must prove that a tokenizer whose encode method raises is never called, manual BOS remains, a long prompt is never left-truncated, EOG trimming is exact, and masks cover only actual response positions.

Run the post-construction tokenizer/model contract check after GRPOTrainer.__init__, because trainer construction can repair PAD/EOS fields. Require tokenizer/model/generation config to report PAD 1, BOS 0, canonical EOS 1, and the expected vocabulary; separately retain native EOG list [1, 130073] in the generation recipe. Use positional masks and right-padding; token value 1 alone cannot distinguish PAD from a real terminal.

Use these initial config values for a single admitted arm: remove_unused_columns=False, max_prompt_length=2048, max_completion_length=192, per_device_train_batch_size=1, generation_batch_size=G, gradient_accumulation_steps=1, steps_per_generation=1, num_iterations=1, shuffle_dataset=False, dataloader_num_workers=0, beta=0, temperature and sampling values explicitly hashed, and one fixed num_generations=G. The current profile’s G=2 and G=4 are separate candidate arms; measure both under the lead-owned live probe, then select exactly one for a training identity. A G=4 arm may provide more group signal, while G=2 costs less memory; this is a resource hypothesis, not a quality result.

The old smoke explicitly selected TRL loss_type=bnpo (rl_smoke.py around line 837), while TRL 0.24.0 defaults to dapo. The first behavior-compatible recipe should set loss_type=bnpo explicitly, with scale_rewards=group, rather than relying on a default. A later GRPO/DAPO objective is a separate hashed treatment. Do not describe BNPO as paper-objective equivalence. Avoid steps_per_generation greater than 1 or num_iterations greater than 1 until stale-rollout/importance-sampling and checkpoint behavior have separate evidence.

The stock RepeatSampler uses a seeded torch generator and repeated groups (utils.py around lines 1672-1765), but its generator state is not a named campaign checkpoint field. For the first run use a CampaignRepeatSampler with selected-ID manifest order, contiguous G copies per prompt, no partial group, and a serialized sampler identity. A deterministic stateless epoch permutation can be added later only with a resume fixture. Do not combine the optional E1 ordered-difficulty scheduler from rl_smoke.py with the first protocol/mechanism baseline; it is a separate treatment arm.

## PRM03 reward

CampaignPRM03Reward should receive context, target_operation, and target_body_text as lists aligned with each generated completion. It should:

1. Reject any generated ID list that is noninteger, out of range, lacks terminal 1, ends in 130073, contains a native CONTROL ID before terminal, or exceeds 192. Return reward 0 and increment the corresponding failure counter.
2. Decode IDs before the canonical terminal using the pinned tokenizer with special-token skipping disabled and cleanup disabled. Parse the resulting exact wire text with parse_output; never search for a substring terminal.
3. Map parsed output to a semantic region: no-op means context.region_old (both [NO_EDIT] and an unchanged full-copy body); delete means []; replace means the parsed body lines. Derive the expected region from the out-of-band row label: no-op uses context.region_old, delete uses [], replace uses target_body_text.split("\\n") with empty internal lines preserved.
4. Set exact = 1 only when protocol parsing is accepted and semantic lines match exactly, including whitespace. Compute the legacy line-F1 only after protocol validity; preserve rstrip per line and trailing-empty-line removal from rl_smoke.py lines 131-149. Set reward = exact + 0.2 * line_f1; malformed or noncanonical output gets 0 rather than a shaping reward.
5. Record protocol_valid, exact_region, predicted_noop, operation, expected_operation, line_f1, reward, canonical EOS/noncanonical EOG/control/cap counters, family, package, row ID, and a bounded output hash. Use a per-attempt telemetry sink, never the legacy global STASH.

Required reward fixtures are exact replace, whitespace-only mismatch, no-op [NO_EDIT], unchanged full-copy no-op, nonempty delete, empty-range no-op, missing/duplicate terminal, early canonical EOS, terminal 130073, CONTROL ID before terminal, and literal marker text. Include a context-manifest stale/self-consistency fixture: a context record whose own content hash matches an old snapshot must still fail when its captured version/source identity is not current.

## Tokenizer and PEFT transition gate

The lead reported that a disposable RL memory probe stopped after generation with AttributeError("_flag_for_generation"), before any policy result. No policy memory, timing, gradient, or quality result from that probe is accepted.

The future live path must restore the pinned tokenizer contract once immediately after model loading, then assert the contract again after GRPOTrainer construction. It must not repeatedly call the tokenizer-restoration helper or mutate PAD/BOS/EOS fields between candidates. Generation must run inside the existing training-configuration guard; after generation, call the pinned FastLanguageModel.for_training transition exactly once before the differentiable policy pass. The implementation fixture must verify that this call reaches the actual wrapped model and that expected Unsloth generation attributes (including _flag_for_generation where present) are forwarded through the PEFT wrapper. If the attribute is absent or the transition fails, stop before policy accounting and emit a failed live receipt. Do not mask the failure by manually adding a private attribute or by silently retrying with changed tokenizer IDs.

## Identity, checkpoint, control, and development reads

Build a canonical seven-field identity because campaign_checkpoint.IDENTITY_FIELDS requires parent, tokenizer, renderer, data, source, policy, and schedule (campaign_checkpoint.py lines 19-20 and 68-73). The identity must include:

- parent: kind=merged_sft, future accepted manifest SHA, merged-weight SHA, base model revision, and accepted SFT recipe identity;
- tokenizer: revision, tokenizer JSON/config hashes, vocabulary, BOS/EOS/PAD, native EOG list;
- renderer: PRM03 schema/renderer/tokenization policy, terminal, and NO_EDIT;
- data: candidate or admitted rows SHA, context-manifest SHA, selected-ID SHA, ordered-ID hash, train split, and row count;
- source: protocol SHA, future trainer SHA, TRL version, and GRPO source SHA;
- policy: r16/a16, seven modules, 294 attachments, 25,116,672 trainable parameters, 2048/192/2240 caps, G, BNPO/group/beta=0;
- schedule: seed 3407, sampler ID, generation batch, steps_per_generation=1, num_iterations=1, full/evaluation/decision steps, and UTC deadline.

Do not substitute the raw Midtrain path or a disposable profile receipt in parent. Do not attach an RL adapter to an unidentified merged sidecar. The parent manifest must identify merge inputs, bytes, tokenizer contract, and SFT identity.

Reuse checkpoint_callback only with full-state settings. Its on_train_begin requires an evaluator unless explicitly pending, save_steps == full_every, save_only_model == false, and ignore_data_skip == false. A full checkpoint must contain the adapter/model files plus optimizer.pt, scheduler.pt, rng_state.pth, and trainer_state.json; seal_checkpoint records identity and an inventory hash. Supply an explicit sampler state containing selected-ID order, seed, epoch, current index, G, batch geometry, and sampler ID. Light adapter archives are useful for inspection, but they cannot serve as resume points.

Use a CPU fake-model interruption fixture: run to step k, save, restart from the full checkpoint, and compare adapter tensors, optimizer/scheduler/RNG state, trainer state, consumed selected IDs, reward records, and telemetry with an uninterrupted run. The fixture must run with steps_per_generation=1 and num_iterations=1; a future buffered rollout mode needs a separate resume proof.

Attach control_callback with a fresh attempt telemetry file, an explicit UTC deadline, stop/decision steps, and a reserve for full persistence. It emits train_begin, per-step duration/resources, trainer metrics, and train_end. Because the callback cannot interrupt a hung kernel, the launch recipe needs an independent supervisor timeout. A trainer exit is not a promotion signal.

Nominate development evaluation steps in the recipe and require those steps to coincide with full checkpoint boundaries. development_evaluator must remain on the predeclared dev panel and write case IDs, package/family denominators, prompt/target NLL token denominators, protocol-valid, exact, strict-no-op, false-suggestion, edit, cap, canonical-EOS, noncanonical-EOG, and CONTROL counts. Preserve RNG around evaluation. Never read the final set during training or use an unlisted dev case. A scheduled readout can be at steps such as [50, 100, ...] only if those values are explicitly in the actual recipe and fit the time budget.

## Future tests and acceptance gates

A future implementation packet should add experiments/training/test_campaign_rl_train.py and a CPU check script. It must pass:

- protocol reward rows listed above, including no-op/delete geometry and exact whitespace;
- fixed-ID generation with forbidden tokenizer encode, manual BOS, no truncation, native EOG accounting, and response-only masks;
- G=2 and G=4 contiguous sampler groups, no partial group, deterministic replay, selected-ID/hash identity;
- nested prompt envelope and all reward metadata surviving TRL Dataset/DataLoader with remove_unused_columns=false;
- post-GRPOTrainer PAD/BOS/EOS/config assertion and PEFT transition/attribute-forwarding fixture;
- all-equal reward group logging with nonzero frac_reward_zero_std;
- checkpoint interruption/resume evidence and full inventory;
- callback deadline/decision behavior and development denominator files.

Live acceptance remains blocked on the lead-owned RL profile against the future merged-SFT parent. That probe must report synchronized generation, recompute and backward timings, allocated/reserved/peak memory, prompt/generated token denominators, G=2/G=4, canonical EOS, noncanonical EOG, CONTROL and cap counters, finite loss, and finite nonzero adapter gradients. The current profile is a mechanism probe and makes no reward, GRPO update, or quality claim.

## Hazards that must remain visible

- All G rewards can be equal, especially for no-op rows, giving zero group advantage. Record zero-standard-deviation groups and no-op share; do not infer learning from a nonzero mean reward.
- G=2 has higher reward variance; G=4 costs more memory. Select using matched live measurements and keep G in the policy identity.
- A native EOG 130073 is a stopping event, not a valid PRM03 terminal. A CONTROL token can be hidden by ordinary decoding and still invalidate the raw ID sequence.
- EOS and PAD both equal 1. Position masks, not ID tests, define active response tokens.
- Line-F1 can give shaping credit to a malformed answer if computed before protocol validation; invalid protocol must return zero.
- BNPO, GRPO, and DAPO normalizations differ. The legacy smoke uses BNPO while TRL 0.24.0 defaults to DAPO; keep the first choice explicit and treat other objectives as separate arms.
- Reused rollouts with multiple optimizer iterations introduce stale-policy ratios and complicate resume. Keep them disabled in the first sustained run.
- Distributed reward gathering must preserve prompt groups. The first admitted run should be one device until cross-rank group alignment is tested.
- The 2240 managed context limit is a hard geometry check. Do not truncate target or evidence to make an overlong row fit.
- Current candidate-only rows and the Midtrain base are not training admission. Parent merge identity, context-manifest freshness, row admission, live resource lease, and final host/profile gates remain external prerequisites.
- Timings, memory, candidate-count choice, and feasible step budget depend on the lead-owned live probe and the eventual merged-SFT parent.

The old rl_smoke.py remains useful only as a control for the line-F1 shaping and optional E1 scheduler tests. Its dataset discovery, legacy zeta2 parsing, text re-tokenization, 480/170 filters, global STASH, and scalar generation assumptions cannot be carried into PRM03 training.
