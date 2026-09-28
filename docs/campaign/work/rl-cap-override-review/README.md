# RL-08 d CUDA cap review

Status: `prepared_pending_root_review`. This packet is CPU-only source and metadata review. It did not launch CUDA, SSH, a model, or a framework, and it did not read model weight bytes or alter the existing checkpoint, recipe, EXEC, or state.

## Finding

The current frozen entry does not support a runtime-only cap override.

- In `campaign_rl_entry.py` (`3d3a5626a9fbe19ccbe7d9614497dfd2e58d52431f811489d072852015f9230d`), `_cuda_memory_fraction` accepts values through the already-coded hard limit `0.80` (lines 75-76 and 119-130).
- `_validate_entry_contract` validates `recipe.cuda_memory_fraction` and `identity.policy.cuda_memory_fraction`, then rejects any unequal pair (lines 504-513). Its runtime packet contains the single validated value at line 516.
- `preflight_entry` places that packet in the admission at lines 729-731 and 766-779. `run` takes `admission["runtime"]["cuda_memory_fraction"]` at line 1016 and passes it directly to `_load_live_model` at lines 1034-1038. `_configure_cuda_allocator` calls `set_per_process_memory_fraction` and records the cap at lines 856-873.
- There is no CLI, environment, or separate receipt input that can replace this value. `campaign_rl_launch.py` (`fa245326052bfae2997d91d80bbf8f45c055d35ffcd0b4b2b7d4440045ffb36a`) only preflights the recipe and supervises the command (lines 81-115).

The current d recipe (`6ee66fb910403f05e70520cda9b1649b58ba121663a37bb5185fb26e5356728b`) has both values at `0.75`. Editing only the top-level value to `0.80` fails the equality gate. Editing `identity.policy.cuda_memory_fraction` changes the canonical identity from `48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2` to `ebf964b456eff1c2df2b3064277afe346db9a358eb47b5a902a203822b5b4818` for this exact d identity.

`campaign_checkpoint.py` (`64202f4e5b88c94a74794442be3a4955f07d195079c9d64bbcff2850dbadd0ba`) defines `policy` as one of the checkpoint `IDENTITY_FIELDS` (line 19); `verify_checkpoint` compares the entire manifest identity to the requested identity (lines 68-73 and 114-123). Therefore the existing full100 checkpoint cannot be resumed under an unannounced `.80` identity.

## Observed boundary

The exact supervision log is:

`/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-d-host-supervision/process.log`

It is 13,219 bytes, SHA256 `4a5ebe848d61d46424fc4e104f0e1947311b7ef31ebc6c11a7dd20a51626c0af`. Its terminal traceback is at line 76, after the process reached the backward path. The failed allocation was 3.56 GiB with GPU total capacity 31.84 GiB, physical free 9.58 GiB, PyTorch cap 23.88 GiB (`0.75`), allocated 16.19 GiB, and reserved but unallocated 4.28 GiB. The reported failure is allocator-cap/OOM; the run did not terminate from the host guard or a driver event.

The exact host supervision stream is 24 lines, SHA256 `2f1a2fe54d930d44545d85e89785abd68870c0fb6805b70288bb9f1701d02b3c`. Its last sample was 9,299 MiB available. Earlier samples contain page reads; the terminal samples contain no driver events. This supports a CUDA allocation failure at the observed boundary, while it does not prove that a `.80` cap will survive every future kernel workspace or length mix.

The frozen RL data geometry is prompt 2,048 plus completion 192, context 2,240 (`campaign_rl_data.py` lines 48-50), with d policy `candidate_count=4`, `per_device_train_batch_size=4`, and `gradient_accumulation_steps=8`. The observed 3.56 GiB request is the only concrete allocation bound in this review; it is not a proof of the worst allocation for every 2,240-token mb4 backward pass.

Arithmetic from the observed snapshot is favorable but conditional:

- `.80 * 31.84 GiB = 25.472 GiB`, adding about `1.592 GiB` to the `.75` cap.
- Using the logged allocated plus reserved-unallocated values, the apparent `.75` headroom is `23.88 - 16.19 - 4.28 = 3.41 GiB`, about `0.15 GiB` below the 3.56 GiB request.
- The corresponding `.80` apparent headroom is `25.472 - 16.19 - 4.28 = 5.002 GiB`, about `1.44 GiB` above that request.

These calculations are a feasibility screen, not a capacity guarantee. Fragmentation, temporary workspaces, and a different 2,240-token batch can consume the margin.

## Recommended route: explicit new derived checkpoint container

Use the existing strict validators and create a fresh, separately named full100 checkpoint container. Preserve the original full100 checkpoint byte-for-byte and leave its `.75` identity intact. The new container may change only the declared resource identity metadata:

1. Copy the complete full100 checkpoint into a new exclusive directory. Require the copy to be full and at step 100 before changing metadata.
2. Derive a new recipe identity by changing only `identity.policy.cuda_memory_fraction` from `0.75` to `0.80`. The d identity becomes `ebf964b456eff1c2df2b3064277afe346db9a358eb47b5a902a203822b5b4818`; all parent, tokenizer, renderer, data, source, schedule, optimizer geometry, sampler, and prompt fields remain unchanged.
3. In the new copy, change only the identity object in `campaign-state.json` and `campaign-manifest.json`; recompute the `campaign-state.json` inventory entry and write the new manifest. Keep every optimizer, scheduler, RNG, adapter, trainer state, sampler, and other file byte/hash unchanged. The original manifest/state remain the source of the migration comparison.
4. Bind old identity SHA `48028a...a0f2`, derived identity SHA `ebf964...4818`, source manifest SHA `2fa678487c5a760a2d04b798fc29787cd300752e7b6bd3445fdb8a9b09399ebd`, source state SHA `930385c7715c91d3be3e9a570ee8de752d12f04ef1e6db55b10aab05900b9ed8`, step 100, source cursor 800, selected index 425, and consumed rows 25,600 in an explicit migration receipt. The receipt must list the exact allowed metadata delta and every unchanged file inventory hash.
5. Derive a new recipe from d with both top-level and policy cap fields set to `0.80`, and `resume_from` pointing to the new container. Do not rewrite d. The new recipe is intentionally a new identity and must not be reported as “same identity” continuation.
6. Run the pure validator and strict `verify_checkpoint(new_container, derived_identity, require_full=True)` before launch. The live preflight must pass `_validate_resume_checkpoint`; its allocator audit must report `.80`. Root still owns the guarded CUDA launch and the decision to stop or continue.

This route requires no production source change and keeps `campaign_checkpoint`'s exact identity comparison meaningful. It preserves optimizer/data/RNG/source-cursor bytes, but it does not preserve the old identity hash; that identity change is the point of the explicit migration. No existing artifact may be modified in place.

A runtime-only override is a possible later source change, but it is not the smallest safe action for this retry. If selected instead, add an exact top-level `runtime_resource_override` object to the recipe and extend `_validate_entry_contract` to require: schema `sepalith.rl.runtime-resource-override.v1`; identity SHA equal to the canonical unchanged `.75` identity; `from_cuda_memory_fraction=0.75`; `to_cuda_memory_fraction=0.80`; a lowercase review-receipt SHA; and a bounded nonempty reason. Return both configured identity cap and effective runtime cap in `entry-preflight.json`; pass only the effective cap to `_load_live_model`; keep checkpoint verification against the unchanged identity; and reject all other fractions, identity hashes, fields, or missing bindings. This needs coordinated edits to the frozen entry source and its tests, so it remains an unintegrated candidate in this packet.

## CPU checks

`test_runtime_cap_override_candidate.py` exercises the unintegrated contract fixture: default `.75`, valid explicit `.80`, wrong identity, nonmaximum/out-of-range values, wrong source value, equality mismatch, exact schema/receipt keys, identity migration, and wrong migration source/target. Result: 9 tests passed in 0.002 seconds.

`source_contract_check.py` parses the frozen entry source and checks the cap equality gate, the `run` to `_load_live_model` propagation, and allocator setup. Result: pass.

The two existing frozen CPU boundary tests also passed with no framework imports or model reads:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<frozen training source> /usr/bin/python3 -m unittest \
  test_campaign_rl_entry.EntryBoundaryTests.test_cuda_memory_fraction_is_required_bounded_and_identity_bound \
  test_campaign_rl_entry.EntryBoundaryTests.test_entry_binds_4k_model_load_and_frozen_rl_context_caps
..
Ran 2 tests in 0.004s
OK
```

The candidate files are deliberately unintegrated review fixtures. Their tests do not validate a live CUDA allocation or a real checkpoint copy.

## Open gates

The `.80` retry is feasible from the one observed cap miss but remains unproven. Root must create and seal the new derived checkpoint container, verify that only the declared metadata identity changed, pass pure and strict checkpoint preflight, confirm the live allocator audit, and apply host/driver supervision. No quality or promotion claim follows from this resource retry.
