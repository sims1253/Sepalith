# C20. Supervised gold for medium and long finish blocks

The user approved this pilot on November 21, 2026, after three models agreed on the design. Start from the adopted [C19 checkpoint 100](../c19/RESULTS.md). Its frozen exec hold-out all-pass rate was 0.399. The gain came from rename propagation. Finish-block all-pass was flat at 0.219, with medium at 0.050 and long at 0.003. In the prior census, all eight samples failed on 89% of 9–16-line targets. C20 tests whether gold targets provide the missing training signal.

## Frozen validation and training data

Freeze validation first, from the uncapped task file pinned by the [C18 pool manifest](../c18/C18-exec-pool.json). Include all gated tasks from the exact functions and packages in C18 validation. Expand the training split the same way, with the exact functions and packages already in C18 TRAIN. Check both function and package identities against each other and against the C18 hold-out identity manifests. Keep the C18 suite and gate fingerprints. The [validation manifest](../c20/validation-manifest.json) records 1,967 tasks, including 994 finish blocks of 3–16 target lines across 130 functions. The original panel had 403 tasks. The 16 validation packages are the same.

Draw one epoch of 4,800 rows using seed 3407. Sort candidates by ID before sampling. Each batch of 16 has the following composition.

| Rows | Source |
| ---: | --- |
| 8 | C18 TRAIN gold finish blocks of 3–16 target lines |
| 2 | C18 TRAIN gold finish blocks of 1–2 target lines |
| 2 | C18 TRAIN gold rename propagation |
| 3 | No-op rows from the pinned arm A SFT TRAIN data |
| 1 | Plain causal continuation from the pinned C18b CPT-text replay |

Keep the C04 no-op share and the maximum of three exposures per no-op. Each row in the frozen mix appears once. The CPT replay uses whole chunks with 64–160 target tokens, with any original EOS preserved. This keeps the plain-text loss from dominating the supervised tokens in a batch. All longer exec training cuts come from the pinned C18 build and pass its per-cut gates. Exclude any no-op or CPT replay from a validation or hold-out package. Use a 2,048-token training context and admit only rows that fit whole. SYNTH v3.1 and the sealed DAT-07 and DAT-08 sets do not enter C20.

The [mix manifest](../c20/mix-manifest.json) pins the seed, inputs, output row file, draw schedule, and length profile. The [pilot recipe](../c20/pilot.json) pins both manifests. The preparation commands use the frozen paths on the campaign drive. Run the build only for a fresh freeze.

```bash
export PYTHONPATH="$PWD/packages/sepalith/src"
C20_PY=/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python
nice -n 10 "$C20_PY" -m sepalith.training.sft.c20.recipe build
"$C20_PY" -m sepalith.training.sft.c20.recipe bind
"$C20_PY" -m sepalith.training.sft.c20.recipe verify
```

## Pilot and selection

Start from C19-100 weights. Initialize the optimizer and sampler from scratch. Use C04's full-weight training contracts through the C20 trainer with micro-batch 1, gradient accumulation 16, bf16, gradient checkpointing, and the exact PRM03 target mask for edit rows. Use Aurora Mix with hidden LR 2.5e-6, side LR 2.5e-7, no weight decay, 32 warmup steps, and cosine decay to 10% of peak. Keep FP32 optimizer state in host RAM and stage one parameter's state at a time on the GPU. The trainer saves full state, including the draw cursor and RNG, every 100 steps and at the 16:40 deadline after the update in flight. Keep every durable checkpoint.

Run enlarged validation, DEV75, and CPT 2K NLL on C19-100 and after steps 100, 200, and 300. Use bf16 greedy generation, batch 32 with the dynamic cache, a 1,024-token cap, and stop IDs 1 and 130073. Compare each save against C19-100 on the same panels. A save qualifies for selection when all four conditions hold.

| Metric | Required result |
| --- | --- |
| Validation all-pass on 3–16-line finish blocks | Gain of at least 2.0 percentage points with CI95 lower bound above zero |
| Validation rename all-pass | Drop of at most 1.0 percentage point |
| DEV75 nofp | Rise of at most 2 cases |
| CPT 2K NLL | Rise of at most 0.03 |

Pair by task ID and bootstrap by function with 2,000 resamples. Select the qualifying save with the highest all-pass rate on the 3–16-line subset. Break ties by earlier step. Run DEV250 only at the selected save and require `score250 >= 108`. If it fails, or no save qualifies, stop training at C19-100. A failed DEV250 does not trigger selection of another save. A crash, nonfinite loss, wrong hash, or R7 disk failure stops the job for the user.

## Final evaluation

Run one final job after a passing pilot. Use the C19 generation and scoring helpers through the C20 evaluator. Keep the frozen hold-out and use the labels `c19-100` and `c20-selected`. Use the SHA256-pinned completions from C19's `final-step-100/completions-c19-selected.jsonl`. The job resumes generation for the selected model from any completed batches. The hold-out does not enter training, gates, or checkpoint selection.

Adopt the selected save only if the paired, function-clustered 95% CI for overall all-pass difference has a lower bound above zero, DEV250 is at least 108, and CPT NLL rise is at most 0.03. Use 2,000 resamples. Report the medium and long finish-block buckets descriptively. Keep C19-100 if the final decision fails.

## Night jobs and resources

Jobs bind cwd, PYTHONPATH, and the SFT interpreter to `/home/m0hawk/sandbox/Sepalith`. Merge this card before enqueueing so the source hashes match. Exec scoring uses rig R 4.6.1 at `/home/m0hawk/.local/share/rig/r/4.6.1`. The smoke requires one training update, a durable save, one resumed update, DEV75, CPT 2K, and greedy exec scoring on 24 validation tasks. Before measuring the pilot baseline, verify the smoke receipt against the bound smoke recipe, both full checkpoints, and the step-2 gate evidence. The final job requires the pilot and skips evaluation if the pilot keeps C19-100. When the queue requests a stop, the pilot asks the trainer to save at the next update boundary. The driver waits for the save, verifies the full checkpoint, and exits with code 75 so the queue can resume the job.

| Job | GPU window estimate | Resumable |
| --- | ---: | --- |
| [Smoke](../c20/jobs/C20-sft-finish-smoke.json) | 45 minutes | No |
| [Pilot](../c20/jobs/C20-sft-finish-pilot.json) | 12 hours | Yes |
| [Final hold-out](../c20/jobs/C20-final-holdout.json) | 4 hours | Yes |

These are planning estimates pending smoke measurements. The C04 and C19 receipts come from a 32 GB GPU. C20 caps the PyTorch allocator at 14,500 MiB and checks the training peak in smoke. The cap leaves space on a 16 GB card for CUDA runtime and non-allocator use. The job still needs sufficient free VRAM and host memory. Keep at least 150 GB free after reserving 19 GB for the next full checkpoint. The smoke and three planned pilot saves use about 95 GB at that estimate, plus 19 GB for each additional deadline save. The CPU preflight checks space for seven saves.

## November 21 pilot result and DEV250 exception

Step 300 passed the pilot validation gates. All-pass on 3–16-line finish blocks rose 2.0 percentage points versus C19-100, with a 95% CI of [0.002, 0.0407] in absolute rate. Rename all-pass did not drop, DEV75 had zero additional nofp cases, and CPT NLL rose 0.0021. DEV250 scored 107 against the required 108, with 72 exact edits and 35 correct no-ops. The pilot recorded `keep_c19`.

On November 21, 2026, the user approved a one-time DEV250 exception for step 300 because the one-point miss was in no-op cases, their lowest-priority error class. The receipt pins the pilot summary, DEV250 result, and checkpoint binding. [DEV250 exception receipt](../c20/dev250-exception.json)

Queue the resumable exception job from the main checkout after the job lands. It writes its verdict under `C20-final-holdout-exception/final-step-300-exception` in the evaluation gates directory. Adopt step 300 only if the paired hold-out all-pass difference versus C19-100 has a 95% CI lower bound above zero and CPT NLL rise is at most 0.03. Otherwise keep C19-100. [Exception hold-out job](../c20/jobs/C20-final-holdout-exception.json)
