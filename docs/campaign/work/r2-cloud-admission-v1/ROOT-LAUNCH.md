Recommend one Anyscale fresh-Midtrain target-only task-SFT control. It answers whether the new CPT stage helps when task data, sample order, loss, optimizer and step horizon are matched. Use the frozen R2 task trainer, not the old cloud SFT entrypoint or rejected theta0 corrective recipe. The new task mixture is still unbound, so this packet is not launch admission.

Use one fixed `g5.2xlarge` head, no workers, no job queue and zero retries. This supplies one A10G and 32 GiB host RAM; the historically used `g5.xlarge` has only 16 GiB host RAM. The larger host is a proposed margin for full checkpoint and data materialization, not measured proof of fit. AWS documents the [G5 shape](https://aws.amazon.com/blogs/aws/new-ec2-instances-g5-with-nvidia-a10g-tensor-core-gpus/). New allocation availability remains untested.

The first useful target is full checkpoint 250, using the unchanged 1000-step cosine horizon. Set mandatory_stop_steps=[250], fresh optimizer, seed3407, BF16, LoRA32/64, LR0.0002, effective batch16 and sequence4096. Select microbatch2/accumulation8 provisionally and bind the same geometry to the comparison arm if exact batch matching is required. The existing target-only gate validates the actual first accumulation group before an update; it is not a worst-case 4096-token backward-memory test. Root must profile complete forward/backward/checkpoint behavior and stop on OOM, invalid target loss, or an upload/deadline failure. Do not silently reduce context, truncate targets, change dtype or resume an old parent to make it fit.

Bind these exact inputs before allocation:

1. Fresh Midtrain weights identity `38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad`, revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`, plus original config/tokenizer files. Tokenizer JSON SHA `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`. These are metadata pins reused from root's current CPT recipe; no weight bytes were read here.
2. The admitted output of `work/r2-task-mixture-v1`, its complete target-only rows, source/context provenance and exact draw schedule. Preserve all reserved validation groups and the complete <=192-token target rule. Mixture source files are active and were not read here.
3. `work/r2-task-trainer-preparation-v1/source-manifest.json` SHA `670d87b52724317782e8f7fc03376926e6ae6115f7b185b75cb6df3ad62e4bf4`. The task adapter SHA is `1d74e9f6e8310eac3a8d0922848fc892c200d048c0ecf290fb74ce3758ba5965`. Bind relocated files and recipe paths to a new cloud source identity. Do not edit this accepted capsule.
4. A pinned image/bootstrap and the recorded Python3.10 package versions in `local-training-package-metadata.json`. The old cloud requirements match the named installed versions but omit an explicit `unsloth_zoo==2026.8.12` pin and do not constitute a complete wheel/source closure. The old image tag `anyscale/ray:2.57.0-py311` alone is insufficient. Existing setup machinery took about six minutes historically; reserve 20–30 minutes for first environment and private model/data staging. No new throughput or setup timing was measured.
5. Root's explicit evaluator binding. The existing `campaign_eval:development_evaluator` is an HF diagnostic only and can run with the same corrected DEV75 at cap192. Native postmerge Q8 DEV remains required for scientific acceptance. A route label alone must not imply that the cloud process is running native evaluation.
6. A new private output prefix in the currently accessible private model repository `scholzmx/sepalith-lora`, or an independently verified private object-store location. Before training, upload a small random sentinel and read it back by exact hash from a separate root client. Upload the full250 checkpoint, adapter, manifests and logs before success; record remote commit/object IDs and verify their checksums after cluster exit. A local checkpoint or SUCCEEDED status alone does not prove persistence. Preserve an incomplete-run artifact on failure without treating it as a full checkpoint.

The template intentionally has no executable payload yet: `root-bound-entry.sh`, the pinned environment and admitted mixture must be supplied by root. Its actual training command must call the reviewed launcher:

```sh
PYTHONPATH="$PAYLOAD/source/experiments/training:$PAYLOAD/source/packages/sepalith/src" \
  "$PYTHON310" -B "$PAYLOAD/source/experiments/training/campaign_launch.py" \
  "$PAYLOAD/recipe.json" --receipt "$RUN/supervision.json"
```

The root entrypoint must verify all staged input hashes, keep setup inside a finite timeout, run this supervised command, upload/check artifacts, then exit. It must not source a shell startup file or embed the HF token in a generated kernel or verbose job configuration. The old uploader proves a historical private-adapter path, but it does not upload this trainer's full checkpoint layout unchanged.

Bound the first allocation to two hours from watchdog arming, provider timeout6900s, checkpoint/upload reserve600s inside that deadline. Setup counts toward the two hours. Never extend past Monday2026-09-14 06:15UTC without a new root decision; the watchdog enforces that cutoff and a six-hour maximum. The initial six-hour limit is a validator ceiling, not this proposal's allocation size. No retry or replacement follows a failed lifecycle or price check.

Root launch sequence after source/payload/cost admission:

```sh
# RUN_NAME must be sepalith-r2-control- followed by a fresh lowercase UUID without hyphens.
# DEADLINE_UTC is at most two hours from now and before Monday06:15UTC.
nohup python3 -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-cloud-admission-v1/deadline_watchdog.py \
  --name "$RUN_NAME" --deadline-utc "$DEADLINE_UTC" --output "$GUARD_DIR" \
  > "$GUARD_LOG" 2>&1 < /dev/null &
# Confirm GUARD_DIR/armed.json and its live PID before the one submission.
/home/m0hawk/.local/bin/anyscale job submit -f "$BOUND_JOB_YAML" \
  --cloud 'Anyscale Cloud' --max-retries 0 --timeout-s 6900
```

Use the same unique name in the YAML and guard. Save the returned job ID without verbose status. The watchdog can resolve the unique name if the submit response is lost; after observing an ID it terminates only that ID. Each API command is bounded20s. It requests termination at the deadline or after repeated status failures, then allows120s for terminal confirmation. It reports unconfirmed termination explicitly. API outage, host loss and provider scheduling can prevent confirmation; the provider job timeout is the independent backstop. No queue is used because timing out a queued job need not terminate its queue cluster. [Anyscale job controls](https://docs.anyscale.com/reference/sdk/job) distinguish a job timeout from a CLI wait timeout. Eight synthetic watchdog tests pass; actual provider termination has not been exercised in this refresh.

Current cost evidence: account credit remains $95.433820846, with no new spend and zero visible GPU nodes. The current published A10G component is1.3635AC/hour, so the two-hour GPU component is2.727AC; that is not an account-specific all-in USD quote. [Pricing](https://www.anyscale.com/pricing) and [usage guidance](https://docs.anyscale.com/administration/billing/usage-dashboard) do not turn estimates into a hard billing stop. Proposed root envelope: node compute at most$8/hour for two hours plus$12 setup/storage/transfer/cleanup reserve = $28, below the existing$60 all-in ceiling. Root must accept a current quote/coverage within that envelope before launch; no additional user billing evidence is needed. The provider has no observed hard dollar-stop. A newly quoted g5.2xlarge price or uncovered fee above the envelope blocks this one allocation.

Kaggle fallback: current quota is30GPU-hours and20TPU-hours, zero used, refresh September19UTC. All26 accessible saved Sepalith kernels have terminal states; one unrelated old notebook status is inaccessible, so this is not an exhaustive interactive-session inventory. The existing repository dataset is confirmed private, and historical CPU output files remain listable. Saved output under `/kaggle/working` has a documented20GB cap; [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) describes persistence and privacy. New exact-input dataset staging historically took15–20minutes, plus3.5minutes of package installation; allow20–35minutes for a new portable payload. CLI2.2.4 supports push timeout and explicit accelerator selection. Current authenticated quota does not prove immediate physical stock or a BF16-capable shape. The frozen trainer requires BF16; T4/P100 do not supply it. A standalone FP16 dense-Llama auxiliary control could be prepared separately, with GradScaler/optimizer/tokenizer and finite-gradient checks, but would be a precision-confounded arm and needs new source admission. Do not run the frozen BF16 control on Kaggle unchanged or use the old GDN failure as proof that all standard models fail. No ready draft-model recipe was found or fabricated.

Azure: exactly two quota refreshes, North Europe and East US. A10/A100/H100 candidate-family quotas remain zero. Nonzero legacy NC/NV allowances do not change the selected BF16 training route. No further Azure sweep or allocation is proposed.
