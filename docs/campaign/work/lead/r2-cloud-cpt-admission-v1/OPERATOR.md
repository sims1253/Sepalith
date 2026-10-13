# Root operator sequence

This packet is preparation only. Do not upload or submit until the root CPT recipe review emits an affirmative receipt.

1. Re-run `verify_inputs.py`, `build_transport_manifest.py`, and the CPU tests. Their hashes must remain the frozen values in `launch-plan.json`.
2. Run `build_payload_manifest.py` after the final source review. Copy its SHA256 into a new binding derived from `binding.template.json`; do not edit the template after freezing.
3. Upload the frozen inputs only after the affirmative root receipt:

   ```bash
   python3 upload_private_inputs.py \
     --root-admission ROOT_RECEIPT.json \
     --expected-root-admission-sha256 ROOT_RECEIPT_SHA256 \
     --execute
   ```

   The script reads `HF_TOKEN` from its process environment, rejects an existing prefix, uploads every byte-pinned file, commits `transport-manifest.json` last, and writes a sanitized receipt. Bind its immutable final revision as `input_revision`.
4. Fill a fresh UUID32, `artifact_prefix=r2-cpt/UUID32`, current timestamps, the root admission hash, the input revision, and the frozen payload-manifest hash. Keep `admitted=true`, USD28, 28,800 provider seconds, 29,100 watchdog seconds, and `max_retries=0`. The first attempt has `resume=null`.
5. Start `payload/root_arm_watchdog.py` before submission with a zero SHA placeholder for the unavailable armed receipt. Once `armed.json` exists, hash it into the job's final binding. The watchdog continues by unique name and switches to job ID after discovery.
6. Place the final files as `binding.root.json`, `root-recipe-admission.json`, and `watchdog-armed.json`, then execute `submit_with_private_environment.py`. It rechecks the live watchdog, payload/root receipts, remaining provider window, live credits, and conservative USD60 ceiling before calling the SDK. It submits the packet directory once with `max_retries=0`; the marker forbids another attempt. `job-template.proposed.yaml` documents the equivalent config. Allocation failure is terminal evidence; do not blind-retry.
7. Observe provider state and private checkpoint receipts. The sidecar uploads only complete, manifest-verified full checkpoints at multiples of 317. A successful provider exit is insufficient: independently read back the final checkpoint and require step 1902, status `schedule_complete`, sampler cursor 30,432, and exact frozen identity.
8. If interrupted, build a new binding whose `resume` points to a committed `checkpoint-receipts/checkpoint-N.json` at N in 317, 634, 951, 1268, or 1585. The stager downloads every inventoried optimizer/RNG/model file and creates the exact `external_interruption` resume binding. Never resume from adapter-only output.

The stager reconstructs the frozen absolute paths with `sudo install` inside the ephemeral Anyscale container. This is necessary because the immutable merged-parent manifest binds those exact paths. It rejects paths outside the three frozen campaign roots, never rewrites the parent manifest or trainer source, and leaves the trainer's data, validation, parent, tokenizer, and source identities unchanged.
