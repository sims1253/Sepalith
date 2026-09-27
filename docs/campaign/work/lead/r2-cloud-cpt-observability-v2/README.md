# CPT bootstrap diagnostic v2

The first paid CPT job, `prodjob_5nza57fm5lfcjfx379ehifckxj`, reached its allocated `g5.2xlarge` node and exited with code 1 before the private persistence sentinel. The provider retained only the wrapper command and `RAY_JOB_ENTRYPOINT_COMMAND_ERROR`; it did not retain the redirected bootstrap log. `runtime-failure-review.json` records that evidence. It does not establish which bootstrap phase failed.

This packet runs the same pinned image and the same seven bootstrap commands as the failed source closure, in the same order. It adds one JSON `START` event and one `END` or `EXCEPTION` event per phase. Failure output includes at most the final 4,096 bytes of the phase log, with Hugging Face token-shaped values redacted. Bootstrap subprocesses receive an environment with every key containing `TOKEN` or `SECRET` removed. The parent diagnostic process uses the private token only after contract validation to commit one sanitized `diagnostic-terminal.json` record to the run-specific private artifact prefix.

The diagnostic contract requires:

- one `g5.2xlarge` head and no workers;
- the exact original image digest;
- a 600-second provider timeout and a 900-second independently armed watchdog;
- zero retries;
- `training_authorized=false` and `input_staging_authorized=false`;
- a fresh 32-hex run ID and private `r2-cpt-bootstrap-diagnostic/<run-id>` prefix;
- a root diagnostic admission that binds this payload manifest;
- an armed-watchdog receipt that binds the same run ID and deadline.

The proposed listed compute exposure is 0.22725 AC for 600 seconds at 1.3635 AC/hour. The full 900-second watchdog envelope is 0.340875 AC. Both fit inside the existing USD28 cloud reservation. Provider accounting can settle after release.

The frozen candidate payload is `payload-manifest.json` SHA256 `53e1a7f4049effb79e81b15e13c70d50bf616a180c3af099a886276e716a2aa9`. No admission, watchdog, provider submission, input staging, training, retry, or new paid allocation was performed while preparing it. Root must generate the binding, admission, and watchdog receipts and must review the exact provider command before a single diagnostic submission.

Run the CPU-only checks with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B -m unittest -v test_bootstrap_diagnostic.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B -m unittest -v test_cloud_cpt_packet.py
```
