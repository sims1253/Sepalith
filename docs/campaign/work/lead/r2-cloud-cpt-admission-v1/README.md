# Cloud CPT admission preparation

The Anyscale A10G route and a CPT-specific cloud packet are prepared within the existing budget, but neither private staging nor launch is admitted yet. The remaining-CPT producer froze its corrected schedule during this review. `launch-plan.json` binds the producer receipt, transport closure, and cloud source closure. `OPERATOR.md` gives the root-only review, staging, admission, launch, persistence, and continuation sequence.

Run the read-only observations with:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 docs/campaign/work/lead/r2-cloud-cpt-admission-v1/live_probe.py
```

After the producer and root freeze the recipe, rerun:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 docs/campaign/work/lead/r2-cloud-cpt-admission-v1/verify_inputs.py
```

The verifier checks all 32 recipe inputs and cross-checks the data manifest, schedule, and coverage. Its pass result establishes byte closure only; it does not authorize private upload or paid launch.

The preferred provider is Anyscale hosted `g5.2xlarge` A10G with an eight-hour timeout, no retry, and a USD28 incremental reservation. The current account balance is USD94.094521118 and the conservative remaining room under the campaign's USD60 ceiling is USD54.094521118. Published A10G compute is 1.3635 Anyscale credits per hour. The fleet endpoint reports zero active nodes; it cannot establish unallocated regional capacity.

The 5.58 GB input closure must use a new prefix in the already-probed private Hugging Face repository. It exceeds the documented 500 MiB Ray job working-directory submission limit. Only the download/upload subprocesses receive `HF_TOKEN`; training receives paths and hashes after the credential-bearing process exits. No upload, provider submission, GPU allocation, or paid action was performed here.

The source packet is `payload-manifest.json` SHA256 `9571d35502e5efcb2867e5d85efb31cded552b55fc6464de6395476c6064dc61`. The private input map is `transport-manifest.json` SHA256 `a9afe5d58034b56e36abf83a0c1d169ec578b79cc215daa32603ccd210e7045b`. The stager reconstructs the immutable original absolute paths inside the ephemeral container because the merged-parent manifest binds them. It rejects every other path root and changes only output/deadline/launch fields in the staged recipe. The trainer source and parent manifest remain byte exact.

The sidecar uploads each complete full checkpoint at 317-step cadence and commits a resume inventory after remote metadata and manifest readback. A later binding can resume only steps 317, 634, 951, 1268, or 1585 with exact optimizer, RNG, identity, schedule, and cursor state. Terminal success requires step 1902 and cursor 30,432. The corrected watchdog uses job names before discovery and job IDs afterward, with an absolute 29,100-second limit.

Kaggle is not currently actionable. Authenticated API access works, but the account has only `m0hawk/sepalith-repo`, no CPT artifact dataset, and the API has no remaining-GPU-hours endpoint. Kaggle documents a weekly quota and possible queue, so the user interface must establish quota before that route can be admitted.
