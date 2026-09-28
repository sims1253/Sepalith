# CPT checkpoint-317 persistence diagnosis

The active sidecar is in a fast failed-upload loop. It read 2,613,640,706 bytes in 7.009 seconds. A complete attempt reads the 613,561,153-byte checkpoint four times before `upload_folder`, or 2,454,244,612 bytes. Local persistence is empty and the private prefix contains only the sentinel, so no checkpoint commit completed.

V4 discards the useful exception: the sidecar keeps only `error_type` in memory, emits nothing, and retries the same step every five seconds. The exact exception cannot be recovered through read-only inspection.

The prepared controller requires a root authorization record with `authorized` changed to `true`. It then verifies the exact job, run, PID 5498, start tick 18486, parent, argv, payload hashes, binding hash, and checkpoint path. It captures only the sidecar's HF token in memory, stops only that exact sidecar with SIGTERM, and runs the existing checkpoint uploader once with a 900-second bound. It writes a sanitized success or failure diagnostic on the node. It does not signal training or create provider resources.

Root command after reviewing and creating the authorization file:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/m0hawk/.local/share/uv/tools/anyscale/bin/python -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cloud-cpt-persistence-diagnosis-v1/root_stop_and_upload_once.py --authorization /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cloud-cpt-persistence-diagnosis-v1/root-authorization.json
```

No repair command has been executed by this worker.
