# Checkpoint-317 one-shot diagnostic controller v2

This replaces the blocked v1 controller. The upload child is base64 embedded with real newlines, and tests compile the exact nested child and remote sources. The controller retains the managed virtual-environment path from sidecar argv element 0. Before any signal, it runs a nonmutating import probe for `artifact_upload` and `huggingface_hub` under that exact interpreter. A failed probe produces a sanitized precondition result and leaves the sidecar running.

The controller additionally redacts HF tokens, bearer values, URL query strings, signed-query fields, and URL user information. It requires a root-created authorization record with `authorized: true`. It checks the exact owned job/run/cluster, PID 5498, start tick 18486, parent PID 4588, argv, source hashes, binding hash, checkpoint, and fresh diagnostic path. It then sends SIGTERM only to that sidecar and invokes `upload_checkpoint` exactly once with a 900-second bound. It never signals training.

Root command after review and authorization:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/m0hawk/.local/share/uv/tools/anyscale/bin/python -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cloud-cpt-persistence-diagnosis-v2/root_stop_and_upload_once_v2.py --authorization /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cloud-cpt-persistence-diagnosis-v2/root-authorization.json
```

No signal or upload has been executed by this preparation.
