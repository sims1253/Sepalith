# Checkpoint-317 one-shot controller v3

The v2 controller exited before any remote write because it assumed binding and payload copies existed under the generated run directory. Read-only inspection proved those paths absent. The authoritative active binding and payload remain in PID 4588's exact Ray submission cwd; the run directory contains artifacts, the managed venv, and checkpoints.

This fresh controller gates PID 4588/start tick 7757, its exact cwd and argv, and the binding plus three payload hashes in that submission cwd. It writes the byte-exact opaque transport under the new name `persistence-staging/tar_persistence_once_v3.py`. Capped redacted SSH stderr is retained if another wrapper exception occurs.

Root creates a fresh authorization from the disabled template and runs:

```bash
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
/home/m0hawk/.local/share/uv/tools/anyscale/bin/python docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-once-v3/root_tar_persistence_once_v3.py once --authorization docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-once-v3/root-authorization-once.json
```

Require `remote_once_finished`, child exit 0, and validated tar and receipt revisions. The template remains unauthorized; root owns execution.
