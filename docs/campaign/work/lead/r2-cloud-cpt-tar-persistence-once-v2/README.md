# Corrected checkpoint-317 one-shot controller v2

The first v1 root attempt exited `precondition_failed` before creating authorization or writing remote state. Read-only observation showed PID 4588 has start tick 7757, cwd basename `s3_dc963855898b4fa1797b256f71b8711bd61b633c`, and exact argv `python -B <cwd>/payload/cloud_entry.py binding.root.json`. The run ID is carried by the verified binding, not by argv.

This fresh controller requires that exact PID/start-tick/cwd/argv tuple. It hashes both the submission cwd and staged run copies of `binding.root.json`, `cloud_entry.py`, `artifact_upload.py`, and `checkpoint_sidecar.py`. It then uses the byte-exact frozen v1 opaque transport under the versioned remote name `persistence-staging/tar_persistence_once_v2.py`.

Root creates a fresh authorization from the disabled template and runs:

```bash
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
/home/m0hawk/.local/share/uv/tools/anyscale/bin/python docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-once-v2/root_tar_persistence_once_v2.py once --authorization docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-once-v2/root-authorization-once.json
```

Require `remote_once_finished`, child exit 0, and validated tar and receipt revisions. `HF_TOKEN` must be present only in root's environment. This preparation does not authorize or execute the command.
