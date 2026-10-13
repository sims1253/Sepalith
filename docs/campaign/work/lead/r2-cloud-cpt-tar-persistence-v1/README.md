# Cloud CPT opaque tar persistence v1

This fresh packet repairs checkpoint persistence for owned Anyscale job `prodjob_neqwkumat4i3rii5yl8af241ks`, run `4f899bbc6e9d46c0a88985d64e6d40e2`. The failed V4 sidecar used `HfApi.upload_folder`, which parsed the checkpoint `README.md` and rejected its intentionally local `base_model` path before uploading any checkpoint bytes.

`tar_persistence.py` treats a completed checkpoint as opaque files. It verifies every manifest-listed file and requires the only additional file to be the original `campaign-manifest.json`. It then builds and reopens `run/persistence-staging/checkpoint-N.tar`, hashes every tar member against the original file inventory, uploads the single tar with `upload_file`, and validates the remote LFS SHA-256 and size. The original checkpoint, its README, and its manifest are never edited. Scratch and receipts stay outside `run/artifacts`, so V4 terminal artifact upload cannot duplicate the tar.

The monitor covers exactly steps 317, 634, 951, 1268, 1585, and 1902. Each step receives at most three attempts with 30- and 120-second backoffs. Exhausted steps are recorded once and abandoned; the monitor cannot re-enter an infinite rehash loop. It also exits at the binding deadline or after the existing `run/stop-sidecar` signal and a final scan. All failures include a redacted message in `persistence-staging/events.jsonl` and the terminal receipt.

The controller requires the exact running provider job, run directory, cloud-entry PID 4588, V4 binding SHA, and three active V4 source hashes. It uses the existing managed venv. It accepts a local `HF_TOKEN` only through process stdin and forwards it only in the child environment. It never writes the token or includes it in argv/output. The stopped PID 5498 sidecar is not restarted. V4 cloud entry will later observe that old sidecar's signal exit and is expected to report provider failure after training; this packet only preserves checkpoints.

Root execution is deliberately split. First, copy the disabled one-shot authorization to a fresh file, set only `authorized` to true, review it, then run:

```bash
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
/home/m0hawk/.local/share/uv/tools/anyscale/bin/python docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-v1/root_tar_persistence_controller.py once --authorization docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-v1/root-authorization-once.json
```

Only after checkpoint 317 reports `remote_once_finished`, child exit 0, a validated tar revision, and a validated receipt revision, copy the disabled monitor authorization to a fresh file, set only `authorized` to true, review it, then run:

```bash
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
/home/m0hawk/.local/share/uv/tools/anyscale/bin/python docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-v1/root_tar_persistence_controller.py monitor --authorization docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-v1/root-authorization-monitor.json
```

Both commands require `HF_TOKEN` in root's environment. This preparation authorizes neither command; the checked-in templates remain `authorized: false`. Root owns each upload and monitor launch.
