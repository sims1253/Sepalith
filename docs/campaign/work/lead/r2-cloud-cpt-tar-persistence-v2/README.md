# Bounded tar persistence monitor v2

This fresh monitor follows the corrected checkpoint-317 one-shot. It uses the same opaque tar and remote receipt format, so a validated one-shot completion is read back without reuploading the tar. Its remote source is versioned as `persistence-staging/tar_persistence_v2.py`, avoiding conflict with either earlier controller source.

The v2 monitor distinguishes three checkpoint states:

- An absent checkpoint directory or absent terminal `campaign-manifest.json` remains pending.
- A present, valid full manifest enters the bounded upload path.
- A present manifest that is unreadable, incomplete, mismatched, or has corrupt files emits one `CHECKPOINT_INVALID` event and is permanently abandoned.

Every upload exception is written immediately with redacted type/message. Each valid step has at most three attempts with 30- and 120-second backoffs. Exhausted steps are never reconsidered. The outer loop stops at `2026-09-14T18:40:37.166121+00:00` or after the existing `stop-sidecar` signal and a final scan. A network operation that started before the deadline can extend past the outer-loop deadline; the existing provider watchdog is the ultimate hard cap. This is recorded in the monitor terminal receipt.

The controller gates the observed PID 4588/start tick 7757, exact cwd/argv, and both submission and staged binding/source hashes. It accepts only the monitor action and sends root's local `HF_TOKEN` through stdin and then the remote child environment.

After root has independently accepted the checkpoint-317 one-shot result, create a fresh authorization from the disabled template and run:

```bash
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
/home/m0hawk/.local/share/uv/tools/anyscale/bin/python docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-v2/root_tar_monitor_controller_v2.py monitor --authorization docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-v2/root-authorization-monitor.json
```

Require `monitor_started` plus the monitor PID, start tick, source SHA, and deadline. This preparation does not authorize or execute the monitor.
