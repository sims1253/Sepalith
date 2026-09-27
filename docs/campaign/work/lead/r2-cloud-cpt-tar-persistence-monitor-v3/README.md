# Bounded tar persistence monitor v3

This fresh monitor combines the v2 corrupt-manifest lifecycle fix with the corrected active submission-cwd identity gate from the checkpoint-317 once-v3 controller. It never expects binding or payload copies under the generated run directory.

The remote source name is `persistence-staging/tar_persistence_monitor_v3.py`. The source reuses a validated once-v3 completion without tar reupload, waits when the checkpoint directory or terminal manifest is absent, and emits one `CHECKPOINT_INVALID` event before abandoning any present corrupt manifest. Valid steps get at most three attempts with 30- and 120-second backoffs. The outer deadline is `2026-09-14T18:40:37.166121+00:00`; an in-flight network call remains bounded by the provider watchdog.

After root accepts the checkpoint-317 once-v3 remote tar and receipt, create a fresh authorization from the disabled template and run:

```bash
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
/home/m0hawk/.local/share/uv/tools/anyscale/bin/python docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-monitor-v3/root_tar_monitor_controller_v3.py monitor --authorization docs/campaign/work/lead/r2-cloud-cpt-tar-persistence-monitor-v3/root-authorization-monitor.json
```

Require `monitor_started`, exact PID/start tick, source SHA, and deadline. Root owns authorization and execution.
