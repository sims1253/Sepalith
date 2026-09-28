# E750 notebook CPU quality root admission review

This packet independently rechecks the frozen `packet-final` source closure, current model stat identity, port, process state, and preserved b4/editor identities without rereading the 2.68 GB model. The model's prior final-path full SHA verification remains bound by the immutable stat identity.

The frozen evaluator preserves all 75 corrected DEV cases at caps 192, 384, and 768. Each uses the same prompt IDs, context 4096, a per-case 120-second offline deadline, and complete EOS/cap/transport/mechanical accounting. Its output is offline CPU quality evidence only; the production five-second profiles remain unchanged.

No per-case launch blocker remains. Two outer controls are absent inside the frozen runner: it does not consume the source-manifest admission field, and it does not enforce the 28,800-second suite maximum as a wall-clock deadline. `stage_and_launch.sh` compensates without changing frozen files by verifying the exact complete remote source closure immediately before launch and staging the separately reviewed `watchdog_remote.py`. The watchdog binds the runner PID/start tick, enforces 28,800 seconds, allows the active 120-second request plus cleanup to finish after TERM, and, if it remains wedged, cleans only server process groups whose PIDs and start ticks match immutable arm launch records before killing the runner group.

Root must run `issue_admission.py` after source review. The proposed draft is deliberately non-admitting. The stage-and-launch wrapper refuses an unadmitted file, a reused run ID, altered packet/source bytes, changed model stat, an occupied port, or an existing owned process/session.
