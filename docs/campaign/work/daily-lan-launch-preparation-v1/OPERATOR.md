This is a preparation candidate for root review. It has not been installed or run on either serving host. Root must admit the daily gateway bytes and verify the final editor/runtime package before deployment.

The desktop serves the selected theta0 Q8 model with CUDA, a 4096-token context, batch and microbatch 256, one slot, a 192-token output cap, and CUDA GraphOpt 0. The notebook connects through SSH to its own loopback port 18403. The launcher binds the desktop native server to 127.0.0.1:18403 and the gateway to 127.0.0.1:18423. It creates no public listener. It uses the accepted, uninstrumented VSIX, SHA b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1.

Root performs these deployment steps once, after the release gates pass:

1. Review source-manifest.json and gateway.patch. The gateway keeps its existing 30-minute default. This launcher explicitly selects an eight-hour bound. The request, response, queue, socket, identity, cancellation, receipt-size, and memory limits remain unchanged. Root must benchmark these daily gateway bytes before promoting them.
2. Initialize the stable CUDA lock inode at /home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock. Use that same lock for every future CUDA launcher. Never unlink or replace it. The daily launcher opens an existing lock and fails if it cannot acquire it. The manual RTX 5090 ledger row must say None before an operator starts a daily session. Root retains ownership of that ledger.
3. Review deployment-admission.example.json. Save a separate admitted deployment receipt with the exact source-manifest hash and the current RESOURCE-LEASES.md SHA256. The first start verifies that ledger hash. Later starts reuse this admitted deployment, verify the same source manifest, acquire the exclusive lock, and reject any occupied or unknown ledger row. They do not require another agent approval. A changed deployment receipt requires a new reviewed state directory.
4. Stage only notebook_setup.py and candidate.vsix into the fresh notebook directory /home/m0hawk/.local/share/sepalith-daily-lan/package. Preserve the existing notebook profile and b4 package. No SSH private key or certificate is copied. Existing SSH authentication and known-host verification remain in force.

The following staging commands are prepared for root to run from this packet directory:

```sh
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes m0hawk@192.168.178.40 'test ! -e /home/m0hawk/.local/share/sepalith-daily-lan/package && mkdir -p /home/m0hawk/.local/share/sepalith-daily-lan/package'
scp -o BatchMode=yes -o StrictHostKeyChecking=yes notebook_setup.py candidate.vsix m0hawk@192.168.178.40:/home/m0hawk/.local/share/sepalith-daily-lan/package/
```

Use the exact notebook initialization command in notebook-init-command.txt. It verifies the helper's own hash and the VSIX hash, then creates settings in the dedicated user-data directory. It refuses to replace different existing settings. Install the VSIX on the notebook with:

```sh
/usr/bin/code --user-data-dir /home/m0hawk/.local/share/sepalith-daily-lan/user-data --extensions-dir /home/m0hawk/.local/share/sepalith-daily-lan/extensions --install-extension /home/m0hawk/.local/share/sepalith-daily-lan/package/candidate.vsix
```

Daily operator steps after root deployment:

1. On the desktop, start the foreground supervisor. Replace ADMITTED-DEPLOYMENT.json with the root receipt path:

```sh
python3 daily_lan.py start --state-dir /home/m0hawk/.local/state/sepalith-daily-lan --admission ADMITTED-DEPLOYMENT.json
```

2. Wait for the printed session path. The launcher verifies source, VSIX, model, runtime files, full native offload, gateway identity, and the actual notebook forward. It uploads a fresh UUID binding before exposing the forward. The native/model hashes occur only during this root-admitted start; preparation did not read those artifacts. Keep the supervisor running.
3. Open the dedicated notebook window:

```sh
/usr/bin/code --user-data-dir /home/m0hawk/.local/share/sepalith-daily-lan/user-data --extensions-dir /home/m0hawk/.local/share/sepalith-daily-lan/extensions --new-window
```

4. In its Command Palette, run `Sepalith: Stop server`, then `Sepalith: Start server`. Require the verified Remote CUDA state. These are existing extension commands. `Alt+.` requests a suggestion. Automatic requests use the retained 1500 ms delay. Scope context is off, matching the selected LAN baseline; debug prompt logging is off. This package adds no instrumentation or diagnostics feature.
5. To stop, press Ctrl+C in the desktop terminal, or use the printed session directory:

```sh
python3 daily_lan.py stop SESSION_DIRECTORY
python3 daily_lan.py status SESSION_DIRECTORY
```

Stop addresses the recorded supervisor through a PID descriptor after checking PID, start tick, and UID. The supervisor stops only its recorded children, in reverse order, then writes terminal.json. It verifies each original identity has exited. The notebook window stays open; its extension cannot use the stopped forward. `Sepalith: Stop server` disconnects the client and never kills the desktop backend by itself.

A broken SSH connection, native/gateway exit, changed process identity, occupied ledger, memory below 8 GiB, full log, operator stop, or eight-hour session deadline ends the session and releases its owned children. A child receives SIGTERM if the supervisor disappears. Children inherit the CUDA lock, so an orphan cannot silently free the lock while still serving. The SSH keepalive bound is approximately 45 seconds, plus scheduling and cleanup. A final SIGKILL follows a child's six-second stop grace. Each child log is capped at 64 MiB; the gateway retains its 32 MiB receipt cap. These caps can end a busy session before eight hours. No automatic restart or model fallback occurs. Start a new session and repeat the two extension commands to accept its new identity.

If status reports that the supervisor disappeared and no terminal receipt exists, root must check the recorded child identities and lock before restart. The launcher does not kill unknown processes or reclaim a held lock. It does not prove an instantaneous GPU release from an HTTP disconnect.

Rollback keeps the protected b4 Q8 model and matched legacy renderer together. Stop the daily supervisor and close its dedicated notebook window. Return to the preserved b4 installation/profile recorded in RUN-03-b4-host-baseline.json and RUN-01-b4-renderer-v3-source-and-package.json. Root supplies that matched rollback as part of REL-08. This packet does not overwrite or substitute its model, VSIX, renderer, or settings.

CPU evidence covers lifecycle, exclusive ownership, busy rejection before artifact access, admission reuse, exact binding identity, atomic replacement, and the gateway duration option. It does not prove notebook installation, SSH forwarding, live CUDA placement, model quality, daily latency, multi-hour stability, or release acceptance. Those remain root integration gates. The final held-out evaluation gates remain unchanged and this launcher has no final-data reader.
