# Step-500 notebook editor test plan

Status: CPU preparation only.  This packet does not install, start, or stop an
editor, notebook process, native server, model, or SSH tunnel.

The test uses the isolated notebook profile
`/home/m0hawk/.local/share/sepalith-r2-step500-checks` and a fresh child run
directory below that profile.  The only editor input is a disposable workspace
created by the accepted harness: `auto-*.R` and `observer-control-*.txt`.
No DEV, final, or authored result is used.  A small explicitly admitted TRAIN
row may be added by root only if its row identity and source hash are recorded;
the default run is synthetic-only.

## Pinned inputs

The current daily VSIX is
`docs/campaign/work/lead/remote-auto350-b/notebook-capsule/candidate.vsix`,
40,832 bytes, SHA-256
`b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1`.
The complete capsule manifest is
`docs/campaign/work/lead/remote-auto350-b/notebook-capsule/capsule-manifest.json`,
SHA-256
`79be4167a2f9159480c8a4ad2747c7019c0e5368d69a1df9e0e474c65e6194a6`.
The manifest validates all 16 capsule files, including the actual VSIX,
`run_remote_editor.mjs`, the repaired renderer observer, the automatic-case
analyzer, and the primary editor harness.

The selected step-500 runtime contract is
`docs/campaign/work/lead/r2-step500-lan-preparation-v1/primary-manifest.json`:
file SHA-256
`9459f19a8a29e1158b4e91169da6a4d26c91f6e625f61c5d954a3d2b68201a92`,
canonical manifest SHA-256
`59971733ec5a09320c991466e0cd378c26ab9b69d897d2fd192ddeeed2e3b36e`,
model SHA-256
`d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db`, and
model size 2,679,710,496 bytes.  Its contract is CUDA, context 4096, output
cap 192, BOS 0, EOS/PAD 1, native EOG `[1, 130073]`, vocabulary 130560,
renderer `zeta2-prm03-v1`, and `postAcceptCooldown=true`.

The selected accepted SFT parent metadata is
`docs/campaign/work/lead/r2-step500-rl-gate-v2/parent-manifest.json`, SHA-256
`05eac983926eacc45b78f59d281eeaeeecfc8c58ed03647b4a8cbe794160afa5`.
It identifies the step-500 merged weights as
`631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c`.
The editor run does not read that file; the root serving admission owns the
model load and must bind a fresh instance to the selected runtime manifest.

## Root run sequence

Run these steps only after the RL lease is released and root has an admitted
step-500 desktop session.  Keep the profile and each run directory private
(`umask 077`).  Replace `<RUN_ID>` with a fresh identifier and never reuse a
completed run directory.

1. Validate this packet without launching anything:

   ```sh
   cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
   /usr/bin/python3 docs/campaign/work/lead/r2-step500-editor-preparation-v1/editor_plan.py --validate
   ```

2. Prepare the remote profile package.  The existing receiver is the selected
   step-500 `notebook_setup.py`, not the stale receiver from another arm:

   ```sh
   ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes m0hawk@192.168.178.40 \
     'umask 077; mkdir -p /home/m0hawk/.local/share/sepalith-r2-step500-checks/package /home/m0hawk/.local/share/sepalith-r2-step500-checks/capsule/primary-editor-harness /home/m0hawk/.local/share/sepalith-r2-step500-checks/runs'
   scp -p -o BatchMode=yes -o StrictHostKeyChecking=yes \
     docs/campaign/work/lead/r2-step500-lan-preparation-v1/notebook_setup.py \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/candidate.vsix \
     m0hawk@192.168.178.40:/home/m0hawk/.local/share/sepalith-r2-step500-checks/package/
   ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes m0hawk@192.168.178.40 \
     '/usr/bin/python3 /home/m0hawk/.local/share/sepalith-r2-step500-checks/package/notebook_setup.py init --self-sha256 7b2e60446ebd274080ad861f96751c784376872a3ac1f41bce423fcdc015e97e'
   ```

   Copy the capsule code and analyzers from the pinned capsule.  Do not copy
   or use its `binding.json` as the selected binding: that file names the old
   theta0 model SHA `22401b9f...`.  Do not use its hard-coded
   `notebook_editor_supervisor.py`; it also names that old binding.  The
   desktop daily supervisor supplies the fresh binding and tunnel, while the
   capsule's `run_remote_editor.mjs` supplies the disposable VS Code host and
   owned child cleanup.

   ```sh
   scp -p -o BatchMode=yes -o StrictHostKeyChecking=yes \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/README.md \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/accepted-observer.mjs \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/analyze_auto.mjs \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/analyze_renderer.mjs \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/check_merged.mjs \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/merged-source-manifest.json \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/observe_renderer.mjs \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/run_remote_editor.mjs \
     m0hawk@192.168.178.40:/home/m0hawk/.local/share/sepalith-r2-step500-checks/capsule/
   scp -p -o BatchMode=yes -o StrictHostKeyChecking=yes \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/primary-editor-harness/package.json \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/primary-editor-harness/extension.js \
     docs/campaign/work/lead/remote-auto350-b/notebook-capsule/primary-editor-harness/acceptance-v2.js \
     m0hawk@192.168.178.40:/home/m0hawk/.local/share/sepalith-r2-step500-checks/capsule/primary-editor-harness/
   ```

3. Start the root-owned desktop session with the selected LAN packet.  The
   resulting session binding must have model SHA `d269a9fb...`, canonical
   manifest SHA `59971733...`, a fresh UUID, and endpoint
   `http://127.0.0.1:18403`.  The daily supervisor uploads that binding to
   `current-binding.json` and reverse-forwards notebook port 18403 to its
   gateway.  Use the root admission and ledger paths supplied at run time:

   ```sh
   cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-step500-lan-preparation-v1
   /usr/bin/python3 daily_lan.py start \
     --state-dir /home/m0hawk/.local/state/sepalith-r2-step500-checks \
     --admission <ROOT-ADMITTED-DEPLOYMENT.json>
   ```

   Before the editor launch, require the receiver's `probe` output to show
   the same fresh instance ID and selected model/server/renderer identity.
   A probe against the stale capsule binding is a configuration failure, not
   evidence for this run.

4. Install and start the real VS Code host in the isolated profile.  The
   launcher repeats the force-install and records `install.json`; the command
   below is a separate visible install check.  The renderer source hash must
   be rechecked on the notebook before launch and must equal the pinned
   `e04cb8ec68016ccf4557690a2c8c908c3534614af1abe4c5c557157548169d78`.

   ```sh
   ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes m0hawk@192.168.178.40 \
     '/usr/bin/code --user-data-dir /home/m0hawk/.local/share/sepalith-r2-step500-checks/user-data --extensions-dir /home/m0hawk/.local/share/sepalith-r2-step500-checks/extensions --install-extension /home/m0hawk/.local/share/sepalith-r2-step500-checks/package/candidate.vsix --force'
   ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes m0hawk@192.168.178.40 \
     'sha256sum /usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js'
   ```

   After the daily supervisor has established the reverse tunnel and written a
   fresh binding, run the capsule launcher in a fresh remote run directory:

   ```sh
   ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes m0hawk@192.168.178.40 \
     'umask 077; mkdir -m 700 /home/m0hawk/.local/share/sepalith-r2-step500-checks/runs/<RUN_ID>; \
      xvfb-run -a --server-args="-screen 0 1280x800x24" node \
      /home/m0hawk/.local/share/sepalith-r2-step500-checks/capsule/run_remote_editor.mjs \
      --code /usr/bin/code \
      --vsix /home/m0hawk/.local/share/sepalith-r2-step500-checks/package/candidate.vsix \
      --vsix-sha256 b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1 \
      --binding /home/m0hawk/.local/share/sepalith-r2-step500-checks/current-binding.json \
      --binding-sha256 <FRESH-BINDING-SHA256> \
      --instance-id <FRESH-INSTANCE-UUID> \
      --debug-port 19403 \
      --renderer-source /usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js \
      --renderer-sha256 e04cb8ec68016ccf4557690a2c8c908c3534614af1abe4c5c557157548169d78 \
      --run-root /home/m0hawk/.local/share/sepalith-r2-step500-checks/runs/<RUN_ID> \
      --timeout-ms 180000 --debounce-ms 350'
   ```

   This is the real VS Code extension host with a disposable Xvfb display.
   It is not a mocked `vscode` API.  The 350 ms arm belongs to the
   `remote-auto350-b` capsule.  A 1500 ms arm is a separate root experiment;
   do not combine its observations with this run.

## Acceptance gates and evidence

The launcher and harness already implement the event sequence.  After a
successful process return, run the pinned analyzers in the remote run root:

```sh
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes m0hawk@192.168.178.40 \
  'node /home/m0hawk/.local/share/sepalith-r2-step500-checks/capsule/analyze_renderer.mjs \
   /home/m0hawk/.local/share/sepalith-r2-step500-checks/runs/<RUN_ID> \
   /home/m0hawk/.local/share/sepalith-r2-step500-checks/runs/<RUN_ID>/renderer-analysis.json'
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes m0hawk@192.168.178.40 \
  'node /home/m0hawk/.local/share/sepalith-r2-step500-checks/capsule/analyze_auto.mjs \
   /home/m0hawk/.local/share/sepalith-r2-step500-checks/runs/<RUN_ID>'
```

Require each item below to be explicitly recorded.  A missed observer window
remains pending; it is not a zero-ghost pass.

| Gate | Procedure | Required evidence |
| --- | --- | --- |
| Install | Match VSIX size/SHA before and after `--install-extension`; retain `install.json` and stderr. | `install.code=0`, no timeout, exact candidate SHA. |
| Start/identity | Harness runs `sepalith.startServer`, checks `/sepalith/runtime`, `/props`, status bar `Sepalith: ready`, settings, fresh binding instance header. | `host-result.json`, `runtime-status.json`, gateway identity records; selected step-500 model/renderer hashes. |
| Ghost positive | Review `control-visible` and screenshots. Require focused, visible, unobscured ghost with the specific view-zone ancestor. | `renderer-analysis.json`: `observer_positive_control=true`; screenshot and frame metadata. Ordinary `.view-lines` text is never ghost evidence. |
| Ghost multiline | Review `control-multiline` before commit. Require all three sentinels in line-level view-zone metadata, then exact document insertion after `inlineSuggest.commit`. | `observer_multiline_control=true`, `control-multiline` commit event, post-commit buffer hash. Missing line metadata is indeterminate. |
| Model acceptance/save | `typing-accept-save` accepts the observed model item, waits for cooldown, calls `document.save()`, then compares the on-disk `auto-accept.R` SHA to the recorded after-buffer SHA. | `auto_commit_result.changed=true`, version increment, `auto_save`, matching file SHA. Do not repair or append a tail. |
| R parse | Parse each accepted synthetic R buffer with `Rscript --vanilla -e 'parse(file=..., keep.source=FALSE)'`; parsing executes no R code. | Per-buffer exit code/output and exact buffer SHA. Parse failure is an invalid-edit measurement. |
| Stale ghost | In `typing-cancel`, require a gateway dispatch before the edit, then match completion ID, request ID, URI/version and frame time. A successor ghost must be distinguished from the old result. | `auto-analysis.json` stale row plus gateway `request_cancelled`/`request_released` chronology. No identity join means pending. |
| Cancel control | The delayed deterministic provider resolves after VS Code cancellation. Require started → cancelled → resolved(`cancelled=true`) and zero stale sentinel frames for at least 5 s of contiguous focused frames. | `observer_cancellation_control=true`; no production cancellation claim follows from this control alone. |
| Cleanup | Let the launcher write `retained-evidence.json` and root supervisor verify only owned editor descendants exited. | `terminal.json` with no owned survivors. Never stop the forwarded native process from the notebook. |

The harness's `typing-pauses` and `typing-switch` cases additionally verify
the fixed debounce arm, cursor/document identity, and no cross-file stale
publication.  Their generated text and request bodies are synthetic or
bounded TRAIN fixtures only.  `run_remote_editor.mjs` and the analyzer both
retain hashes and IDs instead of raw prompt text in control receipts.

## Known limits

The capsule's checked-in `binding.json` and notebook supervisor belong to the
old theta0 arm and must be treated as historical code/data.  The selected
step-500 desktop session must supply a new binding before launch.  The
renderer hash is a notebook-local preflight and may change with a Code update.
The synthetic control proves the observer and VS Code cancellation barrier,
not native task cancellation.  Production cancellation requires the gateway
request ledger and native evidence.  Xvfb frames establish DOM geometry and
screenshots, not physical keyboard latency.  One short editor run does not
establish model quality, representative p95, or release/final-data
acceptance.
