# RUN-01 target-host intake

Observed at `2026-09-12T09:25:53Z` from the local campaign worktree. This is a bounded read-only SSH intake. No package installation, model transfer, process termination, server start, benchmark, or remote file modification was performed. The only SSH-side state change permitted by the packet was the normal `known_hosts` record for this host.

## Connection identity

The connection command used for every remote probe was:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new m0hawk@192.168.178.40 'bash -s'
```

The negotiated host key from a verbose `true` connection was:

```text
Server host key: ssh-ed25519 SHA256:Tn5oE3/sIruL5la/jz4RivzqBO5Wb3dKWt4RD0crq6w
Authenticated to 192.168.178.40 ([192.168.178.40]:22) using "publickey".
```

The bounded `ssh-keyscan` observed the following host key fingerprints. The ED25519 key above is the key actually negotiated and accepted; the other fingerprints are recorded for host-key inventory only:

```text
ECDSA SHA256:wnkhVHgyPOcJphwYY57Mugw+d0wcCqtTEjO18m7/IPs
RSA   SHA256:YEQl3rJqzObE+RLSvNBJGCQoQ5IwQx+ft2K4uUyeW7w
ED25519 SHA256:Tn5oE3/sIruL5la/jz4RivzqBO5Wb3dKWt4RD0crq6w
```

Remote identity:

```text
hostname=m0pad
user=m0hawk
uid=1000
```

## Observed hardware and capacity

```text
OS=CachyOS Linux (ID=cachyos, ID_LIKE=arch, rolling)
kernel=Linux 7.2.3-1-cachyos #1 SMP PREEMPT_DYNAMIC x86_64
CPU=AMD Ryzen 5 PRO 5650U with Radeon Graphics
logical_cpus=12
physical_cores=6
RAM_total=15656516 KiB (~14.93 GiB)
RAM_available=11916584 KiB (~11.37 GiB)
loadavg=0.86 0.49 0.24 (2 runnable / 1398 total at probe)
uptime=375.69 s at probe
```

DMI confirms the notebook form factor:

```text
sys_vendor=HP
product_name=HP EliteBook 845 G8 Notebook PC
product_version=SBKPF
chassis_type=10
bios_version=T82 Ver. 01.16.00
```

The root filesystem, also mounted at `/home` and `/tmp`, reported:

```text
/dev/nvme0n1p2  total=974663348 KiB  used=617636612 KiB  available=354898172 KiB  use=64%
```

That is approximately 930 GiB total and 338.5 GiB available on the observed filesystem. This is capacity evidence, not a reservation for a campaign job.

## Runtime and editor paths

Bounded command/path probes found:

```text
/usr/bin/code                         present, regular file, 308 bytes
/usr/bin/node                         present, v26.8.1
/usr/bin/npm                          present, 12.0.2
/usr/bin/npx                          present
/usr/bin/vulkaninfo                   present
```

The exact source path `/home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith` is present. Its package reports version `0.0.7` and `main: ./dist/extension.js`. The source and package hashes at intake are:

```text
f86262ee5910b023323d3c9b0b8b088f82037fd938b674f283454ad05bc249f9  /home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith/package.json
368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285  /home/m0hawk/Documents/Sepalith/extensions/vscode-sepalith/src/extension.ts
bd0d9edf69283ebdf4e73e0a7b168d2fcf50acbd01f63674cad93ed4fe42fdad  /usr/bin/code
```

`src/extension.ts` is present (30,316 bytes). The declared `dist/` directory and `dist/extension.js` are absent. The extension's `node_modules` directory and `node_modules/.bin/esbuild` are absent. No matching `sepalith` directory was found in the bounded `/home/m0hawk/.vscode/extensions` installation directory. No `/home/m0hawk/.vscode-server` or `/home/m0hawk/.vscode-server/cli/servers` path was found.

No `llama-server` or `llama-cli` was found through `command -v`, `/usr/bin`, `/usr/local/bin`, `/home/m0hawk/.local/bin`, or the known project runtime path `/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453`. The known project runtime directory itself was absent. `vulkaninfo` presence was recorded; no Vulkan enumeration or GPU benchmark was run in this intake.

The bounded process table contained `zcode`, `ZCode`, `zcode-cli`, `zcode-host-loca`, `zcode-node-repl`, and `cosmic-comp` processes. An exact process-name filter found no `code`, `code-server`, `node`, `npm`, `llama-server`, or `llama-cli` process. Process names alone do not establish which editor window owns the active `zcode` processes.

## Smallest next checks

1. After lead admission, install/build the extension dependencies in the target project checkout and verify that the declared `dist/extension.js` exists. Record the build command and hashes; this intake deliberately did not install anything.
2. Supply or build the pinned b10453 `llama-server` at a reviewed path, then run one bounded CPU health/completion smoke with the selected model and no concurrent heavy workload. A CPU `ngram-mod` scout is the smallest speculative check described by `SPECULATIVE-PATHS.md`; it needs paired exact-output traces and host-load recording.
3. For editor acceptance, use an actual desktop VS Code window or a separately authorized Remote SSH editor session. Verify extension activation, loopback sidecar start/readiness, one R-file same-line completion, no-op behavior, cancellation, and server shutdown. SSH shell access alone does not provide GUI/editor acceptance evidence.

The host appears suitable for a bounded CPU scout by observed core/RAM/storage capacity, but no latency, Vulkan, llama.cpp, model-quality, or editor result is claimed here.
