# A100 checkpoint-90 state transport preparation

This packet prepares two independently sealed private-Hub closures. It does not upload, allocate cloud resources, or launch training.

* The checkpoint closure contains the original 12-file full checkpoint plus its original `campaign-manifest.json`: 13 physical files, 17,250,934,153 bytes.
* The runtime closure contains the immutable 16K streaming cache, the pinned checkpoint-66 bootstrap model files required by the current loader, the reviewed A100 DDP preparation source, and control manifests: 36 files, 8,880,462,456 bytes.

The remote prefix is derived from the immutable manifest digest. Each upload re-hashes the source before sending it, verifies the remote LFS SHA-256 or Git blob identity, journals every completed file, and uploads a closure marker last. Restore requires an exact Hub revision and closure SHA-256, streams every file into a fresh temporary directory, verifies it, fsyncs it, and publishes it by atomic rename. `bounded_exec.py` terminates the owned process group on timeout, redacts token-shaped text, and can remove only a caller-named temporary prefix.

The private repository `scholzmx/sepalith-lora` was confirmed readable and private through cached authentication. No token is placed in an argument, manifest, receipt, or log.

## Required root flow

1. Review `transport-spec.json`, `runtime-bundle-spec.json`, source hashes, and tests. Confirm the checkpoint is still the root-accepted sealed checkpoint.
2. Create fresh admission JSONs from both templates. Changing any admitted field causes upload refusal.
3. Run the two upload commands in `commands.json`. Upload is not authorized by this preparation packet.
4. Record each returned immutable Hub revision and closure SHA-256 in a fresh cloud binding.
5. On a fresh cloud volume with at least 55 GiB free, run both restore commands through `bounded_exec.py`. The 55 GiB allowance covers 26.13 GB of restored files plus a conservative second cached copy and metadata.
6. Bind a cloud recipe to the restored checkpoint-66 parent, cache, source, and fresh output paths. Bind a new continuation admission to that exact recipe and restored checkpoint-90 manifest.
7. Choose one execution path explicitly:
   * The current frozen local trainer can continue exactly with one visible GPU and the single `rng_state.pth`.
   * The reviewed A100 DDP packet preserves the 16-row effective batch and prepares per-rank RNG state, but its production `run` entry point deliberately fails closed. Root must complete and review the actual-model integration and the A100 CUDA parity/memory gate before an eight-GPU launch.

The current local trainer initializes from checkpoint 66 before Trainer restores checkpoint 90. The runtime closure therefore includes the pinned checkpoint-66 model/config/tokenizer bootstrap set. Omitting it makes the current loader fail before state restoration.

## Limits

The prior 613,826,560-byte Hub readback took 22.446 seconds and the 613,888,000-byte readback took 19.717 seconds. Extrapolating those measured rates gives about 14.0–15.9 minutes to download 26.13 GB. At the reviewed p4d whole-node rate of 29.6433 AC/hour, that would consume about 6.9–7.9 AC before model initialization if the rate holds. This is an inference; cloud-region locality, LFS cache state, and concurrent traffic can change it. Upload duration and Hub storage charges are unmeasured.

The uploader performs an integrity read and then the Hub client reads the files for upload, so the source-side I/O is approximately 52.26 GB. Restore can consume roughly 52.26 GB at peak because the safe implementation retains the Hub cache while atomically constructing the destination. No tar staging is created.

## Returning a distributed checkpoint

After a root-accepted DDP8 runner seals a future checkpoint, `build_return_spec.py` requires the accepted manifest SHA and enforces the exact rank RNG closure (`rng_state_0.pth` through `rng_state_7.pth`), the rank-0 `rng_state.pth` alias, optimizer/scheduler/cursor continuity, and the admitted draw schedule. `upload_private_return_checkpoint.py` uses a separate exact root admission and closure-last publication. `restore_private_return_checkpoint.py` restores every original byte and verifies that the canonical single-device RNG equals rank 0, allowing a later separately admitted one-GPU resume. These scripts cannot be used yet because no future cloud checkpoint exists and the DDP production runner remains fail-closed.
