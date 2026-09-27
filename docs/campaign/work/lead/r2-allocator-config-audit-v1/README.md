# Allocator configuration audit v1

The active pilot uses the correct Torch 2.11 environment variable name, but it is not an expandable-segments experiment on this WSL host.

Installed Torch `2.11.0+cu130` accepts `PYTORCH_ALLOC_CONF=expandable_segments:True`. A CPU-only parser probe with `CUDA_VISIBLE_DEVICES=''` accepted that setting and the roundup fallback, rejected an invented key, and left `torch.cuda.is_initialized()` false before and after every call.

Installed `unsloth_zoo 2026.8.12` explicitly classifies Windows/WSL expandable segments as unsupported. During import it removes `expandable_segments` from all allocator variables. On a real CUDA WSL path, when the user has not disabled its fallback, it then sets the unified Torch 2.10+ variable to `roundup_power2_divisions:[32:256,64:128,256:64,>:32]`. The active trainer has `WSL_DISTRO_NAME` and `WSL_INTEROP`, requests only `PYTORCH_ALLOC_CONF=expandable_segments:True`, and does not set `UNSLOTH_DISABLE_ALLOC_FALLBACK`; therefore the installed source predicts roundup fallback, not expandable segments.

An isolated import-order probe started with the active requested setting. `unsloth_zoo` removed it before failing because GPUs were hidden, and CUDA stayed uninitialized. The hidden-GPU failure occurs before the later roundup branch, so that probe proves removal but cannot directly observe the CUDA-path fallback. The installed source is the evidence for that later branch.

`/proc/<pid>/environ` exposes the launch environment and is not evidence of Python mutations after import. The four reserved-byte values matching the baseline are consistent with the configuration being removed, but they do not independently prove allocator policy.

`runtime-report.patch` is a fresh, review-only change against the frozen stage-transition v3 trainer. It captures requested environment before importing Unsloth, records effective environment, Torch/Unsloth versions, WSL flags, native allocator backend, and interpretation immediately after import and before model loading. It writes `allocator-runtime.json` and includes the same record in `run-result.json`. The backend string alone cannot distinguish native expandable-segment subconfiguration, so the report keeps the post-import environment as the explicit evidence. No active process was inspected beyond read-only `/proc` metadata or modified.

Torch's CPU-only parser also accepts the exact next candidate `backend:native,roundup_power2_divisions:[32:256,64:128,256:64,>:32],garbage_collection_threshold:0.8`. The installed legitimate fraction APIs are `torch.cuda.memory.set_per_process_memory_fraction(0.95, device=0)` and `get_per_process_memory_fraction(0)`. They call CUDA lazy initialization and were therefore not invoked in this audit. A root-admitted wrapper must call them after Unsloth import and before model allocation, then persist the getter result alongside the post-import allocator environment and backend.
