# Native optimizer mmap resume preparation

This packet replaces only the single-device Transformers 5.5 optimizer/scheduler restore hook. It opens `optimizer.pt` through the inherited, hash-verified native-attestation file descriptor and uses `torch.load(weights_only=True, map_location="cpu", mmap=True)`. The unchanged composite optimizer validates its exact identity and clones each state tensor into owned FP32 storage on the parameter device. The seam rejects sharded, distributed, FSDP, DeepSpeed, XLA, missing-lock, changed-inode, version, and legacy optimizer paths.

Model, trainer state, scheduler semantics, data cursor, and CPU/CUDA RNG restoration remain owned by Transformers and the existing campaign checkpoint contract. Mmap reduces the need for a second 12.2 GB anonymous CPU copy, but it does not avoid reading the pages while device-owned state is materialized. Linux can retain mapped pages in the page cache, and peak RSS/PSS plus GPU transient memory still require a real checkpoint322 probe. Direct CUDA deserialization is excluded because incoming and restored state could overlap on the 32 GB device.

Use the mixin before `FullWeightOptimizerTrainerMixin` and `Trainer`. This is a preparation packet; it does not modify or authorize the running native trainer.
