# Full-weight editing context profile preparation

This packet prepares five independent, root-exclusive forward/backward resource probes at 2K, 4K, 8K, 16K, and 32K. Each process loads the same accepted full checkpoint, restores the saved FP32 tensors and complete hybrid FP32 optimizer state, enables non-reentrant gradient checkpointing, and performs target-only backward on one intact TRAIN example. It never calls `optimizer.step`, never writes a checkpoint, and restores the checkpoint CPU/CUDA RNG state before exit.

The 2K fixture comes from the accepted 15,006-row pool. The longer fixtures are existing TRAIN resource-probe rows and remain outside training admission. Their use here does not admit them or the pending 10,948 provider inputs. The small reviewed profiles establish a 20,191-row review pool: 15,006 original rows, 616+133+4,435 reviewed candidates, and one recovered row. Training admission remains separate.

Earlier fresh-parent resource probes measured roughly 24.20 GB allocated at 4K, 8K, and 16K. The 32K run reached 29.96 GB allocated and 37.35 GB reserved with zero reported free bytes, so it was explicitly not accepted as practical fit. Those runs stepped a freshly built optimizer. This packet instead profiles the future checkpoint322 with its complete optimizer state live and no update, so every cap needs a new measured result.
