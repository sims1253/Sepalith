"""P1 offload-disable shim (from bisect_probes r5; CONFIRMED at 300 steps on T4).
Import BEFORE unsloth to neuter the low-VRAM smart gradient-offload patch
(the mixed BFloat16/Half culprit on 16GB cards).
"""
import torch
import torch.utils.checkpoint as _ckpt_mod
_PRISTINE_CHECKPOINT = getattr(_ckpt_mod, '_old_checkpoint', _ckpt_mod.checkpoint)
_PRISTINE_CKPT_FN = getattr(_ckpt_mod, '_old_CheckpointFunction', _ckpt_mod.CheckpointFunction)
def _plain_checkpoint(function, *args, use_reentrant=None, **kwargs):
    kwargs.pop('use_reentrant', None)
    return _PRISTINE_CHECKPOINT(function, *args, use_reentrant=True, **kwargs)
def _plain_apply(function, preserve_rng_state=True, *args, **kwargs):
    kwargs.pop('use_reentrant', None)
    return _PRISTINE_CHECKPOINT(function, *args, use_reentrant=True, **kwargs)
def _disable_offload(*args, **kwargs):
    _ckpt_mod.checkpoint = _plain_checkpoint
    _ckpt_mod.CheckpointFunction = _PRISTINE_CKPT_FN
    try:
        import transformers.modeling_utils as _mu
        _mu.checkpoint = _plain_checkpoint
    except Exception:
        pass
    return None
import unsloth  # establish the guarded import chain first
import unsloth_zoo.gradient_checkpointing as _gc
_gc.patch_unsloth_smart_gradient_checkpointing = _disable_offload
_gc.unpatch_unsloth_smart_gradient_checkpointing = _disable_offload
_gc.patch_unsloth_gradient_checkpointing = _disable_offload
_gc.unpatch_unsloth_gradient_checkpointing = _disable_offload
_gc.patch_gradient_checkpointing = _disable_offload
_gc.unpatch_gradient_checkpointing = _disable_offload
_gc.unsloth_offloaded_gradient_checkpoint = _plain_checkpoint
_gc.unsloth_gradient_checkpoint = _plain_checkpoint
_gc.unsloth_checkpoint = _plain_checkpoint
_gc.UnslothCheckpointFunction.apply = _plain_apply
try:
    import unsloth.models._utils as _uu
    _uu.patch_unsloth_smart_gradient_checkpointing = _disable_offload
    _uu.unpatch_unsloth_smart_gradient_checkpointing = _disable_offload
    _uu.patch_unsloth_gradient_checkpointing = _disable_offload
    _uu.unpatch_unsloth_gradient_checkpointing = _disable_offload
except Exception as _e:
    print(f'BISECT-P1: _utils pre-bind skipped ({type(_e).__name__})', flush=True)
_disable_offload()
print('BISECT-P1: offload-disable shims installed (plain torch ckpt)', flush=True)
