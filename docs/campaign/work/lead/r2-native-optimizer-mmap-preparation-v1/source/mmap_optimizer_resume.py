#!/usr/bin/env python3
"""Strict single-device Trainer optimizer restore through an attested mmap.

Place this mixin before FullWeightOptimizerTrainerMixin and Trainer in the MRO.
It deliberately replaces only Transformers' optimizer/scheduler load method;
model, Trainer state, data skip, and CPU/CUDA RNG restoration remain upstream.
"""
from __future__ import annotations
import gc,json,math,os
from pathlib import Path
from typing import Any
import torch

SUPPORTED_TRANSFORMERS = "5.5.0"
SUPPORTED_TORCH_PREFIX = "2.11.0"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"

def require(value,message):
    if not value: raise ValueError(message)

def fingerprint(stat):
    return {key:int(getattr(stat,key)) for key in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")}

def tensors(value):
    if torch.is_tensor(value): yield value
    elif isinstance(value,dict):
        for item in value.values(): yield from tensors(item)
    elif isinstance(value,(list,tuple)):
        for item in value: yield from tensors(item)

def attested_fd(path: Path) -> tuple[int,dict]:
    raw=os.environ.get("SEPALITH_NATIVE_STAGE_ATTESTATION")
    require(raw,"native stage attestation missing")
    try: envelope=json.loads(raw)
    except Exception as exc: raise ValueError("native stage attestation is not JSON") from exc
    expected=str(path.resolve());matches=[x for x in envelope.get("files",[]) if x.get("path")==expected]
    require(len(matches)==1,"optimizer has no unique native attestation record")
    record=matches[0];fd=record.get("fd");require(type(fd) is int and fd>=0,"optimizer attestation fd differs")
    held=os.fstat(fd);current=path.stat();require(fingerprint(held)==fingerprint(current)==record.get("fingerprint"),"optimizer attested inode changed")
    import fcntl
    probe=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    try:
        try: fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(probe,fcntl.LOCK_UN);raise ValueError("optimizer attestation does not hold a shared lock")
    finally: os.close(probe)
    require(record.get("bytes")==held.st_size and isinstance(record.get("sha256"),str) and len(record["sha256"])==64,"optimizer attestation identity differs")
    return fd,record

def mmap_load_optimizer_state(path: Path):
    fd,record=attested_fd(path)
    before=os.fstat(fd)
    # /proc/self/fd opens the already verified inode, avoiding a pathname
    # substitution between attestation and torch.load.
    state=torch.load(f"/proc/self/fd/{fd}",map_location="cpu",weights_only=True,mmap=True)
    require(isinstance(state,dict) and isinstance(state.get("state"),dict) and isinstance(state.get("param_groups"),list),"optimizer mmap payload differs")
    require(fingerprint(before)==fingerprint(os.fstat(fd)),"optimizer changed during mmap load")
    mapped=list(tensors(state["state"]));require(mapped,"optimizer mmap contains no tensor state")
    return state,{"path":str(path.resolve()),"sha256":record["sha256"],"bytes":record["bytes"],"mapped_tensors":len(mapped),"mapped_tensor_bytes":sum(x.numel()*x.element_size() for x in mapped)},mapped

def assert_materialized_without_alias(optimizer,mapped):
    restored=list(tensors(optimizer.state));require(restored,"restored optimizer has no tensor state")
    incoming={x.untyped_storage().data_ptr() for x in mapped if x.numel()};aliases=sum(x.numel()>0 and x.untyped_storage().data_ptr() in incoming for x in restored)
    require(aliases==0,"optimizer retained mapped checkpoint tensor storage")
    require(all((not x.is_floating_point()) or x.dtype==torch.float32 for x in restored),"restored floating optimizer state is not FP32")
    return {"restored_tensors":len(restored),"restored_tensor_bytes":sum(x.numel()*x.element_size() for x in restored),"mapped_storage_aliases":aliases}

class MmapOptimizerResumeTrainerMixin:
    """Single-device, non-sharded Transformers 5.5 optimizer mmap restore."""
    mmap_optimizer_load_report: dict|None=None
    def _load_optimizer_and_scheduler(self,checkpoint):
        if checkpoint is None:return
        import transformers
        from transformers.trainer import check_torch_load_is_safe
        require(transformers.__version__==SUPPORTED_TRANSFORMERS,"Transformers version lacks reviewed mmap resume path")
        require(torch.__version__.startswith(SUPPORTED_TORCH_PREFIX),"PyTorch version lacks reviewed mmap semantics")
        require(int(self.args.world_size)==1,"mmap resume is prepared only for single-device Trainer")
        require(not self.is_deepspeed_enabled and not self.is_fsdp_enabled,"mmap resume forbids sharded optimizer ownership")
        require(not getattr(self,"is_fsdp_xla_v1_enabled",False),"mmap resume forbids XLA FSDP")
        checkpoint=Path(checkpoint);optimizer_path=checkpoint/OPTIMIZER_NAME;scheduler_path=checkpoint/SCHEDULER_NAME
        require(optimizer_path.is_file() and scheduler_path.is_file() and not optimizer_path.is_symlink() and not scheduler_path.is_symlink(),"full optimizer/scheduler checkpoint files missing")
        require(self.optimizer is not None and self.lr_scheduler is not None,"optimizer and scheduler must exist before resume restore")
        check_torch_load_is_safe();state,report,mapped=mmap_load_optimizer_state(optimizer_path)
        self.optimizer.load_state_dict(state)
        report.update(assert_materialized_without_alias(self.optimizer,mapped))
        del mapped,state;gc.collect()
        # Scheduler state is tiny and remains on the stock safe loader path.
        check_torch_load_is_safe();scheduler_state=torch.load(scheduler_path,map_location="cpu",weights_only=True)
        self.lr_scheduler.load_state_dict(scheduler_state);del scheduler_state
        report.update({"scheduler_path":str(scheduler_path.resolve()),"mode":"attested_fd_mmap_cpu_then_owned_optimizer_materialization","rng_restore_owned_by_transformers":True,"model_restore_owned_by_transformers":True})
        self.mmap_optimizer_load_report=report
