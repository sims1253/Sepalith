"""Durable checkpoint receipts and evaluation hooks for the Tuesday campaign.

No model or CUDA imports occur until the callback is constructed. Structural
checkpoint verification does not prove faithful resume; run the recipe's
interruption test before admitting a long training job.
"""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile


IDENTITY_FIELDS = {"parent", "tokenizer", "renderer", "data", "source", "policy", "schedule"}
FULL_STATE_FILES = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")


def flush_directory(path):
    try:
        directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise


def flush_files(directory):
    for path in Path(directory).rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Some destination mounts do not support directory fsync.
        flush_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def check_identity(identity):
    missing = IDENTITY_FIELDS - identity.keys()
    if missing or any(identity[k] in (None, "", {}) for k in IDENTITY_FIELDS if k in identity):
        raise ValueError(f"Incomplete training identity: {sorted(missing)}")
    # Canonical serialization rejects nonfinite metrics and non-JSON state.
    return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))


def inventory(directory):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint contains symlink: {path}")
        if path.is_file() and path.name != "campaign-manifest.json":
            if not path.stat().st_size:
                raise ValueError(f"Empty checkpoint file: {path}")
            files[str(path.relative_to(directory))] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return files


def seal_checkpoint(directory, identity, step, *, full, sampler, checkpoint_kind="adapter"):
    directory = Path(directory)
    identity = check_identity(identity)
    if checkpoint_kind == "full_weights":
        from model_layout import validate_dense_weights
        validate_dense_weights(directory)
    elif checkpoint_kind == "adapter":
        if not (directory / "adapter_model.safetensors").is_file() or not (directory / "adapter_config.json").is_file():
            raise ValueError("Campaign LoRA checkpoint must contain adapter weights and config")
    else:
        raise ValueError("Unknown checkpoint kind")
    if full:
        missing = [name for name in FULL_STATE_FILES if not (directory / name).is_file()]
        if missing:
            raise ValueError(f"Checkpoint is not resumable; missing {missing}")
        state = json.loads((directory / "trainer_state.json").read_text())
        if state["global_step"] != step:
            raise ValueError("Trainer step does not match checkpoint identity")
        if not sampler:
            raise ValueError("Full checkpoint requires sampler reconstruction/state metadata")
    write_json(directory / "campaign-state.json", {
        "identity": identity, "step": step, "full": full, "sampler": sampler,
        "resume_validation": "Requires matched interruption/resume receipt; file presence alone is insufficient",
        "checkpoint_kind": checkpoint_kind,
    })
    flush_files(directory)
    files = inventory(directory)
    manifest = {"schema_version": 1, "step": step, "full": full, "identity": identity, "files": files, "checkpoint_kind": checkpoint_kind}
    write_json(directory / "campaign-manifest.json", manifest)
    return manifest


def verify_checkpoint(directory, identity=None, *, require_full=False, expected_checkpoint_kind=None):
    directory = Path(directory)
    manifest = json.loads((directory / "campaign-manifest.json").read_text())
    if identity is not None and manifest["identity"] != check_identity(identity):
        raise ValueError("Checkpoint identity differs from the requested recipe")
    if require_full and not manifest["full"]:
        raise ValueError("A lightweight adapter is not a full resume checkpoint")
    if inventory(directory) != manifest["files"]:
        raise ValueError("Checkpoint bytes differ from the sealed manifest")
    state = json.loads((directory / "campaign-state.json").read_text())
    kind = manifest.get("checkpoint_kind", "adapter")
    if kind not in ("adapter", "full_weights"):
        raise ValueError("Unknown checkpoint kind")
    if expected_checkpoint_kind is not None and kind != expected_checkpoint_kind:
        raise ValueError("Checkpoint kind differs from requested training mode")
    if state.get("checkpoint_kind", "adapter") != kind:
        raise ValueError("Checkpoint kind differs between state and manifest")
    for key in ("identity", "step", "full"):
        if state.get(key) != manifest.get(key):
            raise ValueError("Checkpoint state and manifest disagree: " + key)
    if kind == "full_weights":
        from model_layout import validate_dense_weights
        validate_dense_weights(directory)
    elif not (directory / "adapter_model.safetensors").is_file() or not (directory / "adapter_config.json").is_file():
        raise ValueError("Adapter checkpoint has no adapter weights/config")
    if manifest["full"]:
        missing = [name for name in FULL_STATE_FILES if not (directory / name).is_file()]
        if missing or not state.get("sampler"):
            raise ValueError("Full checkpoint lacks resumable state")
        if json.loads((directory / "trainer_state.json").read_text()).get("global_step") != manifest["step"]:
            raise ValueError("Trainer step differs from sealed checkpoint")
    return manifest


def archive_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    expected = verify_checkpoint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if verify_checkpoint(destination) != expected:
            raise ValueError(f"Archive destination has a different identity: {destination}")
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        if verify_checkpoint(temporary) != expected:
            raise ValueError("Archive verification failed")
        flush_files(temporary)
        flush_directory(temporary)
        os.rename(temporary, destination)
        flush_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


@contextmanager
def preserve_random_state(model):
    import numpy as np
    import torch

    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def checkpoint_callback(*, identity, archive_root, tokenizer, light_every, full_every,
                        evaluator=None, sampler_state=None, allow_pending_evaluation=False,
                        evaluation_allowed=None, evaluation_steps=None, evaluation_context=None,
                        checkpoint_kind="adapter"):
    """Create a TrainerCallback; evaluator runs synchronously in the CUDA owner.

    The caller must configure full saves at full_every and use a matched,
    deterministic sampler. A custom or buffered rollout sampler must supply
    sampler_state and a validated restore implementation in its trainer.
    """
    from transformers import TrainerCallback

    identity = check_identity(identity)
    archive_root = Path(archive_root)
    if light_every < 1 or full_every < 1 or full_every % light_every:
        raise ValueError("Full cadence must be a positive multiple of adapter cadence")
    if evaluation_steps is not None:
        evaluation_steps = frozenset(evaluation_steps)
        if not evaluation_steps or any(type(step) is not int or step < 1 for step in evaluation_steps):
            raise ValueError("Evaluation steps must be explicit positive checkpoint steps")

    class CampaignCheckpointCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if evaluator is None and not allow_pending_evaluation:
                raise ValueError("Install a development evaluator before campaign training")
            if args.save_steps != full_every or str(args.save_strategy) not in ("steps", "SaveStrategy.STEPS"):
                raise ValueError("Trainer full-save cadence differs from campaign cadence")
            if args.save_only_model:
                raise ValueError("save_only_model cannot preserve optimizer/RNG state")
            if args.ignore_data_skip:
                raise ValueError("ignore_data_skip would restart the data sequence on resume")

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % light_every:
                return control
            path = archive_root / ("weights" if checkpoint_kind == "full_weights" else "adapters") / f"checkpoint-{state.global_step}"
            if path.exists():
                raise ValueError(f"Refusing to overwrite an adapter identity: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
            try:
                kwargs["model"].save_pretrained(temporary, safe_serialization=True)
                if tokenizer is not None:
                    tokenizer.save_pretrained(temporary)
                seal_checkpoint(temporary, identity, state.global_step, full=False, sampler=None, checkpoint_kind=checkpoint_kind)
                flush_directory(temporary)
                os.rename(temporary, path)
                flush_directory(path.parent)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return control

        def on_save(self, args, state, control, **kwargs):
            source = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            sampler = sampler_state() if sampler_state is not None else {
                "method": "Trainer reconstructs deterministic sampler and skips consumed batches",
                "seed": args.seed, "data_seed": args.data_seed, "epoch": state.epoch,
                "global_step": state.global_step, "ignore_data_skip": args.ignore_data_skip,
                "dataset_identity": identity["data"],
            }
            seal_checkpoint(source, identity, state.global_step, full=True, sampler=sampler, checkpoint_kind=checkpoint_kind)
            archived = archive_checkpoint(source, archive_root / "full" / source.name)
            request = {"checkpoint": str(archived), "step": state.global_step, "identity": identity,
                       "created_at": datetime.now(timezone.utc).isoformat()}
            if evaluation_steps is not None and state.global_step not in evaluation_steps:
                request["status"] = "archived: checkpoint not nominated for development evaluation"
            elif evaluator is None:
                request["status"] = "pending: no evaluator installed; long-run admission remains blocked"
            elif evaluation_allowed is not None and not evaluation_allowed():
                request["status"] = "deferred: time reserved for checkpoint persistence and shutdown"
            else:
                with preserve_random_state(kwargs["model"]):
                    model = kwargs["model"]
                    with evaluation_context(model) if evaluation_context else nullcontext():
                        result = evaluator(model, tokenizer, archived, state.global_step)
                if not isinstance(result, dict) or "denominators" not in result or "case_ids" not in result:
                    raise ValueError("Development result requires denominators and case IDs")
                request.update(status="evaluated", result=result)
            write_json(archive_root / "evaluations" / f"step-{state.global_step}.json", request)
            return control

    return CampaignCheckpointCallback()
