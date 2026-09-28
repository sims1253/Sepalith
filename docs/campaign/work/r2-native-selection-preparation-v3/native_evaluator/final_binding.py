"""Source and registered-tensor checks for the existing final evaluator.

This module does not load checkpoints, create freeze receipts, or open rows.
Root must derive the expected tensor manifest from the selected artifact in a
separate, admitted load and pin the manifest in its freeze receipt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys


class BindingError(ValueError):
    pass


def digest_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise BindingError("sha256_required")
    return value


SOURCE_HASH_CHUNK_BYTES = 4 * 1024 * 1024


def source_file_sha256(path):
    """Hash source or native-library bytes with at most one 4 MiB read."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(SOURCE_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_closure(manifest, expected_sha256):
    """Verify a root-pinned, explicit graph; never infer admission from a scan.

    Missing external dependencies keep a preparation inventory unadmittable.
    A fresh root-owned process and an independently reviewed graph are trust
    boundaries; this is not a Python sandbox or a proof against monkeypatches.
    """
    if digest_json(manifest) != _sha(expected_sha256):
        raise BindingError("source_closure_digest_mismatch")
    if manifest.get("schema") != "dat08.source-closure.v1":
        raise BindingError("source_closure_schema_mismatch")
    if manifest.get("status") != "root_admitted" or manifest.get("unresolved") != []:
        raise BindingError("source_closure_not_admitted")
    files = manifest.get("files")
    roots = manifest.get("roots")
    if not isinstance(files, dict) or not files or not isinstance(roots, list) or not roots:
        raise BindingError("source_closure_graph_required")
    reachable = set()

    def visit(name):
        if name not in files:
            raise BindingError(f"source_dependency_missing:{name}")
        if name in reachable:
            return
        reachable.add(name)
        entry = files[name]
        if not isinstance(entry.get("dependencies"), list):
            raise BindingError(f"source_dependencies_required:{name}")
        for dependency in entry["dependencies"]:
            visit(dependency)

    for root in roots:
        visit(root)
    if reachable != set(files):
        raise BindingError("source_closure_unreachable_members")
    # An inventory cannot become a complete closure merely by changing its
    # status. Every recorded third-party boundary needs a source-file binding
    # in the admitted graph. Root still reviews dynamic/native dependencies.
    module_members = {}
    for name, entry in files.items():
        for module_name in entry.get("modules", []):
            module_members.setdefault(module_name, set()).add(name)
    for name, entry in files.items():
        for imported in entry.get("external_imports", []):
            targets = module_members.get(imported)
            if not targets or not targets.issubset(set(entry["dependencies"])):
                raise BindingError(f"external_source_binding_missing:{name}:{imported}")
    module_paths = {}
    for name, entry in sorted(files.items()):
        path = Path(entry["path"])
        if not path.is_absolute() or ".." in path.parts:
            raise BindingError(f"source_path_invalid:{name}")
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise BindingError(f"source_path_symlink:{name}")
        if source_file_sha256(path) != _sha(entry["sha256"]):
            raise BindingError(f"source_bytes_mismatch:{name}")
        for module_name in entry.get("modules", []):
            if module_name in module_paths and module_paths[module_name] != path:
                raise BindingError(f"source_module_ambiguous:{module_name}")
            module_paths[module_name] = path
            module = sys.modules.get(module_name)
            if module is not None:
                origin = getattr(module, "__file__", None)
                if not isinstance(origin, str) or Path(origin).resolve() != path.resolve():
                    raise BindingError(f"loaded_source_path_mismatch:{module_name}")
    return {"source_closure_sha256": expected_sha256, "source_files": len(files)}


def verify_freeze_bindings(freeze, tensor_manifest, source_manifest):
    """Require root freeze pins in addition to the existing time/weight gates."""
    tensor_sha = digest_json(tensor_manifest)
    source_sha = digest_json(source_manifest)
    if tensor_sha != _sha(freeze.get("loaded_tensor_manifest_sha256")):
        raise BindingError("frozen_tensor_manifest_mismatch")
    if source_sha != _sha(freeze.get("source_closure_sha256")):
        raise BindingError("frozen_source_closure_mismatch")
    if tensor_manifest.get("weights_sha256") != freeze.get("weights_sha256"):
        raise BindingError("tensor_manifest_weights_mismatch")
    if tensor_manifest.get("schema") != "dat08.loaded-tensors.v1":
        raise BindingError("tensor_manifest_schema_mismatch")
    verify_source_closure(source_manifest, source_sha)
    return tensor_sha


def _tensor_record(tensor, kind):
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.is_meta or tensor.is_quantized or tensor.layout != torch.strided:
        raise BindingError("dense_materialized_tensor_required")
    version = tensor._version
    metadata = {"kind": kind, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    # Hash bounded chunks, including exact bfloat16/float16 bits. No cast or
    # tolerance can admit different loaded values. Scalar/empty tensors work.
    flat = tensor.detach().reshape(-1)
    elements = max(1, (4 * 1024 * 1024) // tensor.element_size())
    digest = hashlib.sha256()
    for offset in range(0, flat.numel(), elements):
        block = flat[offset:offset + elements].to(device="cpu").contiguous()
        digest.update(block.view(torch.uint8).numpy().tobytes())
    if version != tensor._version:
        raise BindingError("tensor_changed_during_hash")
    return {**metadata, "sha256": digest.hexdigest()}


def tensor_manifest_from_model(model, weights_sha256):
    """Inspect all registered parameters AND buffers, including aliases.

    Root may use this with its independent reference load. Calling it on the
    evaluation model does not make that model admitted: the resulting digest
    must equal the separately frozen reference digest. No path is read.
    """
    import torch

    if not isinstance(model, torch.nn.Module):
        raise BindingError("torch_module_required")
    records = {}
    ancestors = set()

    def visit(module, prefix):
        if id(module) in ancestors:
            raise BindingError("cyclic_module_registration")
        ancestors.add(id(module))
        # Inspect registered tensors directly. A custom state_dict override
        # cannot substitute checkpoint tensors for the model's actual state.
        for kind, collection in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, tensor in sorted(collection.items()):
                if tensor is not None:
                    key = prefix + name
                    if key in records:
                        raise BindingError("duplicate_registered_tensor")
                    records[key] = _tensor_record(tensor, kind)
        for name, child in sorted(module._modules.items()):
            if child is not None:
                visit(child, prefix + name + ".")
        ancestors.remove(id(module))

    visit(model, "")
    if not records:
        raise BindingError("registered_tensors_required")
    configs = {}
    for name in ("config", "generation_config"):
        value = getattr(model, name, None)
        if value is not None:
            method = getattr(value, "to_dict", None)
            if not callable(method):
                raise BindingError(f"serializable_model_config_required:{name}")
            configs[name] = method()
    return {
        "schema": "dat08.loaded-tensors.v1",
        "weights_sha256": _sha(weights_sha256),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "config_sha256": digest_json(configs),
        "tensors": records,
    }


def verify_loaded_model(model, manifest, expected_sha256):
    if digest_json(manifest) != _sha(expected_sha256):
        raise BindingError("admitted_tensor_manifest_mismatch")
    observed = tensor_manifest_from_model(model, manifest["weights_sha256"])
    if observed != manifest:
        raise BindingError("loaded_model_tensor_identity_mismatch")
    return {
        "status": "loaded_registered_tensors_match_frozen_manifest",
        "weights_sha256": manifest["weights_sha256"],
        "loaded_tensor_manifest_sha256": digest_json(observed),
        "registered_tensors": len(observed["tensors"]),
        "model_class": observed["model_class"],
        "config_sha256": observed["config_sha256"],
    }
