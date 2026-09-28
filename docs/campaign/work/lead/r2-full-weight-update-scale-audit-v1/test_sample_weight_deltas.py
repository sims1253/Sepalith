import ast
import importlib.util
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("sampler", HERE / "sample_weight_deltas.py")
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)


def fixture(delta=0.01, omit=None):
    values = {}
    for layer in sampler.LAYERS:
        for suffix in sampler.MATRICES.values():
            name = f"model.layers.{layer}.{suffix}"
            if name != omit:
                values[name] = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64).div(100).to(torch.bfloat16)
        for suffix in sampler.NORMS:
            name = f"model.layers.{layer}.{suffix}"
            if name != omit:
                values[name] = torch.ones(64, dtype=torch.bfloat16)
    return values, {k: v.float().add(delta).to(v.dtype) for k, v in values.items()}


def test_bounded_slice_sampler_and_metrics():
    parent, candidate = fixture()
    with tempfile.TemporaryDirectory() as td:
        p, c = Path(td) / "p.safetensors", Path(td) / "c.safetensors"
        save_file(parent, p); save_file(candidate, c)
        result = sampler.sample(p, c, "a" * 64, "b" * 64)
    assert len(result["rows"]) == 27
    assert 0 < result["sampling"]["logical_tensor_payload_bytes"] <= sampler.MAX_LOGICAL_BYTES
    assert all(r["delta_rms"] > 0 for r in result["rows"])
    assert result["bindings"]["parent"]["full_hash_recomputed_by_sampler"] is False


def test_key_mismatch_rejected():
    parent, candidate = fixture()
    candidate.pop(next(iter(candidate)))
    with tempfile.TemporaryDirectory() as td:
        p, c = Path(td) / "p.safetensors", Path(td) / "c.safetensors"
        save_file(parent, p); save_file(candidate, c)
        try: sampler.sample(p, c, "a" * 64, "b" * 64)
        except ValueError as e: assert "key sets differ" in str(e)
        else: raise AssertionError("key mismatch accepted")


def test_source_never_calls_full_tensor_loader():
    tree = ast.parse((HERE / "sample_weight_deltas.py").read_text())
    attrs = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert "get_tensor" not in attrs
    assert "get_slice" in attrs


def test_analysis_metrics_are_derived_from_slice_rows():
    rows = json.loads((HERE / "slice-results.json").read_text())["rows"]
    analysis = json.loads((HERE / "analysis.json").read_text())
    matrix = [r for r in rows if r["kind"] == "matrix"]
    import math
    geometric = math.exp(sum(math.log(r["relative_delta_rms"]) for r in matrix) / len(matrix))
    assert abs(geometric - analysis["measured"]["matrix_relative_delta_rms_geometric_mean"]) < 1e-15
    assert min(r["relative_delta_rms"] for r in matrix) == analysis["measured"]["matrix_relative_delta_rms_min"]
    assert max(r["relative_delta_rms"] for r in matrix) == analysis["measured"]["matrix_relative_delta_rms_max"]
    assert analysis["optimizer_scale_audit"]["explicit_layer_or_depth_scaling"] is False
