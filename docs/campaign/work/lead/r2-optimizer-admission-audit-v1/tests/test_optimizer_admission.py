import copy
import importlib.util
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from optimizer_admission import (  # noqa: E402
    classify_full_weight, classify_lora, ordered_manifest, verify_manifest,
)


def test_full_weight_dispatch_keeps_side_adamw_and_limits_aurora_scope():
    assert classify_full_weight("model.layers.0.mlp.gate_proj.weight", (6144, 2048)).optimizer == "aurora"
    assert classify_full_weight("model.layers.0.mlp.up_proj.weight", (6144, 2048)).optimizer == "aurora"
    assert classify_full_weight("model.layers.0.mlp.down_proj.weight", (2048, 6144)).optimizer == "muon"
    assert classify_full_weight("model.layers.0.self_attn.q_proj.weight", (2048, 2048)).optimizer == "muon"
    assert classify_full_weight("model.embed_tokens.weight", (130560, 2048)).optimizer == "adamw"
    assert classify_full_weight("model.layers.0.input_layernorm.weight", (2048,)).optimizer == "adamw"
    with pytest.raises(ValueError, match="PEFT factor"):
        classify_full_weight("base_model.model.layers.0.mlp.gate_proj.lora_B.default.weight", (6144, 32))


def test_lora_translation_is_explicit_and_original_suffixes_match_none():
    names = [
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight",
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight",
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight",
    ]
    assert not any(n.endswith(("Wg.weight", "Wu.weight")) for n in names)
    assert classify_lora(names[0], (32, 2048)).optimizer == "muon"
    assert classify_lora(names[1], (6144, 32)).optimizer == "aurora"
    assert classify_lora(names[2], (2048, 32)).optimizer == "muon"


def test_ordered_manifest_refuses_reordered_or_changed_dispatch():
    rows = [
        classify_full_weight("model.layers.0.mlp.gate_proj.weight", (6144, 2048), "bf16"),
        classify_full_weight("model.layers.0.mlp.down_proj.weight", (2048, 6144), "bf16"),
    ]
    expected = ordered_manifest(rows)
    verify_manifest(expected, copy.deepcopy(expected))
    reordered = ordered_manifest(list(reversed(rows)))
    with pytest.raises(ValueError, match="state load refused"):
        verify_manifest(expected, reordered)
    corrupted = copy.deepcopy(expected)
    corrupted["ordered_entries"][0]["optimizer"] = "muon"
    with pytest.raises(ValueError, match="self-hash"):
        verify_manifest(expected, corrupted)
    with pytest.raises(ValueError, match="duplicate"):
        ordered_manifest([rows[0], rows[0]])


def test_zero_initialized_b_delays_a_gradient_but_not_b_gradient():
    torch.manual_seed(7)
    x = torch.randn(4, 6)
    a = torch.randn(2, 6, requires_grad=True)
    b = torch.zeros(5, 2, requires_grad=True)
    loss = ((x @ a.T) @ b.T).square().sum() + ((x @ a.T) @ b.T).sum()
    loss.backward()
    assert torch.count_nonzero(a.grad) == 0
    assert torch.count_nonzero(b.grad) > 0


def _load_aurora_module():
    path = pathlib.Path("experiments/training/poc_twin/arms/aurora.py").resolve()
    spec = importlib.util.spec_from_file_location("campaign_aurora", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aurora_resume_reconstructs_identical_next_step_with_same_order():
    aurora = _load_aurora_module()
    torch.manual_seed(9)
    initial = [torch.nn.Parameter(torch.randn(8, 2)), torch.nn.Parameter(torch.randn(4, 4))]
    names = ["layer.gate_proj.weight", "layer.q_proj.weight"]
    kwargs = dict(lr=1e-3, momentum=0.9, ns_steps=1, weight_decay=0.0,
                  aurora_K=1, aurora_names=("gate_proj.weight",))
    opt = aurora.Aurora(list(zip(names, initial)), **kwargs)
    for p in initial:
        p.grad = torch.randn_like(p)
    opt.step()
    saved_params = [p.detach().clone() for p in initial]
    saved_state = copy.deepcopy(opt.state_dict())

    resumed = [torch.nn.Parameter(p.clone()) for p in saved_params]
    opt_r = aurora.Aurora(list(zip(names, resumed)), **kwargs)
    opt_r.load_state_dict(copy.deepcopy(saved_state))
    torch.manual_seed(11)
    grads = [torch.randn_like(p) for p in initial]
    for p, g in zip(initial, grads):
        p.grad = g.clone()
    for p, g in zip(resumed, grads):
        p.grad = g.clone()
    opt.step(); opt_r.step()
    for a, b in zip(initial, resumed):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(
            opt.state[a]["momentum_buffer"], opt_r.state[b]["momentum_buffer"], rtol=0, atol=0
        )


def test_aurora_dispatch_flags_are_not_serialized_so_manifest_is_required():
    aurora = _load_aurora_module()
    params = [torch.nn.Parameter(torch.randn(8, 2)), torch.nn.Parameter(torch.randn(4, 4))]
    names = ["layer.gate_proj.weight", "layer.q_proj.weight"]
    opt = aurora.Aurora(list(zip(names, params)), aurora_names=("gate_proj.weight",))
    state = opt.state_dict()
    assert "_aurora_flags" not in state
    assert opt._aurora_flags == [True, False]
