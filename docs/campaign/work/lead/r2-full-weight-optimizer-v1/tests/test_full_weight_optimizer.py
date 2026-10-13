import copy
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from full_weight_optimizer import (  # noqa: E402
    FullWeightOptimizerTrainerMixin, OptimizerConfig, build_full_weight_optimizer,
    install_on_trainer, inventory_full_weight,
)


class TinyMiniCPM(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(19, 4, dtype=dtype)
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        for name, shape in (("q_proj", (4, 4)), ("k_proj", (2, 4)),
                            ("v_proj", (2, 4)), ("o_proj", (4, 4))):
            setattr(layer.self_attn, name, torch.nn.Linear(shape[1], shape[0], bias=False, dtype=dtype))
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_proj = torch.nn.Linear(4, 8, bias=False, dtype=dtype)
        layer.mlp.up_proj = torch.nn.Linear(4, 8, bias=False, dtype=dtype)
        layer.mlp.down_proj = torch.nn.Linear(8, 4, bias=False, dtype=dtype)
        layer.input_layernorm = torch.nn.LayerNorm(4, elementwise_affine=True, bias=False, dtype=dtype)
        self.model.layers = torch.nn.ModuleList([layer])
        self.model.norm = torch.nn.LayerNorm(4, elementwise_affine=True, bias=False, dtype=dtype)
        self.lm_head = torch.nn.Linear(4, 19, bias=False, dtype=dtype)


def cfg(arm="aurora_mix"):
    return OptimizerConfig(arm=arm, hidden_lr=2e-3, side_lr=7e-4,
                           ns_steps=2, aurora_K=1, weight_decay=0.01)


def gradients(model, seed):
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.randn(p.shape, generator=generator, dtype=torch.float32)
            for name, p in model.named_parameters()}


def apply_gradients(model, values):
    for name, p in model.named_parameters():
        p.grad = values[name].to(p.dtype)


@pytest.mark.parametrize("arm,counts,groups", [
    ("adamw_anchor", {"adamw": 11, "muon": 0, "aurora": 0}, 1),
    ("muon_hybrid", {"adamw": 4, "muon": 7, "aurora": 0}, 2),
    ("aurora_mix", {"adamw": 4, "muon": 5, "aurora": 2}, 2),
])
def test_all_parameters_exactly_once_and_expected_minicpm_dispatch(arm, counts, groups):
    model = TinyMiniCPM()
    optimizer, manifest = build_full_weight_optimizer(model, cfg(arm))
    assert manifest["counts"] == counts
    assert len(optimizer.param_groups) == groups
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len(list(model.parameters())) == len(set(map(id, grouped)))


def test_fail_closed_for_peft_frozen_unknown_2d_and_tied_parameters():
    model = TinyMiniCPM()
    model.model.layers[0].mlp.gate_proj.register_parameter(
        "lora_A_default", torch.nn.Parameter(torch.randn(2, 4)))
    with pytest.raises(ValueError, match="PEFT"):
        inventory_full_weight(model, "aurora_mix")
    model = TinyMiniCPM(); model.model.norm.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="frozen"):
        inventory_full_weight(model, "aurora_mix")
    model = TinyMiniCPM(); model.unreviewed_matrix = torch.nn.Parameter(torch.randn(3, 3))
    with pytest.raises(ValueError, match="unknown trainable 2-D"):
        inventory_full_weight(model, "aurora_mix")
    model = TinyMiniCPM(); model.lm_head.weight = model.model.embed_tokens.weight
    with pytest.raises(ValueError, match="tied/duplicated"):
        inventory_full_weight(model, "aurora_mix")


def test_fp32_state_policy_and_every_weight_receives_update():
    model = TinyMiniCPM(dtype=torch.float32)
    optimizer, _ = build_full_weight_optimizer(model, cfg())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    apply_gradients(model, gradients(model, 31))
    optimizer.step()
    for name, param in model.named_parameters():
        assert not torch.equal(before[name], param), name
    for state in optimizer.state.values():
        for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
            if key in state:
                assert state[key].dtype == torch.float32


def test_bfloat16_parameters_still_use_fp32_gradients_moments_and_orthogonalization_state():
    model = TinyMiniCPM(dtype=torch.bfloat16)
    optimizer, _ = build_full_weight_optimizer(model, cfg())
    apply_gradients(model, gradients(model, 32)); optimizer.step()
    assert optimizer.state
    for state in optimizer.state.values():
        for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
            if key in state:
                assert state[key].dtype == torch.float32


def test_stochastic_rounding_preserves_sub_ulp_update_signal_without_master_copy():
    class SideOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(16384, 1, dtype=torch.bfloat16)
            self.lm_head = torch.nn.Linear(1, 16384, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                for p in self.parameters(): p.fill_(1.0)
    model = SideOnly()
    config = OptimizerConfig(arm="adamw_anchor", hidden_lr=1e-5, side_lr=1e-5,
                             weight_decay=0.0)
    optimizer, _ = build_full_weight_optimizer(model, config)
    before = torch.cat([p.detach().flatten().float() for p in model.parameters()])
    for p in model.parameters(): p.grad = torch.ones_like(p)
    # Round-to-nearest would keep every 1.0 value unchanged for this 1e-5 step.
    deterministic = (before - 1e-5).to(torch.bfloat16).float()
    assert torch.equal(before, deterministic)
    torch.manual_seed(321)
    optimizer.step()
    after = torch.cat([p.detach().flatten().float() for p in model.parameters()])
    changed = torch.count_nonzero(after != before).item()
    assert changed > 0
    assert abs(float((before - after).mean()) - 1e-5) < 4e-6


def test_live_interruption_restore_matches_parameters_state_scheduler_and_next_lr():
    torch.manual_seed(44)
    live = TinyMiniCPM()
    optimizer, manifest = build_full_weight_optimizer(live, cfg())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - 0.04 * step)
    for seed in (51, 52):
        apply_gradients(live, gradients(live, seed)); optimizer.step(); scheduler.step()
    saved_weights = copy.deepcopy(live.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())

    restored = TinyMiniCPM(); restored.load_state_dict(saved_weights)
    restored_optimizer, restored_manifest = build_full_weight_optimizer(restored, cfg())
    assert restored_manifest["ordered_rows_sha256"] == manifest["ordered_rows_sha256"]
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0 - 0.04 * step)
    restored_optimizer.load_state_dict(saved_optimizer)
    restored_scheduler.load_state_dict(saved_scheduler)
    next_grads = gradients(live, 53)
    apply_gradients(live, next_grads); apply_gradients(restored, next_grads)
    optimizer.step(); scheduler.step(); restored_optimizer.step(); restored_scheduler.step()
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    assert [g["lr"] for g in optimizer.param_groups] == [g["lr"] for g in restored_optimizer.param_groups]
    for (name_a, a), (name_b, b) in zip(live.named_parameters(), restored.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        state_a, state_b = optimizer.state[a], restored_optimizer.state[b]
        assert state_a.keys() == state_b.keys()
        for key in state_a:
            if torch.is_tensor(state_a[key]):
                torch.testing.assert_close(state_a[key], state_b[key], rtol=0, atol=0)
            else:
                assert state_a[key] == state_b[key]


def test_bfloat16_interruption_restores_rng_for_exact_stochastic_next_step():
    torch.manual_seed(81)
    live = TinyMiniCPM(dtype=torch.bfloat16)
    optimizer, _ = build_full_weight_optimizer(live, cfg())
    apply_gradients(live, gradients(live, 82)); optimizer.step()
    weights = copy.deepcopy(live.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    rng_state = torch.get_rng_state()
    restored = TinyMiniCPM(dtype=torch.bfloat16); restored.load_state_dict(weights)
    restored_optimizer, _ = build_full_weight_optimizer(restored, cfg())
    restored_optimizer.load_state_dict(optimizer_state)
    values = gradients(live, 83)
    torch.set_rng_state(rng_state)
    apply_gradients(live, values); optimizer.step()
    torch.set_rng_state(rng_state)
    apply_gradients(restored, values); restored_optimizer.step()
    for a, b in zip(live.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_state_load_refuses_arm_config_dispatch_and_group_drift():
    model = TinyMiniCPM(); optimizer, _ = build_full_weight_optimizer(model, cfg())
    apply_gradients(model, gradients(model, 61)); optimizer.step()
    state = optimizer.state_dict()
    other, _ = build_full_weight_optimizer(TinyMiniCPM(), cfg("muon_hybrid"))
    with pytest.raises(ValueError, match="arm differs"):
        other.load_state_dict(state)
    corrupted = copy.deepcopy(state); corrupted["param_groups"][0]["param_names"].reverse()
    with pytest.raises(ValueError, match="group structure"):
        build_full_weight_optimizer(TinyMiniCPM(), cfg())[0].load_state_dict(corrupted)
    corrupted = copy.deepcopy(state); corrupted["_sepalith"]["dispatch_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest differs"):
        build_full_weight_optimizer(TinyMiniCPM(), cfg())[0].load_state_dict(corrupted)


def test_adamw_anchor_matches_torch_adamw_on_float32_reference():
    torch.manual_seed(71)
    actual = TinyMiniCPM(); reference = copy.deepcopy(actual)
    config = OptimizerConfig(arm="adamw_anchor", hidden_lr=8e-4, side_lr=8e-4,
                             weight_decay=0.02, adam_betas=(0.9, 0.98), adam_eps=1e-8)
    optimizer, _ = build_full_weight_optimizer(actual, config)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=8e-4,
                                             betas=(0.9, 0.98), eps=1e-8,
                                             weight_decay=0.02, foreach=False)
    for seed in (72, 73, 74):
        values = gradients(actual, seed)
        apply_gradients(actual, values); apply_gradients(reference, values)
        optimizer.step(); reference_optimizer.step()
    for a, b in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)


def test_trainer_mixin_installs_optimizer_and_exposes_manifest():
    class FakeTrainer(FullWeightOptimizerTrainerMixin):
        def __init__(self):
            self.model = TinyMiniCPM(); self.optimizer = None
            self.full_weight_optimizer_config = cfg()
    trainer = FakeTrainer()
    first = trainer.create_optimizer(); second = trainer.create_optimizer()
    assert first is second
    assert trainer.full_weight_optimizer_manifest["counts"] == {"adamw": 4, "muon": 5, "aurora": 2}


def test_install_on_existing_trainer_must_precede_optimizer_and_scheduler_creation():
    class Trainer:
        def __init__(self):
            self.model = TinyMiniCPM(); self.optimizer = None; self.lr_scheduler = None
    trainer = Trainer(); manifest = install_on_trainer(trainer, cfg())
    assert trainer.optimizer is not None and trainer.full_weight_optimizer_manifest is manifest
    with pytest.raises(ValueError, match="already exists"):
        install_on_trainer(trainer, cfg())
    trainer = Trainer(); trainer.lr_scheduler = object()
    with pytest.raises(ValueError, match="scheduler already exists"):
        install_on_trainer(trainer, cfg())
