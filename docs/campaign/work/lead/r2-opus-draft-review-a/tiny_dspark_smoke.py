"""CPU-only structural smoke for the MiniCPM5 DSpark compatibility seam.

The test uses a randomly initialized five-layer, 16-hidden-dimension
Transformers LlamaForCausalLM.  It does not read campaign weights.  A tiny
DSpark-shaped draft consumes five target tap tensors, runs a forward pass,
and computes CE plus hidden-state L1 loss with an explicit mask.
"""

from __future__ import annotations

import json
from types import SimpleNamespace


def run_smoke() -> dict[str, object]:
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return {"skipped": True, "reason": f"torch/transformers unavailable: {exc}"}

    from minicpm5_dspark_config import build_draft_config, serving_stop_token_ids

    torch.manual_seed(0)
    target_config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=[1, 63],
        attention_dropout=0.0,
    )
    source_had_sliding_window = hasattr(target_config, "sliding_window")
    draft_dict = build_draft_config(
        target_config,
        draft_layers=2,
        block_size=2,
        target_layer_ids=(0, 1, 2, 3, 4),
        num_anchors=1,
        markov_rank=0,
    )
    if "sliding_window" not in draft_dict:
        raise AssertionError("compatibility patch did not add sliding_window")

    target = LlamaForCausalLM(target_config)
    target.eval()
    input_ids = torch.tensor([[0, 4, 7, 9, 11, 13]], dtype=torch.long)
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
    with torch.no_grad():
        target_out = target(input_ids=input_ids, output_hidden_states=True)
    # Llama returns embedding output followed by one state per layer.  The
    # five states below stand in for the configured target layer taps.
    target_hidden = torch.cat(tuple(target_out.hidden_states[index + 1] for index in range(5)), dim=-1)
    target_last = target_out.hidden_states[-1]

    class TinyDSpark(nn.Module):
        def __init__(self, config: dict[str, object]) -> None:
            super().__init__()
            self.config = SimpleNamespace(**config)
            # This direct read is the same compatibility seam as the official
            # Qwen3DSparkAttention constructor.
            self.sliding_window = self.config.sliding_window
            hidden = int(self.config.hidden_size)
            vocab = int(self.config.vocab_size)
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.target_fuse = nn.Linear(len(self.config.target_layer_ids) * hidden, hidden)
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden),
                        nn.Linear(hidden, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, hidden),
                    )
                    for _ in range(int(self.config.num_hidden_layers))
                ]
            )
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def forward(
            self,
            ids: torch.Tensor,
            target_taps: torch.Tensor,
            target_last_hidden: torch.Tensor,
            mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden = self.embed_tokens(ids) + self.target_fuse(target_taps)
            for layer in self.layers:
                hidden = hidden + layer(hidden)
            logits = self.lm_head(hidden)
            labels = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
            keep = mask.bool()
            ce = F.cross_entropy(logits[keep], labels[keep])
            l1 = F.smooth_l1_loss(hidden[keep], target_last_hidden[keep])
            return logits, ce, l1

    draft = TinyDSpark(draft_dict)
    logits, ce, l1 = draft(input_ids, target_hidden, target_last, loss_mask)
    loss = 0.1 * ce + 0.9 * l1
    loss.backward()
    finite = bool(torch.isfinite(loss).item()) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in draft.parameters()
    )
    if logits.shape != (1, 6, 64) or not finite:
        raise AssertionError("tiny DSpark forward/loss smoke failed")
    return {
        "skipped": False,
        "target_forward": True,
        "target_model": "random_tiny_transformers.LlamaForCausalLM",
        "target_config_had_sliding_window": source_had_sliding_window,
        "compat_sliding_window": draft_dict["sliding_window"],
        "draft_forward": True,
        "draft_logits_shape": list(logits.shape),
        "loss_mask_tokens": int(loss_mask.sum().item()),
        "ce_finite": bool(torch.isfinite(ce).item()),
        "l1_finite": bool(torch.isfinite(l1).item()),
        "loss_finite_and_gradients_finite": finite,
        "serving_stop_token_ids": list(serving_stop_token_ids(target_config)),
        "campaign_weights_read": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), sort_keys=True))


if __name__ == "__main__":
    main()
