"""Learned module slots, explicit mixer LoRA, and cache-free decoder input."""
import hashlib
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load, save

from sepalith.memory import ConsumerCompatibility, LatentMemoryManifest

LAYOUT = 'latent-facts-v1:slots-then-query:consecutive-2d-positions:no-cache'


class ModuleEncoder(nn.Module):
    def __init__(self, decoder_width, width=128, layers=2, heads=4, slots=32,
                 max_bytes=1024, dropout=0.1):
        super().__init__()
        self.memory_config = dict(decoder_width=decoder_width, width=width, layers=layers, heads=heads,
                                  slots=slots, max_bytes=max_bytes, dropout=dropout)
        self.max_bytes = max_bytes
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.positions = nn.Embedding(max_bytes, width)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout,
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(slots, width) / math.sqrt(width))
        self.pool = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.projection = nn.Linear(width, decoder_width)
        self.scale = nn.Parameter(torch.ones(decoder_width))

    def forward(self, sources):
        encoded = [[byte + 1 for byte in source.encode('utf-8')] for source in sources]
        # Gate modules fit whole; reject rather than silently drop facts.
        if any(not ids or len(ids) > self.max_bytes for ids in encoded):
            raise ValueError('Module outside the frozen byte budget; no implicit truncation')
        device = self.queries.device
        ids = torch.zeros(len(encoded), max(map(len, encoded)), dtype=torch.long, device=device)
        for i, row in enumerate(encoded):
            ids[i, :len(row)] = torch.tensor(row, device=device)
        mask = ids == 0
        x = self.embedding(ids) + self.positions(torch.arange(ids.shape[1], device=device))
        x = self.encoder(x, src_key_padding_mask=mask)
        q = self.queries.unsqueeze(0).expand(len(sources), -1, -1)
        pooled, _ = self.pool(q, x, x, key_padding_mask=mask, need_weights=False)
        x = self.norm1(q + pooled)
        x = self.norm2(x + self.ffn(x))
        return self.projection(x) * self.scale

    @torch.no_grad()
    def calibrate(self, sources, decoder_embeddings):
        was_training = self.training
        self.eval()
        before = self(sources).float().square().mean().sqrt()
        desired = decoder_embeddings.float().square().mean().sqrt()
        self.scale.mul_(desired / before.clamp_min(1e-12))
        self.train(was_training)
        return {'before_rms': before.item(), 'decoder_rms': desired.item()}


class MixerLoRA(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        # Keep optimizer/master adapter parameters fp32. Cast only the forward.
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.multiplier = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.A.to(x.dtype)),
                                      self.B.to(x.dtype)) * self.multiplier


def target_names(config):
    names = []
    for i, kind in enumerate(config.layer_types):
        if kind == 'linear_attention':
            suffixes = ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj']
            prefix = f'model.layers.{i}.linear_attn'
        elif kind == 'full_attention':
            suffixes = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            prefix = f'model.layers.{i}.self_attn'
        else:
            raise ValueError(f'Unsupported layer: {kind}')
        names.extend(f'{prefix}.{suffix}' for suffix in suffixes)
    return names


def attach_adapters(model, rank=16, alpha=32, dropout=0.05):
    names = target_names(model.config)
    for name in names:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear) or layer.weight.ndim != 2:
            raise ValueError(f'Expected a 2-D linear projection: {name}')
    model._latent_adapter_config = dict(rank=rank, alpha=alpha, dropout=dropout)
    model.requires_grad_(False)
    for name in names:
        parent, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), leaf, MixerLoRA(model.get_submodule(name), rank, alpha, dropout))
    return names


def embedding_forward(model, embeddings, *, past_key_values=None, logits_to_keep=0):
    if past_key_values is not None:
        raise ValueError('Hybrid cache reuse is disabled for this experiment')
    count = embeddings.shape[1]
    positions = torch.arange(count, device=embeddings.device).unsqueeze(0).expand(embeddings.shape[0], -1)
    return model(inputs_embeds=embeddings, position_ids=positions,
                 attention_mask=torch.ones(embeddings.shape[:2], dtype=torch.long, device=embeddings.device),
                 use_cache=False, past_key_values=None, logits_to_keep=logits_to_keep)


def target_loss(model, prompt, target, slots=None):
    if prompt.shape[1] < 1 or target.shape[1] < 1:
        raise ValueError('Nonempty prompt and target required')
    text = model.get_input_embeddings()(torch.cat([prompt, target[:, :-1]], dim=1))
    prefix = 0
    if slots is not None:
        prefix = slots.shape[1]
        text = torch.cat([slots.to(text.dtype), text], dim=1)
    # Only the target prediction positions reach the expensive vocabulary head.
    logits = embedding_forward(model, text, logits_to_keep=target.shape[1]).logits
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))
    return loss, {'scored_tokens': target.numel(), 'decoder_tokens': text.shape[1],
                  'latent_positions': prefix, 'first_scored_position': prefix + prompt.shape[1] - 1}


@torch.no_grad()
def greedy(model, prompt, slots, max_new_tokens, eos_token_id, timings=None):
    generated = []
    ids = prompt
    for step in range(max_new_tokens):
        started = time.monotonic()
        text = model.get_input_embeddings()(ids)
        if slots is not None:
            text = torch.cat([slots.to(text.dtype), text], 1)
        token = embedding_forward(model, text, logits_to_keep=1).logits[:, -1].argmax(-1).item()
        generated.append(token)
        if timings is not None:
            timings.append({'step': step, 'seconds': time.monotonic() - started,
                            'forward_positions': text.shape[1]})
        if token == eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], 1)
    return generated


@torch.no_grad()
def token_embedding_contract(model, ids, atol, rtol):
    model.eval()
    position_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    direct = model(input_ids=ids, position_ids=position_ids,
                   attention_mask=torch.ones_like(ids), use_cache=False).logits
    embedded = embedding_forward(model, model.get_input_embeddings()(ids)).logits
    torch.testing.assert_close(direct, embedded, atol=atol, rtol=rtol)
    same = torch.equal(direct.argmax(-1), embedded.argmax(-1))
    if not same:
        raise ValueError('Token/embedding greedy mismatch')
    generated = []
    current = ids
    for _ in range(3):
        positions = torch.arange(current.shape[1], device=current.device).unsqueeze(0)
        a = model(input_ids=current, position_ids=positions, attention_mask=torch.ones_like(current),
                  use_cache=False, logits_to_keep=1).logits
        b = embedding_forward(model, model.get_input_embeddings()(current), logits_to_keep=1).logits
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        token = a[:, -1].argmax(-1, keepdim=True)
        if not torch.equal(token, b[:, -1].argmax(-1, keepdim=True)):
            raise ValueError('Autoregressive token/embedding mismatch')
        generated.append(token.item())
        current = torch.cat([current, token], dim=1)
    return {'max_absolute_error': (direct - embedded).abs().max().item(), 'generated_ids': generated,
            'greedy_equal': same, 'atol': atol, 'rtol': rtol, 'cache_reuse': False}


def compatibility(encoder, decoder, tokenizer, count, width, dtype, scope):
    return ConsumerCompatibility(encoder, decoder, tokenizer, count, width, dtype, LAYOUT, scope)


def check_tensor(tensor, expected):
    if expected.representation != 'input_embeddings':
        raise ValueError('Only projected input embeddings are supported')
    if tuple(tensor.shape) != (expected.latent_count, expected.latent_width):
        raise ValueError('Payload tensor shape mismatch')
    if str(tensor.dtype).removeprefix('torch.') != expected.dtype or not torch.isfinite(tensor).all():
        raise ValueError('Payload tensor dtype or numeric mismatch')


def save_memory(directory, tensor, expected, source_hashes):
    tensor = tensor.detach().cpu().contiguous()
    check_tensor(tensor, expected)
    payload = save({'slots': tensor})
    payload_hash = hashlib.sha256(payload).hexdigest()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = payload_hash + '.safetensors'
    (root / name).write_bytes(payload)
    manifest = LatentMemoryManifest(expected, name, payload_hash, source_hashes)
    (root / (payload_hash + '.manifest.json')).write_text(manifest.to_json())
    return manifest


def load_memory(directory, manifest, expected, current_hashes):
    manifest.validate_for_consumer(expected, current_hashes=current_hashes)
    root = Path(directory).resolve()
    path = (root / manifest.payload_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Payload escapes artifact root')
    # Hash and deserialize the SAME bytes; no verify/reopen race.
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.payload_sha256:
        raise ValueError('Payload SHA256 mismatch')
    tensors = load(payload)
    if set(tensors) != {'slots'}:
        raise ValueError('Unexpected payload tensors')
    check_tensor(tensors['slots'], expected)
    return tensors['slots']
