"""Public standard-architecture LoRA and independent second-GPU probe."""
import hashlib
import json
import os
from pathlib import Path
import time

MODEL = 'HuggingFaceTB/SmolLM2-135M'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
TEXTS = [
    'mean_finite <- function(x) {\n  x <- x[is.finite(x)]\n  mean(x)\n}\n',
    'sum_positive <- function(x) {\n  sum(x[x > 0], na.rm = TRUE)\n}\n',
    'count_missing <- function(x) {\n  sum(is.na(x))\n}\n',
    'center <- function(x) {\n  x - mean(x, na.rm = TRUE)\n}\n',
]


def probe(report):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['USE_TF'] = '0'
    report['stage'] = 'torch import and device discovery'
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    assert torch.cuda.is_available(), 'No CUDA device'
    report['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    report['checks']['cuda'] = True
    # Two independent streams, one tensor per GPU. This does not claim DDP.
    tensors = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{index}')
        x = torch.ones((256, 256), device=device, dtype=torch.float16)
        tensors.append(x @ x)
    for value in tensors:
        assert value[0, 0].item() == 256
    report['checks']['all_devices_compute'] = True
    report['stage'] = 'load public pinned model'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(MODEL, revision=REVISION,
        allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', '*.model']))
    report['model'] = {'repo': MODEL, 'revision': REVISION,
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir() if p.is_file()}}
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(snapshot, torch_dtype=torch.float16,
        trust_remote_code=False, attn_implementation='eager').to('cuda:0')
    model.requires_grad_(False)
    class LoRA(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.a = torch.nn.Parameter(torch.randn(4, base.in_features, device='cuda:0') * .01)
            self.b = torch.nn.Parameter(torch.zeros(base.out_features, 4, device='cuda:0'))
        def forward(self, x):
            return self.base(x) + (x.float() @ self.a.T @ self.b.T).to(x.dtype)
    adapters = []
    for name, module in list(model.named_modules()):
        if name.endswith(('.q_proj', '.v_proj')) and isinstance(module, torch.nn.Linear):
            parent_name, leaf = name.rsplit('.', 1)
            layer = LoRA(module)
            setattr(model.get_submodule(parent_name), leaf, layer)
            adapters.append((name, layer))
    assert adapters, 'No adapter attachment'
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report['adapter_modules'] = len(adapters)
    batch = tokenizer(TEXTS, padding=True, return_tensors='pt').to('cuda:0')
    labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)
    report['input_sha256'] = hashlib.sha256(json.dumps(TEXTS).encode()).hexdigest()
    model.eval()
    with torch.no_grad(): initial = model(**batch, labels=labels).loss.item()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    report['stage'] = '20 adapter updates'
    losses = []
    start = time.monotonic()
    for _ in range(20):
        optimizer.zero_grad()
        loss = model(**batch, labels=labels).loss
        assert torch.isfinite(loss), 'Nonfinite training loss'
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    report['training_seconds'] = time.monotonic() - start
    with torch.no_grad(): final = model(**batch, labels=labels).loss.item()
    report.update(initial_loss=initial, final_loss=final, losses=losses,
        peak_vram_bytes=torch.cuda.max_memory_allocated(0))
    assert final < initial, 'No in-sample loss improvement'
    report['checks']['adapter_training'] = True
    report['stage'] = 'adapter checkpoint round-trip'
    from safetensors.torch import save_file, load_file
    state = {f'{name}.{key}': getattr(layer, key).detach().cpu().contiguous() for name, layer in adapters for key in ('a', 'b')}
    save_file(state, 'adapter.safetensors')
    loaded = load_file('adapter.safetensors', device='cuda:0')
    with torch.no_grad():
        for name, layer in adapters:
            for key in ('a', 'b'): getattr(layer, key).copy_(loaded[f'{name}.{key}'])
        restored = model(**batch, labels=labels).loss.item()
    assert restored == final, 'Reload changed loss'
    report['checks']['adapter_reload'] = True
    report['adapter_sha256'] = hashlib.sha256(Path('adapter.safetensors').read_bytes()).hexdigest()
    prompt = tokenizer('mean_finite <- function(x) {\n', return_tensors='pt').to('cuda:0')
    with torch.no_grad(): generated = model.generate(**prompt, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    report['generation'] = tokenizer.decode(generated[0], skip_special_tokens=True)
    report['checks']['generation'] = True
    report['stage'] = 'complete'
