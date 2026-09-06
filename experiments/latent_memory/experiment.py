"""Bounded fact-recovery screen. All arms start from the same frozen decoder."""
import gc
import json
import math
from pathlib import Path
import random
import time

import torch
from safetensors.torch import save_file

from .data import digest, input_view, validate_dataset
from .audit import training_order
from .model import (ModuleEncoder, attach_adapters, compatibility, greedy, load_memory,
                    save_memory, target_loss, token_embedding_contract)
from .provenance import file_hash, write_json


def token_ids(tokenizer, text, device):
    return torch.tensor([tokenizer.encode(text, add_special_tokens=False)], device=device)


def context(model, tokenizer, encoder, view, arm, recipe, device):
    # This function accepts an InputView, which has no target/post-edit fields.
    query = token_ids(tokenizer, view.query, device)
    if query.shape[1] > recipe['max_prompt_tokens']:
        raise ValueError('Query exceeds frozen budget')
    slots = None
    source = '\n'.join(module.path + '\n' + module.source for module in view.modules)
    if arm == 'retrieval':
        retrieved = token_ids(tokenizer, source + '\n', device)
        if retrieved.shape[1] > recipe['max_retrieval_tokens']:
            raise ValueError('Exact retrieval does not fit; no hidden truncation')
        query = torch.cat([retrieved, query], 1)
    elif arm == 'latent':
        # V1 screen uses exactly one module. Later multi-module recipes must declare allocation.
        if len(view.modules) != 1:
            raise ValueError('The bounded gate requires one module')
        slots = encoder([source])
    elif arm not in ['local', 'base', 'absent']:
        raise ValueError(f'Unknown context arm: {arm}')
    return query, slots, len(source.encode()) if arm == 'latent' else 0


def parameter_groups(model, encoder, recipe):
    groups = []
    for module, lr in [(model, recipe['adapter_lr']), (encoder, recipe['encoder_lr'])]:
        if module is None:
            continue
        for decay in [True, False]:
            params = [p for n, p in module.named_parameters() if p.requires_grad
                      and (p.ndim >= 2 and not n.endswith('bias') and 'norm' not in n) == decay]
            if params:
                groups.append({'params': params, 'lr': lr, 'initial_lr': lr,
                               'weight_decay': recipe['weight_decay'] if decay else 0.0})
    return groups


def checkpoint(output, model, encoder, optimizer, step, recipe, order, ledger):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    adapter = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    save_file(adapter, output / 'adapter.safetensors')
    if encoder is not None:
        save_file({n: p.detach().cpu().contiguous() for n, p in encoder.state_dict().items()},
                  output / 'encoder.safetensors')
    # Explicit restart artifact, never automatically resumed. weights_only=True can read it.
    torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                'order': order, 'recipe': recipe, 'ledger_totals': ledger.totals}, output / 'training-state.pt')
    hashes = {path.name: file_hash(path) for path in output.iterdir() if path.is_file()}
    write_json(output / 'checkpoint.json', {'step': step, 'files': hashes,
               'base_decoder_identity': model._latent_base_identity,
               'adapter_config': model._latent_adapter_config,
               'encoder_config': encoder.memory_config if encoder else None})
    return hashes


def train_arm(model, tokenizer, rows, arm, recipe, output, ledger, device):
    torch.manual_seed(recipe['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(recipe['seed'])
    names = attach_adapters(model, **recipe['adapter'])
    encoder = ModuleEncoder(model.config.hidden_size, **recipe['encoder']).to(device) if arm == 'latent' else None
    calibration = None
    if encoder is not None:
        calibration_rows = rows[:8]  # training partition only
        sources = [m.path + '\n' + m.source for row in calibration_rows for m in input_view(row).modules]
        ids = token_ids(tokenizer, '\n'.join(sources), device)
        calibration_bytes = sum(len(s.encode()) for s in sources)
        calibration_flops = 2 * sum(p.numel() for p in encoder.parameters()) * calibration_bytes
        ledger.reserve(training_encoder_bytes=calibration_bytes, estimated_training_flops=calibration_flops)
        calibration = encoder.calibrate(sources, model.get_input_embeddings()(ids))
        ledger.record('calibration', arm=arm, packages=[r['package'] for r in calibration_rows],
                      encoder_bytes=calibration_bytes, estimated_flops=calibration_flops, decoder_embedding_tokens=ids.numel(),
                      **calibration)
    parameters = {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad}
    write_json(Path(output) / 'adaptation.json', {'targets': names, 'parameters': parameters,
               'trainable_decoder_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'encoder_count': sum(p.numel() for p in encoder.parameters()) if encoder else 0,
               'calibration': calibration})
    optimizer = torch.optim.AdamW(parameter_groups(model, encoder, recipe))
    order = training_order(len(rows), recipe)
    write_json(Path(output) / 'training-order.json', [rows[i]['id'] for i in order])
    model.train()
    if encoder:
        encoder.train()
    completed = 0
    all_params = [p for group in optimizer.param_groups for p in group['params']]
    # Count target-module calls; each explicit adapter must participate in the actual forward.
    calls = {name: 0 for name in names}
    def hook(name):
        def count(*_):
            calls[name] += 1
        return count
    hooks = [model.get_submodule(name).register_forward_hook(hook(name)) for name in names]
    try:
        for step in range(recipe['steps_per_arm']):
            fraction = step / max(1, recipe['steps_per_arm'] - 1)
            warmup = recipe['warmup_fraction']
            factor = (step + 1) / max(1, math.ceil(warmup * recipe['steps_per_arm'])) if fraction < warmup else (
                recipe['min_lr_fraction'] + (1 - recipe['min_lr_fraction']) *
                0.5 * (1 + math.cos(math.pi * (fraction - warmup) / (1 - warmup))))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * factor
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[step * recipe['effective_batch']:(step + 1) * recipe['effective_batch']]:
                row = rows[index]
                view = input_view(row)
                # Build text and reserve the full batch cost before the encoder forward.
                query, _, _ = context(model, tokenizer, None, view,
                                      'local' if arm == 'latent' else arm, recipe, device)
                encoded_bytes = sum(len((m.path + '\n' + m.source).encode()) for m in view.modules) if encoder else 0
                target = token_ids(tokenizer, row['target'], device)
                target = torch.cat([target, torch.tensor([[tokenizer.eos_token_id]], device=device)], 1)
                length = query.shape[1] + target.shape[1] - 1 + (recipe['encoder']['slots'] if encoder else 0)
                # Conservative parameter-matmul proxy (not a hardware FLOP measurement): 6*N*T
                # for the full decoder despite frozen base; encoder gets the same multiplier.
                base_count = sum(p.numel() for p in model.parameters())
                encoder_count = sum(p.numel() for p in encoder.parameters()) if encoder else 0
                flops = 6 * (base_count * length + encoder_count * encoded_bytes)
                ledger.reserve(scored_tokens=target.numel(), training_decoder_tokens=length,
                               training_encoder_bytes=encoded_bytes, estimated_training_flops=flops)
                if encoder:
                    query, slots, _ = context(model, tokenizer, encoder, view, arm, recipe, device)
                else:
                    slots = None
                loss, details = target_loss(model, query, target, slots)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; failed-batch cap is zero')
                (loss / recipe['effective_batch']).backward()
                losses.append(loss.item())
                ledger.record('train_example', arm=arm, step=step, row=row['id'],
                              encoder_bytes=encoded_bytes, estimated_flops=flops, loss=loss.item(), **details)
            if step == 0:
                missing = [name for name in names if calls[name] == 0 or
                           model.get_submodule(name).B.grad is None or
                           not torch.isfinite(model.get_submodule(name).B.grad).all() or
                           model.get_submodule(name).B.grad.abs().sum().item() == 0]
                if missing:
                    raise RuntimeError(f'Adapter not used by the backend: {missing}')
                ledger.record('adapter_forward_gradient_contract', arm=arm, calls=calls.copy())
            norm = torch.nn.utils.clip_grad_norm_(all_params, recipe['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            completed = step + 1
            peak = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
            ledger.record('optimizer_step', arm=arm, step=completed, mean_loss=sum(losses) / len(losses),
                          grad_norm=float(norm), peak_vram_bytes=peak, totals=ledger.totals.copy())
            if peak > recipe['max_vram_bytes']:
                raise RuntimeError('VRAM cap exceeded')
            if completed % 32 == 0:
                checkpoint(Path(output) / f'checkpoint-{completed:04}', model, encoder, optimizer,
                           completed, recipe, order, ledger)
    finally:
        for handle in hooks:
            handle.remove()
        hashes = checkpoint(Path(output) / 'final', model, encoder, optimizer, completed, recipe, order, ledger)
    model.eval()
    if encoder:
        encoder.eval()
    return encoder, hashes


def paired_interval(a, b, samples, seed):
    if len(a) != len(b) or not a:
        raise ValueError('Paired outcomes require equal nonempty rows')
    differences = [int(x) - int(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(differences, k=len(a))) / len(a) for _ in range(samples))
    return {'difference': sum(differences) / len(a), 'ci95': [draws[int(.025 * samples)],
                                                          draws[min(samples - 1, int(.975 * samples))]]}


def verdict(predictions, recipe):
    by_arm = {}
    for row in predictions:
        by_arm.setdefault(row['arm'], {})[row['id']] = row['exact']
    if set(by_arm) != set(recipe['eval_arms']):
        raise ValueError('Incomplete evaluation arms')
    ids = sorted(by_arm['latent'])
    if any(sorted(rows) != ids for rows in by_arm.values()):
        raise ValueError('Evaluation rows are not paired')
    outcomes = {arm: [rows[i] for i in ids] for arm, rows in by_arm.items()}
    gain = paired_interval(outcomes['latent'], outcomes['local'], recipe['bootstrap_samples'], recipe['bootstrap_seed'])
    shuffle = paired_interval(outcomes['latent'], outcomes['shuffled'], recipe['bootstrap_samples'], recipe['bootstrap_seed'])
    passed = (gain['difference'] >= recipe['min_gain'] and gain['ci95'][0] > 0 and
              shuffle['difference'] >= recipe['min_shuffle_drop'] and shuffle['ci95'][0] > 0)
    return {'status': 'PASS' if passed else 'FAIL', 'n_packages': len(ids),
            'accuracy': {arm: sum(values) / len(values) for arm, values in outcomes.items()},
            'latent_minus_local': gain, 'latent_minus_shuffled': shuffle,
            'plumbing_success': True, 'learned_information_recovery': passed,
            'editing_improvement': 'NOT TESTED', 'automatic_expansion': False,
            'comparison': 'equal optimizer steps and target exposure; NOT FLOP matched'}


@torch.no_grad()
def evaluate(model, tokenizer, encoder, rows, arm, recipe, output, ledger, identities, device):
    model.eval()
    if encoder:
        encoder.eval()
    records = []
    # Fixed cyclic derangement, one independent package per example. No target-based matching.
    for i, row in enumerate(rows):
        view = input_view(row)
        donor = rows[(i + 1) % len(rows)] if arm == 'shuffled' else row
        if arm == 'shuffled':
            from dataclasses import replace
            view = replace(view, modules=input_view(donor).modules)
        representation = 'latent' if arm in ['latent', 'shuffled'] else arm
        start = time.monotonic()
        query, slots, encoded_bytes = context(model, tokenizer, encoder, view, representation, recipe, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        context_seconds = time.monotonic() - start
        memory_manifest = None
        if slots is not None:
            expected = compatibility(identities['encoder'], identities['decoder'], identities['tokenizer'],
                                     slots.shape[1], slots.shape[2], 'bfloat16' if device.type == 'cuda' else 'float32',
                                     donor['repository'] + ':' + donor['workspace_tree'])
            hashes = {m.path: digest(m.source.encode()) for m in view.modules}
            memory_manifest = save_memory(Path(output) / 'memory', slots[0].to(getattr(torch, expected.dtype)), expected, hashes)
            slots = load_memory(Path(output) / 'memory', memory_manifest, expected, hashes).unsqueeze(0).to(device)
        timings = []
        ids = greedy(model, query, slots, recipe['max_new_tokens'], tokenizer.eos_token_id, timings)
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        target = token_ids(tokenizer, row['target'], device)
        target = torch.cat([target, torch.tensor([[tokenizer.eos_token_id]], device=device)], 1)
        loss, details = target_loss(model, query, target, slots)
        record = {'id': row['id'], 'arm': arm, 'memory_donor': donor['id'] if slots is not None else None,
                  'prediction': text, 'prediction_ids': ids, 'target': row['target'],
                  'exact': text == row['target'], 'first_token_correct': ids[0] == target[0, 0].item(),
                  'target_loss': loss.item(), 'query': view.query, 'source_manifest': [m.__dict__ for m in view.modules],
                  'parent_workspace': view.parent_revision, 'memory': memory_manifest.to_dict() if memory_manifest else None,
                  'input_positions': list(range(query.shape[1] + (slots.shape[1] if slots is not None else 0))),
                  'encoder_bytes': encoded_bytes, 'context_seconds': context_seconds,
                  'decoder_forwards': timings, 'elapsed_seconds': time.monotonic() - start,
                  'cache_hit': False, 'cache_policy': 'disabled', 'latency_regime': 'training-window-not-serving-benchmark',
                  **details}
        with (Path(output) / 'predictions.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        records.append(record)
        # Greedy recomputes all prefix states for every token. Charge those realized forwards.
        prefix_length = query.shape[1] + (slots.shape[1] if slots is not None else 0)
        ledger.record('eval_example', arm=arm, row=row['id'], encoder_bytes=encoded_bytes,
                      decoder_forward_tokens=sum(prefix_length + n for n in range(len(ids))) + details['decoder_tokens'],
                      generated_tokens=len(ids), scored_tokens=target.numel(), elapsed_seconds=record['elapsed_seconds'])
        ledger.reserve()  # wall-clock check during evaluation as well
    return records


def run_screen(frozen_root, output, ledger, device):
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    root, output = Path(frozen_root), Path(output)
    freeze = json.loads((root / 'freeze.json').read_text())
    recipe = json.loads((root / 'recipe.json').read_text())
    dataset = json.loads((root / 'dataset.json').read_text())
    validate_dataset(dataset)
    tokenizer = AutoTokenizer.from_pretrained(freeze['checkpoint_path'], local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError('Pinned tokenizer requires an EOS token')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    predictions = []
    base_identity = digest([freeze['checkpoint_files'], freeze['backend']])
    def load_decoder():
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            freeze['checkpoint_path'], local_files_only=True, dtype=dtype,
            attn_implementation='eager', output_loading_info=True)
        if any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise ValueError(f'Checkpoint failed exact load: {loading}')
        model = model.to(device)
        model.config.use_cache = False
        model._latent_base_identity = base_identity
        return model
    model = load_decoder()
    fixture = token_ids(tokenizer, 'f_abcd <- function(x = 73) x\n', device)
    contract = token_embedding_contract(model, fixture, recipe['atol'], recipe['rtol'])
    write_json(output / 'real-decoder-contract.json', contract)
    ledger.record('real_decoder_contract', decoder_forward_tokens=2 * fixture.shape[1] +
                  2 * sum(fixture.shape[1] + i for i in range(3)), **contract)
    identities = {'decoder': base_identity, 'tokenizer': freeze['checkpoint_files']['tokenizer.json']}
    predictions += evaluate(model, tokenizer, None, dataset['eval'], 'base', recipe, output, ledger, identities, device)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    for arm in recipe['arms']:
        model = load_decoder()
        arm_output = output / arm
        arm_output.mkdir()
        encoder, hashes = train_arm(model, tokenizer, dataset['train'], arm, recipe, arm_output, ledger, device)
        identities = {'decoder': digest([base_identity, hashes['adapter.safetensors']]),
                      'encoder': digest([hashes.get('encoder.safetensors'), recipe['encoder'], 'utf8-bytes-plus1-v1']),
                      'tokenizer': freeze['checkpoint_files']['tokenizer.json']}
        for evaluation_arm in (['latent', 'absent', 'shuffled'] if arm == 'latent' else [arm]):
            predictions += evaluate(model, tokenizer, encoder, dataset['eval'], evaluation_arm, recipe,
                                    output, ledger, identities, device)
        del model, encoder
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result = verdict(predictions, recipe)
    result['compute'] = ledger.totals
    result['first_token_accuracy'] = {arm: sum(r['first_token_correct'] for r in predictions if r['arm'] == arm) /
                                      len(dataset['eval']) for arm in recipe['eval_arms']}
    write_json(output / 'result.json', result)
    return result


def restore_checkpoint(directory, model, encoder=None):
    """Restore trained weights into the same declared architecture, without resuming training."""
    from safetensors.torch import load_file
    root = Path(directory)
    manifest = json.loads((root / 'checkpoint.json').read_text())
    for name, expected in manifest['files'].items():
        if Path(name).name != name or file_hash(root / name) != expected:
            raise ValueError('Checkpoint payload identity mismatch')
    if getattr(model, '_latent_base_identity', None) != manifest['base_decoder_identity']:
        raise ValueError('Checkpoint base decoder differs')
    if getattr(model, '_latent_adapter_config', None) != manifest['adapter_config']:
        raise ValueError('Checkpoint adapter configuration differs')
    if (encoder.memory_config if encoder else None) != manifest['encoder_config']:
        raise ValueError('Checkpoint encoder configuration differs')
    adapter = load_file(root / 'adapter.safetensors')
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(adapter) != set(expected):
        raise ValueError('Checkpoint adapter targets differ')
    with torch.no_grad():
        for name, tensor in adapter.items():
            if tensor.shape != expected[name].shape:
                raise ValueError('Checkpoint adapter shape mismatch')
            expected[name].copy_(tensor)
    if encoder is not None:
        encoder.load_state_dict(load_file(root / 'encoder.safetensors'), strict=True)
    elif 'encoder.safetensors' in manifest['files']:
        raise ValueError('Latent checkpoint requires its encoder')
    return manifest
