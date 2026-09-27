#!/usr/bin/env python3
"""Merge one verified SFT adapter on CPU into a fresh native artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from campaign_checkpoint import verify_checkpoint


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def tensor_digest(tensor, torch):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required'
    assert args.output.is_absolute() and not args.output.exists()
    temporary = args.output.with_name(args.output.name + '.preparing')
    assert not temporary.exists()
    recipe = json.loads(args.recipe.read_text())
    checkpoint = verify_checkpoint(args.checkpoint, recipe['identity'], require_full=True)
    base = Path(recipe['model_path'])
    for item in recipe['inputs']:
        if Path(item['path']).parent == base:
            assert digest(Path(item['path'])) == item['sha256'], item['path']
    import subprocess
    existing = args.output.parent
    while not existing.exists():
        existing = existing.parent
    filesystem = subprocess.check_output(['df', '-T', '-P', str(existing)], text=True).splitlines()[-1].split()[1]
    assert filesystem == 'ext4', filesystem
    started = time.time()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from campaign_tokenizer_contract import restore_pinned_tokenizer_contract
    from campaign_sft import assert_post_trainer_pinned_identity

    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    reference = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True, trust_remote_code=False)
    assert tokenizer.get_vocab() == reference.get_vocab()
    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16,
                                               device_map='cpu', local_files_only=True,
                                               trust_remote_code=False)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2516756480
    model = PeftModel.from_pretrained(model, str(args.checkpoint), is_trainable=False,
                                      local_files_only=True)
    actual_adapter = get_peft_model_state_dict(model)
    saved_adapter = load_file(str(args.checkpoint / 'adapter_model.safetensors'), device='cpu')
    assert set(actual_adapter) == set(saved_adapter) and len(actual_adapter) == 588
    for key in saved_adapter:
        assert torch.equal(actual_adapter[key], saved_adapter[key]), key
        assert torch.isfinite(actual_adapter[key]).all(), key
    restore_pinned_tokenizer_contract(model, tokenizer, reference_tokenizer=reference, prompt_rows=())
    # Independently calculate the exact BF16 merge for every attached module.
    expected = {}
    for name, module in model.named_modules():
        if not hasattr(module, 'lora_A'):
            continue
        assert list(module.lora_A) == ['default'] and list(module.lora_B) == ['default']
        assert module.r['default'] == 32 and module.lora_alpha['default'] == 64
        left, right = module.lora_B['default'].weight, module.lora_A['default'].weight
        update = left.float() @ right.float() * 2.0
        weight = module.base_layer.weight
        merged = (weight.float() + update).to(weight.dtype)
        assert torch.isfinite(merged).all(), name
        # PeftModel wraps the HF model under base_model.model.
        expected[name.removeprefix('base_model.model.') + '.weight'] = tensor_digest(merged, torch)
    assert len(expected) == 294
    model = model.merge_and_unload(safe_merge=True)
    weights = dict(model.named_parameters())
    for key, sha in expected.items():
        assert tensor_digest(weights[key], torch) == sha, key
    assert sum(parameter.numel() for parameter in model.parameters()) == 2516756480
    assert not any('lora_' in name for name in weights)
    contract = assert_post_trainer_pinned_identity(model, tokenizer, reference, ())
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(temporary), safe_serialization=True, max_shard_size='8GB')
    tokenizer.save_pretrained(str(temporary))
    exported = AutoTokenizer.from_pretrained(str(temporary), local_files_only=True, trust_remote_code=False)
    assert exported.get_vocab() == reference.get_vocab()
    assert (exported.bos_token_id, exported.eos_token_id, exported.pad_token_id) == (0, 1, 1)
    del weights, model, actual_adapter, saved_adapter
    inventory = []
    for path in sorted(temporary.glob('*.safetensors')):
        inventory.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
    assert len(inventory) == 1
    canonical_inventory = json.dumps(inventory, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    manifest = {
        'schema_version': 'sepalith.merged-sft.parent-manifest.v1',
        'status': 'prepared; root selection and load/generation admission required',
        'kind': 'merged_sft', 'merged_model_path': str(args.output),
        'base_model_revision': recipe['identity']['parent']['revision'],
        'sft_identity': recipe['identity'], 'sft_checkpoint': str(args.checkpoint),
        'sft_checkpoint_manifest': checkpoint, 'merged_weights_sha256': inventory[0]['sha256'],
        'weight_inventory': inventory,
        'weight_inventory_sha256': hashlib.sha256(canonical_inventory).hexdigest(),
        'config_sha256': digest(temporary / 'config.json'),
        'generation_config_sha256': digest(temporary / 'generation_config.json'),
        'tokenizer': {'tokenizer_json_sha256': digest(temporary / 'tokenizer.json'),
                      'tokenizer_config_sha256': digest(temporary / 'tokenizer_config.json'),
                      'vocab_size': 130560, 'bos_id': 0, 'eos_id': 1, 'pad_id': 1,
                      'native_eog_ids': [1, 130073]},
        'merge_verification': {'device': 'cpu', 'dtype': 'bfloat16', 'adapter_tensors_exact': 588,
                               'independent_lora_matrix_merges_exact': 294,
                               'tokenizer_contract': contract, 'cuda_started': False},
        'source_script_sha256': digest(Path(__file__)), 'recipe_sha256': digest(args.recipe),
        'elapsed_seconds': time.time() - started,
        'role': 'Intermediate runtime artifact until root explicitly selects the common parent.'
    }
    (temporary / 'parent-manifest.preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for path in temporary.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    temporary.rename(args.output)
    fd = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({'status': 'CPU merge prepared', 'output': str(args.output),
                      'weights_sha256': inventory[0]['sha256'], 'elapsed_seconds': manifest['elapsed_seconds'],
                      'common_parent_accepted': False}))


if __name__ == '__main__':
    main()
