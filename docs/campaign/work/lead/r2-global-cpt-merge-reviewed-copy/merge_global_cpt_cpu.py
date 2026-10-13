#!/usr/bin/env python3
"""Merge one verified CPT adapter on CPU into a fresh native artifact."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import time

import sys
FROZEN_SOURCE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-r2-cpt-global-v1/snapshots/aaacb632811c5a4639fefe52023e72b72e2b960523dbffe10d0187fd6cef8da8/source')
DEPENDENCY_PINS = {'experiments/training/campaign_checkpoint.py': '64202f4e5b88c94a74794442be3a4955f07d195079c9d64bbcff2850dbadd0ba', 'experiments/training/campaign_control.py': '9a5b0614f134ecf6b804d2f0d5e37a4fecf2b1dc53589b92f17580678f9dbdac', 'experiments/training/campaign_cpt.py': '199eeab534a4d56bea443d8c7fde215eae041bb99912dba34cd607ed17922269', 'experiments/training/campaign_cpt_data.py': '3de53b0bf35f4371a644e5c4bd64050daa07538e25b70932c905e94b0fcb3ca0', 'experiments/training/campaign_cpt_eval.py': 'eab4edee75534b5e0da0784a1c608c7deb095155a7743d2ea658ce3d82fdd6a1', 'experiments/training/campaign_cpt_parent.py': 'a6ee0074b82e4a0b89932540fe2dd774a80d22d889df74f52323633e6b05e6fd', 'experiments/training/campaign_launch.py': 'c54e97c8f31f61eaba8781c757fe5bfbcbbe8891ed8c9c4c583db19b37265905', 'experiments/training/campaign_sft.py': 'b2d7d40f1c901629ac05cf1741725a9ea8277fba442ff89b93bf978abe5653d4', 'experiments/training/campaign_sft_data.py': '7474b60846ac9a572d9d545c9819db2b081db5deb9f18d200a85d6e6c9239c9f', 'experiments/training/campaign_tokenizer_contract.py': 'd01721f8716a78904268b8df94d1d86d86161d3f91f7de4bae3f5529d13f9958', 'experiments/training/merge_cpt_cpu.py': '743a1cff106db4c26793ab34edd15c55baa9c0266814d49c6ddff79894ed8d78', 'experiments/training/target_only_gate.py': '181a241204c0d5756babde51c3f20394378096855e5a41e4569cc2ef926c9a91', 'packages/sepalith/src/sepalith/__init__.py': 'c1748c8b1945c2167e77c1d2d962a3a6f53770104f195e10a84e5138bfe8c0a0', 'packages/sepalith/src/sepalith/campaign_protocol.py': '5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156'}
for relative, expected_sha in DEPENDENCY_PINS.items():
    dependency = FROZEN_SOURCE / relative
    if dependency.is_symlink() or hashlib.sha256(dependency.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Frozen merge dependency differs: ' + relative)
sys.path.insert(0, str(FROZEN_SOURCE / 'experiments/training'))
sys.path.insert(0, str(FROZEN_SOURCE / 'packages/sepalith/src'))

from campaign_checkpoint import verify_checkpoint
from campaign_cpt import _check_identity_contract
from campaign_cpt_parent import check_merged_cpt_parent


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def tensor_digest(tensor, torch):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


GLOBAL_SOURCE = 'aaacb632811c5a4639fefe52023e72b72e2b960523dbffe10d0187fd6cef8da8'
GLOBAL_ROWS = 'aac7e1b6140b011043f25a22970c0e8c300fde65316f7e5fbcbda7d747d4f703'
GLOBAL_DRAWS = 'b8a756534875ebb629db03213fa6155140ac9aaeaf75fa4bf3b9f63c33df88e0'
VALIDATION_ROWS = 'efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8'

def validate_global_metadata(recipe, checkpoint, campaign_state):
    """Check the one admitted global stage and its complete upstream lineage."""
    def require(ok, reason):
        if not ok:
            raise ValueError('Global CPT merge: ' + reason)
    identity = recipe['identity']
    require(recipe['stage'] == 'cpt_raw_r_v1', 'stage')
    require(identity['source'] == GLOBAL_SOURCE, 'source is not the frozen global source')
    require(identity['policy']['initialization'] == 'new_lora_on_merged_cpt_parent', 'initialization')
    require(recipe['train_rows']['sha256'] == identity['data']['train_rows_sha256'] == GLOBAL_ROWS, 'global training rows')
    require(recipe['draw_schedule']['sha256'] == identity['data']['draw_schedule_sha256'] == GLOBAL_DRAWS, 'global schedule')
    require(recipe['validation_rows']['sha256'] == identity['data']['validation_rows_sha256'] == VALIDATION_ROWS, 'validation rows')
    require(recipe['parameters']['max_steps'] == 1187 and identity['schedule'] == recipe['parameters'], 'fixed global horizon')
    _check_identity_contract(recipe, recipe['parameters'])
    check_merged_cpt_parent(recipe, verify_model_files=False)
    step = checkpoint['step']
    require(checkpoint.get('full') is True and type(step) is int and 0 < step <= 1187, 'full checkpoint step')
    require(checkpoint['identity'] == identity == campaign_state['identity'], 'checkpoint identity')
    require(campaign_state['step'] == step and campaign_state.get('full') is True, 'campaign state step/full')
    sampler = campaign_state['sampler']
    require(sampler['consumed_draws'] == step * 16, 'source cursor')
    require(sampler['schedule_sha256'] == GLOBAL_DRAWS and sampler['split_id'] == recipe['split_id'], 'sampler schedule/split')
    return sampler


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
    campaign_state = json.loads((args.checkpoint / 'campaign-state.json').read_text())
    prior_sampler = validate_global_metadata(recipe, checkpoint, campaign_state)
    for role in ('train_rows', 'draw_schedule', 'validation_rows'):
        item = recipe[role]
        path = Path(item['path'])
        if not path.is_absolute() or path.is_symlink() or digest(path) != item['sha256']:
            raise ValueError('Global CPT data input differs: ' + role)
    base = Path(recipe['model_path'])
    required_names = {'model.safetensors', 'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'}
    base_inputs = {Path(item['path']).name: item for item in recipe['inputs'] if Path(item['path']).parent == base}
    assert required_names <= set(base_inputs), 'Merged parent input closure is incomplete'
    assert base_inputs['model.safetensors']['sha256'] == recipe['identity']['parent']['weights_sha256']
    assert base_inputs['tokenizer.json']['sha256'] == '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
    assert base_inputs['tokenizer_config.json']['sha256'] == 'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b'
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
    # Accepted e11ca338 correction: PEFT casts delta to the base dtype.
    # Accumulate in FP32, then round the merged result to BF16 once.
    model.float()
    model = model.merge_and_unload(safe_merge=True)
    model.to(dtype=torch.bfloat16)
    weights = dict(model.named_parameters())
    for key, sha in expected.items():
        assert tensor_digest(weights[key], torch) == sha, key
    assert sum(parameter.numel() for parameter in model.parameters()) == 2516756480
    assert not any('lora_' in name for name in weights)
    contract = assert_post_trainer_pinned_identity(model, tokenizer, reference, ())
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(temporary), safe_serialization=True, max_shard_size='8GB')
    tokenizer.save_pretrained(str(temporary))
    # Tokenizer serialization is not evidence of original file identity.
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        shutil.copyfile(base / name, temporary / name)
        assert digest(temporary / name) == digest(base / name)
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
        'schema_version': 'sepalith.merged-cpt.parent-manifest.v1',
        'status': 'prepared; root selection and load/generation admission required',
        'kind': 'merged_cpt', 'merged_model_path': str(args.output),
        'base_model_revision': recipe['identity']['parent']['revision'],
        'cpt_identity': recipe['identity'], 'cpt_checkpoint': str(args.checkpoint),
        'upstream_merged_cpt_parent': recipe['merged_cpt_parent'],
        'merge_stage': 'global_cpt_from_frozen_aaacb632',
        'cpt_checkpoint_manifest': checkpoint, 'merged_weights_sha256': inventory[0]['sha256'],
        'weight_inventory': inventory,
        'weight_inventory_sha256': hashlib.sha256(canonical_inventory).hexdigest(),
        'config_sha256': digest(temporary / 'config.json'),
        'generation_config_sha256': digest(temporary / 'generation_config.json'),
        'tokenizer': {'tokenizer_json_sha256': digest(temporary / 'tokenizer.json'),
                      'tokenizer_config_sha256': digest(temporary / 'tokenizer_config.json'),
                      'vocab_size': 130560, 'bos_id': 0, 'eos_id': 1, 'pad_id': 1,
                      'native_eog_ids': [1, 130073]},
        'merge_verification': {'device': 'cpu', 'dtype': 'bfloat16', 'adapter_tensors_exact': 588,
                               'accumulation_dtype': 'float32', 'rounding': 'one final BF16 cast',
                               'independent_lora_matrix_merges_exact': 294,
                               'tokenizer_contract': contract, 'cuda_started': False},
        'source_script_sha256': digest(Path(__file__)), 'recipe_sha256': digest(args.recipe),
        'recipe_path': str(args.recipe),
        'checkpoint_manifest_sha256': digest(args.checkpoint / 'campaign-manifest.json'),
        'source_cursor': prior_sampler['consumed_draws'],
        'previous_sampler': prior_sampler,
        'tokenizer_original_bytes_restored': True,
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
