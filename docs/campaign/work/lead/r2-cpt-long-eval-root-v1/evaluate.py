"""Root-owned, single-model CPT evaluation; no training or promotion."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

LEAD = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead')
SOURCE = LEAD / 'r2-full-weight-cpt-lr-pilot-v3'
SOURCE_SHA = 'e60c7248b60d13dd295ece934e637fe14219b0861dff1811d82be8e676520bf9'
PANELS = LEAD / 'r2-cpt-long-holdout-preparation-v1/evaluator-configs.json'
PANELS_SHA = 'c20e6261515451b5e6f4dbd84c06117a9b946cb80d7c874dff869a70d978863f'
REFERENCE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-CPT-global-a-250-merged')
TOKENIZER_SHA = '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked(path, expected):
    assert digest(path) == expected, f'identity differs: {path}'


def preflight(binding):
    assert binding['schema'] == 'sepalith.cpt.long-eval-root-binding.v1'
    assert binding['status'] == 'admitted' and binding['training_authorized'] is False
    checked(Path(__file__), binding['runner_sha256'])
    checked(SOURCE / 'source-manifest.json', SOURCE_SHA)
    for item in json.loads((SOURCE / 'source-manifest.json').read_text())['files']:
        checked(SOURCE / item['path'], item['sha256'])
    checked(PANELS, PANELS_SHA)
    panels = json.loads(PANELS.read_text())['panels']
    assert set(panels) == {'8k', '16k'}
    for panel in panels.values():
        checked(Path(panel['validation_rows']['path']), panel['validation_rows']['sha256'])
        assert panel['validation_batch_size'] == 1
    model = Path(binding['model_path'])
    assert model.is_absolute() and model.is_dir()
    assert {'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'} <= set(binding['model_files'])
    for name, expected in binding['model_files'].items():
        assert Path(name).name == name
        checked(model / name, expected)
    checked(model / 'tokenizer.json', TOKENIZER_SHA)
    checked(REFERENCE / 'tokenizer.json', TOKENIZER_SHA)
    assert not (model / 'adapter_config.json').exists()
    anchor = json.loads((SOURCE / 'recipe.lr3e-5.template.json').read_text())['validation']
    checked(Path(anchor['path']), anchor['sha256'])
    panels['anchor2k'] = {'validation_rows': anchor, 'validation_batch_size': 1,
                          'parameters': {'max_sequence_tokens': 2048}, 'materialized_rows': True}
    return model, panels


def run(binding_path, output):
    binding = json.loads(binding_path.read_text())
    assert not output.exists(), 'fresh output required'
    model_path, panels = preflight(binding)
    assert os.environ.get('SEPALITH_CUDA_LOCK_FD'), 'root CUDA guard required'
    os.fstat(int(os.environ['SEPALITH_CUDA_LOCK_FD']))
    sys.path.insert(0, str(SOURCE / 'source/experiments/training'))
    os.environ['UNSLOTH_RETURN_LOGITS'] = '0'
    from unsloth import FastLanguageModel
    import torch
    import transformers
    from transformers import set_seed
    from campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from trainer_tokenizer_alignment import embedding_identity, restore_trainer_eog_alignment, assert_runtime_tokenizer
    from campaign_cpt_eval import package_holdout_evaluator
    from importlib.metadata import version

    assert transformers.__version__ == '5.5.0' and version('unsloth') == '2026.8.18'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    set_seed(3407)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path), max_seq_length=16384, dtype=torch.bfloat16,
        load_in_4bit=False, full_finetuning=True, float32_mixed_precision=False,
        fast_inference=False, trust_remote_code=False, use_gradient_checkpointing=True)
    reference = load_pinned_reference_tokenizer(REFERENCE)
    repair = restore_trainer_eog_alignment(model, tokenizer, reference)
    assert repair['repair']['vocab_mapping_unchanged']
    embeddings = embedding_identity(model)
    model.config.use_cache = False
    FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    model.train()
    assert len(list(model.parameters())) == 381
    assert sum(p.numel() for p in model.parameters()) == 2516756480
    output.mkdir(parents=True)
    audits = []
    for label in ('anchor2k', '8k', '16k'):
        audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings, 'before_' + label))
        started = time.time()
        result = package_holdout_evaluator(panels[label])(model, tokenizer, model_path, binding['checkpoint_step'])
        result['length_stratum'] = label
        result['elapsed_seconds'] = time.time() - started
        result['binding_sha256'] = digest(binding_path)
        audits.append(assert_runtime_tokenizer(model, tokenizer, reference, embeddings, 'after_' + label))
        with (output / (label + '.json')).open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps({'panel': label, 'metrics': result['metrics'], 'denominators': result['denominators']}), flush=True)
    with (output / 'result.json').open('x') as stream:
        json.dump({'status': 'evaluations_complete', 'binding_sha256': digest(binding_path),
                   'tokenizer_audits': audits, 'training_performed': False,
                   'promotion_authorized': False}, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.binding, args.output)
