#!/usr/bin/env python3
"""Package root-selected merged weights with the byte-pinned tokenizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--prepared', type=Path, required=True)
p.add_argument('--selection', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
a = p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert a.output.is_absolute() and not a.output.exists()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

selection = json.loads(a.selection.read_text())
assert selection['status'] == 'accepted'
prep_path = a.prepared / 'parent-manifest.preparation.json'
prep = json.loads(prep_path.read_text())
assert selection['selected_checkpoint'] == prep['sft_checkpoint']
assert prep['merge_verification']['independent_lora_matrix_merges_exact'] == 294
assert prep['merge_verification']['adapter_tensors_exact'] == 588
base = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native')
expected_tokenizers = {
    'tokenizer.json': '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
    'tokenizer_config.json': 'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b',
}
for name, expected in expected_tokenizers.items(): assert sha(base / name) == expected
for item in prep['weight_inventory']:
    f = a.prepared / item['path']
    assert f.stat().st_size == item['bytes'] and sha(f) == item['sha256']
for name, field in [('config.json', 'config_sha256'), ('generation_config.json', 'generation_config_sha256')]:
    assert sha(a.prepared / name) == prep[field]
a.output.parent.mkdir(parents=True, exist_ok=True)
assert subprocess.check_output(['findmnt', '-T', str(a.output.parent), '-n', '-o', 'FSTYPE'], text=True).strip() == 'ext4'
a.output.mkdir()
for item in prep['weight_inventory']:
    shutil.copyfile(a.prepared / item['path'], a.output / item['path'])
for name in ['config.json', 'generation_config.json']:
    shutil.copyfile(a.prepared / name, a.output / name)
for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja']:
    shutil.copyfile(base / name, a.output / name)

sys.path[:0] = [str(a.source / 'experiments/training'), str(a.source / 'packages/sepalith/src')]
from transformers import AutoTokenizer
from sepalith.campaign_protocol import PromptContext, encode_prompt
reference = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(str(a.output), local_files_only=True, trust_remote_code=False)
assert tokenizer.get_vocab() == reference.get_vocab()
assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (0, 1, 1)
panel = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl')
assert sha(panel) == 'b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21'
rows = [json.loads(line) for line in panel.read_text().splitlines()]
assert len(rows) == 75
for row in rows:
    context = PromptContext.from_mapping(row['context'])
    assert encode_prompt(context, tokenizer) == encode_prompt(context, reference)

manifest = dict(prep)
manifest.update(status='accepted', merged_model_path=str(a.output),
                role='Selected theta0 weight and tokenizer bytes. Live load/generation and RL admission remain separate gates.',
                root_selection={'path': str(a.selection.resolve()), 'sha256': sha(a.selection)},
                preparation={'path': str(prep_path), 'sha256': sha(prep_path)},
                tokenizer_packaging={'source': str(base), 'reason': 'Preserve original tokenizer bytes after framework save normalization.',
                                     'original_prepared_tokenizer': prep['tokenizer'],
                                     'dev_prompt_identity_cases': 75, 'full_vocab_equal': True,
                                     'source_prepared_artifact_unchanged': True})
manifest['tokenizer'] = dict(prep['tokenizer'],
                            tokenizer_json_sha256=expected_tokenizers['tokenizer.json'],
                            tokenizer_config_sha256=expected_tokenizers['tokenizer_config.json'])
manifest['packaging_script_sha256'] = sha(__file__)
manifest_path = a.output / 'parent-manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
for f in a.output.iterdir():
    if f.is_file():
        with f.open('rb') as stream: os.fsync(stream.fileno())
fd = os.open(a.output, os.O_RDONLY | os.O_DIRECTORY)
try: os.fsync(fd)
finally: os.close(fd)
fd = os.open(a.output.parent, os.O_RDONLY | os.O_DIRECTORY)
try: os.fsync(fd)
finally: os.close(fd)
from campaign_rl_entry import verify_merged_parent_manifest
audit = verify_merged_parent_manifest({'path': str(manifest_path), 'sha256': sha(manifest_path)})
print(json.dumps({'status': 'selected_parent_bytes_verified', 'parent_manifest': str(manifest_path),
                  'parent_manifest_sha256': sha(manifest_path), 'merged_weights_sha256': audit['merged_weights_sha256'],
                  'live_gate_complete': False}))
