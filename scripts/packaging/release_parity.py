"""CPU-only PRM03 render and GGUF metadata check. Never loads model tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from sepalith.campaign_protocol import PromptContext, render_prompt

EXTENSION_REF = 'c205e41f8f3f5ad860b62ea491315dcedb15b6c9'
EXTENSION = 'extensions/vscode-sepalith/'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_file(ref: str, name: str) -> bytes:
    return subprocess.check_output(['git', 'show', f'{ref}:{EXTENSION}{name}'])


def compare_prompts(contexts: list[dict], actual: list[str]) -> list[str]:
    expected = [render_prompt(PromptContext.from_dict(c)) for c in contexts]
    if actual != expected:
        raise ValueError('extension prompt differs from campaign PRM03 render')
    return [digest(p.encode()) for p in expected]


def validate_request(rendered: dict) -> None:
    if rendered['tokenize'] != {'content': rendered['prompt'], 'add_special': False,
                                'parse_special': False, 'with_pieces': False}:
        raise ValueError('extension tokenization settings differ')
    expected = {'prompt': [0, 42, 43], 'n_predict': 1024, 'temperature': 0, 'top_k': 1,
                'top_p': 1, 'min_p': 0, 'typical_p': 1, 'repeat_penalty': 1,
                'presence_penalty': 0, 'frequency_penalty': 0, 'dry_multiplier': 0,
                'xtc_probability': 0, 'mirostat': 0, 'ignore_eos': False, 'stop': [],
                'seed': 3407, 'cache_prompt': True, 'return_tokens': True,
                'stream': False, 'n_probs': 0, 'samplers': ['top_k']}
    if rendered['request'] != expected:
        raise ValueError('extension completion contract differs')


def metadata(path: Path, prompts: list[str]) -> dict:
    from gguf import GGUFReader
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    reader = GGUFReader(path)
    get = lambda key: reader.fields[key].contents()
    bos, eos = get('tokenizer.ggml.bos_token_id'), get('tokenizer.ggml.eos_token_id')
    tokens = get('tokenizer.ggml.tokens')
    if bos != 0 or eos != 1 or len(tokens) != 130560 or tokens[130073] != '<|im_end|>':
        raise ValueError('GGUF tokenizer identity differs from PRM03')
    template = get('tokenizer.chat_template')
    wrapped = [ImmutableSandboxedEnvironment().from_string(template).render(
        bos_token=tokens[bos], eos_token=tokens[eos], messages=[{'role': 'user', 'content': p}],
        add_generation_prompt=True, enable_thinking=False) for p in prompts]
    return {'path': str(path), 'bytes': path.stat().st_size, 'bos': bos, 'eos': eos,
            'vocab_size': len(tokens), 'chat_template_sha256': digest(template.encode()),
            'tokenizer_sha256': digest(json.dumps({k: v.contents() for k, v in reader.fields.items()
                if k.startswith('tokenizer.') and k != 'tokenizer.chat_template'},
                sort_keys=True, ensure_ascii=False).encode()),
            'chat_wrapping_matches_raw_prm03': all(w == tokens[bos] + p for w, p in zip(wrapped, prompts)),
            'serving_requirement': 'native /completion token IDs; never apply chat_template'}


def check_tiers(paths: list[Path], evidence: dict) -> None:
    expected = {t['gguf']['sha256'] for t in evidence['tiers']}
    actual = [digest(p.read_bytes()) for p in paths]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('GGUF inputs must be exactly the approved release tiers')


def run(ref: str, paths: list[Path], prompt_file: Path, evidence: Path) -> dict:
    check_tiers(paths, json.loads(evidence.read_text()))
    fixture = json.loads(git_file(ref, 'tests/fixtures/prm03-golden.json'))
    contexts = list(fixture['contexts'].values()) + json.loads(prompt_file.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ('prm03.ts', 'completion.ts'):
            (root / name).write_bytes(git_file(ref, 'src/' + name))
        (root / 'check.mts').write_text('''import { readFileSync } from 'node:fs';
import { promptContextFromJson, renderPrompt } from './prm03.ts';
import { tokenizeBody, promptIds, completionBody, SUGGESTION_MAX_TOKENS } from './completion.ts';
const cs = JSON.parse(readFileSync(0, 'utf8'));
console.log(JSON.stringify(cs.map(c => {
 const prompt = renderPrompt(promptContextFromJson(c));
 return {prompt, tokenize: JSON.parse(tokenizeBody(prompt)),
 request: JSON.parse(completionBody(promptIds([42, 43]), SUGGESTION_MAX_TOKENS))};
})));
''')
        rendered = json.loads(subprocess.check_output(['node', str(root / 'check.mts')],
                              input=json.dumps(contexts).encode()))
    hashes = compare_prompts(contexts, [r['prompt'] for r in rendered])
    for r in rendered:
        validate_request(r)
    models = [metadata(p, [r['prompt'] for r in rendered[-8:]]) for p in paths]
    if len({m['tokenizer_sha256'] for m in models}) != 1:
        raise ValueError('shipped tokenizers differ')
    return {'schema': 'sepalith.c17.serving-parity.v1', 'extension_commit': ref,
            'contexts_checked': len(hashes), 'prompt_hashes_sha256': digest(json.dumps(hashes).encode()),
            'render_and_request_parity': 'pass', 'models': models,
            'limits': ['No server tokenization or inference run', 'VS Code and Positron share this renderer',
                       'Zed PRM03 renderer unavailable; release blocked for Zed']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension-ref', default=EXTENSION_REF)
    p.add_argument('--prompts', type=Path, default=Path('docs/training/plan-20260927/c17/prompts.json'))
    p.add_argument('--gguf', action='append', type=Path, required=True)
    p.add_argument('--quant-evidence', type=Path,
                   default=Path('docs/training/plan-20260927/receipts/C17-quant-evidence.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.extension_ref, a.gguf, a.prompts, a.quant_evidence), indent=2) + '\n')


if __name__ == '__main__':
    main()
