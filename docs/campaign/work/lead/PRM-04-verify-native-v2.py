"""Independent full-fixture Python/HF versus TypeScript/native token comparison."""
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
MODEL = Path('/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain')
GGUF = MODEL.with_name('minicpm5-2b-midtrain-vocab-only.gguf')
BINARY = Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-tokenize')
sys.path.insert(0, str(EXEC / 'packages/sepalith/src'))
from transformers import AutoTokenizer
from sepalith.campaign_protocol import (
    PromptContext, TERMINAL, build_training_row, encode_prompt, parse_output, render_prompt, serialize_target,
)

fixture_module = EXEC / 'packages/sepalith/tests/test_campaign_protocol.py'
spec = importlib.util.spec_from_file_location('prm04_fixture', fixture_module)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
base = module.context()
empty = base.to_dict()
empty.update(region_old=[], cursor={'region_line_index': -1, 'code_point_column': None, 'utf16_column': None})
empty['replacement_range']['end'] = dict(empty['replacement_range']['start'])
empty = PromptContext.from_mapping(empty)
cases = []
for ident, context, operation, new in (
    ('noop', base, 'no_op', list(base.region_old)),
    ('replace', base, 'replace', ['  😀 <- 2  ', '  tail  ']),
    ('leading-lf', base, 'replace', ['', '  leading  ']),
    ('trailing-lf', base, 'replace', ['trailing  ', '']),
    ('delete', base, 'delete', []),
    ('insert', empty, 'replace', ['x <- 1', 'y <- 2']),
    ('empty-noop', empty, 'no_op', []),
    ('literal-markers', base, 'replace', ["x <- '</s>'", "y <- '<s>'", 'z <- ">>>>>>> UPDATED"']),
):
    cases.append({'id': f'PRM04-NATIVE-{ident}', 'context': context.to_dict(), 'operation': operation,
                  'region_new': new, 'raw_outputs': [serialize_target(operation, new), 'incomplete',
                      'wrong\n>>>>>>> UPDATED', '\n>>>>>>> UPDATED', '\n\n>>>>>>> UPDATED',
                      'x\n=======\n>>>>>>> UPDATED', 'x\n>>>>>>> UPDATED\ntrailing-junk']})
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
texts = {TERMINAL}
expected = []
for case in cases:
    context = PromptContext.from_mapping(case['context'])
    row = build_training_row(context, operation=case['operation'], region_new=case['region_new'],
                             tokenizer=tokenizer, row_id=case['id'], family='functional',
                             package_id='synthetic-prm04-native')
    texts.update((row['prompt_text'], row['target_text'], row['prompt_text'] + row['target_text']))
    expected.append({'id': case['id'], 'prompt': render_prompt(context),
                     'prompt_ids': encode_prompt(context, tokenizer), 'row': row,
                     'outputs': [asdict(parse_output(raw, context)) for raw in case['raw_outputs']]})
native, comparisons = {}, []
for text in sorted(texts):
    result = subprocess.run([str(BINARY), '-m', str(GGUF), '--ids', '--stdin', '--no-escape', '--offline',
                             '--no-bos', '--no-parse-special'], input=text, text=True,
                            capture_output=True, check=True, timeout=30)
    ids = ast.literal_eval(result.stdout.strip())
    hf = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
    assert ids == hf, 'Full-fixture native/HF token mismatch'
    assert tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text
    native[text] = ids
    comparisons.append({'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'tokens': len(ids), 'equal': True})
bridge = ROOT / 'work/lead/PRM-04-native-bridge.ts'
result = subprocess.run(['node', '--no-warnings=MODULE_TYPELESS_PACKAGE_JSON', str(bridge)],
                        input=json.dumps({'cases': cases, 'native_tokens': native}), text=True,
                        capture_output=True, check=True, timeout=60)
actual = json.loads(result.stdout)
assert actual == json.loads(json.dumps(expected)), 'Python/TypeScript full fixture mismatch'
receipt = {'task': 'PRM-04', 'owner': 'lead', 'observed_at': datetime.now(timezone.utc).isoformat(),
           'status': 'verified_full_fixture_parity', 'fixture_cases': len(cases),
           'native_HF_text_comparisons': len(comparisons), 'lossless_roundtrips': len(comparisons),
           'Python_TS_parser_outcomes': sum(len(case['raw_outputs']) for case in cases),
           'Python_TS_full_training_rows': len(actual), 'comparisons': comparisons,
           'fixture_rows': [{'id': item['id'], 'prompt_tokens_with_BOS': len(item['prompt_ids']),
                             'total_training_tokens': len(item['row']['input_ids']),
                             'prompt_sha256': hashlib.sha256(item['prompt'].encode()).hexdigest()} for item in actual],
           'source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                             (bridge, Path(__file__), EXEC / 'packages/sepalith/src/sepalith/campaign_protocol.py',
                              EXEC / 'extensions/vscode-sepalith/src/campaign_protocol.ts')},
           'scope': 'Synthetic functional fixtures; no final data, model forward, serving or CUDA. Native tokenizer uses metadata-only GGUF.',
           'remaining': 'Live CPU transport passed in separate receipt; editor application integration remains a separate gate.'}
(ROOT / 'receipts/PRM-04-lead-native-fixtures-v2.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: receipt[k] for k in ('status', 'fixture_cases', 'native_HF_text_comparisons',
                                        'lossless_roundtrips', 'Python_TS_parser_outcomes', 'Python_TS_full_training_rows')}))
