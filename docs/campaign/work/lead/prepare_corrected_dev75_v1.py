"""Version six proven DEV finish targets; preserve the original panel and other 69 lines."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys

ROOT = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
WORK = ROOT/'docs/campaign/work/lead/corrected-dev75-v1'
PANEL = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl')
TOKENIZER = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
PINS = {
    PANEL: 'b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21',
    TOKENIZER: '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
    TOKENIZER.parent/'tokenizer_config.json': 'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b',
    EXEC/'packages/sepalith/src/sepalith/campaign_protocol.py': '5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156',
    ROOT/'docs/campaign/work/finish-boundary-repair-v2/finish_boundary_repair_v2.py': '7fabe102485241a33e2af1178a6e2f05d8c06e1ef832f852c44e4d58faa08b5b',
    Path('/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/scenarios.py'): 'cccf8ddfff0ae1f64a0113c9612386227df66f9fff4701608c2320bd8eb0250c',
}
IDS = {'e623a61b5a4c066358a477f2','4f08633513b5c525240d2540','d11581e9cfa4e3971aa1466e','157517ba47dbab157f7c361a','f43de3e77f2d92ed7b223464','04834fef4fe59742f13677a9'}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def main():
    for path, expected in PINS.items():
        assert sha(path.read_bytes()) == expected, path
    protocol = module('corrected_dev_protocol', EXEC/'packages/sepalith/src/sepalith/campaign_protocol.py')
    repair = module('corrected_dev_geometry', ROOT/'docs/campaign/work/finish-boundary-repair-v2/finish_boundary_repair_v2.py')
    scenarios = module('corrected_dev_r', Path('/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/scenarios.py'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER.parent), local_files_only=True, trust_remote_code=False)
    output, evidence, seen = [], [], set()
    for raw in PANEL.read_bytes().splitlines(keepends=True):
        case = json.loads(raw)
        assert case['id'] not in seen and case['split'] == 'dev' and case['admitted_for_training'] is False
        seen.add(case['id'])
        if case['id'] not in IDS:
            assert case['family'] != 'finish_block'
            output.append(raw)
            continue
        context = protocol.PromptContext.from_mapping(case['context'])
        old = copy.deepcopy(case['source_provenance'])
        source = old['selection_source']
        assert case['family'] == 'finish_block' and case['operation'] == 'replace'
        assert old['pre_edit_document']['source_constructor'] == 'finish_block_v5_prefix'
        assert old['finish_splice'] == {'literal_source_splice_verified': True, 'outer_closing_brace_in_label': False}
        assert old['source_identity']['row_verified'] is True
        assert old['source_identity']['split'] == 'dev_group'
        assert not case['context']['suffix_lines']
        before = source['document_text']
        assert '\r' not in before
        assert sha(before.encode()) == source['content_sha256'] == old['pre_edit_document']['content_sha256']
        target = case['target_body_text']
        assert '\n' in target and target.rsplit('\n', 1)[-1].strip(' \t') == ''
        assert sha(target.encode()) == old['target_body_sha256']
        assert '\n'.join(case['region_new']) == target
        old_row = protocol.build_training_row(context, operation='replace', region_new=case['region_new'], tokenizer=tokenizer, row_id=case['id'], family=case['family'], package_id=case['package_id'], split='dev')
        assert sha(old_row['prompt_text'].encode()) == case['prompt_sha256']
        assert sha(old_row['target_text'].encode()) == case['target_sha256']
        splice = lambda text: repair.apply_document_replacement(before, case['context']['replacement_range'], text, region_old=case['context']['region_old'], utf16_to_codepoint_column=protocol.utf16_to_codepoint_column)
        old_post = splice(target)
        new_target = target + '}'
        new_post = splice(new_target)
        assert new_post == old_post + '}'
        assert scenarios.parser.parse(old_post.encode()).root_node.has_error
        assert not scenarios.parser.parse(new_post.encode()).root_node.has_error
        row = protocol.build_training_row(context, operation='replace', region_new=new_target.split('\n'), tokenizer=tokenizer, row_id=case['id'], family=case['family'], package_id=case['package_id'], split='dev')
        assert row['prompt_text'] == old_row['prompt_text']
        assert row['input_ids'][:row['target_start']] == old_row['input_ids'][:old_row['target_start']]
        change = {'id':case['id'], 'parent_line_sha256':sha(raw), 'old_target_body_sha256':sha(target.encode()), 'new_target_body_sha256':sha(new_target.encode()), 'old_target_sha256':case['target_sha256'], 'new_target_sha256':sha(row['target_text'].encode()), 'new_post_document_sha256':sha(new_post.encode()), 'target_body_tokens_before':old_row['target_body_token_count'], 'target_body_tokens_after':row['target_body_token_count'], 'new_document_parse_ok':True, 'original_target_final_lf':target.endswith('\n'), 'trailing_horizontal_whitespace_preserved':target.rsplit('\n',1)[-1]}
        evidence.append(change)
        case.update(region_new=new_target.split('\n'), target_body_text=new_target, target_sha256=change['new_target_sha256'], target_body_token_count=row['target_body_token_count'], target_terminal_token_count=row['target_terminal_token_count'])
        case['source_provenance'] = {
            'origin':'source-supported-finish-boundary-correction-v1',
            'parent_provenance':old,
            'selection_source':source,
            'pre_edit_document':old['pre_edit_document'],
            'source_identity':old['source_identity'],
            'target_body_sha256':change['new_target_body_sha256'],
            'finish_splice':{'literal_source_splice_verified':True,'outer_closing_brace_in_label':True},
            'correction':change,
            'materialization':'Append exactly one source outer brace at the proven represented-document EOF; no prompt changes.'
        }
        output.append((json.dumps(case,sort_keys=True,ensure_ascii=False)+'\n').encode())
    assert len(seen)==75 and len(evidence)==6 and {x['id'] for x in evidence}==IDS
    payload=b''.join(output)
    WORK.mkdir(exist_ok=False)
    out=WORK/'dev75-corrected-finish-v1.jsonl'
    out.write_bytes(payload)
    report={'status':'prepared_versioned_dev_target_correction_pending_independent_review','input_pins':{str(k):v for k,v in PINS.items()},'output':{'path':str(out),'sha256':sha(payload),'bytes':len(payload)},'counts':{'rows':75,'changed_finish':6,'other_lines_byte_identical':69,'edit_rows':43,'noop_rows':32},'changes':evidence,'rule':'Only six previously audited finish targets gain their known outer brace; all inputs and predictions remain unchanged.','training_use':False,'final_content_read':False,'official_historical_scores_overwritten':False}
    (WORK/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':report['output'],'counts':report['counts']}))

if __name__ == '__main__':
    main()
