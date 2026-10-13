#!/usr/bin/env python3
"""CPU-only fixed DEV diagnosis; no dataset materialization or model imports."""
import hashlib, importlib.util, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.dont_write_bytecode = True
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
OUT = Path(__file__).resolve().parent
DEV = PLAN/'work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl'
RESULT = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/DAT-08-native-dev-rehearsal-v3-a/dev-results/results.json')
SOURCE = Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source')
PROTOCOL = SOURCE/'packages/sepalith/src/sepalith/campaign_protocol.py'
def sha(b): return hashlib.sha256(b).hexdigest()
def pin(p): return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
def canon(v): return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',',':'),allow_nan=False).encode()
def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
assert sha(PROTOCOL.read_bytes())=='5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156'
protocol=load_module('r2_pinned_protocol', PROTOCOL)
sys.path.insert(0,'/home/m0hawk/Documents/Sepalith/.venv/lib/python3.10/site-packages')
from tree_sitter import Language, Parser
import tree_sitter_r
parser=Parser(Language(tree_sitter_r.language()))
def parse_ok(text): return not parser.parse(text.encode()).root_node.has_error
assert parse_ok('f <- function(x) { x + 1 }') and not parse_ok('f <- function(x) { x + 1')
def applied(row, parsed):
    c=row['context']; ss=row['source_provenance']['selection_source'];document=ss['document_text'];rr=c['replacement_range']
    assert sha(document.encode())==ss['content_sha256']==rr['content_sha256']
    lines=document.split('\n')
    def offset(pos):
        line=pos['line'];column=protocol.utf16_to_codepoint_column(lines[line],pos['character'])
        return sum(len(s)+1 for s in lines[:line])+column
    start,end=offset(rr['start']),offset(rr['end']);assert start<=end;assert document[start:end]=='\n'.join(c['region_old'])
    replacement=document[start:end] if parsed.operation=='no_op' else '' if parsed.operation=='delete' else '\n'.join(parsed.body)
    return document[:start]+replacement+document[end:]
rows=[json.loads(line) for line in DEV.read_text().splitlines()]; data={r['id']:r for r in rows};result=json.loads(RESULT.read_text());actual=result['results']
assert len(data)==len(rows)==len(actual)==75 and set(data)=={r['id'] for r in actual}
assert result['status']=='complete'
families=defaultdict(Counter);cases=[];finish=[];computed=Counter()
for x in actual:
    r=data[x['id']];assert r['split']=='dev' and r['admitted_for_training'] is False
    assert r['prompt_sha256']==x['prompt_sha256']; assert sha(canon(r['region_new']))==x['target_sha256']
    context=protocol.PromptContext.from_mapping(r['context']);assert sha(protocol.render_prompt(context).encode())==x['prompt_sha256']
    ids=x['returned_token_ids'];assert len(ids)==x['cap']['returned_tokens_including_terminal']
    parsed=protocol.parse_output(x['decoded_body_text'],context)
    valid=protocol.valid_generation_tokens(ids) and parsed.status=='accepted'
    assert valid==x['quality']['protocol_valid']
    actual_region=list(context.region_old) if valid and parsed.operation=='no_op' else list(parsed.body) if valid else None
    noop=r['operation']=='no_op';exact=actual_region==r['region_new'] if valid else False
    good_noop=noop and valid and parsed.operation=='no_op'; edit_exact=not noop and exact;fp=noop and valid and parsed.operation!='no_op'
    assert edit_exact==x['quality']['edit_exact'];assert good_noop==x['quality']['strict_noop_correct'];assert fp==x['quality']['noop_false_positive']
    # Stored terminal text tokens exclude the separate canonical EOS ID1.
    target_total=r['target_body_token_count']+r['target_terminal_token_count']+1
    v={'cases':1,'correct':edit_exact or good_noop,'caps':x['cap']['hit'],'protocol_valid':valid,'false_suggestions':fp,'gold_including_EOS_over_192':target_total>192}
    families[x['family']].update({k:int(z) for k,z in v.items()})
    computed.update({'edit_exact':int(edit_exact),'strict_noop_correct':int(good_noop),'false_suggestions':int(fp),'protocol_valid':int(valid),'cap_hit':int(x['cap']['hit'])})
    case={'id':x['id'],'family':x['family'],'correct':v['correct'],'protocol_valid':valid,'cap_hit':x['cap']['hit'],'parser_operation':parsed.operation,'prompt_tokens':x['hf_prompt_tokens_with_bos'],'target_tokens_including_EOS':target_total,'generated_tokens_including_EOS':len(ids),'raw_text_sha256':x['raw_text_sha256'],'native_region_sha256':x['target_sha256'],'panel_target_text_sha256':r['target_sha256']}
    if x['family']=='finish_block':
        f={'id':x['id'],'protocol_valid':valid,'target_tokens_including_EOS':target_total,'prompt_tokens':case['prompt_tokens'],'prefix_lines':len(r['context']['prefix']),'history_items':len(r['context']['history']),'scope_lines':len(r['context']['scope_lines']),'suffix_lines':len(r['context']['suffix_lines']),'selector_omissions':r['selection']['omissions'],'selector_overflow':r['selection']['overflow'],'predicted_applied_R_parse':None}
        gold=protocol.parse_output(r['target_body_text']+'\n>>>>>>> UPDATED',context);assert gold.status=='accepted';f['gold_applied_R_parse']=parse_ok(applied(r,gold));assert f['gold_applied_R_parse']
        if valid:
            document=applied(r,parsed);f['predicted_applied_R_parse']=parse_ok(document);f['applied_R_sha256']=sha(document.encode())
        finish.append(f)
    cases.append(case)
for k,v in computed.items():assert result['counts'][k]==v
assert computed=={'edit_exact':26,'strict_noop_correct':25,'false_suggestions':5,'protocol_valid':69,'cap_hit':6}
assert sum(v['gold_including_EOS_over_192'] for v in families.values())==3
report={'schema':1,'scope':'DEV diagnosis only; never a TRAIN artifact','status':'passed','checks':{'unique_ids':75,'exact_id_join':75,'prompt_hash_and_render_match':75,'native_region_target_hash_match':75,'token_count_match':75,'protocol_and_quality_recomputed':75,'aggregate_fields_match':5,'gold_finish_applied_R_parse':6,'parser_positive_negative_controls':2},'native_counts':dict(computed),'family_counts':families,'cases':cases,'finish_context_and_parse':finish,'input_pins':[pin(DEV),pin(RESULT),pin(PROTOCOL)],'limits':['Tree-sitter parse is syntactic only; no R execution or semantic-equivalence claim.','Invalid protocol predictions were not applied; 3/6 finish buffers are eligible for prediction parse.','Panel target_sha256 hashes text, while native result target_sha256 hashes canonical region lines; both meanings retained.']}
(OUT/'native-dev-case-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':report['checks'],'families':families,'finish_parse':[f['predicted_applied_R_parse'] for f in finish],'input_pins':report['input_pins']}))
