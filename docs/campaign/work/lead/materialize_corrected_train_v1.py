"""Materialize the reviewed finite TRAIN correction without changing source inputs."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import time
import resource

ROOT=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
EXEC=Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
DATA=Path('/mnt/e/sepalith/campaign-20260915/data-work')
OUT=ROOT/'docs/campaign/work/lead/finish-corrected-train-v1'
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0')
PINS={
 DATA/'DAT-04B-completion-batch.jsonl':'42743e70dbde54dd0f4adab42ebd0c2b0a0e7d3a4c4596752ffca272b20adc25',
 DATA/'SFT-inputs-v1/train-token-rows.jsonl':'7641bbdc8f609aca1e0ad72177561edddfdbdae1b49a8444c15470652bf5ebb6',
 ROOT/'docs/campaign/work/finish-target-overlay-v1/finish-target-overlay.jsonl':'03cff340de0c2b0e641a0399ddc8de727c39c9aa5e51aeed984c6420117054cf',
 TOKENIZER/'tokenizer.json':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
 TOKENIZER/'tokenizer_config.json':'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b',
 EXEC/'packages/sepalith/src/sepalith/campaign_protocol.py':'5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156',
}

def sha(b):return hashlib.sha256(b).hexdigest()

def rows(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for raw in handle:
            digest.update(raw)
            yield raw,json.loads(raw)
    assert digest.hexdigest()==PINS[path],path

def main():
    started=time.monotonic()
    for p in [TOKENIZER/'tokenizer.json',TOKENIZER/'tokenizer_config.json',EXEC/'packages/sepalith/src/sepalith/campaign_protocol.py']:
        assert sha(p.read_bytes())==PINS[p]
    sys.path.insert(0,str(EXEC/'packages/sepalith/src'))
    from sepalith.campaign_protocol import PromptContext,build_training_row,render_prompt,validate_training_row
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True,use_fast=True,trust_remote_code=False)
    assert len(tokenizer)==130560 and tokenizer.bos_token_id==0 and tokenizer.eos_token_id==1
    overlay={}
    for raw,row in rows(ROOT/'docs/campaign/work/finish-target-overlay-v1/finish-target-overlay.jsonl'):
        ident=row['key']['original_id'];assert ident not in overlay
        if not row['lineage']['matched_train_identity']:continue
        overlay[ident]={'eligible':row['eligibility']['strict_v2'],'old_line_sha256':row['lineage']['sft_line_sha256'],'old_body_sha256':row['original']['target_body_sha256'],'packet_line_sha256':row['key']['packet_raw_line_sha256'],'repaired':row['repaired'],'reasons':row['eligibility']['rejection_reasons']}
    assert len(overlay)==4289
    contexts={}
    for raw,row in rows(DATA/'DAT-04B-completion-batch.jsonl'):
        ident=row['row_ref']['row_id']
        if ident not in overlay:continue
        assert row['row_ref']['split']=='train_group'
        assert sha(raw)==overlay[ident]['packet_line_sha256']
        contexts[ident]=row['result']['context']
    assert contexts.keys()==overlay.keys()
    OUT.mkdir(exist_ok=False)
    target=OUT/'train-token-rows.jsonl'
    changed=[];excluded=[];unchanged=0;seen=set();families=Counter();ops=Counter();lengths=Counter();out_hash=hashlib.sha256();max_total=0
    try:
        with target.open('xb') as output:
            for raw,row in rows(DATA/'SFT-inputs-v1/train-token-rows.jsonl'):
                ident=row['id'];assert ident not in seen and row['split']=='train';seen.add(ident)
                if row['family']=='finish_block':
                    patch=overlay[ident]
                    assert sha(raw)==patch['old_line_sha256'] and sha(row['target_body_text'].encode())==patch['old_body_sha256']
                    context=PromptContext.from_mapping(contexts[ident])
                    assert render_prompt(context)==row['prompt_text']
                    if not patch['eligible']:
                        assert patch['reasons']==['strict_v2_rejected:target_not_lf_terminated']
                        excluded.append({'id':ident,'original_line_sha256':sha(raw),'reasons':patch['reasons']});continue
                    fixed=patch['repaired']
                    assert fixed['target_body_text']==row['target_body_text']+'}'
                    assert sha(fixed['target_body_text'].encode())==fixed['target_body_sha256']
                    assert context.replacement_range.content_sha256==fixed['strict_v2_evidence']['replacement_content_sha256']
                    old=row
                    row=build_training_row(context,operation='replace',region_new=fixed['target_body'],tokenizer=tokenizer,row_id=ident,family=old['family'],package_id=old['package_id'],split='train')
                    assert row['prompt_text']==old['prompt_text'] and row['target_start']==old['target_start']
                    assert row['input_ids'][:row['target_start']]==old['input_ids'][:old['target_start']]
                    assert set(row)==set(old)
                    raw_new=(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode()
                    changed.append({'id':ident,'original_line_sha256':sha(raw),'new_line_sha256':sha(raw_new),'old_body_sha256':patch['old_body_sha256'],'new_body_sha256':fixed['target_body_sha256'],'packet_line_sha256':patch['packet_line_sha256'],'target_tokens_before':old['target_token_count'],'target_tokens_after':row['target_token_count'],'prompt_ids_unchanged':True})
                    raw=raw_new
                else:
                    assert ident not in overlay;unchanged+=1
                validate_training_row(row)
                assert len(row['input_ids'])<=4096
                assert row['input_ids'].count(0)==1 and row['input_ids'].count(1)==1
                families[row['family']]+=1;ops[row['target_operation']]+=1
                lengths['long' if len(row['input_ids'])>2048 else 'short']+=1
                max_total=max(max_total,len(row['input_ids']))
                output.write(raw);out_hash.update(raw)
            output.flush()
            import os
            os.fsync(output.fileno())
        assert (len(seen),len(changed),len(excluded),unchanged)==(11764,4051,238,7475)
        assert sum(families.values())==11526 and ops['no_op']==1140
    except BaseException:
        (OUT/'FAILED').write_text('Not admitted: materialization stopped before final manifest.\n')
        raise
    report={'status':'corrected_train_rows_materialized_pending_independent_review','input_pins':{str(k):v for k,v in PINS.items()},'script_sha256':sha(Path(__file__).read_bytes()),'output':{'path':str(target),'sha256':out_hash.hexdigest(),'bytes':target.stat().st_size,'rows':11526},'counts':{'original':11764,'changed':4051,'excluded':238,'unchanged_byte_identical':7475,'noops':1140},'family_counts':dict(families),'operation_counts':dict(ops),'length_buckets':dict(lengths),'max_total_tokens':max_total,'changed':changed,'excluded':excluded,'geometry_review':'docs/campaign/receipts/DAT-04-finish-overlay-lead-review.json','scope':'TRAIN only; original files immutable; no model weights loaded or launch admitted.','elapsed_seconds':time.monotonic()-started,'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    summary={k:v for k,v in report.items() if k not in ['changed','excluded']}
    (ROOT/'docs/campaign/receipts/DAT-04-corrected-train-root-materialization.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'output':report['output'],'counts':report['counts'],'elapsed':report['elapsed_seconds']}))

if __name__=='__main__':main()
