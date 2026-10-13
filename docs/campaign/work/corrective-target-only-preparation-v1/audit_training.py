"""Exact TRAIN-only byte/token/mask audit; no model loading or optimizer use."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parents[1]
SOURCE=Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/26a07c58eebc1769a224a193a7b163e989125e4117236cc367b940e706d7af6f/source/experiments/training')
PROTOCOL=Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src/sepalith/campaign_protocol.py')
sys.path[:0]=[str(HERE/'candidate'),str(SOURCE)]
from campaign_sft_data import target_only_collator
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    assert digest(PROTOCOL)=='5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156'
    spec=importlib.util.spec_from_file_location('audited_protocol',PROTOCOL)
    protocol=importlib.util.module_from_spec(spec);sys.modules[spec.name]=protocol;spec.loader.exec_module(protocol)
    row_file=CAMPAIGN/'work/lead/finish-corrected-train-v1/train-token-rows.jsonl'
    draw_file=CAMPAIGN/'work/finish-corrected-schedule-v1/finish-corrected-draws-200.json'
    assert digest(draw_file)=='017fc74ccb6fad43d2bf263d0b3d892fa13bd3fce37bff9ae6a6572fde912d31'
    draws=json.loads(draw_file.read_text())['row_ids']
    wanted=set(draws[:800]);selected={};samples={};counts=collections.Counter();ops=collections.Counter()
    totals=collections.Counter();row_hash=hashlib.sha256();batch=[];checks=0
    def check_batch(batch):
        nonlocal checks
        output=target_only_collator(batch)
        for i,r in enumerate(batch):
            start,end=r['target_start'],len(r['input_ids'])
            assert output['labels'][i,:start].eq(-100).all()
            assert output['labels'][i,start:end].tolist()==r['input_ids'][start:]
            assert output['labels'][i,end:].eq(-100).all()
            assert output['attention_mask'][i,:end].eq(1).all()
            assert output['attention_mask'][i,end:].eq(0).all()
            assert output['labels'][i,end-1]==1
            checks+=1
    with row_file.open('rb') as f:
        for raw in f:
            row_hash.update(raw);r=json.loads(raw);protocol.validate_training_row(r)
            assert r['split']=='train';assert len(r['input_ids'])<=4096
            counts[r['family']]+=1;ops[r['target_operation']]+=1
            totals['rows']+=1;totals['context_tokens_visible']+=r['target_start'];totals['old_prompt_supervision_tokens']+=r['target_start']-1
            totals['target_supervision_including_eos']+=len(r['input_ids'])-r['target_start']
            totals['eos_supervision_tokens']+=1;totals['max_sequence_tokens']=max(totals['max_sequence_tokens'],len(r['input_ids']))
            if r['id'] in wanted:selected[r['id']]={'family':r['family'],'target':len(r['input_ids'])-r['target_start'],'prompt':r['target_start']-1,'length':len(r['input_ids'])}
            if len(samples.setdefault(r['family'],[]))<2:samples[r['family']].append(r)
            batch.append(r)
            if len(batch)==4:check_batch(batch);batch=[]
    if batch:check_batch(batch)
    assert row_hash.hexdigest()=='e2408c5177e3134189c4b86f7247db41fc0c1ea5aab9d9260a3dea55906786fe'
    assert checks==11526 and counts['finish_block']==4051 and counts['no_op']==1140
    from transformers import AutoTokenizer
    tokenizer_dir=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0')
    tokenizer_pins={name:digest(tokenizer_dir/name) for name in ['tokenizer.json','tokenizer_config.json']}
    assert tokenizer_pins=={'tokenizer.json':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81','tokenizer_config.json':'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b'}
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True,trust_remote_code=False)
    token_checks=[]
    for family,rows in sorted(samples.items()):
        for r in rows:
            encode=lambda text:tokenizer.encode(text,add_special_tokens=False,split_special_tokens=True)
            prompt,target=encode(r['prompt_text']),encode(r['target_text'])
            assert r['input_ids']==[0]+prompt+target+[1]
            assert encode(r['prompt_text']+r['target_text'])==prompt+target
            assert r['target_start']==len(prompt)+1
            assert target==r['target_body_tokens']+r['target_terminal_tokens']
            token_checks.append({'id':r['id'],'family':family,'prompt_tokens':len(prompt),'target_plus_eos':len(target)+1,'prompt_sha256':hashlib.sha256(r['prompt_text'].encode()).hexdigest(),'target_sha256':hashlib.sha256(r['target_text'].encode()).hexdigest()})
    prefixes={}
    for step in [25,50]:
        rows=[selected[i] for i in draws[:step*16]]
        prefixes[str(step)]={'draws':len(rows),'families':dict(collections.Counter(r['family'] for r in rows)),
                            'supervised_target_tokens_including_eos':sum(r['target'] for r in rows),
                            'masked_old_prompt_supervision_tokens':sum(r['prompt'] for r in rows),
                            'long_rows':sum(r['length']>2048 for r in rows)}
    timing={}
    for label in ['a','b']:
        p=Path(f'/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT-finish-correction-{label}/telemetry.jsonl')
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        updates=[r for r in rows if r.get('event')=='optimizer_step']
        timing[label]={'telemetry_sha256':digest(p),'optimizer_updates':len(updates),'optimizer_seconds_sum':sum(r['seconds'] for r in updates),
                       'first_step':updates[0]['step'],'last_step':updates[-1]['step'],'seconds_scope':'saved callback update timings; excludes model load, full archive, evaluation, outer guard'}
    result={'status':'pass','train_sha256':row_hash.hexdigest(),'train_mask_rows_checked':checks,'totals':dict(totals),'families':dict(counts),'operations':dict(ops),'schedule_prefixes':prefixes,'tokenizer_pins':tokenizer_pins,'token_roundtrips':token_checks,'prior_timing':timing,'model_weights_read_or_loaded':False,'training_or_optimizer_steps_run':False,'gpu_or_network':False,'dev_or_final_content_read':False}
    (HERE/'training-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'pass','rows':checks,'tokenizer_roundtrips':len(token_checks),'prefixes':prefixes,'prior_timing':timing},indent=2))


if __name__=='__main__':main()
