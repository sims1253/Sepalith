#!/usr/bin/env python3
"""Build the bounded 25% semantic-no-op sensitivity schedule from metadata."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
BASE = PLAN / 'work/r2-task-mixture-v1'
ROWS = BASE / 'verified-corrected-short-token-rows.jsonl'
PROV = BASE / 'verified-corrected-short-provenance.jsonl'
SCHEDULE = BASE / 'verified-corrected-short-draw-manifest-noop25.json'
DAT02 = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
SAMPLER = EXEC / 'experiments/training/campaign_sampling.py'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb',buffering=4*1024*1024) as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
    provs=[]; rows=[]
    with PROV.open() as pf, ROWS.open() as rf:
        for rr,pp in zip(rf,pf): rows.append(json.loads(rr));provs.append(json.loads(pp))
    assert len(rows)==len(provs)==8526
    metadata=[]
    for row,prov in zip(rows,provs):
        assert row['id']==prov['id'] and prov['split_binding']=='bound_cpt_train'
        total=len(row['input_ids']); target=row['target_token_count']+1
        metadata.append({'row_id':row['id'],'family':row['family'],'source_id':prov['group_id'],'package_id':row['package_id'],'split':'train','semantic_noop':row['target_operation']=='no_op','operation':row['target_operation'],'prompt_tokens':row['target_start'],'target_tokens':target,'total_tokens':total,'length_bucket':'short' if total<=2048 else 'long','naturally_long':total>2048,'source_kind':'ordinary','provenance':f"verified-{prov['source_class']}:{prov['group_id']}"})
    dat02=json.loads(DAT02.read_text())
    sampling=load_module('dat10_noop25_sampler',SAMPLER)
    sched=sampling.build_draw_manifest(metadata,max_steps=1000,effective_batch=16,split_id=dat02['split_id'],seed=3407,token_rows_sha256=digest(ROWS),requested_draws=16000,noop_fraction=.25,family_ceiling=.25,small_pack_cap=3,ordinary_replay_cap=8,naturally_long_fraction=.20,short_max_tokens=2048,long_max_tokens=4096)
    sampling.validate_draw_manifest(sched)
    SCHEDULE.write_text(json.dumps(sched,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':'PASS','path':str(SCHEDULE),'sha256':digest(SCHEDULE),'draw_count':len(sched['row_ids']),'status_schedule':sched['status'],'achieved_mixture':sched['achieved_mixture']},sort_keys=True))
if __name__=='__main__':main()
