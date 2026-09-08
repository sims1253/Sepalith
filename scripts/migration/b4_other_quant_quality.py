#!/usr/bin/env python3
"""Paired quality of three frozen b4 exports; no automatic release decision."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/eval'))
import eval_scenarios as scenario
import eval_noop_fp as noop
from s1_gpu import GpuServer,DeadlineExceeded,check_offload,tokenize,write

ARMS=['Q8_0','Q6_K_imatrix','IQ4_XS_imatrix']


def cases(assets):
    return [json.loads(l) for l in (assets/'cases.jsonl').read_text().splitlines()]


def prepare(run,assets):
    rows=cases(assets)
    if len(rows)!=513 or len({(r['kind'],r['id']) for r in rows})!=513:
        raise ValueError('Require 255 scenario and258 unique no-op cases')
    if sum(r['kind']=='scenario' for r in rows)!=255:
        raise ValueError('Scenario cohort changed')
    for r in rows:
        if r['kind']=='scenario':scenario.scenarios.validate_example(r['scenario'])
        elif r['kind']!='noop':raise ValueError('Unknown case kind')
    write(run/'prepared.json',dict(arms=ARMS,case_ids=[[r['kind'],r['id']] for r in rows],
        expected_rows=1539,ctx=8192,max_seconds=5370,
        scope='Current frozen b4 export comparison; not causal proof of imatrix benefit',
        gates='Observed scenario exact delta >=-1pp; noop a/c/d proposal delta descriptive; intent judge separate',
        calibration='Existing imatrix covers only8x512chunks; no release-grade coverage claim'))


def measure(run,assets):
    p=json.loads((run/'prepared.json').read_text());rows=cases(assets);counts={};server=None
    def expire(*_):raise DeadlineExceeded('Quant quality exceeded90minute bound')
    old=signal.signal(signal.SIGALRM,expire);deadline=time.monotonic()+p['max_seconds'];signal.alarm(p['max_seconds'])
    try:
        with (run/'requests.jsonl').open('x') as out:
            for arm in p['arms']:
                server=GpuServer(assets/(arm+'.gguf'),18475,['-lv','4','--seed','20260905'],ctx=p['ctx'],
                    server=Path(os.environ['S1_RUNTIME'])/'llama-server',foreground=True,log_path=run/('server-'+arm+'.log'))
                try:
                    server.start(ready_timeout=180);check_offload(server.log_path.read_text())
                    # Check the whole frozen cohort before any scored generation.
                    current={}
                    for r in rows:
                        key=(r['kind'],r['id']);n=tokenize(server.port,r['prompt']);limit=640 if r['kind']=='scenario' else 320
                        if n+limit+16>p['ctx']:raise ValueError('Prompt exceeds context: '+str(key))
                        if key in counts and n!=counts[key]:raise ValueError('Cross-arm tokenization mismatch')
                        current[key]=n
                    if not counts:counts=current
                    with (run/'token-audit.jsonl').open('a') as audit:
                        for (kind,identity),n in current.items():audit.write(json.dumps(dict(arm=arm,kind=kind,id=identity,tokens=n))+'\n')
                    for r in rows:
                        key=(r['kind'],r['id']);limit=640 if r['kind']=='scenario' else 320
                        stops=[scenario.STOP] if r['kind']=='scenario' else noop.EXT_STOPS
                        body=json.dumps(dict(prompt=r['prompt'],max_tokens=limit,temperature=0,stop=stops,stream=False)).encode()
                        req=urllib.request.Request(f'http://127.0.0.1:{server.port}/v1/completions',data=body,headers={'Content-Type':'application/json'})
                        started=time.monotonic()
                        with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
                        rec=dict(arm=arm,kind=r['kind'],id=r['id'],tokenized_prompt_tokens=current[key],
                            served_prompt_tokens=data.get('usage',{}).get('prompt_tokens'),response=data,wall_s=time.monotonic()-started,at=time.time())
                        choice=data['choices'][0]
                        if rec['served_prompt_tokens']!=current[key] or choice.get('finish_reason') not in ('stop','length'):
                            write(run/'invalid-response.json',rec);raise ValueError('Invalid quality response')
                        if r['kind']=='scenario':
                            pred=scenario.parse_pred('zeta2',choice['text']);gt=[l.rstrip() for l in r['scenario']['region_new']]
                            while gt and not gt[-1]:gt.pop()
                            passed,kind,reason=scenario.validator_verdict(r['scenario'],pred)
                            rec.update(exact=pred==gt,valid_pass=passed,fail_kind=kind,valid_reason=reason,family=r['scenario']['family'])
                        else:
                            pred=noop.parse_prediction(choice['text']);rec.update(proposal=bool(pred),expectation=r['expectation'],cls=r['cls'])
                        out.write(json.dumps(rec,allow_nan=False)+'\n');out.flush()
                    print(json.dumps(dict(arm=arm,rows=len(rows))),flush=True)
                finally:
                    signal.alarm(0);server.stop();server=None;signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    finally:
        signal.alarm(0)
        try:
            if server is not None:server.stop()
        finally:signal.signal(signal.SIGALRM,old)


def paired(base,candidate):
    if len(base)!=len(candidate) or not base:raise ValueError('Invalid paired cohort')
    loss=sum(a and not b for a,b in zip(base,candidate));gain=sum(b and not a for a,b in zip(base,candidate));d=loss+gain
    p=min(1,2*sum(math.comb(d,k) for k in range(min(loss,gain)+1))/2**d) if d else 1
    return dict(n=len(base),baseline_positive=sum(base),candidate_positive=sum(candidate),losses=loss,gains=gain,
        delta_pp=100*(gain-loss)/len(base),mcnemar_exact_two_sided=p)


def evaluate(run):
    p=json.loads((run/'prepared.json').read_text());rows=[json.loads(l) for l in (run/'requests.jsonl').read_text().splitlines()]
    if [(r['arm'],r['kind'],r['id']) for r in rows]!=[(a,k,i) for a in p['arms'] for k,i in p['case_ids']]:
        raise ValueError('Incomplete, duplicate or reordered quality coverage')
    if any(r.get('error') or r['served_prompt_tokens']!=r['tokenized_prompt_tokens'] for r in rows):raise ValueError('Invalid quality request')
    summary={};comparisons={}
    for arm in p['arms']:
        rs=[r for r in rows if r['arm']==arm];sc=[r for r in rs if r['kind']=='scenario'];no=[r for r in rs if r['kind']=='noop' and r['expectation']=='no_proposal']
        summary[arm]=dict(scenarios=len(sc),exact=sum(r['exact'] for r in sc),valid=sum(r['valid_pass'] for r in sc),
            noop_scored=len(no),noop_false_suggestions=sum(r['proposal'] for r in no),generation_limit_rows=sum(r['response']['choices'][0]['finish_reason']=='length' for r in rs))
        summary[arm]['scenario_by_family']={family:dict(n=sum(r['family']==family for r in sc),
            exact=sum(r['exact'] for r in sc if r['family']==family),valid=sum(r['valid_pass'] for r in sc if r['family']==family)) for family in sorted({r.get('family','unknown') for r in sc})} if all('family' in r for r in sc) else {}
        summary[arm]['noop_by_class']={cls:dict(n=sum(r.get('cls')==cls for r in rs),
            proposals=sum(r.get('proposal',False) for r in rs if r.get('cls')==cls)) for cls in sorted({r['cls'] for r in rs if r['kind']=='noop' and 'cls' in r})}
        if arm==p['arms'][0]:base_sc,base_no=sc,no;continue
        comparisons[arm]=dict(exact=paired([r['exact'] for r in base_sc],[r['exact'] for r in sc]),
            valid=paired([r['valid_pass'] for r in base_sc],[r['valid_pass'] for r in sc]),
            noop_restraint=paired([not r['proposal'] for r in base_no],[not r['proposal'] for r in no]))
        comparisons[arm]['scenario_exact_within_1pp_observed']=comparisons[arm]['exact']['delta_pp']>=-1
        comparisons[arm]['scenario_valid_within_1pp_observed']=comparisons[arm]['valid']['delta_pp']>=-1
    write(run/'evaluation.json',dict(rows=len(rows),arms=summary,paired=comparisons,
        inference_boundary='Paired p-values are unadjusted descriptive tests, not a statistical noninferiority guarantee; package dependence remains'))
    write(run/'verdict.json',dict(verdict='PAIRED-EXPORT-QUALITY-MEASURED',adoption='NOT-ASSESSED',
        boundary='Intent LLM-judge gate remains separate. Frozen export comparison; historical receipts do not hash the f16 parent. Existing imatrix coverage is only8x512chunks. No publication or calibrated-quant promotion.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','measure','evaluate']);p.add_argument('--run',type=Path,required=True);p.add_argument('--assets',type=Path)
    a=p.parse_args();evaluate(a.run) if a.action=='evaluate' else globals()[a.action](a.run,a.assets)
