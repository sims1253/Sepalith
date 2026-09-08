#!/usr/bin/env python3
"""Measure a complete b4 V1a baseline without silent point rejection."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import urllib.request

from s1_gpu import GpuServer, DeadlineExceeded, check_offload, tokenize, write
import episode_metrics
import judge_loop


class RequestFailure(BaseException):
    """Escape the historical judge loop's broad skip-on-error handler."""


def select(path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows=[r for r in rows if r.get('n_points',0)>=5]
    random.Random(3).shuffle(rows)
    chosen=rows[:60]
    if len(chosen)!=60 or len({(r['key'],r.get('variant',1)) for r in chosen})!=60:
        raise ValueError('Expected 60 unique episode identities')
    return chosen


def prepare(run, assets):
    chosen=select(assets/'trajectories.jsonl')
    for episode in chosen:
        for p in episode['points'][:30]:
            if p['label'] not in ('typing','noop') or not isinstance(p['prompt'],str):
                raise ValueError('Invalid episode point')
    write(run/'prepared.json',dict(episodes=[dict(key=r['key'],variant=r.get('variant',1),points=min(30,len(r['points']))) for r in chosen],
        expected_points=sum(min(30,len(r['points'])) for r in chosen),ctx=32768,max_tokens=160,
        max_seconds=7170,scope='New complete b4 V1a baseline; not paired with historical skipped-point runs'))


def measure(run, assets):
    p=json.loads((run/'prepared.json').read_text());chosen=select(assets/'trajectories.jsonl')
    server=GpuServer(Path(os.environ['V1A_MODEL']),18473,['-lv','4','--seed','20260905'],ctx=p['ctx'],
        server=Path(os.environ['S1_RUNTIME'])/'llama-server',foreground=True,log_path=run/'server.log')
    def expire(*_):raise DeadlineExceeded('V1a exceeded two-hour bound')
    old=signal.signal(signal.SIGALRM,expire);signal.alarm(p['max_seconds'])
    original=judge_loop.complete
    current={}
    try:
        server.start(ready_timeout=300);check_offload(server.log_path.read_text())
        with (run/'requests.jsonl').open('x') as requests, (run/'episodes.jsonl').open('x') as episodes:
            def complete(port,prompt,max_tokens=160):
                try:
                    if current['point'] > 0 and judge_loop.HIST_MARK not in prompt:
                        raise ValueError('Proposal history was not injected into prompt')
                    count=tokenize(port,prompt)
                    if count+max_tokens+16>p['ctx']:
                        raise ValueError('History-expanded prompt exceeds context reserve')
                    body=json.dumps(dict(prompt=prompt,max_tokens=max_tokens,temperature=0.0)).encode()
                    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=body,headers={'Content-Type':'application/json'})
                    start=time.monotonic()
                    with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
                    actual=data.get('usage',{}).get('prompt_tokens')
                    row=dict(**current,prompt=prompt,tokenized_prompt_tokens=count,served_prompt_tokens=actual,
                             response=data,wall_s=time.monotonic()-start,at=time.time())
                    requests.write(json.dumps(row,allow_nan=False)+'\n');requests.flush()
                    if actual!=count:raise ValueError('Served prompt count differs from tokenizer')
                    choice=data['choices'][0]
                    if choice.get('finish_reason') not in ('stop','length'):
                        raise ValueError('Unexpected completion finish reason')
                    current['point']+=1
                    return choice['text']
                except Exception as e:
                    raise RequestFailure(str(e)) from e
            judge_loop.complete=complete
            for index,episode in enumerate(chosen):
                current.clear();current.update(episode=index,key=episode['key'],variant=episode.get('variant',1),point=0)
                result=judge_loop.run_trajectory(episode,server.port,cap_points=30)
                if len(result['points'])!=min(30,len(episode['points'])):
                    raise ValueError('Judge loop dropped an episode point')
                episodes.write(json.dumps(result,allow_nan=False)+'\n');episodes.flush()
                print(json.dumps(dict(episode=index,points=len(result['points']),stats=result['stats'])),flush=True)
    finally:
        signal.alarm(0)
        try:server.stop()
        finally:judge_loop.complete=original;signal.signal(signal.SIGALRM,old)


def evaluate(run):
    p=json.loads((run/'prepared.json').read_text())
    episodes=[json.loads(l) for l in (run/'episodes.jsonl').read_text().splitlines()]
    requests=[json.loads(l) for l in (run/'requests.jsonl').read_text().splitlines()]
    expected=[(r['key'],r['variant'],r['points']) for r in p['episodes']]
    if [(r['key'],r.get('variant',1),len(r['points'])) for r in episodes]!=expected:
        raise ValueError('Incomplete episode coverage')
    expected_requests=[(i,k,v,j) for i,(k,v,n) in enumerate(expected) for j in range(n)]
    if [(r['episode'],r['key'],r['variant'],r['point']) for r in requests]!=expected_requests:
        raise ValueError('Incomplete request coverage')
    if any(r['served_prompt_tokens']!=r['tokenized_prompt_tokens'] for r in requests):
        raise ValueError('Prompt count mismatch')
    summary=episode_metrics.summarize('b4-complete-32k-gpu',episodes,str(run/'episodes.jsonl'))
    if summary['stats_mismatch_rows']:raise ValueError('Episode aggregate disagrees with decisions')
    summary['generation_limit_rows']=sum(r['response']['choices'][0]['finish_reason']=='length' for r in requests)
    summary['request_count']=len(requests)
    write(run/'evaluation.json',summary)
    write(run/'verdict.json',dict(verdict='B4-EPISODE-BASELINE-MEASURED',adoption='NOT-ASSESSED',
        boundary='Existing lexical acceptance heuristic and simulator clock, not human preference; complete 32K context regime differs from historical skipped-point 2K slots. No retraining or paired model comparison.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','measure','evaluate'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--assets',type=Path)
    a=p.parse_args();evaluate(a.run) if a.action=='evaluate' else globals()[a.action](a.run,a.assets)
