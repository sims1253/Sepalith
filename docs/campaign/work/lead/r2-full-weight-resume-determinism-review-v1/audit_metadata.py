#!/usr/bin/env python3
"""Read only tiny proof metadata; never open checkpoint payloads."""
import argparse, json, re
from pathlib import Path

MAX_READ = 16 * 1024 * 1024

def read_json(path):
    path=Path(path)
    if path.stat().st_size > MAX_READ: raise ValueError(f"refusing large read:{path}")
    return json.loads(path.read_text())

def audit(root, source, process_log):
    root,source,process_log=Path(root),Path(source),Path(process_log)
    manifests={lane:read_json(root/lane/'checkpoint-1/campaign-manifest.json') for lane in ('uninterrupted','interrupted')}
    states={lane:read_json(root/lane/'checkpoint-1/trainer_state.json') for lane in manifests}
    a,b=manifests['uninterrupted'],manifests['interrupted']
    names=sorted(set(a['files'])|set(b['files']))
    compare={name:a['files'].get(name)==b['files'].get(name) for name in names}
    text=process_log.read_text() if process_log.stat().st_size<=MAX_READ else (_ for _ in ()).throw(ValueError('refusing large log'))
    observations=[]
    for match in re.finditer(r"\{'loss': '([^']+)', 'grad_norm': '([^']+)'",text):
        observations.append({'loss_display':match.group(1),'grad_norm_display':match.group(2)})
    code=source.read_text()
    return {
      'schema':'sepalith.sft11.resume-determinism-metadata-audit.v1',
      'checkpoint_step':1,
      'identity_equal':a['identity']==b['identity'],
      'manifest_file_comparison':compare,
      'payloads_never_opened':True,
      'step1':{
        lane:{'loss':states[lane]['log_history'][0]['loss'],'grad_norm':states[lane]['log_history'][0]['grad_norm']} for lane in states},
      'gradient_norm_absolute_difference':abs(states['uninterrupted']['log_history'][0]['grad_norm']-states['interrupted']['log_history'][0]['grad_norm']),
      'process_log_first_updates':observations[:3],
      'source_facts':{
        'set_seed_default_nondeterministic': 'set_seed(seed)' in code and 'set_seed(seed, deterministic=True)' not in code,
        'no_torch_deterministic_algorithms': 'use_deterministic_algorithms' not in code,
        'stochastic_round_uses_cuda_rng': 'torch.rand_like(probability_upper)' in (source.parent/'full_weight_optimizer.py').read_text(),
      },
    }

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--process-log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); result=audit(a.root,a.source,a.process_log);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
