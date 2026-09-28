#!/usr/bin/env python3
"""Exercise real launch dispatch and full CPU preflight except the 5GB weight hash."""
import json,sys
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'source/experiments/training'))
import campaign_launch
import campaign_expanded_sft
import campaign_sft
recipe=json.loads((WORK/'recipe.json').read_text())
entry=campaign_launch.training_entrypoint(recipe)
expected=WORK/'source/experiments/training/campaign_expanded_sft.py'
if entry.resolve()!=expected.resolve():raise RuntimeError(f'launch dispatch mismatch: {entry}')
weight=(Path(recipe['model_path'])/'model.safetensors').resolve();original=campaign_sft.verified_file;skipped=[]
def verified_without_large_weight(record):
 path=Path(record['path'])
 if path.resolve()==weight:
  if not path.is_file() or record['sha256']!=recipe['identity']['parent']['weights_sha256']:raise ValueError('parent weight metadata mismatch')
  skipped.append(str(path));return path
 return original(record)
campaign_sft.verified_file=verified_without_large_weight
rows,draws,schedule,exposure=campaign_expanded_sft.preflight(recipe)
if skipped!=[str(weight)]:raise RuntimeError(f'unexpected skip set: {skipped}')
print(json.dumps({'status':'PASS','launch_entrypoint':str(entry),'stage':recipe['stage'],'rows':len(rows),'draws':len(draws),'updates':len(exposure),'split_id':schedule['split_id'],'CUDA_started':False,'model_weight_bytes_read':False,'root_remaining_check':'hash exact 5GB parent model input in unmodified preflight'},sort_keys=True))
