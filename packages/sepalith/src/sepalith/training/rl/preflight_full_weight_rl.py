#!/usr/bin/env python3
"""CPU preflight for a root-admitted dense GRPO recipe and binding."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from sepalith.training.checkpoint.campaign_checkpoint import write_json
from sepalith.training.rl.campaign_rl_train import preflight_rl_recipe
from sepalith.training.rl.full_weight_rl_production import validate_production_binding

def sha(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--recipe',type=Path,required=True);ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--receipt',type=Path,required=True);a=ap.parse_args(argv)
 recipe=json.loads(a.recipe.read_text());binding=validate_production_binding(json.loads(a.binding.read_text()))
 if recipe.get('production_binding_sha256')!=sha(a.binding):raise ValueError('recipe production binding hash differs')
 report=preflight_rl_recipe(recipe)
 identity=report['identity'];parent=identity['parent'];data=identity['data'];identity['policy']
 exact={'source_manifest_sha256':identity['source']['manifest_sha256'],'model_manifest_sha256':parent['manifest_sha256'],'model_weights_sha256':parent['model_weights_sha256'],'tokenizer_json_sha256':parent['tokenizer_json_sha256'],'reward_buffer_manifest_sha256':data['reward_buffer_manifest_sha256'],'prompt_context_manifest_sha256':data['context_sha256']}
 for name,value in exact.items():
  if binding[name]!=value:raise ValueError(f'production binding differs from recipe at {name}')
 result={'schema':'sepalith.rl11.full-weight-production-preflight.v1','status':'PASS','recipe_sha256':sha(a.recipe),'binding_sha256':sha(a.binding),'identity':identity,'records':report['records'],'geometry':report['geometry'],'launch_authorized':False,'note':'CPU preflight does not admit model loading, rollout, or optimizer training'}
 write_json(a.receipt,result);print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
