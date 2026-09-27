#!/usr/bin/env python3
import hashlib,json,os,pathlib,sys
sys.path.insert(0,'/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src')
from sepalith.runner import Runner
W=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-remaining-v1')
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
admission=json.loads((W/'root-admission.json').read_text())
expected={'schema':'sepalith.cpt.remaining-root-admission.v1','status':'admitted','task':'SFT-11-CPT-REMAINING','recipe_sha256':sha(W/'recipe.json'),'runner_recipe_sha256':sha(W/'runner-recipe.json'),'snapshot':json.loads((W/'runner-recipe.json').read_text())['snapshot']}
if admission != expected: raise SystemExit('root admission does not exactly bind frozen packet')
runner=Runner('/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-cpt-remaining-v1')
runner.resume()
attempt=runner.run_next()
print(json.dumps({'attempt':attempt,'plan':runner.plan()},indent=2))
raise SystemExit(0 if attempt else 2)
