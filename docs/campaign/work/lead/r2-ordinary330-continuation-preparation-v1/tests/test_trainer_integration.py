#!/usr/bin/env python3
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'source/experiments/training/full_weight_cpt_trainer.py';s=p.read_text()
for fragment in ('resolve_resume_identity(recipe_path,identity(recipe),resume,ordinary_canary_transition_admission)','verify_checkpoint(resume, resume_identity','trainer.train(resume_from_checkpoint=str(resume)','sampler["resume_lineage"]=resume_lineage','seal_checkpoint(source, identity(recipe)','"ordinary_canary_resume_lineage":resume_lineage','--ordinary-canary-transition-admission'):
 assert fragment in s,fragment
assert 'FullWeightTrainer(model=model' in s and 'full_weight_optimizer_config' in s
print('PASS transition before verify, Trainer full resume, destination identity and lineage propagation')
