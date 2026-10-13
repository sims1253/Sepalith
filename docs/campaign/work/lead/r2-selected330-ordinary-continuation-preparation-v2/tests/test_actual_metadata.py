#!/usr/bin/env python3
import hashlib,importlib.util,json,tempfile,sys
from pathlib import Path
PACKET=Path(__file__).resolve().parents[1];SRC=PACKET/'source/experiments/training';sys.path.insert(0,str(SRC));import ordinary_canary_resume as O
spec=importlib.util.spec_from_file_location('trainer',SRC/'full_weight_cpt_trainer.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
recipe=Path('docs/campaign/work/lead/r2-native-cpt194-root-admission-v1/runtime-recipe.json').resolve();prod=T.identity(json.loads(recipe.read_text()));resume=O.SOURCE_CHECKPOINT
before={n:hashlib.sha256((resume/n).read_bytes()).hexdigest() for n in ['campaign-manifest.json','campaign-state.json']}
with tempfile.TemporaryDirectory() as td:
 ad=Path(td)/'admission.json';v=json.loads((PACKET/'selected-packed330-transition-admission.template.json').read_text());v.update(status='admitted',launch_authorized=True,bound_recipe_sha256=O.sha(recipe));ad.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');ident,lineage=O.resolve_resume_identity(recipe,prod,resume,ad)
after={n:hashlib.sha256((resume/n).read_bytes()).hexdigest() for n in before};state=json.loads((resume/'campaign-state.json').read_text())
assert before==after and O.canonical_sha(ident)=='6d5a7a2dff56d74e1d31662e60a2dff6bd81b46827bbf4760bf16d530fd1f0cf'
print(json.dumps({'status':'PASS','checkpoint_manifest_sha256':before['campaign-manifest.json'],'campaign_state_sha256':before['campaign-state.json'],'source_identity_sha256':O.canonical_sha(ident),'destination_identity_sha256':lineage['destination_identity_sha256'],'source_arm':lineage['source_arm'],'destination_execution':lineage['destination_execution'],'step':state['step'],'cursor':state['sampler']['cursor'],'metadata_unchanged':True,'payload_files_read':False},sort_keys=True))
