#!/usr/bin/env python3
import json,sys,tempfile,hashlib
from pathlib import Path
PACKET=Path(__file__).resolve().parents[1];SRC=PACKET/'source/experiments/training';sys.path.insert(0,str(SRC))
import full_weight_cpt_trainer as T
import ordinary_canary_resume as O
recipe=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-native-cpt194-root-admission-v1/runtime-recipe.json');value=json.loads(recipe.read_text());prod=T.identity(value)
cp=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-varlen-canary-322-ordinary_reference-v1/runtime/checkpoint-330');mp=cp/'campaign-manifest.json';sp=cp/'campaign-state.json';before=(O.sha(mp),O.sha(sp))
with tempfile.TemporaryDirectory(prefix='ordinary330-actual-') as td:
 ad=Path(td)/'admission.json';ad.write_text(json.dumps({'schema':O.SCHEMA,'status':'admitted','decision':'ordinary_canary_to_production','launch_authorized':True,'bound_recipe_sha256':O.sha(recipe),'checkpoint':str(cp),'checkpoint_manifest_sha256':O.sha(mp),'checkpoint_step':330,'source_identity_sha256':O.canonical_sha(O.canary_identity(prod)),'destination_identity_sha256':O.canonical_sha(prod),'allowed_policy_delta':O.ORDINARY_POLICY,'payload_action':'load_original_full_state_without_rewrite_or_copy'})+'\n')
 ident,lineage=O.resolve_resume_identity(recipe,prod,cp,ad);assert ident==O.canary_identity(prod);assert lineage['source_checkpoint_manifest_sha256']==before[0]
assert before==(O.sha(mp),O.sha(sp))
print(json.dumps({'status':'PASS','checkpoint_manifest_sha256':before[0],'campaign_state_sha256':before[1],'step':330,'cursor':4224,'source_identity_sha256':O.canonical_sha(ident),'destination_identity_sha256':O.canonical_sha(prod),'original_metadata_unchanged':True,'payload_files_read':False},sort_keys=True))
