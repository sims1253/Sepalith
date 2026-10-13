"""Relocate only parent file paths, preserving the exact task checkpoint identity."""
import argparse,copy,hashlib,json
from pathlib import Path
from selection_contract import check_task_recipe,path

def relocate(recipe,original_recipe_record,local_parent):
 check_task_recipe(recipe);r=copy.deepcopy(recipe);old=Path(r['model_path']);local=path(local_parent)
 for item in r['inputs']:
  q=Path(item['path'])
  if q.parent==old:item['path']=str(local/q.name)
 r['model_path']=str(local)
 r['merge_parent_relocation']={'source_recipe':original_recipe_record,'original_parent_path':str(old),'local_parent_path':str(local),'parent_file_sha256_unchanged':True,'checkpoint_identity_unchanged':True}
 if r['identity']!=recipe['identity']:raise ValueError('relocation changed checkpoint identity')
 check_task_recipe(r);return r
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint-recipe',type=Path,required=True);a.add_argument('--local-parent',required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 if not x.checkpoint_recipe.is_absolute() or not x.output.is_absolute() or x.output.exists():raise ValueError('explicit source and fresh output required')
 data=x.checkpoint_recipe.read_bytes();r=relocate(json.loads(data),{'path':str(x.checkpoint_recipe),'sha256':hashlib.sha256(data).hexdigest()},x.local_parent)
 x.output.write_text(json.dumps(r,indent=2)+'\n')
