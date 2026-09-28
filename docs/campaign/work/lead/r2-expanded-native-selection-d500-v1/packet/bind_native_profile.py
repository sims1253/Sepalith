"""Root metadata binding after verified merge and Q8 integrity review. No weight reads."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'native_evaluator'))
from selection_binding import validate_profile
from selection_contract import check_task_recipe

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,expected):
 if not p.is_absolute() or sha(p)!=expected:raise ValueError('root metadata pin differs')
 return json.loads(p.read_text())
def build(parent,integrity,parent_sha,integrity_sha):
 if parent['schema_version']!='sepalith.r2-task-sft.parent-manifest.v1' or parent['kind']!='merged_task_sft':raise ValueError('task merge manifest required')
 if parent.get('tokenizer_original_bytes_restored') is not True:raise ValueError('original tokenizer bytes required')
 ident=parent['sft_identity'];step=parent['sft_checkpoint_manifest']['step']
 if parent['sft_checkpoint_manifest']['identity']!=ident or parent['sft_checkpoint_manifest']['full'] is not True or parent['source_cursor']!=step*16:raise ValueError('checkpoint lineage differs')
 if integrity.get('status')!='Q8_integrity_verified_pending_native_quality' or integrity['parent_manifest_sha256']!=parent_sha:raise ValueError('independent export integrity binding required')
 q=integrity['q8'];verify=parent['merge_verification']
 if verify.get('accumulation_dtype')!='float32' or verify.get('rounding')!='one final BF16 cast' or verify['independent_lora_matrix_merges_exact']!=294:raise ValueError('accepted merge arithmetic required')
 p=json.loads((H/'profile-baseline.json').read_text());revision='r2-expanded-'+parent['recipe_sha256'][:12]+'-step'+str(step)+'-q8_0'
 p['model'].update(name='model-Q8_0.gguf',sha256=q['sha256'],bytes=q['bytes'],url=None)
 p['model_profile'].update(modelSha256=q['sha256'],modelRevision=revision)
 p['selection']={'purpose':'development_selection_only','task_source':ident['source'],'checkpoint_step':step,'parent_kind':ident['parent']['kind'],'revision':revision,'q8_path':q['path'],'tokenizer_dir':parent['merged_model_path'],'merged_parent_manifest_sha256':parent_sha,'export_integrity_receipt_sha256':integrity_sha,'task_recipe_sha256':parent['recipe_sha256'],'checkpoint_manifest_sha256':parent['checkpoint_manifest_sha256']}
 validate_profile(p);return p

def main():
 a=argparse.ArgumentParser(description=__doc__)
 for n in ('parent-manifest','integrity-receipt','output'):a.add_argument('--'+n,type=Path,required=True)
 for n in ('parent-manifest-sha256','integrity-receipt-sha256'):a.add_argument('--'+n,required=True)
 x=a.parse_args();p=build(load(x.parent_manifest,x.parent_manifest_sha256),load(x.integrity_receipt,x.integrity_receipt_sha256),x.parent_manifest_sha256,x.integrity_receipt_sha256)
 if x.output.exists():raise ValueError('fresh profile path required')
 x.output.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'profile':str(x.output),'sha256':sha(x.output),'root_native_admission_still_required':True}))
if __name__=='__main__':main()
