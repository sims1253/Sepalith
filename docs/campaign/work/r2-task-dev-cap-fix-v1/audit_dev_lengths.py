import os,json,sys,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from transformers import AutoTokenizer
from sepalith.campaign_protocol import PromptContext,build_training_row
w=Path(__file__).resolve().parent;r=json.loads((w.parent/'r2-local-task-stage-preparation-v3/task-recipe.template.json').read_text());panel=Path(r['development_panel']['path']);assert hashlib.sha256(panel.read_bytes()).hexdigest()==r['development_panel']['sha256']
t=AutoTokenizer.from_pretrained('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native',local_files_only=True,trust_remote_code=False)
results=[]
for line in panel.read_text().splitlines():
 c=json.loads(line);assert c['split']=='dev';row=build_training_row(PromptContext.from_mapping(c['context']),operation=c['operation'],region_new=c['region_new'],tokenizer=t,row_id=c['id'],family=c['family'],package_id=c['package_id'],split='dev')
 prompt=row['target_start'];reference=len(row['input_ids'])-prompt
 results.append({'id':c['id'],'prompt_tokens':prompt,'reference_tokens':reference,'reference_over_generation_cap':reference>192,'context_fits':max(len(row['input_ids']),prompt+192)<=4096})
assert sorted(x['id'] for x in results)==sorted(r['development_case_ids']);assert all(x['context_fits'] for x in results)
result={'cases':len(results),'reference_over_generation_cap':sum(x['reference_over_generation_cap'] for x in results),'max_reference_tokens':max(x['reference_tokens'] for x in results),'max_prompt_tokens':max(x['prompt_tokens'] for x in results),'all_context_fits':True,'generation_cap':192,'results':results}
(w/'actual-dev-length-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='results'}))
