from collections import Counter
import hashlib, json
from pathlib import Path
from campaign_rl_contexts import render_prompt
from sepalith.campaign_protocol import PromptContext
root=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1')
meta=json.loads((root/'materialization.json').read_text())
selected=json.loads((root/'selected-train-ids.json').read_text())['row_ids']
sidecar_ids=[]; forbidden=[]; bad_render=[]; bad_shape=[]; avail=Counter(); versions=Counter(); source_files=Counter(); families=Counter()
for line in (root/'context-sidecar.jsonl').open(encoding='utf-8'):
 x=json.loads(line); sidecar_ids.append(x['row_id']); families[x['family']]+=1
 def walk(v, path='context'):
  if isinstance(v,dict):
   for k,c in v.items():
    if str(k).casefold() in {'reward','score','advantage','return','target','target_text','target_body','target_tokens','target_terminal_tokens','region_new','model_target','corpus_target','teacher','generated','completion'}: forbidden.append(path+'.'+str(k))
    walk(c,path+'.'+str(k))
  elif isinstance(v,list):
   for i,c in enumerate(v): walk(c,f'{path}[{i}]')
 walk(x['context'])
 try:
  if render_prompt(PromptContext.from_mapping(x['context'])) != json.loads((root/'eligible-train-rows.jsonl').read_text(encoding='utf-8').splitlines()[len(sidecar_ids)-1])['prompt_text']:
   bad_render.append(x['row_id'])
 except Exception: bad_render.append(x['row_id'])
 geom=x['selection_geometry']; avail[geom['availability']]+=1; versions[geom['context_range']['document_version']]+=1; source_files[x['source_identity']['source_ref']['file']]+=1
print(json.dumps({
 'selected_count':len(selected),'sidecar_count':len(sidecar_ids),'ids_exact_order':selected==sidecar_ids,
 'forbidden_context_keys':len(forbidden),'bad_render':len(bad_render),'bad_shape':len(bad_shape),
 'availability':avail,'document_versions':versions,'source_files':len(source_files),
 'family_counts':families,'first_id':sidecar_ids[0],'last_id':sidecar_ids[-1],
 'artifact_meta_matches': {k: hashlib.sha256((root/v).read_bytes()).hexdigest()==meta[k]['sha256'] for k,v in [('rows','eligible-train-rows.jsonl'),('sidecar','context-sidecar.jsonl'),('selected_ids','selected-train-ids.json')]}
}, sort_keys=True))
