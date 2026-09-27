"""Stream the frozen shard; emit explicit complete pass plus named minimal replay."""
import collections,copy,hashlib,json,os,random,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'source/experiments/training'))
from campaign_cpt_data import validate_materialized_row,validate_draw_schedule

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def write(name,v):
 p=H/name;p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');return {'path':str(p),'sha256':sha(p)}
def main():
 if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 started=time.monotonic();p=H.parent/'r2-cpt-global-shard-v1/shard/cpt_train.jsonl'
 expected='aac7e1b6140b011043f25a22970c0e8c300fde65316f7e5fbcbda7d747d4f703'
 h=hashlib.sha256();rows={};packages=set();groups=set();totals=collections.Counter()
 with p.open('rb') as f:
  for line in f:
   h.update(line);r=json.loads(line);validate_materialized_row(r)
   assert r['cpt_partition']=='cpt_train' and r['row_id'] not in rows
   stats={'code_tokens':len(r['input_ids'])-2-r['overlap_context_tokens'],'loss_tokens':r['supervised_tokens'],'input_tokens':len(r['input_ids']),'terminal_document_eos':int(r['is_document_end'])}
   rows[r['row_id']]=stats;totals.update(stats);packages.add(r['package']);groups.add(r['group_id'])
 assert h.hexdigest()==expected and len(rows)==18991
 ids=list(rows);random.Random(3407).shuffle(ids);replay=ids[:(-len(ids))%16];draws=ids+replay
 schedule={'schema':'sepalith.cpt.draws.v1','split_id':'DAT-02-global-v2-285001f3d93e9f1871df','method':'one_pass_plus_named_replay_v1','seed':3407,'max_steps':len(draws)//16,'effective_batch':16,'token_rows_sha256':expected,'row_ids':draws,'replay_row_ids':replay}
 validate_draw_schedule(schedule,[{'id':x} for x in rows],token_rows_sha256=expected,max_steps=1187)
 rec=write('draws-1187.json',schedule)
 recipe=json.loads((H.parent/'lead/r2-cpt-broad-a/recipe.json').read_text());oldbase=recipe['model_path'];oldtrain=recipe['train_rows'];olddraw=recipe['draw_schedule']
 recipe.update(id='SFT-11-cpt-global-stage-candidate',model_path='/ROOT_MUST_BIND/merged-cpt-parent',output_dir='/ROOT_MUST_BIND/fresh-global-training',archive_dir='/ROOT_MUST_BIND/fresh-global-checkpoints',resume_from=None,launch_authorized=False,deadline='2026-09-14T06:15:00Z',max_attempt_seconds=5400,decision_steps=[250],checkpoint={'light_every':250,'full_every':250,'evaluation_steps':[250,750,1187]})
 recipe['parameters']['max_steps']=1187;recipe['identity']['schedule']=copy.deepcopy(recipe['parameters']);recipe['identity']['policy'].update(initialization='new_lora_on_merged_cpt_parent',status='candidate_requires_root_selection_merge_source_budget_admission')
 recipe['identity']['parent']={'revision':recipe['identity']['parent']['revision'],'weights_sha256':'ROOT_MUST_BIND','merged_cpt_manifest_sha256':'ROOT_MUST_BIND','previous_checkpoint_step':None,'previous_source_cursor':None}
 recipe['train_rows']={'path':str(p),'sha256':expected};recipe['draw_schedule']=rec
 recipe['identity']['data'].update(train_rows_sha256=expected,draw_schedule_sha256=rec['sha256'],train_package_ids=sorted(packages))
 recipe['identity']['source']='ROOT_MUST_BIND_CANDIDATE_SNAPSHOT'
 recipe['merged_cpt_parent']={'manifest':None,'previous_recipe':None,'previous_checkpoint_manifest':None,'merge_script_sha256':sha(H/'source/experiments/training/merge_cpt_cpu.py')}
 recipe['inputs']=[r for r in recipe['inputs'] if not r['path'].startswith(oldbase+'/') and r!=oldtrain and r!=olddraw]
 recipe['inputs'] += [recipe['train_rows'],rec,{'path':str(p.parent/'manifest.json'),'sha256':sha(p.parent/'manifest.json')}]
 write('recipe-template.json',recipe)
 extra=dict(sum((collections.Counter(rows[r]) for r in replay),collections.Counter()))
 exposure={'status':'CPU_prepared_not_launch_admitted','rows':len(rows),'draws':len(draws),'steps':1187,'packages':len(packages),'groups':len(groups),'one_pass':dict(totals),'replay_rows':[{'row_id':r,**rows[r]} for r in replay],'replay_totals':extra,'scheduled_totals':dict(totals+collections.Counter(extra)),'omitted_rows':0,'incomplete_effective_batch':False,'method':schedule['method'],'seed':3407,'fresh_scheduler':'cosine1187steps,warmup_ratio0.03=>36steps,peak1e-4; fresh optimizer/LoRA; not continuation of previous LR horizon','seconds':time.monotonic()-started}
 write('exposure.json',exposure);print(json.dumps({k:v for k,v in exposure.items() if k!='replay_rows'}))
if __name__=='__main__':main()
