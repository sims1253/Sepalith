"""TRAIN metadata exposure audit and novel-source roster; no raw-source payload reads."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,os,sys
EXEC=Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
sys.path[:0]=[str(EXEC/'experiments/training'),str(EXEC/'packages/sepalith/src')]
from campaign_completion_batch import _is_eligible_audit_row
HERE=Path(__file__).resolve().parent
PLAN=HERE.parents[1]
DATA=Path('/mnt/e/sepalith/campaign-20260915/data-work')
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def summary(rows):
 return {'rows':len(rows),'unique_rows':len({x['row_id'] for x in rows}),'families':dict(Counter(x['family'] for x in rows)),'packages':len({x['package_id'] for x in rows}),'prompt_tokens':sum(x['prompt_tokens'] for x in rows),'target_tokens_including_eos':sum(x['target_tokens'] for x in rows),'total_tokens':sum(x['total_tokens'] for x in rows)}
def main():
 schedule=DATA/'SFT-inputs-v1/draws-3000.json';metadata=DATA/'SFT-inputs-v1/sampler-metadata.json';corrected=PLAN/'work/finish-corrected-schedule-v1/finish-corrected-sampler-metadata.json';audit=DATA/'DAT-03-row-audit.jsonl'
 expected={schedule:'a439fe8af68b4929a7747d4a82b9e0b26a05fcd3973f8a3c81148a84581d7f94',metadata:'56ab7f6b8bb4dd01a83292686ca2be6639581dbc49d521b79bbf499e9f037877',corrected:'a18b608cf548913bf55d233c649d7086ba6eb844ef6a40db357a57f4e0318d68'}
 for p,h in expected.items():assert sha(p)==h,str(p)
 s=load(schedule);m=load(metadata);c=load(corrected)['records'];draws=s['draws'][:16000];seen={x['row_id'] for x in draws};all_admitted={x['row_id'] for x in m}
 original_unused=[x for x in m if x['row_id'] not in seen];corrected_unused=[x for x in c if x['row_id'] not in seen]
 exposure={'selected1000_prefix':summary(draws),'available_original':summary(m),'unseen_original':summary(original_unused),'available_corrected':summary(c),'unseen_corrected':summary(corrected_unused),'seen_repeat_histogram':dict(Counter(Counter(x['row_id'] for x in draws).values())),'actual_sampler_consumed_draws_receipt':str(PLAN/'receipts/SFT-10-step1000-review.json')}
 assert load(PLAN/'receipts/SFT-10-step1000-review.json')['sampler_consumed_draws']==16000
 (HERE/'exposure.json').write_text(json.dumps(exposure,indent=2)+'\n')
 (HERE/'unseen-admitted-corrected-metadata.json').write_text(json.dumps({'status':'289_unseen_rows_same_family_not_broader_corpus','source':str(corrected),'source_sha256':expected[corrected],'rows':corrected_unused},indent=2)+'\n')
 attempted=set(all_admitted)
 with (DATA/'DAT-04-structured-batch-broad-v2/selected-audit-rows.jsonl').open() as f:
  for line in f:attempted.add(json.loads(line)['row_id'])
 with (DATA/'DAT-04B-completion-batch.jsonl').open() as f:
  for line in f:attempted.add(json.loads(line)['row_ref']['row_id'])
 supported={'rename_propagation','pipe_rewrite','na_rm_propagation','format_propagation','no_op','roxygen_drafting'}
 limits={family:1024 for family in supported};limits['finish_block']=4096
 seen_groups=set();counts=Counter();unused=Counter();eligible=Counter();selected_counts=Counter();groups=Counter();selected=[];by_source=Counter();digest=hashlib.sha256();total=0
 with audit.open('rb',buffering=4*1024*1024) as f:
  for line in f:
   total+=1;digest.update(line);row=json.loads(line)
   if row['split']!='train_group':continue
   if row['row_id'] in seen:seen_groups.add(row['group_id'])
   family=row['family'];counts[family]+=1
   if row['row_id'] in attempted:continue
   unused[family]+=1
   canonical='finish_block' if family.startswith('finish_block') else family
   if canonical not in limits:continue
   # Match accepted builders: DAT03 legacy boundary flags were superseded by fresh source reconstruction.
   if canonical=='finish_block' and not _is_eligible_audit_row(row):continue
   eligible[canonical]+=1
   cap=8 if canonical=='finish_block' else 4
   key=(canonical,row['group_id'])
   if selected_counts[canonical]>=limits[canonical] or groups[key]>=cap:continue
   groups[key]+=1;selected_counts[canonical]+=1;by_source[row['file']]+=1
   selected.append({**row,'r2_family':canonical,'r2_admission':'metadata_candidate_only; source validators, full targets, tokenization and registry required'})
 assert digest.hexdigest()=='9878a70d5d822f1192ccfc2956346738d668dc38175871556d1af1e16dbccadd'
 assert all(x['split']=='train_group' and x['row_id'] not in attempted for x in selected)
 with (HERE/'novel-train-audit-roster.jsonl').open('w') as f:
  for row in selected:f.write(json.dumps(row,separators=(',',':'))+'\n')
 result={'status':'novel_TRAIN_metadata_roster_not_training_admission','audit':str(audit),'audit_sha256':digest.hexdigest(),'audit_rows':total,'split_id':s['split_id'],'all_candidate_train_counts':dict(counts),'not_previously_attempted_counts':dict(unused),'existing_builder_candidate_counts':dict(eligible),'selected':dict(selected_counts),'selected_total':len(selected),'selected_groups':len({x['group_id'] for x in selected}),'groups_not_exposed_in_selected1000':len({x['group_id'] for x in selected}-seen_groups),'selected_source_files':dict(by_source),'token_counts':'unknown until full PRM03 target tokenization; no estimates represented as measured','train_draw_schedule_created':False,'reason':'Existing admitted pool has only 289 unseen finish rows; novel roster is not admitted and cannot enter sampler yet.'}
 (HERE/'novel-roster-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'exposure':exposure,'novel_summary':result}))
if __name__=='__main__':main()
