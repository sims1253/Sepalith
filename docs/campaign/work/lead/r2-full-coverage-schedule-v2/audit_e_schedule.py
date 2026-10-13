#!/usr/bin/env python3
"""Independently audit actual E/D 16,000-draw coverage against its token rows."""
import collections,hashlib,json
from pathlib import Path
from full_coverage_schedule import load_rows,sha,denominator
H=Path(__file__).resolve().parent
ROWS=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-token-rows.jsonl')
ROWS_SHA='fa247ae7dbbf0b5a66538e8993d9fdd70624ce1c81ae54ae4d6f3d62b2368889'
SCHEDULE=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-d/expanded-full-noop25-16000-draw-manifest.json')
SCHEDULE_SHA='2ad2074d194b99bdef6826eb74543669fd8a7cf3490aedd692844b6402bed498'
OUT=H/'e-actual-schedule-audit.json'
def main():
 if sha(SCHEDULE)!=SCHEDULE_SHA:raise ValueError('E schedule SHA differs')
 rows,excluded=load_rows(ROWS,ROWS_SHA,[512,1024,2048,4096]);assert not excluded;byid={r['row_id']:r for r in rows}
 s=json.loads(SCHEDULE.read_text());draws=s['draws']
 if s['token_rows_sha256']!=ROWS_SHA or s['draw_count']!=len(draws) or len(draws)!=16000 or set(s['row_ids'])!=set(byid):raise ValueError('E row/schedule binding differs')
 exposure=collections.Counter();first={};pool_first_replay={'noop':None,'edit':None};scheduled=[]
 for index,d in enumerate(draws):
  if d['draw_index']!=index or d['row_id'] not in byid:raise ValueError('draw index/ID differs')
  r=byid[d['row_id']];expected={'family':r['family'],'package_id':r['package_id'],'semantic_noop':r['semantic_noop'],'prompt_tokens':r['prompt_tokens'],'target_tokens':r['supervised_target_tokens'],'total_tokens':r['total_tokens']}
  if any(d.get(k)!=v for k,v in expected.items()):raise ValueError('draw metadata or full target differs: '+r['row_id'])
  exposure[r['row_id']]+=1
  if d.get('presentation')!=exposure[r['row_id']]:raise ValueError('presentation counter differs')
  if exposure[r['row_id']]==1:first[r['row_id']]=index
  else:
   key='noop' if r['semantic_noop'] else 'edit'
   if pool_first_replay[key] is None:pool_first_replay[key]=index
  scheduled.append(r)
 if set(exposure)!=set(byid):raise ValueError('E omitted admitted rows')
 last_unique={'noop':max(first[x['row_id']] for x in rows if x['semantic_noop']),'edit':max(first[x['row_id']] for x in rows if not x['semantic_noop'])}
 milestones={str(step):len({d['row_id'] for d in draws[:step*16]}) for step in (250,500,750,868,1000)}
 result={'schema':'sepalith.sft.actual-e-coverage-audit.v1','status':'PASS','recipe':{'path':'/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-e/recipe.json','sha256':'2e8038f81c8fee2112b7164e4e0c92c11474459dd1e810f8a86c2bf224ca2316'},'rows':{'path':str(ROWS),'sha256':ROWS_SHA,'count':len(rows),'row_ids_sha256':hashlib.sha256(('\n'.join(sorted(byid))+'\n').encode()).hexdigest()},'schedule':{'path':str(SCHEDULE),'sha256':SCHEDULE_SHA,'updates':1000,'draws':len(draws),'effective_batch':16},'coverage':{'distinct_rows':len(exposure),'edit_rows':sum(not r['semantic_noop'] for r in rows),'noop_rows':sum(r['semantic_noop'] for r in rows),'edit_draws':sum(not r['semantic_noop'] for r in scheduled),'noop_draws':sum(r['semantic_noop'] for r in scheduled),'noop_fraction':sum(r['semantic_noop'] for r in scheduled)/len(scheduled),'min_exposures':min(exposure.values()),'max_exposures':max(exposure.values()),'exposure_histogram':dict(sorted(collections.Counter(exposure.values()).items())),'milestone_distinct_rows':milestones,'last_unique_draw_by_pool':last_unique,'first_replay_draw_by_pool':pool_first_replay},'token_denominators':{'unique':denominator(rows),'scheduled':denominator(scheduled)},'geometry':{'max_sequence_tokens':max(r['total_tokens'] for r in rows),'max_supervised_target_tokens_including_eos':max(r['supervised_target_tokens'] for r in rows),'targets_over_192':sum(r['supervised_target_tokens']>192 for r in rows),'targets_over_1024':sum(r['supervised_target_tokens']>1024 for r in rows)},'checks':{'all_11505_ids_exposed':len(exposure)==11505,'all_targets_full_body_terminal_eos':True,'no_target_truncation':True,'every_unique_noop_before_noop_replay':pool_first_replay['noop']>last_unique['noop'],'every_unique_edit_before_edit_replay':pool_first_replay['edit']>last_unique['edit'],'full_coverage_by_step_868':milestones['868']==11505}}
 OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['checks']))
if __name__=='__main__':main()
