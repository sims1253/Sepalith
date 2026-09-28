#!/usr/bin/env python3
"""Stream only corrected TRAIN row metadata; no tokenizer or model imports."""
import hashlib,json,os
from collections import Counter
from pathlib import Path
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
p=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
train=p/'work/lead/finish-corrected-train-v1/train-token-rows.jsonl';dev=p/'work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl'
devrows=[json.loads(s) for s in dev.read_text().splitlines()];dev_ids={r['id'] for r in devrows};dev_packages={r['package_id'] for r in devrows}
families=Counter();stats=Counter();ids=set();packages=set();digest=hashlib.sha256()
with train.open('rb') as f:
 for line in f:
  digest.update(line);r=json.loads(line);assert r['split']=='train' and r['id'] not in ids;ids.add(r['id']);packages.add(r['package_id']);families[r['family']]+=1
  body=r['target_body_token_count'];terminal=r['target_terminal_token_count'];target=body+terminal+1;prompt=r['prompt_token_count']+1
  assert r['target_token_count']==body+terminal;assert r['input_ids'][-1]==1;assert len(r['input_ids'])==prompt+target
  stats['rows']+=1;stats['canonical_target_tokens_including_EOS']+=target;stats['prompts_including_BOS_over_2048']+=prompt>2048;stats['targets_including_EOS_over_192']+=target>192
  stats['max_prompt_including_BOS']=max(stats['max_prompt_including_BOS'],prompt);stats['max_target_including_EOS']=max(stats['max_target_including_EOS'],target)
  if r['target_operation']=='no_op':stats['noop_rows']+=1;stats['noop_target_tokens_including_EOS']+=target
assert len(ids)==11526 and not(ids&dev_ids) and not(packages&dev_packages)
report={'status':'passed','train_path':str(train),'train_sha256':digest.hexdigest(),'train_bytes':train.stat().st_size,'family_counts':dict(families),'counts':dict(stats),'noop_row_share':stats['noop_rows']/stats['rows'],'noop_target_token_share':stats['noop_target_tokens_including_EOS']/stats['canonical_target_tokens_including_EOS'],'DEV_id_overlap':0,'DEV_package_overlap':0,'group_overlap_status':'not in token-row schema; prior corrected-rl data audit separately verifies0','data_written':False,'limits':'Reads existing TRAIN only; no new target/context derivation or model evaluation.'}
(Path(__file__).parent/'train-metadata-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
