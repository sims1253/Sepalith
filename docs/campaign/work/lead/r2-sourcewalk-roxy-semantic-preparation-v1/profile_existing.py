#!/usr/bin/env python3
"""Profile frozen shard metadata only; does not open candidate/token payloads."""
import collections, glob, hashlib, json
from pathlib import Path

BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1')
OUT=Path(__file__).with_name('metadata-profile.json')
rows=[]; input_family=collections.Counter(); converted_family=collections.Counter(); excluded=collections.Counter()
for name in sorted(glob.glob(str(BASE/'shard-*'/'structured-materialization-v1'/'summary.json'))):
 p=Path(name); o=json.loads(p.read_text())
 sf={}
 for x in o['source_files']:
  family='no_op' if x['path'].endswith('/no_op.jsonl') else 'roxygen_drafting'
  input_family[family]+=x['selected_records']; sf[family]={k:x[k] for k in ('path','sha256','bytes','selected_records')}
 converted_family.update(o['converted_families']); excluded.update(o['exclusion_reasons'])
 rows.append({'shard':o['shard'],'status':o['status'],'input_rows':o['input_rows'],'converted':o['converted'],'excluded':o['excluded'],'source_files':sf,'summary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'candidate_packets':o['candidate_packets'],'exclusions':o['exclusions'],'repair_queue':o['repair_queue']})
out={'schema':'sepalith.dat10.sourcewalk_semantic_metadata_profile.v1','payload_files_opened':False,'shards':len(rows),'input_rows':sum(x['input_rows'] for x in rows),'input_by_family':dict(input_family),'converted_rows':sum(x['converted'] for x in rows),'converted_by_family':dict(converted_family),'excluded_rows':sum(x['excluded'] for x in rows),'exclusion_reasons':dict(excluded),'all_status_source_checked_not_admitted':all(x['status']=='source_checked_candidates_not_training_registry' for x in rows),'shard_profiles':rows,'need_for_bulk_semantic_replay':'producer status explicitly withholds training registry admission; converted roxygen outside reviewed10017 lacks independent full-source semantic evidence, and all4227 noops lack independent provenance replay'}
OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:out[k] for k in ('shards','input_rows','converted_rows','excluded_rows','input_by_family','converted_by_family')},sort_keys=True))
