#!/usr/bin/env python3
import hashlib,json,math,os
PLAN='/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb'
E='/mnt/e/sepalith/campaign-20260915/data-work/Noop-pool-coverage-review-v1'

def sha(p):
 h=hashlib.sha256();n=0
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b);n+=len(b)
 return {'path':p,'bytes':n,'sha256':h.hexdigest()}
def proj(add_edits,add_noops=0):
 n=1094+add_noops;e=13912+add_edits;t=n+e
 batches=math.ceil(e/12)
 return {'unique_rows':t,'unique_noops':n,'unique_edits':e,'unique_noop_share':n/t,'minimum_4_noop_12_edit_batches_for_all_edits':batches,'scheduled_draws':16*batches,'scheduled_noop_draws':4*batches,'noop_replay_draws_after_unique_coverage':4*batches-n,'edit_alignment_replays':12*batches-e,'mean_noop_exposures':4*batches/n,'unique_noops_needed_for_5pct':max(0,math.ceil(.05*t)-n)}
current=json.load(open(E+'/audit-census.json'));cand=json.load(open(E+'/supported-noop-summary.json'));alt=json.load(open(E+'/alternate-kind-noop-summary.json'));allsrc=json.load(open(E+'/source-walk-all-family-summary.json'));div=json.load(open(E+'/current-diversity.json'))
review={
 'schema':'sepalith.dat10.noop-pool-coverage-review.v1','status':'independent_review_complete_no_training_admission','created_at':'2026-09-14T17:55:00Z',
 'scope':{'train_only':True,'dev_opened':False,'final_opened':False,'gpu_or_model_load':False,'cloud':False,'cpu_threads_max':2,'bulk_outputs':E},
 'finding':{'current_unique_noops':1094,'current_unique_rows':15006,'current_unique_share':1094/15006,'pool_exhausted':False,'strongest_unused_source_supported_pool':{'rows':4227,'structurally_unique_contexts':4220,'exact_context_duplicate_extras':7,'direct_license_and_source_validation_rows':4227,'training_admitted':0},'conclusion':'Resolve and deduplicate the existing source-supported no-op pool before deriving synthetic post-edit no-ops.'},
 'current_diversity':div['current_noop_diversity'],
 'inventory_reconciliation':{
  'all_metadata_noops':11219,'heldout_dev_rows':8,'heldout_cpt_validation_rows':534,'train_cpt_rows':10677,
  'current_inventory_overlap':1089,'current_supplemental_rows_outside_inventory':5,
  'old_gate_materialized_rows':550,'old_gate_existing_prompt_duplicate_rows':1,'old_gate_source_license_support_pending_rows':550,
  'early_alternate_kind_support_queue_rows':425,
  'later_source_walk_converted_rows':4227,'later_source_walk_context_duplicate_extras':7,
  'later_alternate_kind_support_queue_rows':4384,
  'unresolved_metadata_only_rows':[
   {'row_id':'3ad92a51cf87976341746d08','status':'source_checked','reason':'not found in current/gate/early-support/later-converted/later-support sets'},
   {'row_id':'6243ac16509dc953c518db36','status':'unattempted_source_walk','reason':'not found in current/gate/early-support/later-converted/later-support sets'}],
  'reconciles_inventory_train_cpt_denominator':1089+550+425+4227+4384+2==10677,
  'inventory_train_cpt_plus_five_supplemental_current_rows':10677+5,
  'note':'The 5 supplemental current rows are outside the 11,219-row metadata inventory, so 10,677 inventory TRAIN/CPT rows reconcile as 1,089+550+425+4,227+4,384+2.'
 },
 'unused_supported_diversity':{'rows':4227,'unique_contexts':4220,'unique_packages':2195,'unique_groups':2195,'unique_raw_source_lines':4227,'normalized_source_files':4141,'normalized_source_hashes':4139,'region_old_zero_width_rows':1650,'region_old_one_line_rows':2577,'history_events_zero_rows':4227,'current_package_overlap':701,'new_packages_over_current':1494,'union_packages_if_admitted':2404,'all_current15006_id_overlap':0,'roxy10017_id_overlap':0,'roxy_safe8597_id_overlap':0,'validation_failures':0},
 'queues':{
  'structural_prompt_duplicate_extras':{'rows':7,'action':'deterministically retain one only after exact renderer confirms equal prompt and target'},
  'source_geometry_duplicate_extras':{'rows':2,'action':'inspect with the structural duplicate groups; do not count as diversity'},
  'old_gate':{'rows':550,'potential_exact_prompt_unique_rows':549,'action':'source/license replay and current-union exact prompt/target/source dedup required'},
  'early_alternate_kind':{'rows':425,'action':'semantic source-kind support required'},
  'later_alternate_kind':{'rows':4384,'action':'semantic source-kind support required; not a length exclusion or automatic invalid'},
  'unresolved_metadata_only':{'rows':2,'action':'named source replay before classification'},
 },
 'all_source_walk':allsrc['counts'],
 'length_evidence':{'old_gate_noops':current['candidate_gate_noops']['sequence_length_buckets'],'early_support_prompt_chars':current['support_queue_noops']['prompt_char_buckets'],'all_source_walk_token_audit':allsrc['token_audit_all_families'],'interpretation':'No converted source-walk row exceeded 4096 sequence tokens and tokenizer excluded zero rows. The 4384 no-op support exclusions are semantic-kind gates. The old 573-row gate no-op set is <=1024 sequence tokens. No no-op was found excluded by an old length cap.'},
 'projections':{'add_root_approx_8530_edits':proj(8530),'add_roxy_safe_review_union_8597_edits':proj(8597),'add_all_roxy10017_review_rows':proj(10017),'add_roxy8597_and_structurally_unique_noops4220':proj(8597,4220)},
 'sampling_recommendation':{
  'preserve_all_eligible_unique_rows':True,'noop_25pct_status':'existing candidate schedule, not established optimum',
  'recommendation':'First admit only rows passing exact renderer/token/source/license/dedup gates. Then compare deterministic coverage-first schedules at 15%, 20%, and 25% no-op draw share using TRAIN-only signal checks. Report unique and replay exposures separately, exhaust each unique no-op before replay, stratify by source group and geometry, and cap per-row exposure imbalance.',
  'why':'With 8,597 added edit rows and no new no-ops, a 25% 4:12 schedule needs 7,504 no-op draws from 1,094 unique rows (6,410 replays; mean 6.86 exposures). If 4,220 structurally unique candidates pass admission, the same schedule needs 2,190 no-op replays (mean 1.41 exposures).'
 },
 'admission_gates':[
  'Re-bind every candidate to exact DAT-02 TRAIN group and CPT-train partition; keep DEV/CPT-validation/final out.',
  'Replay the pinned prompt renderer and tokenizer; assert exact [NO_EDIT] body, protocol terminator, EOS, and token geometry.',
  'Deduplicate exact prompt, prompt-target, raw line, source identity, and normalized source geometry against current15006 and the selected roxygen union.',
  'Retain one deterministic representative from each of the seven identical structural-context pairs pending exact rendered-prompt confirmation.',
  'Keep 4,384 later and 425 early alternate-kind rows in named support queues until semantics are proven from source; absence of a length failure is not admission.',
  'Root must review before any training or schedule change.'
 ],
 'derivation_contingency':{'needed_now':False,'policy_if_verified_pool_later_exhausted':'Only derive a follow-up no-op from a TRAIN edit whose exact observed history can be replayed to a pinned post-edit source and whose family predicate mechanically proves the requested transformation is already satisfied. Preserve original source/history/cursor/range hashes; require unchanged applied buffer, exact [NO_EDIT] protocol, parser validity where available, and dedup. Limit to mechanically checkable structured families; do not derive finish_block or roxygen no-ops from subjective correctness. No derived row is admitted by this review.'},
 'artifacts':{}
}
for name in ['audit-census.json','current-diversity.json','supported-noop-summary.json','supported-noop-candidates-metadata.jsonl','alternate-kind-noop-summary.json','alternate-kind-noop-support-queue.jsonl','source-walk-all-family-summary.json','audit.time','candidate-scan-v2.time']:
 review['artifacts'][name]=sha(E+'/'+name)
out=PLAN+'/docs/campaign/work/lead/r2-noop-pool-coverage-review-v1/review.json';open(out,'w').write(json.dumps(review,indent=2,sort_keys=True)+'\n')
print(json.dumps(review['projections'],indent=2))
