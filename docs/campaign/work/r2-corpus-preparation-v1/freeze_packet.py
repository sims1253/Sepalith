"""Pin completed preparation artifacts and write the owned return receipt."""
import datetime,json
from pathlib import Path
import raw_cpt_v2 as c
HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parent.parent

def main():
    sources={p.name:{'sha256':c.sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.md')}
    source_manifest=HERE/'source-manifest.json'
    source_manifest.write_text(json.dumps({'schema':1,'files':sources},indent=2)+'\n')
    excluded={'artifact-manifest.json'}
    files={str(p.relative_to(HERE)):{'sha256':c.sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.rglob('*'))
           if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts}
    artifact_manifest=HERE/'artifact-manifest.json'
    artifact_manifest.write_text(json.dumps({'schema':1,'scope':'Completed preparation artifacts; initial aborted inventory files are provenance only, not an admitted corpus.', 'files':files},indent=2)+'\n')
    profile=json.loads((HERE/'profile-shard-v2-2k/manifest.json').read_text())
    broader=json.loads((HERE/'broader-stage-manifest.json').read_text())
    previous=json.loads((HERE/'sft-options-intermediate-receipt.json').read_text())
    result={
        'historical_raw_R_CPT_performed':False,
        'initialization':{'proposal':'fresh Midtrain; root owns real weight binding', 'revision':'8dc5f6055b90fe4b9422340810b270b9569f37f3', 'historical_parent_weight_sha256_from_receipts':'38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad'},
        'protected_SFT1000':{'path':'/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/SFT-primary-3000-a/full/checkpoint-1000','manifest_sha256':'230abf65a8d168bde4a66dbbabb67223f3c0631e3db933b031f63d3dadc971d3','worker_weight_access':False},
        'historical_training':'Full-text PRM03 task SFT; prompt token totals include renderer text and repeats, not distinct raw-R tokens.',
        'sft_exposure_artifact':str(HERE/'exposure.json'),
        'sft_expansion_candidates':6690,'sft_expansion_groups':2311,'sft_expansion_new_groups_vs_SFT1000':1239,
        'sft_expansion_manifest_sha256':c.sha(HERE/'sft-expansion-manifest.json'),
        'raw_normalized_root':'/mnt/h/sepalith/normalized',
        'global_split_path':'/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json',
        'global_split_sha256':'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09',
        'mapped_raw_TRAIN_packages':10719,'CPT_train_groups':10163,'CPT_validation_groups':556,
        'partition_path':str(HERE/'cpt-train-group-partition.json'),'partition_sha256':c.sha(HERE/'cpt-train-group-partition.json'),
        'profile_2k_counts':profile['counts'],'profile_manifest_sha256':c.sha(HERE/'profile-shard-v2-2k/manifest.json'),
        'profile_duplicate_exclusions':{'within_CPT_validation':1,'TRAIN_duplicate_of_CPT_validation':3},
        'broader':{k:v for k,v in broader.items() if k!='group_code_token_counts'},
        'label_policy':'BOS + payload + EOS on every chunk; mask BOS, prior-code-token overlap and nonterminal EOS; supervise each new code token once and actual final EOS once/document. Pad labels by attention positions, not EOS/PAD token ID.',
        'source_manifest_sha256':c.sha(source_manifest),'artifact_manifest_sha256':c.sha(artifact_manifest),
        'source_files':sources,
        'validation':{'focused_tests':'7 groups PASS in0.361s','negative_validator_tests':'1 unchanged positive and5 corruptions PASS in2.061s',
            'profile_all_rows':json.loads((HERE/'profile-v2-independent-validation.json').read_text()),
            'broader_all_rows':json.loads((HERE/'broader-independent-validation.json').read_text())},
        'raw_payload_stage_seconds':599.6164232849842,
        'CPT_validation_for_broader_training':str(HERE/'profile-shard-v2-2k/cpt_validation.jsonl'),
        'broader_training_rows':str(HERE/'broader-shard-v1-2k/cpt_train.jsonl'),
        'broader_one_pass_schedule':str(HERE/'broader-draws-one-pass.json'),
        'CPU_processes_terminal':True,'model_weights_read_or_hashed':False,'training_launched':False,
        'final_content_accessed':False,'accepted_inputs_modified':False,
    }
    receipt={'task':'SFT-R2-DATA / DAT10 raw-R CPT preparation','status':'verified','owner':'freeze_binding',
        'started':'2026-09-13T17:58:07Z','ended':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dependencies_checked':previous.get('dependencies_checked',[]),
        'action':'Audited selected SFT exposure; froze6,690new task-SFT metadata candidates; built and independently validated TRAIN-only raw-R 2K profile and38.9M-token broader corpus with a separate TRAIN-group validation partition.',
        'commands_or_method':['prepare_options.py metadata audit; materialize_candidates.py prepared but root owns execution.',
            'inventory_raw_train_v2.py: two CPU affinity cores,eight I/O workers; bounded partial TRAIN-only enumeration, stopped after enough stage metadata.',
            'raw_cpt.py --output profile-shard-v1; derive_profile_v2.py',
            'raw_cpt_broader.py --metadata broader-raw-file-snapshot.jsonl --output broader-shard-v1-2k --train-bytes 125829120 --validation-bytes 0 --reserve-documents profile-shard-v1/documents.jsonl --package-code-token-cap 400000',
            'finalize_broader.py; validate_cpt_shard.py against both closed shards; test_raw_cpt_v2.py; test_cpt_validation_negative.py'],
        'result':result,
        'acceptance':'pass — bounded CPU preparation and exact token/source/label checks. This receipt does not admit a GPU run or claim scientific improvement.',
        'changed_files':[str(HERE/path) for path in files]+[str(artifact_manifest),str(CAMPAIGN/'receipts/SFT-R2-DATA-corpus-options.json')],
        'artifacts':[str(source_manifest),str(artifact_manifest),str(HERE/'CPT-PLAN.md')],
        'unresolved':['Root must review actual trainer/collator/fused-loss masks and denominator before update1, then admit fixed data and model identities.',
            'The full10,719-package raw corpus was not read/tokenized. The fixed broader shard is bounded and comes from a partial metadata inventory.',
            'No complete held-out source-file hash inventory or cross-package near-duplicate proof; campaign non-TRAIN groups are excluded before source traversal and available parent hashes are guarded.',
            'Raw source files were UTF-8/tokenizer validated, not R-executed or universally R-parse validated. License filtering conservatively excludes unknown/custom expressions.',
            'Changing from the profiling dataset to the larger dataset requires an explicit reviewed sampler/data transition; the old cursor must not be relabeled as continuous.',
            'Later SFT should retain CPT validation-group exclusion; authored/non-normalized SFT identities need their DAT02 join.',
            'Fresh Midtrain→CPT→corrected broader target-only SFT→conditional RL remains root-owned. The prior25-step corrective result is not global evidence against this branch.'],
        'next':'Root admit the2Kprofile runtime gate, choose fixed CPT token/step budget from measured throughput, then select the broader manifest and draw schedule within the initial four-hour CPT ceiling.',
        'lease_released':'yes — all worker metadata/materialization/validation CPU processes terminal; no CUDA, server, network or model launch.'}
    receipt_path=CAMPAIGN/'receipts/SFT-R2-DATA-corpus-options.json'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'receipt':str(receipt_path),'receipt_sha256':c.sha(receipt_path),
        'source_manifest_sha256':c.sha(source_manifest),'artifact_manifest_sha256':c.sha(artifact_manifest),
        'broader_stage_manifest_sha256':c.sha(HERE/'broader-stage-manifest.json'),'files':len(files)}))

if __name__=='__main__':main()
