"""Metadata-only CPT transition gate; model bytes are checked only at admitted launch."""
import hashlib
import json
from pathlib import Path
from campaign_checkpoint import digest

POLICY = 'new_lora_on_merged_cpt_parent'
MIDTRAIN_SOURCE = '5149a4065285c681c2230352e5847b861e2ad2e416c8c263131a557e48bd23ca'
GLOBAL_CPT_SOURCE = 'aaacb632811c5a4639fefe52023e72b72e2b960523dbffe10d0187fd6cef8da8'
TOKENIZER_SHA = '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
TOKENIZER_CONFIG_SHA = 'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b'

def require(ok, why):
    if not ok: raise ValueError('Merged CPT parent: ' + why)

def record(value):
    require(isinstance(value, dict) and set(value)=={'path','sha256'}, 'record schema')
    p=Path(value['path'])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(), 'regular absolute metadata input')
    require(digest(p)==value['sha256'], 'metadata hash mismatch')
    return p

def check_merged_cpt_parent(recipe, *, verify_model_files=False):
    identity=recipe['identity']; binding=recipe.get('merged_cpt_parent')
    require(isinstance(binding,dict), 'binding is absent')
    manifest=json.loads(record(binding['manifest']).read_text())
    previous=json.loads(record(binding['previous_recipe']).read_text())
    checkpoint=json.loads(record(binding['previous_checkpoint_manifest']).read_text())
    require(manifest.get('schema_version')=='sepalith.merged-cpt.parent-manifest.v1' and manifest.get('kind')=='merged_cpt', 'manifest kind')
    require(previous['stage']=='cpt_raw_r_v1' and previous['identity']['source'] in
            (MIDTRAIN_SOURCE, GLOBAL_CPT_SOURCE), 'prior stage/source')
    require(previous['identity']['policy']['initialization'] in
            ('new_lora_on_midtrain', 'new_lora_on_merged_cpt_parent'),
            'prior initialization')
    require(checkpoint.get('full') is True and type(checkpoint.get('step')) is int and checkpoint['step']>0, 'prior full checkpoint')
    require(checkpoint['identity']==previous['identity']==manifest['cpt_identity'], 'prior identity')
    require(manifest['cpt_checkpoint_manifest']==checkpoint, 'embedded checkpoint differs')
    require(manifest['checkpoint_manifest_sha256']==binding['previous_checkpoint_manifest']['sha256'], 'checkpoint file pin')
    require(manifest['recipe_sha256']==binding['previous_recipe']['sha256'] and manifest['recipe_path']==binding['previous_recipe']['path'], 'previous recipe pin')
    require(str(Path(manifest['cpt_checkpoint'])/'campaign-manifest.json')==binding['previous_checkpoint_manifest']['path'], 'checkpoint path')
    require(recipe.get('resume_from') != manifest['cpt_checkpoint'], 'prior-stage optimizer resume is forbidden')
    # Other resume paths still pass the unchanged full checkpoint/exact identity gate in preflight.
    require(manifest['source_cursor']==checkpoint['step']*16, 'prior source cursor')
    sampler=manifest['previous_sampler']
    require(sampler['consumed_draws']==manifest['source_cursor'] and sampler['schedule_sha256']==previous['draw_schedule']['sha256'] and sampler['split_id']==previous['split_id'], 'previous sampler lineage')
    merge_script = record(binding['merge_script'])
    require(manifest['source_script_sha256']==binding['merge_script']['sha256'], 'merge source pin')
    require(manifest.get('tokenizer_original_bytes_restored') is True, 'tokenizer byte preservation')
    verify=manifest['merge_verification']
    require(verify['device']=='cpu' and verify['dtype']=='bfloat16' and verify['adapter_tensors_exact']==588 and verify['independent_lora_matrix_merges_exact']==294 and verify['cuda_started'] is False, 'merge verification')
    require(verify.get('accumulation_dtype')=='float32' and verify.get('rounding')=='one final BF16 cast', 'merge rounding policy')
    inventory=manifest['weight_inventory']
    require(len(inventory)==1 and inventory[0]['path']=='model.safetensors' and inventory[0]['bytes']>0, 'one merged weight file')
    canonical=json.dumps(inventory,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    require(hashlib.sha256(canonical).hexdigest()==manifest['weight_inventory_sha256'], 'weight inventory digest')
    require(identity['parent']['weights_sha256']==manifest['merged_weights_sha256']==inventory[0]['sha256'], 'merged weights identity')
    require(identity['parent']['revision']==manifest['base_model_revision']==previous['identity']['parent']['revision'], 'base revision')
    require(identity['parent']['merged_cpt_manifest_sha256']==binding['manifest']['sha256'], 'parent lineage binding')
    require(identity['parent']['previous_checkpoint_step']==checkpoint['step'], 'selected checkpoint step')
    require(identity['parent']['previous_source_cursor']==manifest['source_cursor'], 'selected cursor')
    require(manifest['tokenizer']['tokenizer_json_sha256']==TOKENIZER_SHA and manifest['tokenizer']['tokenizer_config_sha256']==TOKENIZER_CONFIG_SHA, 'original tokenizer pins')
    root=Path(recipe['model_path']); require(root.is_absolute() and str(root)==manifest['merged_model_path'], 'merged model path')
    expected={'model.safetensors':manifest['merged_weights_sha256'], 'config.json':manifest['config_sha256'], 'generation_config.json':manifest['generation_config_sha256'], 'tokenizer.json':TOKENIZER_SHA,'tokenizer_config.json':TOKENIZER_CONFIG_SHA}
    inputs={r['path']:r['sha256'] for r in recipe['inputs']}
    for name,sha in expected.items():
        require(inputs.get(str(root/name))==sha, 'missing merged file pin '+name)
        require(isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha), 'invalid SHA256')
        if verify_model_files:
            require((root/name).is_file() and not (root/name).is_symlink() and digest(root/name)==sha, 'merged file bytes '+name)
    require(not verify_model_files or not (root/'adapter_config.json').exists(), 'merged parent contains adapter config')
    return {'kind':'merged_cpt','previous_step':checkpoint['step'],'previous_source_cursor':manifest['source_cursor'],'optimizer':'fresh','lora':'fresh','model_bytes_verified':verify_model_files}
