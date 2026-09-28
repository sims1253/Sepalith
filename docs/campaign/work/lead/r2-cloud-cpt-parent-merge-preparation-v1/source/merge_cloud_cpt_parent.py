#!/usr/bin/env python3
"""Merge a root-selected cloud CPT LoRA checkpoint into its exact native parent."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

NATIVE_TOKENIZER_SHA = '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
MUTATED_CLOUD_TOKENIZER_SHA = 'd5ede0bcd21e0676a58b176937262fb80a06a32b5cb4ed8bfed8b7d11e45b0e1'
MODEL_ELEMENTS = 2_516_756_480
ADAPTER_TENSORS = 588
MERGED_MATRICES = 294

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def tensor_digest(t, torch) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def merge_matrix_once(base, left, right, scale, torch):
    """FP32 LoRA addition followed by one cast to the original base dtype."""
    assert left.ndim == right.ndim == base.ndim == 2
    assert left.shape[1] == right.shape[0]
    assert left.shape[0] == base.shape[0] and right.shape[1] == base.shape[1]
    return (base.float() + (left.float() @ right.float()) * float(scale)).to(base.dtype)

def fs_type(path: Path) -> str:
    current=path
    while not current.exists(): current=current.parent
    return subprocess.check_output(['df','-T','-P',str(current)],text=True).splitlines()[-1].split()[1]

def require_hash(path: Path, expected: str):
    got=digest(path); assert got == expected, f'hash mismatch: {path}: {got}'

def verify_binding(binding_path: Path):
    b=json.loads(binding_path.read_text())
    assert b['schema']=='sepalith.cloud-cpt-parent-merge.binding.v1'
    assert b['status']=='root_admitted_candidate_merge'
    assert b['selected_checkpoint_step'] in (1585,1902)
    assert b['candidate']['status']=='verified_readback'
    assert b['candidate']['step']==b['selected_checkpoint_step']
    assert b['candidate']['full'] is True
    assert b['candidate']['source_identity']=='7e4e80059416b3783c783a532d95b40af6067076e7a0a23d64e3b571f478c9df'
    assert b['candidate']['draw_schedule_sha256']=='914cf353199f45697c028da3a6f06b732fd57bd265869890b75bf43151851b58'
    assert b['candidate']['train_rows_sha256']=='836b61c5b7d5545063aeaa3054c62915b8a4d24574908cade4c2d3eed1537a87'
    assert b['parent']['weights_sha256']=='5f12810692a90ebb32f589a25e5adcedac53d9fb6edd9166b8245fdec6e3dd1c'
    assert b['parent']['manifest_sha256']=='c5ddf086518e66a6709d8e4a16b88f7399d4e87ea477e24f3988ffcdecb1366e'
    assert b['tokenizer']['native_tokenizer_json_sha256']==NATIVE_TOKENIZER_SHA
    assert b['tokenizer']['cloud_checkpoint_tokenizer_json_sha256']==MUTATED_CLOUD_TOKENIZER_SHA
    assert b['tokenizer']['native_eog_ids']==[1,130073]
    assert (b['tokenizer']['bos_id'],b['tokenizer']['eos_id'],b['tokenizer']['pad_id'])==(0,1,1)
    assert b['merge']['device']=='cpu' and b['merge']['threads']==2
    assert b['merge']['method']=='streaming_per_matrix_fp32_add_one_bf16_cast_then_unload'
    assert digest(Path(__file__))==b['merge_source_sha256']
    admission=Path(b['root_admission']['path'])
    require_hash(admission,b['root_admission']['sha256'])
    a=json.loads(admission.read_text())
    assert a['status']=='admitted' and a['action']=='merge_cloud_cpt_parent_candidate'
    assert a['selected_checkpoint_step']==b['selected_checkpoint_step']
    assert a['binding_review_sha256']==b['binding_review_sha256']
    return b

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--binding',type=Path,required=True)
    ns=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only merge requires CUDA_VISIBLE_DEVICES empty'
    b=verify_binding(ns.binding.resolve())
    checkpoint=Path(b['candidate']['checkpoint_path']); base=Path(b['parent']['path']); output=Path(b['output'])
    readback=Path(b['candidate']['verified_readback_receipt']['path']); require_hash(readback,b['candidate']['verified_readback_receipt']['sha256'])
    rr=json.loads(readback.read_text()); assert rr['acceptance']['status']=='pass' and Path(rr['artifacts']['extracted_directory'])==checkpoint
    assert checkpoint.is_absolute() and base.is_absolute() and output.is_absolute()
    assert output.parent==Path('/mnt/e/sepalith/campaign-20260915/models')
    assert not output.exists(); temporary=output.with_name('.'+output.name+'.preparing'); assert not temporary.exists()
    assert fs_type(output) in ('9p','v9fs'), fs_type(output)
    # Verify all checkpoint payload bytes from the immutable manifest before importing ML libraries.
    manifest_path=checkpoint/'campaign-manifest.json'; require_hash(manifest_path,b['candidate']['campaign_manifest_sha256'])
    manifest=json.loads(manifest_path.read_text())
    assert manifest['full'] is True and manifest['step']==b['selected_checkpoint_step']
    assert manifest['identity']['source']==b['candidate']['source_identity']
    for name, rec in manifest['files'].items():
        p=checkpoint/name; assert p.is_file() and p.stat().st_size==rec['bytes']; require_hash(p,rec['sha256'])
    require_hash(checkpoint/'adapter_config.json',b['candidate']['adapter_config_sha256'])
    require_hash(checkpoint/'adapter_model.safetensors',b['candidate']['adapter_weights_sha256'])
    require_hash(checkpoint/'tokenizer.json',MUTATED_CLOUD_TOKENIZER_SHA)
    # The cloud tokenizer is evidence only. It is never loaded or copied.
    parent_manifest=base/'parent-manifest.preparation.json'; require_hash(parent_manifest,b['parent']['manifest_sha256'])
    for name, rec in b['parent']['files'].items():
        p=base/name; assert p.is_file() and p.stat().st_size==rec['bytes']; require_hash(p,rec['sha256'])
    cfg=json.loads((checkpoint/'adapter_config.json').read_text())
    assert Path(cfg['base_model_name_or_path'])==base
    assert cfg['peft_type']=='LORA' and cfg['r']==32 and cfg['lora_alpha']==64 and cfg['lora_dropout']==0
    assert set(cfg['target_modules'])=={'gate_proj','v_proj','o_proj','k_proj','down_proj','up_proj','q_proj'}
    started=time.time()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    torch.set_num_threads(2); torch.set_num_interop_threads(1); assert not torch.cuda.is_available()
    reference=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
    assert digest(base/'tokenizer.json')==NATIVE_TOKENIZER_SHA
    assert len(reference)==130560 and (reference.bos_token_id,reference.eos_token_id,reference.pad_token_id)==(0,1,1)
    model=AutoModelForCausalLM.from_pretrained(str(base),dtype=torch.bfloat16,device_map='cpu',local_files_only=True,trust_remote_code=False)
    assert sum(p.numel() for p in model.parameters())==MODEL_ELEMENTS
    model=PeftModel.from_pretrained(model,str(checkpoint),is_trainable=False,local_files_only=True)
    actual=get_peft_model_state_dict(model); saved=load_file(str(checkpoint/'adapter_model.safetensors'),device='cpu')
    assert len(actual)==len(saved)==ADAPTER_TENSORS and set(actual)==set(saved)
    for key in sorted(saved): assert torch.equal(actual[key],saved[key]) and torch.isfinite(actual[key]).all(), key
    del actual, saved
    expected={}
    for name,module in model.named_modules():
        if not hasattr(module,'lora_A'): continue
        assert list(module.lora_A)==['default'] and list(module.lora_B)==['default']
        assert module.r['default']==32 and module.lora_alpha['default']==64
        weight=module.base_layer.weight
        merged=merge_matrix_once(weight,module.lora_B['default'].weight,module.lora_A['default'].weight,2.0,torch)
        assert torch.isfinite(merged).all(),name
        key=name.removeprefix('base_model.model.')+'.weight'; expected[key]=tensor_digest(merged,torch)
        with torch.no_grad(): weight.data=merged
    assert len(expected)==MERGED_MATRICES
    assert model.config.tie_word_embeddings is False
    model=model.unload(); weights=dict(model.named_parameters())
    assert not any('lora_' in n for n in weights) and len(expected)==MERGED_MATRICES
    for key,sha in expected.items(): assert tensor_digest(weights[key],torch)==sha,key
    assert sum(p.numel() for p in model.parameters())==MODEL_ELEMENTS
    temporary.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(str(temporary),safe_serialization=True,max_shard_size='8GB')
    # Preserve native raw tokenizer/config/template bytes; never serialize the mutated checkpoint tokenizer.
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        shutil.copyfile(base/name,temporary/name); require_hash(temporary/name,b['parent']['files'][name]['sha256'])
    exported=AutoTokenizer.from_pretrained(str(temporary),local_files_only=True,trust_remote_code=False)
    assert len(exported)==130560 and (exported.bos_token_id,exported.eos_token_id,exported.pad_token_id)==(0,1,1)
    assert digest(temporary/'tokenizer.json')==NATIVE_TOKENIZER_SHA
    inventory=[]
    for p in sorted(temporary.glob('*.safetensors')): inventory.append({'path':p.name,'bytes':p.stat().st_size,'sha256':digest(p)})
    assert len(inventory)==1
    out_manifest={'schema':'sepalith.cloud-cpt.merged-parent-candidate.v1','status':'candidate_unselected','step':b['selected_checkpoint_step'],'checkpoint_path':str(checkpoint),'checkpoint_manifest_sha256':b['candidate']['campaign_manifest_sha256'],'source_identity':b['candidate']['source_identity'],'draw_schedule_sha256':b['candidate']['draw_schedule_sha256'],'train_rows_sha256':b['candidate']['train_rows_sha256'],'base_parent':b['parent'],'merge':{'method':b['merge']['method'],'device':'cpu','adapter_tensors_exact':ADAPTER_TENSORS,'merged_matrices_exact':MERGED_MATRICES,'accumulation_dtype':'float32','output_dtype':'bfloat16','weight_inventory':inventory,'weight_inventory_sha256':canonical_sha(inventory)},'tokenizer':{'tokenizer_json_sha256':digest(temporary/'tokenizer.json'),'tokenizer_config_sha256':digest(temporary/'tokenizer_config.json'),'chat_template_sha256':digest(temporary/'chat_template.jinja'),'vocab_size':130560,'bos_id':0,'eos_id':1,'pad_id':1,'native_eog_ids':[1,130073],'cloud_tokenizer_loaded':False},'binding_sha256':digest(ns.binding.resolve()),'source_sha256':digest(Path(__file__)),'elapsed_seconds':time.time()-started}
    (temporary/'parent-manifest.preparation.json').write_text(json.dumps(out_manifest,indent=2,sort_keys=True)+'\n')
    for p in temporary.iterdir():
        if p.is_file():
            with p.open('rb') as f: os.fsync(f.fileno())
    fd=os.open(temporary,os.O_RDONLY|os.O_DIRECTORY); os.fsync(fd); os.close(fd)
    temporary.rename(output)
    fd=os.open(output.parent,os.O_RDONLY|os.O_DIRECTORY); os.fsync(fd); os.close(fd)
    # Reopen all small identity files and the output weight header; full output hash was computed above.
    require_hash(output/'tokenizer.json',NATIVE_TOKENIZER_SHA)
    assert digest(output/'model.safetensors')==inventory[0]['sha256']
    print(json.dumps({'status':'candidate_unselected','step':b['selected_checkpoint_step'],'output':str(output),'weights_sha256':inventory[0]['sha256']}))
if __name__=='__main__': main()
