#!/usr/bin/env python3
"""Evaluate a preserved full SFT checkpoint without resuming training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--recipe', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
recipe = json.loads(a.recipe.read_text())
source = Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots') / recipe['identity']['source'] / 'source'
sys.path[:0] = [str(source / 'experiments/training'), str(source / 'packages/sepalith/src')]
from campaign_checkpoint import verify_checkpoint, write_json, digest
from campaign_eval import development_evaluator
from campaign_sft import assert_post_trainer_pinned_identity, training_configuration_guard, TARGET_MODULES, restore_trainer_eog_alignment
from campaign_tokenizer_contract import load_pinned_reference_tokenizer, restore_pinned_tokenizer_contract
assert a.output.is_absolute() and not a.output.exists()
assert subprocess.check_output(['findmnt', '-T', str(a.output.parent), '-n', '-o', 'FSTYPE'], text=True).strip() == 'ext4'
manifest = verify_checkpoint(a.checkpoint, recipe['identity'], require_full=True)
state = json.loads((a.checkpoint / 'campaign-state.json').read_text())
step = state['step']
assert state['full'] and state['sampler']['consumed_draws'] == step * 16
assert not (a.checkpoint.parent.parent / 'evaluations' / f'cases-step-{step}.json').exists()
for pid in [3273988,3274675,3274676,3312632,3312634]: assert not Path(f'/proc/{pid}').exists()
mem = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True).splitlines()
assert len(mem) == 1 and float(mem[0]) < 4096
for item in recipe['inputs']:
    if Path(item['path']).parent == Path(recipe['model_path']): assert digest(Path(item['path'])) == item['sha256']
a.output.mkdir()
write_json(a.output / 'launch.json', {'task':'SFT-10','owner':'lead','pid':os.getpid(),'step':step,
           'checkpoint':str(a.checkpoint),'checkpoint_manifest_sha256':digest(a.checkpoint/'campaign-manifest.json'),
           'source_snapshot':recipe['identity']['source'],'helper_sha256':digest(Path(__file__)),
           'scope':'Evaluation only, no optimizer/training updates. Fresh CUDA process after prior runtime failure.'})
try:
    from unsloth import FastLanguageModel
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    torch.manual_seed(3407)
    probe = torch.randn(256,256,dtype=torch.bfloat16)
    expected = probe.float() @ probe.float().T
    actual = (probe.cuda() @ probe.cuda().T).float().cpu()
    torch.cuda.synchronize()
    nmse = float((actual-expected).square().mean()/expected.square().mean())
    assert torch.isfinite(actual).all() and nmse < 1e-5
    write_json(a.output/'cuda-health.json', {'device':torch.cuda.get_device_name(0),'bf16_matmul_nmse':nmse,
               'finite':True,'synchronized':True,'cuda_allocator_fraction':0.75,'cuda_allocator_cap_bytes':int(torch.cuda.get_device_properties(0).total_memory*0.75),'scope':'Small forward check; no proof of backward stability.'})
    del actual, expected, probe
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=recipe['model_path'],max_seq_length=4096,
                   dtype=torch.bfloat16,load_in_4bit=False,trust_remote_code=False)
    reference = load_pinned_reference_tokenizer(Path(recipe['model_path']))
    restore_pinned_tokenizer_contract(model, tokenizer, reference_tokenizer=reference, prompt_rows=[])
    model = FastLanguageModel.get_peft_model(model,r=32,lora_alpha=64,lora_dropout=0,target_modules=TARGET_MODULES,
                   bias='none',use_gradient_checkpointing='unsloth',random_state=3407)
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict, get_peft_model_state_dict
    saved = load_file(str(a.checkpoint/'adapter_model.safetensors'),device='cpu')
    set_peft_model_state_dict(model,saved,adapter_name='default')
    loaded = get_peft_model_state_dict(model)
    assert set(saved) == set(loaded) and len(saved) == 588
    for name in saved: assert torch.equal(saved[name],loaded[name].detach().cpu()), name
    write_json(a.output/'load-audit.json', {'adapter_tensors_exact':588,
               'contract':assert_post_trainer_pinned_identity(model,tokenizer,reference,[])})
    del saved, loaded
    restore_trainer_eog_alignment(model,tokenizer,reference)
    with training_configuration_guard(model,FastLanguageModel.for_training):
        model.eval()
        summary = development_evaluator(recipe)(model,tokenizer,a.checkpoint,step)
    write_json(a.output/'terminal.json', {'status':'complete','step':step,'summary':summary,
               'terminal_contract':assert_post_trainer_pinned_identity(model,tokenizer,reference,[])})
    print(json.dumps({'status':'complete','step':step,'counts':summary['counts'],'denominators':summary['denominators']}))
except Exception as error:
    write_json(a.output/'failure.json', {'status':'failed','type':type(error).__name__,'message':str(error)})
    raise
