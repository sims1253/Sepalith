"""Disposable complete-target training memory profile; never saves model weights."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

W = Path(__file__).resolve().parent
SOURCE = W.parents[1] / 'r2-task-trainer-review-v2/source/experiments/training'
sys.path.insert(0, str(SOURCE))
DATA = W.parent / 'r2-data-expansion-audit-v1/expanded-corrected-short-token-rows.jsonl'
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == 'fa247ae7dbbf0b5a66538e8993d9fdd70624ce1c81ae54ae4d6f3d62b2368889'
rows = [json.loads(line) for line in DATA.read_text().splitlines()]
long_sequence = max(rows, key=lambda r: len(r['input_ids']))
long_target = max(rows, key=lambda r: len(r['input_ids']) - r['target_start'])
selected = [long_sequence, long_target]
del rows
from unsloth import FastLanguageModel
import torch
from campaign_sft_data import target_only_collator
from campaign_tokenizer_contract import load_pinned_reference_tokenizer, restore_pinned_tokenizer_contract

torch.set_num_threads(4)
model_path = '/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-merged'
started = time.monotonic()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path, max_seq_length=4096, dtype=torch.bfloat16,
    load_in_4bit=False, trust_remote_code=False, use_gradient_checkpointing=True)
reference = load_pinned_reference_tokenizer(Path(model_path))
audit = restore_pinned_tokenizer_contract(model, tokenizer, reference_tokenizer=reference, prompt_rows=selected)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=64, lora_dropout=0,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    bias='none', use_gradient_checkpointing=True, random_state=3407)
FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False
parameters = [p for p in model.parameters() if p.requires_grad]
assert sum(p.numel() for p in parameters) == 50233344
optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=0, fused=True)
load_seconds = time.monotonic() - started
records = []
for batch_size in (2, 4):
    chosen = selected * (batch_size // 2)
    batch = {k: v.cuda() for k, v in target_only_collator(chosen).items()}
    expected = sum(len(r['input_ids']) - r['target_start'] for r in chosen)
    assert int((batch['labels'] != -100).sum()) == expected
    for repeat in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        output = model(**batch, use_cache=False)
        loss = output.loss
        assert torch.isfinite(loss).item()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        assert torch.isfinite(grad_norm).item()
        optimizer.step()
        torch.cuda.synchronize()
        record = dict(batch_size=batch_size, repeat=repeat, loss=float(loss.detach()),
                      grad_norm=float(grad_norm), seconds=time.monotonic()-tick,
                      supervised_tokens=expected,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        records.append(record)
        with (W/'steps.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        del output, loss
    del batch
(W/'result.json').write_text(json.dumps(dict(status='completed_disposable_profile',
    load_seconds=load_seconds, tokenizer_audit=audit, records=records,
    selected_rows=[dict(id=r['id'], sequence_tokens=len(r['input_ids']),
                        target_tokens=len(r['input_ids'])-r['target_start']) for r in selected],
    model_weights_saved=False,
    limitation='Four disposable optimizer steps; not a trained checkpoint or sustained throughput estimate.'), indent=2)+'\n')
