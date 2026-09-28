"""Run only after root supervisor termination; no model tensor is loaded."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,importlib.util,json,math,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NATIVE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
WORK=ROOT/'work/lead/target-only25-a'
OUT=NATIVE/'training/SFT-target-only25-a'
CHECKPOINT=NATIVE/'checkpoints/SFT-target-only25-a/full/checkpoint-25'
SNAPSHOT=NATIVE/'runner-target-only-v1/snapshots/dcccf3ed2de98386223a0d06c994747e025363ebed80d6adc9d96bfee21a3e43/source'

def pin(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return True

def main():
    assert (WORK/'supervisor-terminal.json').is_file(),'root supervisor is still active; no artifact hashing permitted'
    recipe=json.loads((WORK/'recipe.json').read_text());terminal=json.loads((OUT/'terminal.json').read_text())
    assert recipe['identity']['source']=='dcccf3ed2de98386223a0d06c994747e025363ebed80d6adc9d96bfee21a3e43'
    assert pin(WORK/'recipe.json')['sha256']=='d9fcd250546338040e553e65e353c75bce0e16849a35ef88ff68fa651a8cc0ac'
    source_manifest=json.loads((SNAPSHOT.parent/'manifest.json').read_text())
    assert source_manifest['id']==recipe['identity']['source']
    source_pins=[]
    for declared in source_manifest['files']:
        actual=pin(SNAPSHOT/declared['path']);assert actual['sha256']==declared['sha256'];source_pins.append(actual)
    assert terminal['step']==25 and terminal['identity']==recipe['identity']
    assert terminal['status']=='lead_decision'
    gate=json.loads((OUT/'target-only-pre-update-gate.json').read_text())
    assert gate['status']=='pass' and gate['before_optimizer_step']==0 and gate['actual_fused_calls']==1
    assert gate['prompt_supervision_tokens']==gate['padding_supervision_tokens']==0 and gate['eos_supervised'] is True
    assert gate['target_denominator']==1081 and gate['microbatch_target_denominators']==[128,560,198,195]
    assert abs(gate['fused_loss']-gate['reference_loss'])<=gate['absolute_tolerance']
    assert gate['batch_identity_sha256']==json.loads((HERE/'startup-batch-corroboration.json').read_text())['admitted_first16_batch_sha256']
    telemetry=[json.loads(line) for line in (OUT/'telemetry.jsonl').read_text().splitlines()]
    updates=[r for r in telemetry if r.get('event')=='optimizer_step']
    assert [r['step'] for r in updates]==list(range(1,26)) and finite(telemetry)
    assert updates[-1]['stop_reason']=='lead_decision'
    assert (OUT/'target-only-pre-update-gate.json').stat().st_mtime<=datetime.fromisoformat(updates[0]['at']).timestamp()
    state=json.loads((CHECKPOINT/'campaign-state.json').read_text())
    manifest=json.loads((CHECKPOINT/'campaign-manifest.json').read_text())
    trainer=json.loads((CHECKPOINT/'trainer_state.json').read_text())
    assert manifest['full'] is True and manifest['step']==25 and manifest['identity']==recipe['identity']
    assert trainer['global_step']==25 and state['step']==25
    assert state['sampler']['consumed_draws']==400
    assert state['sampler']['schedule_sha256']==recipe['draw_schedule']['sha256']
    files=[]
    for name,expected in sorted(manifest['files'].items()):
        p=CHECKPOINT/name;assert p.is_file() and not p.is_symlink()
        got=pin(p);assert got['sha256']==expected['sha256'] and got['bytes']==expected['bytes'];files.append(got)
    required={'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'}
    assert required<=set(manifest['files'])
    actual_files={str(p.relative_to(CHECKPOINT)) for p in CHECKPOINT.rglob('*') if p.is_file() and p.name!='campaign-manifest.json'}
    assert actual_files==set(manifest['files'])
    checkpoint={'manifest':pin(CHECKPOINT/'campaign-manifest.json'),'files':files,'full':True,'step':25,'sampler':state['sampler'],
                'all_file_hashes_and_sizes_match':True,'model_weights_loaded':False}
    # State contents are separate from model weights. weights_only=True prevents
    # arbitrary pickle class execution; mmap avoids copying optimizer tensors.
    import torch
    torch.set_num_threads(1)
    optimizer=torch.load(CHECKPOINT/'optimizer.pt',map_location='cpu',weights_only=True,mmap=True)
    steps=Counter(float(entry['step']) for entry in optimizer['state'].values())
    assert set(steps)=={25.0} and sum(steps.values())==588
    assert all(torch.isfinite(v).all() for entry in optimizer['state'].values() for k,v in entry.items() if torch.is_tensor(v))
    groups=optimizer['param_groups'];assert sum(len(g['params']) for g in groups)==588
    scheduler=torch.load(CHECKPOINT/'scheduler.pt',map_location='cpu',weights_only=True)
    assert scheduler['last_epoch']==25
    expected_lr=5e-5*0.5*(1+math.cos(math.pi*(25-6)/(200-6)))
    assert all(math.isclose(group['lr'],expected_lr,rel_tol=1e-12) for group in groups)
    checkpoint['optimizer']={'parameter_states':sum(steps.values()),'steps':dict(steps),'all_state_tensors_finite':True,'group_learning_rates':[g['lr'] for g in groups]}
    checkpoint['scheduler']={'state':scheduler,'independent_cosine_lr':expected_lr,'warmup_steps':6,'horizon':200}
    del optimizer
    # Only the standard NumPy array/dtype reconstruction classes are admitted;
    # this reads saved RNG state on CPU and never installs it in this process.
    import numpy as np
    reconstruct=np.core.multiarray._reconstruct
    allowed=[np.ndarray,np.dtype,type(np.dtype('uint32')),
             (reconstruct,'numpy.core.multiarray._reconstruct'),(reconstruct,'numpy._core.multiarray._reconstruct')]
    with torch.serialization.safe_globals(allowed):
        rng=torch.load(CHECKPOINT/'rng_state.pth',map_location='cpu',weights_only=True)
    assert {'python','numpy','cpu','cuda'}<=set(rng)
    assert len(rng['python'])==3 and len(rng['python'][1])==625
    assert rng['numpy'][0]=='MT19937' and rng['numpy'][1].shape==(624,)
    assert rng['cpu'].dtype==torch.uint8 and rng['cpu'].numel()>0
    cuda_states=rng['cuda'] if isinstance(rng['cuda'],list) else [rng['cuda']]
    assert cuda_states and all(t.dtype==torch.uint8 and t.numel()>0 for t in cuda_states)
    checkpoint['rng']={'file':pin(CHECKPOINT/'rng_state.pth'),'keys':sorted(rng),'python_mt_state_length':625,
        'numpy_mt_state_length':624,'cpu_rng_bytes':rng['cpu'].numel(),'cuda_rng_bytes':[t.numel() for t in cuda_states],
        'loaded_on_cpu_only':True,'state_installed_or_replay_test_run':False}
    readout=load_module('independent_corrective_readout',ROOT/'work/corrective-sft-readout/read_corrective_sft.py')
    readout.DEFAULT_PROTOCOL=SNAPSHOT/'packages/sepalith/src/sepalith/campaign_protocol.py'
    cases_path=CHECKPOINT.parent.parent/'evaluations/cases-step-25.json'
    cases=json.loads(cases_path.read_text());assert cases['status']=='complete' and len(cases['results'])==75
    score=readout.readout(WORK/'recipe.json',cases_path,25,HERE/'development-readout.json')
    assert score['status']=='complete_readout' and not score['validation']['issues'] and not score['evaluation']['embedded_summary_mismatches']
    from tokenizers import Tokenizer
    tokenizer_path=Path(recipe['model_path'])/'tokenizer.json'
    tok=Tokenizer.from_file(str(tokenizer_path));decode_checks=[]
    protocol=readout.load_module('exact_new_protocol',readout.DEFAULT_PROTOCOL)
    panel,_=readout.read_panel(recipe,protocol);byid={p.row['id']:p for p in panel};finish=[]
    for result in cases['results']:
        ids=result['generated_ids'];raw_ids=ids[:-1] if ids and ids[-1]==1 else ids
        assert tok.decode(raw_ids,skip_special_tokens=False)==result['raw_output']
        decode_checks.append({'id':result['id'],'generated_tokens':len(ids),'raw_sha256':hashlib.sha256(result['raw_output'].encode()).hexdigest()})
        case=byid[result['id']]
        if case.row['family']=='finish_block':
            classified=readout.classify_result(result,case,protocol,512)
            if classified['protocol_valid']:
                _,after,geometry=readout.apply_prediction(case,classified['_parsed'],protocol)
                p=HERE/(result['id']+'.after.R');p.write_text(after)
                finish.append({'id':result['id'],'after_file':pin(p),'geometry':geometry})
    result={'status':'independent_terminal_review_complete','at':datetime.now(timezone.utc).isoformat(),
            'root_supervisor_terminal':json.loads((WORK/'supervisor-terminal.json').read_text()),
            'runtime_gate':gate,'telemetry':{'updates':25,'contiguous':True,'finite':True,'update_seconds_sum':sum(r['seconds'] for r in updates),
                'source_cursor':400,'last_update':updates[-1],'trainer_metrics':[r for r in telemetry if r.get('event')=='trainer_metrics']},
            'checkpoint':checkpoint,'development':score,'generated_ids_raw_decode_checks':decode_checks,'exact_finish_buffers':finish,
            'source_snapshot':{'manifest':pin(SNAPSHOT.parent/'manifest.json'),'files_verified':len(source_pins),'files':source_pins},
            'input_pins':[pin(WORK/'recipe.json'),pin(readout.DEFAULT_PROTOCOL),pin(ROOT/'work/corrective-sft-readout/read_corrective_sft.py')],
            'terminal_artifact_pins':[pin(OUT/n) for n in ['admitted-recipe.json','load-audit.json','post-trainer-contract.json','train-begin-tokenizer-contract.json','target-only-pre-update-gate.json','telemetry.jsonl','terminal.json']],
            'model_weights_loaded':False,'R_execution':False,'GPU_or_network_or_process_control':False,'final_access':False}
    (HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'counts':score['counts'],'denominators':score['denominators'],'finish_parse':score['finish_block_audit']['r_parse_ok'],'finish_exact':score['finish_block_audit']['exact_region']},indent=2))


if __name__=='__main__':main()
