#!/usr/bin/env python3
"""Closed smoke metadata review; tensors and serialized optimizer/RNG stay unread."""
import collections, datetime, json, math, os, statistics
from pathlib import Path
from audit_and_prepare import HERE, NATIVE, COR, SOURCE, sha, read, write, check

def main():
    if hasattr(os,'sched_setaffinity'): os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    t=NATIVE/'training/SFT11-CPT-smoke25-c'; c=NATIVE/'checkpoints/SFT11-CPT-smoke25-c'
    root=HERE.parent/'lead/r2-cpt-smoke-c'; guard=NATIVE/'training/SFT11-CPT-smoke25-c-host-supervision'
    terminal=read(t/'terminal.json'); host=read(guard/'terminal.json')
    check(terminal['step']==25 and terminal['status']=='lead_decision','trainer terminal')
    check(host['status']=='completed' and host['child_exit_code']==0 and host['reason'] is None,'guard terminal')
    recipe=read(root/'recipe.json'); identity=recipe['identity']
    check(identity['source']==SOURCE and terminal['identity']==identity and terminal['resumed_from_step'] is None,'source/fresh identity')
    checkpoint=c/'full/checkpoint-25'; manifest=read(checkpoint/'campaign-manifest.json'); state=read(checkpoint/'campaign-state.json')
    check(manifest['identity']==state['identity']==identity and manifest['full'] and state['full'] and state['step']==manifest['step']==25,'full checkpoint identity')
    check(state['sampler']['consumed_draws']==400 and state['sampler']['schedule_sha256']==recipe['draw_schedule']['sha256'],'cursor')
    files={}
    for name,pin in manifest['files'].items():
        p=checkpoint/name
        check(p.is_file() and p.stat().st_size==pin['bytes'],'checkpoint file size '+name)
        measured=False
        if name in ('campaign-state.json','trainer_state.json','adapter_config.json'):
            check(sha(p)==pin['sha256'],'checkpoint metadata hash '+name); measured=True
        files[name]={**pin,'actual_size_checked':True,'payload_sha256_independently_measured':measured}
    for f in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin','trainer_state.json']:
        check(f in files,'full resume artifact presence '+f)
    check(read(checkpoint/'trainer_state.json')['global_step']==25,'trainer state step')
    gate=read(t/'first-step-supervision-check.json')
    check(gate['status']=='verified' and gate['causal_loss_denominator']==2212 and gate['global_step']==0,'actual first collator gate')
    baseline=read(t/'step-0-cpt-validation-baseline.json')['evaluation']
    result=read(c/'evaluations/step-25.json'); after=result['result']
    check(result['status']=='evaluated' and after['step']==25,'evaluation terminal')
    check(baseline['denominators']==after['denominators'] and baseline['case_ids']==after['case_ids'],'validation denominator/case identity')
    before_nll=baseline['metrics']['mean_causal_nll']; after_nll=after['metrics']['mean_causal_nll']
    check(math.isfinite(before_nll) and math.isfinite(after_nll),'finite validation')
    events=[json.loads(l) for l in (t/'telemetry.jsonl').read_text().splitlines()]
    steps=[e for e in events if e['event']=='optimizer_step']
    metrics=[e for e in events if e['event']=='trainer_metrics' and 'loss' in e['metrics']]
    check([e['step'] for e in steps]==list(range(1,26)) and [e['step'] for e in metrics]==list(range(1,26)),'contiguous25 updates/metrics')
    check(all(math.isfinite(v) for e in metrics for v in e['metrics'].values() if isinstance(v,(int,float))),'finite training metrics')
    draws=read(Path(recipe['draw_schedule']['path']))['row_ids'][:400]
    tokens={}
    with Path(recipe['train_rows']['path']).open() as f:
        for line in f:
            row=json.loads(line)
            if row['row_id'] in draws: tokens[row['row_id']]=row['supervised_tokens']
    check(len(tokens)==len(draws)==400,'source-derived400 unique draws')
    loss_tokens=sum(tokens[r] for r in draws)
    host_rows=[json.loads(l) for l in (guard/'host-memory.jsonl').read_text().splitlines()]
    durations=[s['seconds'] for s in steps]
    check(all(math.isfinite(x) and x>=0 for x in durations),'finite timing')
    load=read(t/'load-audit.json')
    check(load['dtype']=='torch.bfloat16' and load['trainable_parameters']==50233344 and len(load['modules'])==294 and load['gradient_checkpointing_requested'] is True,'actual load contract')
    pins={}
    for p in [root/'recipe.json',t/'terminal.json',guard/'terminal.json',t/'telemetry.jsonl',t/'first-step-supervision-check.json',t/'post-trainer-contract.json',t/'load-audit.json',t/'step-0-cpt-validation-baseline.json',c/'evaluations/step-25.json',checkpoint/'campaign-manifest.json',checkpoint/'campaign-state.json',checkpoint/'trainer_state.json',guard/'host-memory.jsonl',root/'host-rss-sampler/terminal.json',root/'host-rss-sampler/configuration.json']:
        pins[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
    sampler_terminal=read(root/'host-rss-sampler/terminal.json')
    report={'status':'verified_smoke25_terminal_and_metadata_not_continuation_or_edit_acceptance','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':SOURCE,'trainer_terminal':{k:v for k,v in terminal.items() if k!='identity'},'guard_terminal':host,'source_derived_consumed_draws':400,'source_derived_loss_tokens':loss_tokens,'actual_first_collator_gate':gate,'validation':{'before_nll':before_nll,'after_nll':after_nll,'absolute_delta':after_nll-before_nll,'relative_delta':after_nll/before_nll-1,'denominators':after['denominators'],'document_count':len(after['case_ids']),'validation_packages':len(identity['data']['validation_package_ids']),'same_case_ids':True,'scope':'package-heldout TRAIN causal NLL; no edit quality measurement'},'training':{'optimizer_updates':25,'finite_step_loss_metrics':25,'loss_min':min(e['metrics']['loss'] for e in metrics),'loss_max':max(e['metrics']['loss'] for e in metrics),'optimizer_callback_seconds_sum':sum(durations),'optimizer_callback_seconds_median':statistics.median(durations),'loss_tokens_per_callback_second':loss_tokens/sum(durations),'host_guard_seconds':host['seconds'],'loss_tokens_per_guard_second':loss_tokens/host['seconds'],'peak_process_rss_bytes':max(e['resources']['process_peak_rss_bytes'] for e in steps),'peak_cuda_allocated_bytes':max(e['resources']['cuda_attempt_peak_allocated_bytes'] for e in steps),'peak_cuda_reserved_bytes':max(e['resources']['cuda_attempt_peak_reserved_bytes'] for e in steps),'timing_scope':'Source callback intervals can include intervening checkpoint work; guard duration also includes startup and step-zero/25 validation. Neither is a broad-stage guarantee.'},'host':{'windows_samples':len(host_rows),'available_mib_min':min(r['AvailableMBytes'] for r in host_rows),'available_mib_max':max(r['AvailableMBytes'] for r in host_rows),'page_reads_per_second_max':max(r['PageReadsPersec'] for r in host_rows),'sampler_terminal':sampler_terminal},'checkpoint':{'path':str(checkpoint),'cursor':state['sampler'],'files':files,'limitation':'Declared tensor/optimizer/RNG/scheduler hashes carried from closed writer manifest; only file sizes and selected JSON hashes checked here. No serialized state loaded. Resume50 and resource release remain root-owned.'},'pins':pins}
    write('smoke-c-review.json',report)
    print(json.dumps({k:report[k] for k in ('status','validation','training','host')}))

if __name__=='__main__':main()
