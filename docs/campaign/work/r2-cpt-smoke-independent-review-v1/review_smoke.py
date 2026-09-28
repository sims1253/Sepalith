"""Read closed smoke receipts and compute CPU-only exposure accounting."""
import datetime,hashlib,json,math,os,statistics
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parent.parent
ROOT=CAMPAIGN/'work/lead/r2-cpt-smoke-a'
CORPUS=CAMPAIGN/'work/r2-corpus-preparation-v1'
NATIVE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
ATTEMPT=NATIVE/'runner-r2-cpt-v1/attempts/8de4e4e2031241e4b84e8a658d8822b8'
TRAIN=NATIVE/'training/SFT11-CPT-smoke25-a'
GUARD=NATIVE/'training/SFT11-CPT-smoke25-a-host-supervision'
SNAPSHOT=NATIVE/'runner-r2-cpt-v1/snapshots/8df4c01e161fddb4f0e3c6d3b7b0899c6683b51971253806f9455dea66ec2b1d'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text())
def readlines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def stamp(value):return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()

def main():
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    terminal=load(GUARD/'terminal.json')
    if terminal['child_exit_code'] is None:raise ValueError('host child is not terminal')
    selected={
        'training':['load-audit.json','first-step-supervision-check.json','post-trainer-contract.json',
                    'train-begin-tokenizer-contract.json','step-0-cpt-validation-baseline.json',
                    'declared-token-exposure.json','admitted-recipe.json','telemetry.jsonl'],
        'guard':['terminal.json','launch.json','preflight.json','host-memory.jsonl','low-memory-details.jsonl',
                 'post-load-cache-release.json','process.log','cuda-lease.json'],
        'attempt':['execution.json','supervision.json','runtime.json','recipe.json','01-guarded-cpt-smoke25.log'],
        'root':['recipe.json','runner-recipe.json','supervisor-launch.json','snapshot.json','raw-shard-root-audit.json'],
    }
    source_roots={'training':TRAIN,'guard':GUARD,'attempt':ATTEMPT,'root':ROOT}
    evidence={}
    for category,names in selected.items():
        dest=HERE/'evidence'/category;dest.mkdir(parents=True,exist_ok=True)
        for name in names:
            src=source_roots[category]/name
            if not src.exists():continue
            raw=src.read_bytes()
            if len(raw)>16*1024*1024:raise ValueError('unexpectedly large bounded evidence')
            (dest/name).write_bytes(raw)
            evidence[str((dest/name).relative_to(HERE))]={'source':str(src),'sha256':sha(dest/name),'bytes':len(raw)}
    recipe=load(ROOT/'recipe.json');identity=recipe['identity'];supervision=load(ATTEMPT/'supervision.json')
    source_manifest=load(SNAPSHOT/'manifest.json');source_checks=[]
    for record in source_manifest['files']:
        name=record['path'];expected=record['sha256']
        if not name.endswith('.py'):raise ValueError('unexpected source manifest file')
        actual=sha(ATTEMPT/'source'/name)
        if actual!=expected or sha(SNAPSHOT/'source'/name)!=expected:raise ValueError('declared source changed')
        source_checks.append({'path':name,'sha256':expected})
    if identity['source']!=source_manifest['id']:raise ValueError('source identity differs')
    if sha(ROOT/'recipe.json')!=supervision['recipe_sha256']:raise ValueError('supervised recipe differs')
    if load(TRAIN/'admitted-recipe.json')!=recipe:raise ValueError('loaded recipe differs')
    for key in ('train_rows','validation_rows','draw_schedule'):
        if sha(recipe[key]['path'])!=recipe[key]['sha256']:raise ValueError('frozen data input differs')
    draw_raw=load(recipe['draw_schedule']['path']);draws=draw_raw.get('row_ids',draw_raw.get('schedule',{}).get('row_ids'))
    if draws is None:raise ValueError('draw schema unrecognized')
    train_rows={r['row_id']:r for r in readlines(recipe['train_rows']['path'])}
    validation_rows=readlines(recipe['validation_rows']['path'])
    telemetry=readlines(TRAIN/'telemetry.jsonl');steps=[r for r in telemetry if r['event']=='optimizer_step']
    metrics=[r for r in telemetry if r['event']=='trainer_metrics']
    if [r['step'] for r in steps]!=list(range(1,len(steps)+1)):raise ValueError('optimizer step gap')
    for r in steps:
        if not math.isfinite(r['seconds']) or r['seconds']<=0:raise ValueError('nonfinite step timing')
    if telemetry[0]['identity']!=identity:raise ValueError('train-begin identity differs')
    gate=load(TRAIN/'first-step-supervision-check.json')
    first=draws[:recipe['parameters']['per_device_batch']]
    denominator=sum(sum(t!=-100 for t in train_rows[i]['labels'][1:]) for i in first)
    if gate['rows']!=first or gate['causal_loss_denominator']!=denominator:raise ValueError('first batch denominator/IDs differ')
    baseline=load(TRAIN/'step-0-cpt-validation-baseline.json')['evaluation']
    val_den=sum(sum(t!=-100 for t in row['labels'][1:]) for row in validation_rows)
    if baseline['denominators']['validation_loss_tokens']!=val_den:raise ValueError('holdout denominator differs')
    if not math.isfinite(baseline['metrics']['mean_causal_nll']):raise ValueError('baseline NLL not finite')
    committed_draws=len(steps)*16;prefix=[train_rows[i] for i in draws[:committed_draws]]
    durations=[r['seconds'] for r in steps];steady=durations[1:]
    step_inputs=sum(len(r['input_ids']) for r in prefix)
    step_losses=sum(r['supervised_tokens'] for r in prefix)
    host=readlines(GUARD/'host-memory.jsonl');floor=load(GUARD/'preflight.json')['minimum_free_mib']
    low=[r for r in host if r['AvailableMBytes']<floor]
    load_audit=load(TRAIN/'load-audit.json')
    checkpoint_root=NATIVE/'checkpoints/SFT11-CPT-smoke25-a'
    # Names only: no checkpoint tensor or optimizer file is opened or hashed.
    checkpoint_names=sorted(str(p.relative_to(checkpoint_root)) for p in checkpoint_root.rglob('*')) if checkpoint_root.exists() else []
    summary={
        'status':'host_guard_stopped_before_step25','attempt_id':ATTEMPT.name,
        'source_id':identity['source'],'declared_source_files_verified':source_checks,
        'recipe_sha256':supervision['recipe_sha256'],'recipe_id':recipe['id'],
        'planned_schedule_steps':recipe['parameters']['max_steps'],'decision_stop_steps':recipe['decision_steps'],
        'actual_supervisor':load(ROOT/'supervisor-launch.json'),'host_guard_terminal':terminal,
        'model_identity_scope':'Declared parent hash and path match admitted recipe; worker did not read weights or independently bind loaded tensors.',
        'parent':identity['parent'],'load_audit':{'dtype':load_audit['dtype'],'trainable_parameters':load_audit['trainable_parameters'],'attached_modules':len(load_audit['modules'])},
        'tokenizer':identity['tokenizer'],'first_batch':{'row_ids':first,'causal_loss_tokens':denominator,'scope':'Actual collated tensors checked before train; this is not an independent fused-loss-versus-reference comparison.'},
        'baseline':{'mean_causal_nll':baseline['metrics']['mean_causal_nll'],'denominators':baseline['denominators'],'documents':len(baseline['case_ids']),'packages':len({r['package'] for r in validation_rows})},
        'optimizer_steps_recorded':len(steps),'first_optimizer_event':steps[0] if steps else None,
        'last_optimizer_event':steps[-1] if steps else None,'trainer_loss_metric_rows':metrics,
        'source_derived_committed_draws':committed_draws,'source_derived_prefix_input_tokens':step_inputs,
        'source_derived_prefix_loss_tokens':step_losses,
        'cursor_scope':'Derived from exact sequential source and completed step callbacks, not a persisted checkpoint cursor. Additional partial-step reads are unknown.',
        'timing':{'optimizer_step_seconds_sum':sum(durations),'first_step_seconds':durations[0] if durations else None,
            'mean_step_seconds':statistics.mean(durations) if durations else None,
            'steady_steps_count':len(steady),'steady_step_mean_seconds':statistics.mean(steady) if steady else None,
            'steady_step_median_seconds':statistics.median(steady) if steady else None,
            'source_derived_loss_tokens_per_optimizer_second':step_losses/sum(durations) if durations else None,
            'scope':'CPU monotonic on_step_begin to on_step_end. Excludes load, step-zero validation, host guard startup, final checkpoint/evaluation and release; first-step setup or compilation was not measured separately.'},
        'GPU_observed_peaks':{k:max(r['resources'][k] for r in steps) for k in ('cuda_attempt_peak_allocated_bytes','cuda_attempt_peak_reserved_bytes')} if steps else {},
        'host':{'samples':len(host),'soft_floor_mib':floor,'min_available_mib':min(r['AvailableMBytes'] for r in host),
            'max_available_mib':max(r['AvailableMBytes'] for r in host),'low_samples':len(low),'last_two_samples':host[-2:],
            'max_page_reads_per_second':max(r['PageReadsPersec'] for r in host),'max_pages_input_per_second':max(r['PagesInputPersec'] for r in host),
            'scope':'Windows host counters from guard; Linux MemAvailable is a different measure.'},
        'training_terminal_exists':(TRAIN/'terminal.json').exists(),'checkpoint_names_only':checkpoint_names,
        'step25_nll':None,'nll_change':None,'saved_full_checkpoint':False,
        'training_finite_loss_evidence':'No trainer loss metric row before the20-step logging boundary. Baseline fused validation loss is finite; optimizer callback timing/resource telemetry is finite.',
        'acceptance':'First-update and collator startup components observed; requested complete25-step smoke/checkpoint/resume evidence failed to materialize because host guard stopped at17.',
        'quality_limit':'CPT package-held-out next-token NLL is diagnostic and does not measure PRM03 edit quality.',
        'process_release':'Root-owned; terminal child returncode is not independent proof all CUDA owners have exited.',
    }
    (HERE/'smoke-readout.json').write_text(json.dumps(summary,indent=2)+'\n')
    (HERE/'evidence-manifest.json').write_text(json.dumps({'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':evidence},indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ('status','optimizer_steps_recorded','first_batch','baseline','timing','host','source_derived_committed_draws','source_derived_prefix_loss_tokens','checkpoint_names_only')}))

if __name__=='__main__':main()
