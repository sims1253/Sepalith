"""Account for every frozen broader-corpus row, including the six-row tail."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
CORPUS=HERE.parent/'r2-corpus-preparation-v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    path=CORPUS/'broader-shard-v1-2k/cpt_train.jsonl'
    schedule_path=CORPUS/'broader-draws-one-pass.json'
    schedule=json.loads(schedule_path.read_text())
    if sha(path)!=schedule['training_rows_sha256']:raise ValueError('broader row identity changed')
    order=schedule['row_ids'];stats={}
    for line in path.open():
        row=json.loads(line)
        stats[row['row_id']]={'row_id':row['row_id'],'document_id':row['document_id'],'package':row['package'],
            'input_tokens':len(row['input_ids']),'loss_tokens':sum(t!=-100 for t in row['labels'][1:]),
            'code_tokens':row['token_end']-row['token_start'],'terminal_document_eos':int(row['is_document_end'])}
    if len(order)!=len(set(order)) or set(order)!=set(stats):raise ValueError('one-pass schedule omits or repeats rows')
    full_steps,remainder=divmod(len(order),16);prefix_count=full_steps*16
    def totals(ids):return {k:sum(stats[i][k] for i in ids) for k in ('input_tokens','loss_tokens','code_tokens','terminal_document_eos')}
    all_totals=totals(order);prefix=totals(order[:prefix_count]);tail=totals(order[prefix_count:])
    if all_totals['code_tokens']!=38905169 or all_totals['loss_tokens']!=38919267:raise ValueError('token totals changed')
    padded_tokens=sum(len(order[i:i+2])*max(stats[j]['input_tokens'] for j in order[i:i+2]) for i in range(0,len(order),2))
    result={'status':'verified_exposure_report_no_source_or_schedule_changes',
        'training_rows':str(path),'training_rows_sha256':sha(path),'schedule':str(schedule_path),'schedule_sha256':sha(schedule_path),
        'rows':len(order),'unique_rows':len(stats),'full_effective16_steps':full_steps,'full_steps_rows':prefix_count,
        'one_pass_rows_remaining':remainder,'all_rows_totals':all_totals,'full_steps_totals':prefix,'tail_totals':tail,
        'tail_rows':[stats[i] for i in order[prefix_count:]],
        'dynamic_padding_for_declared_microbatch2_order':{'padded_input_tokens':padded_tokens,
            'actual_nonpadding_tokens':all_totals['input_tokens'],'nonpadding_fraction':all_totals['input_tokens']/padded_tokens,
            'scope':'Exact length arithmetic for sequential pairs; no runtime tensor or GPU measurement.'},
        'current_source_constraint':{'source':'8df4c01e161fddb4f0e3c6d3b7b0899c6683b51971253806f9455dea66ec2b1d',
            'draw_validator':'campaign_cpt_data.py validate_draw_schedule requires len(row_ids)==max_steps*16',
            'checkpoint_cursor':'campaign_cpt.py stores consumed_draws=global_step*16',
            'true_partial_tail_admitted_by_current_source':False},
        'preferred_true_one_pass_policy':{'updates':full_steps+1,'last_update_rows':remainder,
            'last_update_microbatches_at_size2':3,'last_update_loss_tokens':tail['loss_tokens'],
            'final_actual_cursor':len(order),'required_changes':'Root-reviewed partial-tail schedule contract; actual six-row loss normalization across three microbatches; cursor derived from actual consumed rows; source/data identity and resume checks updated. No patch or launch is supplied here.'},
        'unchanged_source_choices':[
            {'updates':full_steps,'draws':prefix_count,'unexposed_tail_rows':remainder,'unexposed_tail_loss_tokens':tail['loss_tokens'],
             'label':'Explicit incomplete pass; preserve and report six unexposed rows. Must not call this a complete one-pass exposure.'},
            {'updates':full_steps+1,'draws':(full_steps+1)*16,'required_named_replay_rows':16-remainder,
             'label':'All rows plus ten explicitly named replay rows; not a pure one-pass corpus. New root-admitted schedule required.'}],
        'checkpoint_resumption_limit':'The stopped smoke has no saved checkpoint. A future profiling checkpoint also has a different dataset identity; broader-stage initialization requires root selection and an explicit transition.'}
    (HERE/'long-stage-one-pass-exposure.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('tail_rows','unchanged_source_choices','preferred_true_one_pass_policy')}))

if __name__=='__main__':main()
