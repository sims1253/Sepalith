import pathlib,json,sys,hashlib,collections,statistics
sys.path.insert(0,'/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src')
from sepalith import campaign_protocol as protocol
w=pathlib.Path(__file__).resolve().parent
src=w.parents[1]/'corrected-dev75-v1/dev75-corrected-finish-v1.jsonl'
# Use the exact admitted DEV label artifact, independently of the preliminary client summary.
src=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl')
assert hashlib.sha256(src.read_bytes()).hexdigest()=='7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035'
labels={r['id']:r for r in map(json.loads,src.read_text().splitlines())}
records=list(map(json.loads,(w/'requests.jsonl').read_text().splitlines()))
assert len(records)==225 and len({(r['row_id'],r['cap']) for r in records})==225
out=[]
for r in records:
 label=labels[r['row_id']];ctx=protocol.PromptContext.from_dict(label['context']);parsed=protocol.parse_output(r.get('raw_text',''),ctx)
 valid=r['protocol_status']=='accepted' and parsed.status=='accepted';noop=valid and parsed.operation=='no_op';region=list(ctx.region_old) if noop else list(parsed.body)
 out.append({'id':r['row_id'],'cap':r['cap'],'family':label['family'],'valid':valid,'exact_edit':valid and label['operation']!='no_op' and region==label['region_new'],'correct_noop':label['operation']=='no_op' and noop,'false_suggestion':label['operation']=='no_op' and valid and not noop,'wall_ms':r['combined_case_wall_ms'],'tokens':r.get('returned_token_count'),'region':region,'parser_operation':parsed.operation})
summary={}
for cap in [192,384,512]:
 rows=[r for r in out if r['cap']==cap];summary[cap]={k:sum(r[k] for r in rows) for k in ['valid','exact_edit','correct_noop','false_suggestion']};summary[cap].update(cases=75,edit_cases=43,noop_cases=32,median_wall_ms=statistics.median(r['wall_ms'] for r in rows))
 summary[cap]['finish']={k:sum(r[k] for r in rows if r['family']=='finish_block') for k in ['valid','exact_edit']}
old=json.loads(pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-task-global500-native-a/dev-results/results.json').read_text());old={r['id']:r for r in old['results']}
parity=sum(r.get('raw_text')==old[r['row_id']]['raw_text'] and r.get('returned_token_ids')==old[r['row_id']]['returned_token_ids'] for r in records if r['cap']==192)
assert summary[192]['correct_noop']==26 and summary[192]['false_suggestion']==4 and summary[192]['exact_edit']==26
result={'status':'root_context_aware_scoring_complete','summary':summary,'baseline192_raw_and_token_parity':parity,'preliminary_summary_invalid':'Client token-only parser does not classify unchanged replacement bodies as no_op; summary.json no-op counts are invalid and superseded here. Raw inference records are unchanged.','protocol_sha256':hashlib.sha256(pathlib.Path(protocol.__file__).read_bytes()).hexdigest(),'rows':out}
(w/'context-aware-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
