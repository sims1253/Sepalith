import pathlib,json,sys,argparse
from types import SimpleNamespace
import probe
sys.path.insert(0,'/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src')
from sepalith import campaign_protocol as protocol
w=pathlib.Path(__file__).resolve().parent
arm=sys.argv[1];entry=next(x for x in json.loads((w/'arms.json').read_text()) if x['arm']==arm)
rows=json.loads((w/'dev-cases.json').read_text());out=w/arm;out.mkdir()
with (out/'dev.jsonl').open('x') as f:
 for row in rows:
  r=probe.run_case('http://127.0.0.1:18414',row,'cold',1,192,4096,5000)
  ctx=protocol.PromptContext.from_dict(row['context']);parsed=protocol.parse_output(r.get('raw_text',''),ctx)
  valid=r['protocol_status']=='accepted' and parsed.status=='accepted';noop=valid and parsed.operation=='no_op';actual=list(ctx.region_old) if noop else list(parsed.body)
  exact=valid and '\n'.join(actual)==row['target_body_text']
  r.update(family=row['family'],package_id=row['package_id'],context_protocol_valid=valid,expected_noop=row['target_operation']=='no_op',exact_edit=exact and row['target_operation']!='no_op',correct_noop=noop and row['target_operation']=='no_op',false_suggestion=valid and not noop and row['target_operation']=='no_op')
  f.write(json.dumps(r)+'\n');f.flush()
args=SimpleNamespace(url='http://127.0.0.1:18414',arm=arm,model_path=entry['path'],model_sha256=entry['sha256'],panel=str(w/'panel.jsonl'),manifest=str(w/'manifest.json'),row_limit=None,cap=192,context=4096,reps=1,deadline_ms=5000)
r=probe.run_probe(args);(out/'train.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'arm':arm,'dev_cases':len(rows),'train_requests':len(r['requests'])}),flush=True)
