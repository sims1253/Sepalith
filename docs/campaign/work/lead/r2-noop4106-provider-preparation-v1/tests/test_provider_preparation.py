import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import build_plan,run_lane,prepare_fallback,finalize_policy,bind_commands

def dump(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:
  for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
def js(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n')
def h(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class ProviderPreparation(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir='/mnt/e/sepalith/campaign-20260915/tmp');self.root=Path(self.temp.name);self.inputs=self.root/'inputs';self.inputs.mkdir()
  supported=[];holds=[];candidates=[]
  for i in range(4106):
   shard=0 if i<2053 else 1;rid=f'supported-{i:04d}';candidates.append({'row_id':rid,'shard':shard,'status':'provenance_supported_candidate_root_review_required'})
   pre='a'*64 if i<2 else hashlib.sha256(rid.encode()).hexdigest();workspace='/workspace-a' if i!=1 else '/workspace-b'
   supported.append((shard,{'row_id':rid,'preedit_sha256':pre,'cursor':{'line':4 if i<2 else i,'character':0},'path':'R/shared.R' if i<2 else f'R/{rid}.R','workspace_root':workspace,'absolute_document_path':f'{workspace}/R/source.R'}))
  for i in range(121):
   shard=0 if i<61 else 1;rid=f'hold-{i:03d}';candidates.append({'row_id':rid,'shard':shard,'status':'hold_independent_provenance_failure'});holds.append((shard,{'row_id':rid,'status':'hold'}))
  dump(self.root/'candidates.jsonl',candidates);self.candidates=self.root/'candidates.jsonl';js(self.root/'coverage.json',{'partial':False,'completed_shards':41,'pending_shards':[]});self.coverage=self.root/'coverage.json'
  per=[]
  for shard in (0,1):
   p=[x for s,x in supported if s==shard];q=[x for s,x in holds if s==shard];dump(self.inputs/f'shard-{shard:04d}.jsonl',p);dump(self.inputs/f'shard-{shard:04d}.sidecar.jsonl',[{'row_id':x['row_id']}for x in p]);dump(self.inputs/f'shard-{shard:04d}.holds.jsonl',q)
   per.append({'shard':shard,'candidates':len(p)+len(q),'prediction_inputs':len(p),'holds':len(q),'prediction_sha256':h(self.inputs/f'shard-{shard:04d}.jsonl'),'sidecar_sha256':h(self.inputs/f'shard-{shard:04d}.sidecar.jsonl'),'holds_sha256':h(self.inputs/f'shard-{shard:04d}.holds.jsonl')})
  duplicate=[{'schema':'sepalith.dat10.sourcewalk-noop.duplicate-geometry.v1','geometry':{'preedit_sha256':'a'*64,'cursor':{'line':4,'character':0},'path':'R/shared.R'},'count':2,'row_ids':['supported-0000','supported-0001'],'workspace_roots':['/workspace-a','/workspace-b'],'absolute_document_paths':['/workspace-a/R/source.R','/workspace-b/R/source.R'],'disposition':'retain_all_until_provider_prompt_target_dedup'}]
  dump(self.inputs/'duplicate-geometries.jsonl',duplicate)
  js(self.inputs/'manifest.json',{'schema':'sepalith.dat10.sourcewalk-noop-expansion-preparation.v3','status':'complete_review_only','full_41_shard_closure':True,'candidate_rows':4227,'prediction_inputs':4106,'holds':121,'duplicate_geometry':{'policy':'retain_all_until_provider_prompt_target_dedup','artifact':'duplicate-geometries.jsonl','sha256':h(self.inputs/'duplicate-geometries.jsonl'),'groups':1,'rows_in_groups':2,'excess_rows':1},'shards':per})
 def tearDown(self):self.temp.cleanup()
 def plan(self):
  out=self.root/'plan.json'
  with mock.patch.object(build_plan,'COVERAGE_SHA',h(self.coverage)),mock.patch.object(build_plan,'CANDIDATES_SHA',h(self.candidates)):value=build_plan.build(self.inputs,self.coverage,self.candidates,out)
  return value,out
 def test_plan_enumerates_all_shards_and_balances_two_distinct_cores(self):
  plan,path=self.plan();self.assertEqual(len(plan['entries']),41);self.assertEqual(plan['nonempty_shards'],2);self.assertEqual(sum(plan['lane_rows']),4106);self.assertEqual({x['core']for x in plan['entries']if x['rows']},{4,6});self.assertEqual(plan['duplicate_geometry_groups'],1);self.assertEqual(plan['duplicate_geometry_rows'],2);self.assertFalse(plan['target_or_gold_used']);self.assertTrue(path.is_file())
 def test_command_binder_embeds_exact_plan_hash_and_lane_cores(self):
  plan,path=self.plan();output=self.root/'bound-commands.json';argv=['bind_commands.py','--plan',str(path),'--output-root',str(self.root/'render'),'--output',str(output)]
  with mock.patch.object(sys,'argv',argv):bind_commands.main()
  bound=json.loads(output.read_text());self.assertEqual(bound['plan_sha256'],h(path));self.assertEqual([x['core']for x in bound['lanes']],[4,6]);self.assertTrue(all(h(path) in x['command']for x in bound['lanes']));self.assertFalse(bound['execution_authorized'])
 def test_tampered_duplicate_geometry_evidence_is_rejected(self):
  dup=self.inputs/'duplicate-geometries.jsonl';row=json.loads(dup.read_text());row['geometry']['path']='R/other.R';dump(dup,[row]);manifest=json.loads((self.inputs/'manifest.json').read_text());manifest['duplicate_geometry']['sha256']=h(dup);js(self.inputs/'manifest.json',manifest)
  with mock.patch.object(build_plan,'COVERAGE_SHA',h(self.coverage)),mock.patch.object(build_plan,'CANDIDATES_SHA',h(self.candidates)):
   with self.assertRaisesRegex(ValueError,'member mismatch'):build_plan.build(self.inputs,self.coverage,self.candidates,self.root/'bad-plan.json')
 def test_lane_runner_invokes_only_planned_shards_and_checks_terminal(self):
  source=self.inputs/'shard-0000.jsonl';plan={'schema':'sepalith.dat10.noop4106.provider-run-plan.v1','status':'prepared_no_launch','phase':'render16','context_size':16384,'generation_reserve':2048,'input_root':str(self.inputs),'provider_source_manifest_sha256':run_lane.SOURCE_SHA,'run_shard_sha256':run_lane.RUN_SHA,'provider_runtime_sha256':run_lane.RUNTIME_FILES,'tokenizer_sha256':run_lane.TOKENIZER_SHA,'entries':[{'shard':0,'lane':0,'core':4,'path':source.name,'sha256':h(source),'rows':2053},{'shard':1,'lane':1,'core':6,'path':'shard-0001.jsonl','sha256':h(self.inputs/'shard-0001.jsonl'),'rows':2053}]};plan_path=self.root/'runner-plan.json';js(plan_path,plan);output=self.root/'render16'
  def fake(command,env):
   shard=int(command[2]);core=int(command[3]);context=int(command[4]);reserve=int(command[5]);out=Path(command[-1]);out.mkdir(parents=True,exist_ok=True);result=out/f'shard-{shard:04d}.jsonl';dump(result,[{'row_id':f'x-{i}'}for i in range(2053)]);js(out/f'shard-{shard:04d}.terminal.json',{'status':'complete','shard':shard,'core':core,'context_size':context,'generation_reserve':reserve,'input':{'sha256':h(source)},'output':{'rows':2053,'sha256':h(result)}});return mock.Mock(returncode=0)
  argv=['run_lane.py','--plan',str(plan_path),'--plan-sha256',h(plan_path),'--lane','0','--output',str(output)]
  with mock.patch.object(sys,'argv',argv),mock.patch.object(run_lane.subprocess,'run',side_effect=fake) as call:run_lane.main()
  self.assertEqual(call.call_count,1);self.assertEqual(json.loads((output/'lane-0.terminal.json').read_text())['shards'],[0])
 def test_16k_to_32k_fallback_preserves_all_4106_and_121_upstream_holds(self):
  plan,plan_path=self.plan();r16=self.root/'render16';r16.mkdir();held={'supported-0000','supported-0001','supported-2053'}
  for item in plan['entries']:
   if not item['rows']:continue
   src=[json.loads(x)for x in (self.inputs/item['path']).read_text().splitlines()];result=[{'row_id':x['row_id'],'status':'hold' if x['row_id']in held else'supported'}for x in src];out=r16/f"shard-{item['shard']:04d}.jsonl";dump(out,result);js(r16/f"shard-{item['shard']:04d}.terminal.json",{'status':'complete','context_size':16384,'generation_reserve':2048,'input':{'sha256':item['sha256']},'output':{'sha256':h(out),'rows':item['rows']}})
  fallback=self.root/'fallback';argv=['prepare_fallback.py','--plan',str(plan_path),'--plan-sha256',h(plan_path),'--render16',str(r16),'--output',str(fallback)]
  with mock.patch.object(sys,'argv',argv):prepare_fallback.main()
  fm=json.loads((fallback/'manifest.json').read_text());self.assertEqual(fm['rows'],3)
  r32=self.root/'render32';r32.mkdir()
  for item in fm['entries']:
   if not item['rows']:continue
   src=[json.loads(x)for x in (fallback/item['path']).read_text().splitlines()];result=[{'row_id':x['row_id'],'status':'hold' if x['row_id']=='supported-2053' else'supported'}for x in src];out=r32/f"shard-{item['shard']:04d}.jsonl";dump(out,result);js(r32/f"shard-{item['shard']:04d}.terminal.json",{'status':'complete','context_size':32768,'generation_reserve':2048,'input':{'sha256':item['sha256']},'output':{'sha256':h(out),'rows':item['rows']}})
  final=self.root/'final';argv=['finalize_policy.py','--plan16',str(plan_path),'--fallback',str(fallback),'--render16',str(r16),'--render32',str(r32),'--output',str(final)]
  with mock.patch.object(sys,'argv',argv):finalize_policy.main()
  result=json.loads((final/'manifest.json').read_text());self.assertTrue(result['exact_accounting']);self.assertEqual(result['provider_supported'],4105);self.assertEqual(result['provider_holds_after_32k'],1);self.assertFalse(result['target_truncated']);self.assertEqual(result['selected']['rows'],4106)

if __name__=='__main__':unittest.main(verbosity=2)
