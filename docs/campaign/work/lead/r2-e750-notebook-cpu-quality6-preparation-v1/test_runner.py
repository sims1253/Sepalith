import importlib.util,json,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parent
import sys;sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('suite',P/'run_suite.py');suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
class RunnerTest(unittest.TestCase):
 def rows(self):return [json.loads(x) for x in (P/'prepared-inputs/dev75-tokenized.jsonl').read_text().splitlines()]
 def fake(self,incomplete_at=None):
  count={'n':0}
  def run(endpoint,case,protocol,tokenizer,deadline,before_request=None):
   count['n']+=1
   if before_request:before_request(None)
   incomplete=count['n']==incomplete_at
   return {'id':case['id'],'family':case['family'],'response_complete':not incomplete,'status':'failed' if incomplete else 'accepted','failure_class':'transport' if incomplete else None,'quality':{'protocol_valid':not incomplete,'edit_exact':not case['expected_noop'] and not incomplete,'strict_noop_correct':case['expected_noop'] and not incomplete,'noop_false_positive':False},'cap':{'hit':False,'limit':suite.transport.COMPLETION_CAP},'eos':{'status':'not_observed' if incomplete else 'canonical_eos'}}
  return run
 def test_all_caps_preserve_same_75_prompts(self):
  rows=self.rows();old=suite.transport._run_case
  try:
   suite.transport._run_case=self.fake()
   for cap in suite.CAPS:
    with tempfile.TemporaryDirectory() as td:
     out=Path(td)/'result.json';r=suite.evaluate('stub',cap,rows,object(),lambda _:None,out)
     self.assertEqual(r['status'],'complete');self.assertEqual(r['denominators']['attempted'],75);self.assertEqual(r['expected_ids'],[x['id'] for x in rows]);self.assertEqual(r['counts']['canonical_eos'],75);self.assertEqual({x['prompt_ids_sha256'] for x in rows},{x['prompt_ids_sha256'] for x in self.rows()})
  finally:suite.transport._run_case=old
 def test_incomplete_transport_preserves_denominator_and_stops(self):
  rows=self.rows();old=suite.transport._run_case
  try:
   suite.transport._run_case=self.fake(3)
   with tempfile.TemporaryDirectory() as td:
    r=suite.evaluate('stub',192,rows,object(),lambda _:None,Path(td)/'r.json')
    self.assertEqual(r['status'],'incomplete_preserved');self.assertEqual(r['denominators']['attempted'],3);self.assertEqual(len(r['unattempted_ids']),72);self.assertEqual(r['counts']['transport_failed'],1)
  finally:suite.transport._run_case=old
 def test_all_actual_prompts_fit_largest_cap(self):
  rows=self.rows();self.assertEqual(max(len(x['prompt_ids']) for x in rows),2619);self.assertTrue(all(len(x['prompt_ids'])+768<=4096 for x in rows))
 def test_cap_termination_accounting_fields(self):
  r=suite.summary(384,self.rows(),[],'partial');self.assertEqual(set(['canonical_eos','noncanonical_eog','cap_hit','transport_failed','mechanical_failed']).issubset(r['counts']),True)
 def test_six_core_deadline_and_fresh_namespace(self):
  source=(P/'run_suite.py').read_text();packet=json.loads((P/'packet.json').read_text())
  self.assertEqual(packet['cpu_list'],[0,2,4,6,8,10]);self.assertEqual(packet['maximum_threads'],6);self.assertEqual(packet['offline_case_deadline_seconds'],300);self.assertEqual(packet['production_reference_case_deadline_seconds'],5)
  self.assertIn("'--cpu-list','0,2,4,6,8,10'",source);self.assertIn("'-t','6','-tb','6'",source);self.assertIn('tokenizer,300,before_request',source)
 def test_admission_and_watchdog_bindings(self):
  a=json.loads((P/'root-admission.template.json').read_text());w=(P/'watchdog_remote.py').read_text()
  self.assertEqual(a['status'],'ROOT_MUST_ADMIT');self.assertEqual(a['maximum_seconds'],28800);self.assertFalse(a['source_reviewed'])
  for v in ('MAX_SECONDS=28800','TERM_GRACE_SECONDS=150','arm_servers(run_root)','os.killpg(pid,signal.SIGTERM)'):self.assertIn(v,w)
if __name__=='__main__':unittest.main()
