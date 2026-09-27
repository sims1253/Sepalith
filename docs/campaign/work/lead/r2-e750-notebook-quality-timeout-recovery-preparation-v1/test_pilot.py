import hashlib,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parent
class PilotTests(unittest.TestCase):
 def test_failed_row_is_exact_original_and_max_prompt(self):
  got=json.loads((P/'failed-row.json').read_text());rows=[json.loads(x) for x in (P.parent/'r2-e750-notebook-cpu-quality-preparation-v1/prepared-inputs/dev75-tokenized.jsonl').read_text().splitlines()]
  want=next(x for x in rows if x['id']=='dat07-existing-719cd49683667d0fb86fb2fa')
  self.assertEqual(got,want);self.assertEqual(len(got['prompt_ids']),2619)
 def test_thread_arms_cover_physical_cores_first(self):
  x=json.loads((P/'packet.json').read_text());self.assertEqual(x['arms'],[{'threads':4,'cpu_list':'0,2,4,6'},{'threads':6,'cpu_list':'0,2,4,6,8,10'},{'threads':8,'cpu_list':'0,2,4,6,8,10,1,3'}])
  self.assertEqual(x['cap'],192);self.assertEqual(x['case_deadline_seconds'],300);self.assertFalse(x['production_latency_evidence'])
 def test_runner_keeps_identity_and_cleanup_contract(self):
  s=(P/'run_failed_row_pilot.py').read_text()
  for v in ("transport.COMPLETION_CAP=CAP","'-ngl','0'","common.statid(base['model']['path'])","common.cleanup(proc,tick)","result=transport._run_case","CASE_DEADLINE=300"):
   self.assertIn(v,s)
 def test_admission_is_not_preissued(self):
  x=json.loads((P/'root-admission.template.json').read_text());self.assertEqual(x['status'],'ROOT_MUST_ADMIT');self.assertFalse(x['source_reviewed']);self.assertIsNone(x['run_id'])
if __name__=='__main__':unittest.main()
