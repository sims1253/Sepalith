import hashlib,json,unittest
from pathlib import Path
ROOT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic-provider-integration-v1/final-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class OutputTests(unittest.TestCase):
 def test_manifest_and_rows(self):
  m=json.loads((ROOT/'manifest.json').read_text());self.assertEqual(m['candidate_rows'],133);self.assertEqual(m['excluded_rows'],0);self.assertFalse(m['training_admission'])
  for name,e in m['outputs'].items():
   self.assertEqual(sha(ROOT/name),e['sha256'])
   with (ROOT/name).open() as f:self.assertEqual(sum(1 for _ in f),e['rows'])
 def test_exact_ids_and_no_target_selection_fields(self):
  rows=[json.loads(x) for x in (ROOT/'candidate-provenance.jsonl').read_text().splitlines()];self.assertEqual(len({x['row_id'] for x in rows}),133);self.assertTrue(all(x['selection_target_or_gold_used'] is False and x['target_truncated'] is False for x in rows))
 def test_evidence_and_namespace_identity(self):
  rows=[json.loads(x) for x in (ROOT/'candidate-provenance.jsonl').read_text().splitlines()];self.assertTrue(all(x['selected_reference_count']>=1 and len(x['namespace_sha256'])==64 for x in rows))
 def test_ledger_conservation(self):
  rows=[json.loads(x) for x in (ROOT/'exclusion-ledger.jsonl').read_text().splitlines()];self.assertEqual(len(rows),133);self.assertEqual({x['status'] for x in rows},{'candidate'})
if __name__=='__main__':unittest.main(verbosity=2)
