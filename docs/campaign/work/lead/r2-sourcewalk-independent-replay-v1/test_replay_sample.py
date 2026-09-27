import importlib.util, json
from pathlib import Path
import unittest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('replay_sample',HERE/'replay_sample.py');m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m)

def evidence(family='no_op'):
 e={x:True for x in ('raw_line_hash_match','raw_source_file_hash_match','raw_metadata_match','global_train','protected_disjoint','normalized_source_hash_match','normalized_source_reconstruction_match','source_parse_ok','license_hash_match','license_fields_match','strict_protocol_ok')}
 e.update(family=family,reference_analyzer='complete_not_applicable_noop',noop_family_predicate='exact_unchanged_source_supported_kind')
 return e

class ReplayTest(unittest.TestCase):
 def test_noop_uses_family_specific_predicate(self):
  self.assertEqual(m.decide(evidence()),'provenance_supported_candidate_root_review_required')
  e=evidence();e['noop_family_predicate']='failed';self.assertEqual(m.decide(e),'hold_noop_family_predicate')
 def test_roxy_missing_analyzer_never_passes(self):
  e=evidence('roxygen_drafting');e['reference_analyzer']='pending_full_source_semantic_follow_on';e['unsupported_claim_analyzer']='pending_full_source_semantic_follow_on'
  self.assertEqual(m.decide(e),'provenance_pass_semantic_analyzer_queued')
 def test_any_identity_failure_holds(self):
  for key in ('raw_line_hash_match','global_train','protected_disjoint','license_hash_match','strict_protocol_ok'):
   e=evidence();e[key]=False;self.assertEqual(m.decide(e),'hold_independent_provenance_failure',key)
 def test_reconstruction_is_family_specific(self):
  raw={'prefix':['a'],'region_old':['b'],'region_new':['c'],'suffix':['d']}
  before,after=m.reconstructed_source(raw,'roxygen_drafting');self.assertEqual(before,b'a\nb\nd\n');self.assertEqual(after,b'a\nc\nd\n')
  self.assertEqual(m.reconstructed_source(raw,'no_op'),(before,before))
 def test_actual_sample_contract(self):
  p=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v1/attempt-06/manifest.json');o=json.loads(p.read_text())
  self.assertLessEqual(o['rows'],100);self.assertLessEqual(o['packages'],20);self.assertEqual(o['strict_validator']['sha256'],m.STRICT_SHA)
  self.assertEqual(sum(o['families'].values()),o['rows']);self.assertEqual(sum(o['status_counts'].values()),o['rows'])
  self.assertTrue(all(x['passed'] for x in o['negative_controls'].values()))

if __name__=='__main__':unittest.main()
