import copy, hashlib, json, subprocess, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BIND=ROOT/'binding.candidate.json'
class TestPreparation(unittest.TestCase):
 def test_binding_invariants(self):
  d=json.loads(BIND.read_text()); self.assertEqual(d['selection'],{'primary':'E750_Q8','rollback':'b4_Q8','authority':'root_only_after_all_release_gates'})
  self.assertEqual([x['max_output_tokens'] for x in d['output_cap_candidates']],[192,384,768])
  self.assertEqual([x['existing_accepted_vsix_compatible'] for x in d['output_cap_candidates']],[True,False,False])
  self.assertEqual(d['candidate']['artifacts']['model']['sha256'],'9b11c5275202b83c6e6299890582fc55526113586c4ae3f8daf3bc0ac1ab54bd')
  self.assertFalse(d['release_constraints']['sealed_final_access'])
 def test_context_budget_is_explicit(self):
  d=json.loads(BIND.read_text())
  for x in d['output_cap_candidates']:
   self.assertEqual(x['maximum_prompt_tokens_including_manual_bos']+x['max_output_tokens'],4096)
 def test_routes_are_exact(self):
  d=json.loads(BIND.read_text()); self.assertEqual(d['candidate']['editor_route']['requests'],['POST /tokenize','POST /completion'])
  self.assertEqual(d['candidate']['editor_route']['completion_prompt_type'],'integer_token_ids_with_manual_bos0')
 def test_commands_cannot_launch(self):
  c=json.loads((ROOT/'commands.json').read_text()); self.assertFalse(c['execution_authorized'])
  self.assertNotIn('REL-02',json.dumps(c)); self.assertNotIn('final-results',json.dumps(c).lower())
  self.assertEqual(sorted(c['cap_arm_templates']),['192','384','768'])
 def test_b4_is_isolated(self):
  d=json.loads(BIND.read_text()); r=d['rollback']
  self.assertEqual(r['port'],18403); self.assertIn('sepalith-b4-rollback',r['dedicated_user_data']); self.assertNotEqual(r['artifacts']['vsix']['sha256'],d['candidate']['artifacts']['vsix']['sha256'])
 def test_larger_caps_fail_closed_on_editor(self):
  d=json.loads(BIND.read_text())
  for cap in (384,768):
   x=next(y for y in d['output_cap_candidates'] if y['max_output_tokens']==cap)
   self.assertIn('fresh_reviewed_vsix',x['required_before_editor_check'])
 def test_packet_self_hash_excludes_manifest(self):
  m=json.loads((ROOT/'source-manifest.json').read_text())
  for x in m['files']:
   p=ROOT/x['path']; self.assertEqual(p.stat().st_size,x['bytes']); self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),x['sha256'])
if __name__=='__main__': unittest.main()
