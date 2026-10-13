import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path

ROOT=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PACKET=ROOT/'docs/campaign/work/lead/r2-cloud-cpt1902-merge-review-v1'
EXTRACTED=Path('/mnt/e/sepalith/campaign-20260915/cloud-readback/CPT-v4-tar1902-full-v1/extracted')
EVIDENCE=Path('/mnt/e/sepalith/campaign-20260915/cloud-readback/CPT-v4-tar1902-full-v1/readback-evidence.json')
MERGE=ROOT/'docs/campaign/work/lead/r2-cloud-cpt-parent-merge-preparation-v1/source/merge_cloud_cpt_parent.py'
SPEC=importlib.util.spec_from_file_location('binder',PACKET/'prepare_binding_v2.py');binder=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(binder)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

class ReviewTests(unittest.TestCase):
 def test_wrapper_preserves_original_root_evidence_and_runtime_shape(self):
  w=json.loads((PACKET/'readback-acceptance-wrapper.json').read_text())
  self.assertEqual(w['acceptance']['status'],'pass')
  self.assertEqual(Path(w['artifacts']['extracted_directory']),EXTRACTED)
  for key in ('readback_evidence','root_durable_readback_receipt'):
   rec=w['artifacts'][key];self.assertEqual(sha(rec['path']),rec['sha256']);self.assertEqual(Path(rec['path']).stat().st_size,rec['bytes'])
  self.assertTrue(w['preservation']['original_receipt_immutable']);self.assertFalse(w['preservation']['checkpoint_files_rewritten'])
 def test_registry_exact_terminal_1902_identity(self):
  r=json.loads((PACKET/'candidates-1902.json').read_text());c=r['candidates']['1902'];e=json.loads(EVIDENCE.read_text())
  self.assertEqual((c['step'],c['cursor'],c['full'],c['status']),(1902,30432,True,'verified_readback'))
  self.assertEqual(c['campaign_manifest_sha256'],e['verified_files']['campaign-manifest.json']['sha256'])
  self.assertEqual(c['campaign_state_sha256'],e['verified_files']['campaign-state.json']['sha256'])
  self.assertEqual(c['adapter_config_sha256'],e['verified_files']['adapter_config.json']['sha256'])
  self.assertEqual(c['adapter_weights_sha256'],e['verified_files']['adapter_model.safetensors']['sha256'])
  self.assertEqual(c['source_identity'],e['identities']['source_identity']);self.assertEqual(c['draw_schedule_sha256'],e['identities']['schedule_sha256'])
  self.assertEqual(c['train_rows_sha256'],e['identities']['data_core']['train_rows_sha256'])
  self.assertEqual(r['root_evidence']['tar_sha256'],e['download']['sha256']);self.assertEqual(r['root_evidence']['hub_revision'],e['hub_revision'])
  self.assertEqual(sha(c['verified_readback_receipt']['path']),c['verified_readback_receipt']['sha256'])
 def test_all_expected_extracted_paths_and_sizes_exist_without_payload_hashing(self):
  e=json.loads(EVIDENCE.read_text());self.assertEqual(set(e['verified_files']),{p.name for p in EXTRACTED.iterdir() if p.is_file()})
  for name,rec in e['verified_files'].items():self.assertTrue((EXTRACTED/name).is_file());self.assertEqual((EXTRACTED/name).stat().st_size,rec['bytes'])
 def test_fixed_binder_binds_actual_registry_and_merge_source(self):
  registry=PACKET/'candidates-1902.json';expected=json.loads((PACKET/'candidate-1902-binding-review.json').read_text())
  actual=binder.review(json.loads(registry.read_text()),1902,registry,MERGE)
  self.assertEqual(actual,expected);self.assertEqual(actual['candidate_registry_sha256'],sha(registry));self.assertEqual(actual['merge_source_sha256'],sha(MERGE))
  self.assertEqual(actual['selected_checkpoint_step'],1902);self.assertTrue(actual['candidate_selection_pending']);self.assertEqual(actual['status'],'awaiting_root_admission')
 def test_no_authorization_or_binding_is_present(self):
  self.assertFalse(any(PACKET.glob('*admission*.json')));self.assertFalse(any(PACKET.glob('*binding.json')))

if __name__=='__main__':unittest.main(verbosity=2)
