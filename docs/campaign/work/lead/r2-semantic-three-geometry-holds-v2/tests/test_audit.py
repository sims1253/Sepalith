import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location('audit',ROOT/'audit_provider_geometry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SIDECAR=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic-three-geometry-holds-v1/review-01/training-sidecar.review-only.jsonl')
class TestAudit(unittest.TestCase):
 def test_raw_full_preedit_replays_and_is_target_free(self):
  records=m.rows(SIDECAR)
  for rid in m.IDS:
   probe=m.prepare(records[rid]);self.assertTrue(probe['full_raw_reapplication_exact']);self.assertTrue(probe['target_free']);self.assertIn(probe['expected_document_eol'],('mixed','crlf'))
 def test_actual_primary_path_rejects_all_three(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
   out=Path(td)/'out';subprocess.run([sys.executable,str(ROOT/'audit_provider_geometry.py'),'--sidecar',str(SIDECAR),'--output',str(out)],check=True,capture_output=True,text=True)
   got=subprocess.run(['/home/m0hawk/.bun/bin/bun',str(ROOT/'verify_primary_path.ts'),str(out/'provider-probes.review-only.jsonl'),str(SIDECAR)],check=True,capture_output=True,text=True)
   self.assertEqual(json.loads(got.stdout),{'rows':3,'mixed_eol_rejected':2,'uniform_preedit_raw_apply_mismatch':1,'protocol_holds':3})
   manifest=json.loads((out/'manifest.json').read_text());self.assertEqual((manifest['provider_predictions'],manifest['protocol_holds']),(0,3))
 def test_changed_source_or_ambiguous_target_cannot_be_substituted(self):
  records=m.rows(SIDECAR);x=json.loads(json.dumps(records[m.IDS[0]]));x['source']['sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'source_changed'):m.prepare(x)
if __name__=='__main__':unittest.main(verbosity=2)
