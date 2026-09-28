import hashlib,json,subprocess,unittest,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent
CAPS=(192,384,768)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
class PacketTest(unittest.TestCase):
 def test_accepted_inputs_unchanged(self):
  self.assertEqual(sha(P.parent/'extension-source-snapshots/41a509aaa6b2205add579e12db9afbdad8d4c40155baf798d4f263009989ebfc/manifest.json'),'eceaebd37be18f58a96ea62af4c91b70d37f9840a380f7b45030e8ebab028587')
  self.assertEqual(sha(P.parent/'r2-expanded-native-selection-e750-v1/source-manifest.json'),'7a83644819b88ec8c9565635946dcfdf7f29745209bb7de3e1e0b2ccab628712')
 def test_exact_arm_profiles(self):
  base=None
  for cap in CAPS:
   arm=P/f'arms/cap-{cap}';d=json.loads((arm/'native_evaluator/profile.json').read_text());self.assertEqual(d['output'],cap);self.assertEqual(d['model_profile']['maxOutputTokens'],cap);self.assertEqual(d['context'],4096);self.assertEqual(d['case_deadline_seconds'],5)
   self.assertEqual(d['model']['sha256'],'9b11c5275202b83c6e6299890582fc55526113586c4ae3f8daf3bc0ac1ab54bd')
   stripped=json.loads(json.dumps(d));stripped['output']=0;stripped['model_profile']['maxOutputTokens']=0
   if base is None:base=stripped
   self.assertEqual(stripped,base)
 def test_code_equal_across_arms(self):
  paths=[]
  for root in ('native_evaluator','native_controller'):
   paths += [p.relative_to(P/'arms/cap-192') for p in (P/'arms/cap-192'/root).glob('*.py')]
  for rel in paths:
   hashes={sha(P/f'arms/cap-{cap}'/rel) for cap in CAPS};self.assertEqual(len(hashes),1,rel)
 def test_root_templates_bind_cap_profile_and_inputs(self):
  for cap in CAPS:
   arm=P/f'arms/cap-{cap}';a=json.loads((arm/'root-admission.template.json').read_text());prof=json.loads((arm/'native_evaluator/profile.json').read_text())
   self.assertEqual(a['status'],'ROOT_MUST_ADMIT');self.assertEqual(a['output_cap'],cap);self.assertEqual(a['profile_sha256'],canon(prof))
   for path,digest in a['input_files'].items():self.assertEqual(sha(path),digest)
   self.assertEqual(a['data_files'][str(P.parent/'corrected-dev75-v1/dev75-corrected-finish-v1.jsonl')],'7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035')
 def test_prepared_closures_complete_but_unadmitted(self):
  for cap in CAPS:
   d=json.loads((P/f'arms/cap-{cap}/native_controller/source-closure.prepared.json').read_text());self.assertEqual(d['unresolved'],[]);self.assertEqual(d['status'],'preparation_only');self.assertGreater(len(d['files']),5000)
 def test_vsix_isolated_minimal(self):
  v=P/'vscode-sepalith-e750-cap-flex-v1-0.0.8.vsix';self.assertEqual(sha(v),'f634633a583dd4be3201880b4c64818ebaed3f4e5a9ceb417c9a7adfd9a1e778')
  with zipfile.ZipFile(v) as z:self.assertEqual(set(z.namelist()),{'extension.vsixmanifest','[Content_Types].xml','extension/package.json','extension/readme.md','extension/dist/extension.js'})
 def test_manifests_cap_only(self):
  profiles=[]
  for cap in CAPS:
   m=json.loads((P/f'manifests/e750-q8-cap-{cap}.manifest.json').read_text());self.assertEqual(m['modelProfile']['maxOutputTokens'],cap);self.assertEqual(m['modelProfile']['contextSize'],4096);profiles.append(m)
   self.assertEqual(m['model']['sha256'],m['modelProfile']['modelSha256'])
  for m in profiles:m['build']='x';m['modelProfile']['maxOutputTokens']=0
  self.assertEqual(profiles[0],profiles[1]);self.assertEqual(profiles[1],profiles[2])
 def test_no_execution_authorized(self):
  c=json.loads((P/'commands.json').read_text());self.assertFalse(c['execution_authorized']);self.assertTrue(c['no_final'])
if __name__=='__main__':unittest.main()
