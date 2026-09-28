import ast,copy,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'source'))
import edit_context_profile as profile
class Contract(unittest.TestCase):
 def test_all_actual_fixtures_are_complete_target_only_without_truncation(self):
  manifest=json.loads((P/'fixture-manifest.json').read_text());self.assertEqual([x['cap']for x in manifest['rows']],[2048,4096,8192,16384,32768])
  for item in manifest['rows']:
   row,start=profile.validate_fixture(item['path'],item['sha256'],item['cap']);self.assertEqual(len(row['input_ids']),item['sequence_tokens']);self.assertEqual(len(row['input_ids'])-start,item['supervised_tokens'])
  self.assertEqual(manifest['review_pool_rows'],20191);self.assertEqual(manifest['pending_provider_inputs'],10948);self.assertFalse(manifest['pending_provider_training_admission'])
 def test_truncation_missing_eos_and_prompt_supervision_fail(self):
  source=json.loads((P/'fixtures/edit-2048.jsonl').read_text())
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   q=Path(td)/'row.jsonl';bad=copy.deepcopy(source);bad['input_ids'].append(7);bad['labels'].append(7);bad['attention_mask'].append(1);q.write_text(json.dumps(bad)+'\n')
   with self.assertRaisesRegex(ValueError,'token arrays'):profile.validate_fixture(q,hashlib.sha256(q.read_bytes()).hexdigest(),2048)
   bad=copy.deepcopy(source);bad['input_ids'][-1]=7;q.write_text(json.dumps(bad)+'\n')
   with self.assertRaisesRegex(ValueError,'BOS/EOS'):profile.validate_fixture(q,hashlib.sha256(q.read_bytes()).hexdigest(),2048)
   bad=copy.deepcopy(source);bad['labels'][0]=bad['input_ids'][0];q.write_text(json.dumps(bad)+'\n')
   with self.assertRaisesRegex(ValueError,'target-only'):profile.validate_fixture(q,hashlib.sha256(q.read_bytes()).hexdigest(),2048)
 def test_runtime_has_full_state_no_update_and_rng_restore_gates(self):
  tree=ast.parse((P/'source/edit_context_profile.py').read_text());calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)];attrs=[n.func.attr for n in calls if isinstance(n.func,ast.Attribute)]
  self.assertIn('backward',attrs);self.assertNotIn('step',attrs);source=(P/'source/edit_context_profile.py').read_text();self.assertIn("optimizer.load_state_dict(state)",source);self.assertIn("torch.cuda.set_rng_state_all",source);self.assertIn("parameters_unchanged",source);self.assertIn("EXPECTED_TENSORS=381",source);self.assertIn("load_in_4bit=False",source);self.assertIn("use_gradient_checkpointing=True",source)
 def test_real_source_manifest_and_byte_tamper(self):
  audit=profile.verify_source(P/'source-manifest.json');self.assertEqual(audit['files'],7)
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   root=Path(td);(root/'x.py').write_text('one\n');manifest={'schema':'sepalith.sft11.full-weight-edit-context-profile-source.v1','files':[{'path':'x.py','bytes':4,'sha256':hashlib.sha256(b'one\n').hexdigest()}]};mp=root/'source-manifest.json';mp.write_text(json.dumps(manifest));self.assertEqual(profile.verify_source(mp)['files'],1);(root/'x.py').write_text('two\n')
   with self.assertRaisesRegex(ValueError,'source file identity differs'):profile.verify_source(mp)
 def test_admission_rejects_wrong_checkpoint_fixture_or_source(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   root=Path(td);cp=root/'checkpoint-1';cp.mkdir();(cp/'campaign-manifest.json').write_text('{}\n');fixture=P/'fixtures/edit-2048.jsonl';source=P/'profile-admission.template.json';ad=root/'ad.json';value={'schema':'sepalith.sft11.full-weight-edit-context-profile-admission.v1','status':'admitted','launch_authorized':True,'cap':2048,'checkpoint':str(cp),'checkpoint_manifest_sha256':profile.sha(cp/'campaign-manifest.json'),'fixture':str(fixture.resolve()),'fixture_sha256':profile.sha(fixture),'source_manifest_sha256':profile.sha(source)};ad.write_text(json.dumps(value));self.assertIn('sha256',profile.load_admission(ad,cp,fixture,2048,source));value['cap']=4096;ad.write_text(json.dumps(value))
   with self.assertRaisesRegex(ValueError,'paths differ'):profile.load_admission(ad,cp,fixture,2048,source)
if __name__=='__main__':unittest.main(verbosity=2)
