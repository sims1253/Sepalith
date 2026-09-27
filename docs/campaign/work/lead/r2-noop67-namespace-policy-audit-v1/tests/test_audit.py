import hashlib,importlib.util,json,pathlib,subprocess,tempfile,unittest
P=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class T(unittest.TestCase):
 def test_exact_67_classification(self):
  x=json.loads((P/'classification.json').read_text());self.assertEqual(x['rows'],67);self.assertEqual(x['counts'],{'semantic_unavailable_ambiguous_explicit_origin':5,'semantic_unavailable_conditional':27,'semantic_unavailable_malformed_directive':19,'semantic_unavailable_syntax_or_encoding_parse_error':16})
 def test_snapshots_are_exact(self):
  request=json.loads((P/'namespace-request.json').read_text())['namespaces'];out=[json.loads(x) for x in (P/'namespace-parse-output.jsonl').read_text().splitlines() if x]
  self.assertEqual(len(request),37);self.assertEqual(len(out),37)
  for a,b in zip(request,out):self.assertEqual(a['expected_sha256'],b['parsed_sha256']);self.assertTrue(b['stat_stable'])
 def test_fixture_hashes(self):
  m=json.loads((P/'fixtures/manifest.json').read_text())
  for x in m['namespace_examples'].values():self.assertEqual(sha(P/x['fixture_path']),x['source_sha256'])
 def test_no_target_fields_drive_inputs(self):
  xs=[json.loads(x) for x in (P/'fixtures/noop-train-inputs.jsonl').read_text().splitlines()]
  forbidden={'target','target_text','gold','label','no_op'}
  self.assertTrue(all(not(forbidden&set(x)) for x in xs))
 def test_all_entrypoints_compile_or_import(self):
  with tempfile.TemporaryDirectory(dir='/mnt/e/sepalith/campaign-20260915/tmp') as d:
   for i,name in enumerate(('prepare67_inputs.py','run_policy_ladder.py','tests/test_audit.py')): subprocess.run(['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B','-c',f'import py_compile;py_compile.compile({str(P/name)!r},cfile={str(pathlib.Path(d)/str(i))!r},doraise=True)'],check=True)
 def test_ladder_join_is_id_exact(self):
  spec=importlib.util.spec_from_file_location('ladder',P/'run_policy_ladder.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  supported,held=m.split_stage([{'row_id':'a'},{'row_id':'b'}],[{'row_id':'b','status':'hold'},{'row_id':'a','status':'supported'}]);self.assertEqual([x['row_id'] for x in supported],['a']);self.assertEqual([x['row_id'] for x in held],['b'])
  with self.assertRaisesRegex(ValueError,'ids'):m.split_stage([{'row_id':'a'}],[{'row_id':'x','status':'supported'}])
if __name__=='__main__':unittest.main()
