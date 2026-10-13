import copy, importlib.util, json, pathlib, tempfile, unittest
P=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v',P/'verify_registry.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class T(unittest.TestCase):
 def pair(self):
  pre='a'*64; rid='r1'; inp={'row_id':rid,'preedit_sha256':pre,'cursor':{'line':3,'character':2},'path':'R/x.R'}
  ctx={'schema_version':'sepalith.prompt.prm03.v1','path':'R/x.R','prefix':['a'],'region_old':[],'suffix_lines':['b'],'history':[{'x':1}],'cursor':{'region_line_index':-1},'replacement_range':{'content_sha256':pre,'start':{'line':3,'character':2},'end':{'line':3,'character':2},'uri':'file:///workspace/R/x.R'}}
  out={'row_id':rid,'selected_context':ctx}; prov={'row_id':rid,'preedit_sha256':pre,'source_identity':{'row_id':rid,'source_sha256':'b'*64,'source_path':'/source/R/x.R'}}
  return inp,out,prov
 def test_identity_layers_preserved(self):
  got=v.validate_pair(*self.pair()); self.assertNotEqual(got['original_source_sha256'],got['current_preedit_sha256']); self.assertEqual(got['selected_window']['history_items'],1)
 def test_cursor_mismatch_rejected(self):
  i,o,p=self.pair();o['selected_context']['replacement_range']['start']['line']=4
  with self.assertRaisesRegex(ValueError,'cursor'):v.validate_pair(i,o,p)
 def test_preedit_mismatch_rejected(self):
  i,o,p=self.pair();p['preedit_sha256']='c'*64
  with self.assertRaisesRegex(ValueError,'preedit'):v.validate_pair(i,o,p)
 def test_mode_selects_legacy_context(self):
  i,o,p=self.pair();ctx=o.pop('selected_context');o['full_context']=ctx;o['bounded_context']={**ctx,'prefix':[]};p['mode']='full_document';self.assertEqual(v.selected(o,p)['prefix'],['a']);p['mode']='complete_span';self.assertEqual(v.selected(o,p)['prefix'],[])
 def test_target_fields_are_not_required(self):
  i,o,p=self.pair(); self.assertNotIn('target',json.dumps([i,o,p])); v.validate_pair(i,o,p)
 def test_registry_static_accounting(self):
  x=json.loads((P/'registry.json').read_text());self.assertEqual(sum(c['candidate_rows'] for c in x['cohorts']),5185);self.assertEqual([c['candidate_rows'] for c in x['cohorts']],[616,133,4435,1])
 def test_every_entrypoint_compiles(self):
  import py_compile;
  with tempfile.TemporaryDirectory(dir='/mnt/e/sepalith/campaign-20260915/tmp') as d: py_compile.compile(str(P/'verify_registry.py'),cfile=str(pathlib.Path(d)/'verify.pyc'),doraise=True)
if __name__=='__main__':unittest.main()
