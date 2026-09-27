import importlib.util,hashlib,json,subprocess,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).parent
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,HERE/file);value=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(value);return value
m=module('semantic','analyze_semantics.py');q=module('queue_runner','run_semantic_queue.py')
class Tests(unittest.TestCase):
 def test_target_name_and_crlf_are_exact(self):
  target="#' @param x value\n";self.assertEqual(m.target_name(target+'f <- function(x) x\n',target),'f');self.assertIsNone(m.target_name(target+'x <- 1\n',target))
  self.assertEqual(m.occurrence(b"#' x\r\nf <- function() 1\r\n",b"#' x\nf <- function() 1\n")[:2],(1,'uniform_crlf_to_lf'))
  self.assertEqual(m.occurrence(b"#'  x\n",b"#' x\n")[0],0)
 def test_ellipsis_is_a_documented_formal(self):
  self.assertEqual(m.doc_params("#' @param x,... values\n"),{'x','...'})
  item={'target_occurrences':1,'target_definition_name':'f','documented_params':['x'],'imported_symbols':[],'prior_reviewed_recovery':False}
  scope=self.scope();scope['function_formals']=['x','...'];reasons,_,_=m.decide(item,scope,'f <- function(x, ...) x\n');self.assertIn('undocumented_function_formal:...',reasons)
 def scope(self):
  return {'status':'scope_inventory_complete','target_definition_span':[3,3],'function_formals':['x'],'variable_references':[],'call_heads':['helperA'],'base_bound':[],'top_level_definitions':['helperA','helperB','f'],'top_level_definition_spans':{'helperA':[1,1],'helperB':[2,2],'f':[3,3]},'top_level_definition_counts':{'helperA':1,'helperB':1,'f':1},'function_dependencies':{'helperA':{'codetools_ok':True,'variables':[],'call_heads':['helperB'],'base_bound':[]},'helperB':{'codetools_ok':True,'variables':['threshold'],'call_heads':[],'base_bound':[]}}}
 def test_recursive_helper_dependency_and_unresolved_default_hold(self):
  item={'target_occurrences':1,'target_definition_name':'f','documented_params':['x'],'imported_symbols':[],'prior_reviewed_recovery':False};source='helperA <- function(x) helperB(x)\nhelperB <- function(x=threshold) x\nf <- function(x) helperA(x)\n'
  reasons,resolution,closure=m.decide(item,self.scope(),source)
  self.assertIn('unresolved_global_or_nse_requires_occurrence_evidence:threshold',reasons);self.assertEqual([x['name'] for x in closure['required_helper_spans']],['helperA','helperB']);self.assertIn('threshold',resolution['visited_dependency_names'])
 def test_missing_or_false_target_span_holds(self):
  item={'target_occurrences':1,'target_definition_name':'f','documented_params':['x'],'imported_symbols':[],'prior_reviewed_recovery':False};s=self.scope();s['call_heads']=[];s['target_definition_span']=None
  self.assertIn('target_definition_span_missing_or_not_source_backed',m.decide(item,s,'f <- function(x) x\n')[0])
 def test_r_reads_exact_temp_bytes_and_emits_real_spans(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);src=d/'x.R';src.write_text('helper <- function(z) z\nf <- function(x, y=threshold) helper(x) + y\n');inp=d/'in.json';out=d/'out.jsonl';inp.write_text(json.dumps({'source_groups':[{'source_path':str(src),'source_sha256':m.sha(src),'rows':[{'row_id':'r','target_definition_name':'f'}]}]}))
   done=subprocess.run(['Rscript','--vanilla',str(HERE/'semantic_scope.R'),str(inp),str(out)],capture_output=True,text=True,timeout=30);self.assertEqual(done.returncode,0,done.stderr);row=json.loads(out.read_text())
   self.assertEqual(row['parsed_source_sha256'],m.sha(src));self.assertEqual(row['target_definition_span'],[2,2]);self.assertEqual(row['top_level_definition_spans']['helper'],[1,1]);self.assertIn('threshold',row['function_dependencies']['f']['variables'])
 def test_wrong_packet_pin_fails_before_semantics(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);ledger=d/'ledger';ledger.write_text('');packet=d/'packets';packet.write_text('')
   done=subprocess.run(['python3',str(HERE/'analyze_semantics.py'),'--provenance-ledger',str(ledger),'--candidate-packets',str(packet),'--expected-candidate-packets-sha256','0'*64,'--output',str(d/'out')],capture_output=True,text=True)
   self.assertNotEqual(done.returncode,0);self.assertIn('differs from provenance binding',done.stderr)
 def test_resume_binds_analyzer_and_helper_hash(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);ledger=d/'ledger';ledger.write_text('{}\n');manifest={'status':'complete_review_only','inputs':{'provenance_ledger':{'sha256':'p'},'candidate_packets':[{'sha256':'c'}]},'code':{'analyzer_sha256':'a','scope_helper_sha256':'h'},'exact_id_closure':True,'output':{'path':str(ledger),'bytes':ledger.stat().st_size,'sha256':q.sha(ledger)}};(d/'manifest.json').write_text(json.dumps(manifest))
   self.assertTrue(q.reusable(d,'p','c','a','h'));self.assertFalse(q.reusable(d,'p','c','changed','h'));self.assertFalse(q.reusable(d,'p','c','a','changed'))
 def test_namespace_unstable_is_not_accepted(self):
  self.assertFalse(m.namespace_ok({'stat_stable':False}));self.assertFalse(m.namespace_ok({}));self.assertTrue(m.namespace_ok({'stat_stable':True}))
if __name__=='__main__':unittest.main()
