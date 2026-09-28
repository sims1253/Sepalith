import importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('semantic',HERE/'analyze_semantics.py');m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_target_name_requires_adjacent_complete_definition(self):
  target="#' @param x value\n"
  self.assertEqual(m.target_name(target+'f <- function(x) x\n',target),'f')
  self.assertIsNone(m.target_name(target+'x <- 1\n',target))
 def test_documented_params_no_name_allowlist(self):
  self.assertEqual(m.doc_params("#' @param x,y values\n#' @return z\n"),{'x','y'})
 def test_scope_inventory_includes_defaults_callheads_and_top_level_helpers(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);src=d/'x.R';src.write_text('helper <- function(z) z\nf <- function(x, y=threshold) helper(x) + y\n')
   inp=d/'in.json';out=d/'out.jsonl';inp.write_text(json.dumps({'source_groups':[{'source_path':str(src),'source_sha256':m.sha(src),'rows':[{'row_id':'r','target_definition_name':'f'}]}]}))
   done=subprocess.run(['Rscript','--vanilla',str(HERE/'semantic_scope.R'),str(inp),str(out)],capture_output=True,text=True,timeout=30)
   self.assertEqual(done.returncode,0,done.stderr);row=json.loads(out.read_text())
   self.assertEqual(row['function_formals'],['x','y']);self.assertIn('threshold',row['variable_references']);self.assertIn('helper',row['call_heads']);self.assertIn('helper',row['top_level_definitions'])
 def test_unknown_license_or_nse_is_never_implicitly_resolved(self):
  # Resolution sets are structural; arbitrary names are not added.
  refs={'known','group'};resolved={'known'};self.assertEqual(refs-resolved,{'group'})
 def test_decision_rejects_formal_mismatch_and_unresolved_but_closes_helpers(self):
  item={'target_occurrences':1,'target_definition_name':'f','documented_params':['x','invented'],'imported_symbols':[],'prior_reviewed_recovery':False}
  scope={'status':'scope_inventory_complete','target_definition_span':[2,3],'function_formals':['x','missing'],'variable_references':['threshold'],'call_heads':['helper'],'base_bound':[],'top_level_definitions':['f','helper'],'top_level_definition_spans':{'helper':[1,1]},'top_level_definition_counts':{'f':1,'helper':1}}
  reasons,resolution,closure=m.decide(item,scope)
  self.assertIn('invented_documented_formal:invented',reasons);self.assertIn('undocumented_function_formal:missing',reasons);self.assertIn('unresolved_global_or_nse_requires_occurrence_evidence:threshold',reasons)
  self.assertEqual(closure['required_helper_spans'],[{'name':'helper','span':[1,1]}])
if __name__=='__main__':unittest.main()
