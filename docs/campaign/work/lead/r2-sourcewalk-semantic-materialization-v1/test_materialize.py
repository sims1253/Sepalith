import copy,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('semantic_materializer',HERE/'materialize_semantic.py');m=importlib.util.module_from_spec(spec);sys.modules['semantic_materializer']=m;spec.loader.exec_module(m)

def fixture(root:Path,target_lines=None):
 if target_lines is None:target_lines=["#' @title Target docs","#' @param x input"]
 prefix=['helper_before <- function(z) {','  # helper comment must survive','  z + 1','}','']
 function=['target <- function(x) {','  helper_before(helper_after(x))','}']
 suffix=['','helper_after <- function(y) y * 2']
 lines=prefix+target_lines+function+suffix;source='\n'.join(lines)+'\n';source_path=root/'source.R';source_path.write_text(source)
 target_start=len(prefix);function_start=target_start+len(target_lines)+1;helper_after_start=function_start+len(function)+1
 scope={'status':'scope_inventory_complete','target_definition_span':[function_start,function_start+2],'function_formals':['x'],'variable_references':[],'call_heads':['helper_before','helper_after'],'base_bound':[],'top_level_definitions':['helper_before','helper_after','target'],'top_level_definition_spans':{'helper_before':[1,4],'target':[function_start,function_start+2],'helper_after':[helper_after_start,helper_after_start]},'top_level_definition_counts':{'helper_before':1,'target':1,'helper_after':1},'function_dependencies':{'helper_before':{'codetools_ok':True,'formals':['z'],'variables':[],'call_heads':['+','{'],'base_bound':['+','{']},'helper_after':{'codetools_ok':True,'formals':['y'],'variables':[],'call_heads':['*'],'base_bound':['*']},'target':{'codetools_ok':True,'formals':['x'],'variables':[],'call_heads':['helper_before','helper_after'],'base_bound':[]}},'parsed_source_sha256':hashlib.sha256(source.encode()).hexdigest()}
 item={'target_occurrences':1,'target_definition_name':'target','documented_params':['x'],'imported_symbols':[],'prior_reviewed_recovery':False}
 reasons,resolution,closure=m.SEMANTIC.decide(item,scope,source);assert reasons==[],reasons
 target=('\n'.join(target_lines)+'\n').encode();rid='row-semantic-fixture';source_sha=hashlib.sha256(source.encode()).hexdigest()
 semantic={'row_id':rid,'status':'semantic_supported_context_closure_root_review_required','reasons':[],'source_path':str(source_path),'source_sha256':source_sha,'parse_bytes_sha256':source_sha,'target_occurrences':1,'target_occurrence_method':'exact_bytes','target_definition_name':'target','documented_params':['x'],'imported_symbols':[],'namespace':{'stat_stable':True,'status':'namespace_inventory_complete'},'prior_reviewed_recovery':False,'target_sha256':hashlib.sha256(target).hexdigest(),'target_bytes':len(target),'scope':scope,'reference_resolution':resolution,'context_closure':closure}
 context={'schema_version':'sepalith.prompt.prm03.v1','path':'R/source.R','prefix':['obsolete window'],'selected_references':[],'history':[],'diagnostics':[],'retrieval':[],'scope_mode':'off','scope_lines':[],'suffix_lines':['obsolete suffix'],'region_old':[],'cursor':{'region_line_index':-1,'code_point_column':None,'utf16_column':None},'replacement_range':{'uri':'file:///sepalith/R/source.R','document_version':0,'content_sha256':'0'*64,'start':{'line':target_start,'character':0},'end':{'line':target_start,'character':0}},'document_eol':'lf'}
 packet={'family':'roxygen_drafting','row_ref':{'row_id':rid,'family':'roxygen_drafting','package_id':'fixturepkg','group_id':'g-fixture','source':'scenario_roxygen_drafting','file':'/mnt/h/sepalith/datasets/scenarios_v1/roxygen_drafting.jsonl','line':1,'raw_line_sha256':'1'*64,'split':'train_group'},'validation':{'source_path':str(source_path),'source_sha256':source_sha,'parent_group_split':'train_group','full_buffer_application':True,'normalized_parent_R_parse':True},'result':{'operation':'replace','target_body':target_lines,'context':context}}
 provenance={'row_id':rid,'status':'provenance_pass_semantic_analyzer_queued','family':'roxygen_drafting','package_id':'fixturepkg','license_decision':{'ok':True,'reason':'positive_reviewed_allowed_license'},'source_stat_stable':True,'description_stat_stable':True}
 return semantic,provenance,packet,source,target_lines

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.tokenizer=m.TokenizerAdapter();cls.original_normalized_root=m.NORMALIZED_ROOT;m.NORMALIZED_ROOT=Path(tempfile.gettempdir())
 @classmethod
 def tearDownClass(cls):m.NORMALIZED_ROOT=cls.original_normalized_root
 def test_real_r_helper_closure_protocol_and_exact_tokens(self):
  with tempfile.TemporaryDirectory() as d:
   semantic,provenance,packet,source,target=fixture(Path(d));done=subprocess.run(['Rscript','--vanilla','-e',f'parse(file={json.dumps(str(Path(d)/"source.R"))}, keep.source=TRUE)'],capture_output=True,text=True);self.assertEqual(done.returncode,0,done.stderr)
   row,profile=m.materialize_row(semantic,provenance,packet,self.tokenizer);prompt=row['prompt_text']
   self.assertLess(prompt.index('helper_before <-'),prompt.index('target <-'));self.assertLess(prompt.index('target <-'),prompt.index('helper_after <-'));self.assertIn('# helper comment must survive',prompt)
   self.assertNotIn("#' @title Target docs",prompt);self.assertEqual(row['target_body_text'],'\n'.join(target));self.assertEqual(m.STRICT.validate_token_row(row,full_text=True),[])
   self.assertEqual([x['name'] for x in profile['semantic']['selected_spans_source_order']],['helper_before','target','helper_after']);self.assertTrue(all(len(x['sha256'])==64 for x in profile['semantic']['selected_spans_source_order']))
   self.assertEqual(row['input_ids'][0],0);self.assertEqual(row['input_ids'][-1],1);self.assertEqual(row['target_terminal_tokens'],list(m.STRICT.TERMINAL_IDS));self.assertTrue(profile['semantic']['complete_target_preserved'])
 def test_target_duplication_holds(self):
  with tempfile.TemporaryDirectory() as d:
   semantic,provenance,packet,source,target=fixture(Path(d));source=source+'\n'.join(target)+'\n';Path(semantic['source_path']).write_text(source);digest=hashlib.sha256(source.encode()).hexdigest();semantic['source_sha256']=digest;semantic['parse_bytes_sha256']=digest;semantic['scope']['parsed_source_sha256']=digest;packet['validation']['source_sha256']=digest
   with self.assertRaisesRegex(m.Hold,'target_not_exactly_once'):m.materialize_row(semantic,provenance,packet,self.tokenizer)
 def test_target_leakage_in_preserved_evidence_holds(self):
  with tempfile.TemporaryDirectory() as d:
   semantic,provenance,packet,_,target=fixture(Path(d));packet['result']['context']['selected_references']=[{'path':'R/leak.R','content':'\n'.join(target)+'\n'}]
   with self.assertRaisesRegex(m.Hold,'target_roxygen_leaked_into_prompt'):m.materialize_row(semantic,provenance,packet,self.tokenizer)
 def test_formal_scope_mutation_cannot_reuse_stale_closure(self):
  with tempfile.TemporaryDirectory() as d:
   semantic,provenance,packet,_,_=fixture(Path(d));semantic['scope']['function_dependencies']['helper_before']['formals']=[];semantic['scope']['function_dependencies']['helper_before']['variables']=['x']
   with self.assertRaisesRegex(m.Hold,'semantic_v6_recheck_failed'):m.materialize_row(semantic,provenance,packet,self.tokenizer)
 def test_long_complete_target_is_not_truncated(self):
  target=["#' @title Long target","#' @param x input"]+["#' detail "+str(i) for i in range(1200)]
  with tempfile.TemporaryDirectory() as d:
   semantic,provenance,packet,_,_=fixture(Path(d),target);row,profile=m.materialize_row(semantic,provenance,packet,self.tokenizer)
   self.assertTrue(profile['geometry']['target_gt_1024']);self.assertEqual(row['target_body_text'],'\n'.join(target));self.assertTrue(row['target_body_text'].endswith("#' detail 1199"))
 def test_length_buckets_have_no_exclusion_gap(self):
  self.assertEqual([m.length_bucket(x) for x in (4096,4097,8192,8193,16384,16385,32768,32769)],['le_4096','4097_8192','4097_8192','8193_16384','8193_16384','16385_32768','16385_32768','gt_32768'])
 def test_train_identity_failure_is_explicit_hold(self):
  with tempfile.TemporaryDirectory() as d:
   semantic,provenance,packet,_,_=fixture(Path(d));packet['row_ref']['split']='unknown'
   with self.assertRaisesRegex(m.Hold,'train_split'):m.materialize_row(semantic,provenance,packet,self.tokenizer)
 def test_cli_atomic_denominator_and_input_bindings(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);semantic,provenance,packet,_,_=fixture(root);semantic_path=root/'semantic.jsonl';provenance_path=root/'provenance.jsonl';packet_path=root/'packets.jsonl';accepted_path=root/'accepted.json';semantic_manifest_path=root/'semantic-manifest.json';output=root/'out'
   for path,row in ((semantic_path,semantic),(provenance_path,provenance),(packet_path,packet)):path.write_text(json.dumps(row)+'\n')
   accepted_path.write_text(json.dumps({'ids':[semantic['row_id']]}))
   semantic_manifest_path.write_text(json.dumps({'status':'complete_review_only','exact_id_closure':True,'code':{'analyzer_sha256':m.SEMANTIC_SHA,'scope_helper_sha256':m.SEMANTIC_SCOPE_SHA,'namespace_helper_sha256':m.SEMANTIC_NAMESPACE_SHA},'output':{'sha256':m.sha(semantic_path)}}))
   argv=['materialize_semantic.py','--semantic-manifest',str(semantic_manifest_path),'--expected-semantic-manifest-sha256',m.sha(semantic_manifest_path),'--semantic-ledger',str(semantic_path),'--expected-semantic-sha256',m.sha(semantic_path),'--provenance-ledger',str(provenance_path),'--expected-provenance-sha256',m.sha(provenance_path),'--candidate-packets',str(packet_path),'--expected-packets-sha256',m.sha(packet_path),'--accepted-ids',str(accepted_path),'--expected-accepted-ids-sha256',m.sha(accepted_path),'--output',str(output)]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'NORMALIZED_ROOT',root):m.main()
   manifest=json.loads((output/'manifest.json').read_text());self.assertTrue(manifest['exact_denominator_closure']);self.assertEqual((manifest['materialized_rows'],manifest['hold_rows']),(1,0));self.assertEqual(manifest['outputs']['training-rows.jsonl']['sha256'],m.sha(output/'training-rows.jsonl'))
if __name__=='__main__':unittest.main()
