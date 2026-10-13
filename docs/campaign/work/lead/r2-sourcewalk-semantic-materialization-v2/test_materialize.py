import copy,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('semantic_materializer_v2',HERE/'materialize_semantic.py');m=importlib.util.module_from_spec(spec);sys.modules['semantic_materializer_v2']=m;spec.loader.exec_module(m)

def set_candidate_context(packet,prefix,suffix):
 text='\n'.join([*prefix,'',*suffix]);digest=hashlib.sha256(text.encode()).hexdigest();c=packet['result']['context']
 c['prefix']=list(prefix);c['suffix_lines']=list(suffix);c['region_old']=[];c['cursor']={'region_line_index':-1,'code_point_column':None,'utf16_column':None}
 c['replacement_range']['start']={'line':len(prefix),'character':0};c['replacement_range']['end']={'line':len(prefix),'character':0};c['replacement_range']['content_sha256']=digest
 packet['result']['selection_source']={'text':text,'content_sha256':digest}

def fixture(root:Path,target_lines=None,full_function=True):
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
 context={'schema_version':'sepalith.prompt.prm03.v1','path':'R/source.R','prefix':[],'selected_references':[],'history':[],'diagnostics':[],'retrieval':[],'scope_mode':'off','scope_lines':[],'suffix_lines':[],'region_old':[],'cursor':{'region_line_index':-1,'code_point_column':None,'utf16_column':None},'replacement_range':{'uri':'file:///sepalith/R/source.R','document_version':0,'content_sha256':'0'*64,'start':{'line':0,'character':0},'end':{'line':0,'character':0}},'document_eol':'lf'}
 packet={'family':'roxygen_drafting','row_ref':{'row_id':rid,'family':'roxygen_drafting','package_id':'fixturepkg','group_id':'g-fixture','source':'scenario_roxygen_drafting','file':'/mnt/h/sepalith/datasets/scenarios_v1/roxygen_drafting.jsonl','line':1,'raw_line_sha256':'1'*64,'split':'train_group'},'validation':{'source_path':str(source_path),'source_sha256':source_sha,'parent_group_split':'train_group','full_buffer_application':True,'normalized_parent_R_parse':True},'result':{'operation':'replace','target_body':target_lines,'context':context}}
 set_candidate_context(packet,[],function if full_function else function[:1])
 provenance={'row_id':rid,'status':'provenance_pass_semantic_analyzer_queued','family':'roxygen_drafting','package_id':'fixturepkg','license_decision':{'ok':True,'reason':'positive_reviewed_allowed_license'},'source_stat_stable':True,'description_stat_stable':True}
 return semantic,provenance,packet,source,target_lines,function

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.tokenizer=m.TokenizerAdapter();cls.original_normalized_root=m.NORMALIZED_ROOT;m.NORMALIZED_ROOT=Path(tempfile.gettempdir())
 @classmethod
 def tearDownClass(cls):m.NORMALIZED_ROOT=cls.original_normalized_root
 def test_real_source_contiguous_offset_apply_and_helper_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   semantic,provenance,packet,source,target,function=fixture(Path(d));done=subprocess.run(['Rscript','--vanilla','-e',f'parse(file={json.dumps(str(Path(d)/"source.R"))}, keep.source=TRUE)'],capture_output=True,text=True);self.assertEqual(done.returncode,0,done.stderr)
   row,profile=m.materialize_row(semantic,provenance,packet,self.tokenizer);prompt=row['prompt_text'];context=m.PROTOCOL.PromptContext.from_mapping(packet['result']['context'])
   self.assertEqual(profile['geometry']['mode'],'candidate_contiguous');self.assertTrue(profile['geometry']['application_exact']);self.assertEqual(m.GEOMETRY.context_buffer_text(context.to_dict()),packet['result']['selection_source']['text'])
   self.assertIn('\n'.join(function),prompt);self.assertNotIn("#' @title Target docs",prompt);self.assertIn('<filename>R/source.R#L1-L4',prompt);self.assertIn('<filename>R/source.R#L12-L12',prompt)
   placements={x['name']:x['placement'] for x in profile['semantic']['selected_spans_source_order']};self.assertEqual(placements,{'helper_before':'selected_reference','target':'editable_buffer','helper_after':'selected_reference'})
   self.assertEqual(row['target_body_text'],'\n'.join(target));self.assertEqual(m.STRICT.validate_token_row(row,full_text=True),[])
 def test_stale_content_hash_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_,_=fixture(Path(d));k['result']['context']['replacement_range']['content_sha256']='0'*64
   with self.assertRaisesRegex(m.Hold,'context_content_sha256_mismatch'):m.materialize_row(s,p,k,self.tokenizer)
 def test_stale_range_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_,_=fixture(Path(d));k['result']['context']['replacement_range']['start']['line']=1;k['result']['context']['replacement_range']['end']['line']=1
   with self.assertRaisesRegex(m.Hold,'expected_zero_width_blank_anchor'):m.materialize_row(s,p,k,self.tokenizer)
 def test_omitted_gaps_are_typed_records_not_concatenated_buffer(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_,function=fixture(Path(d));row,profile=m.materialize_row(s,p,k,self.tokenizer);editable=k['result']['selection_source']['text']
   self.assertEqual(editable,'\n'+'\n'.join(function));self.assertNotIn('helper_before <-',editable);self.assertNotIn('helper_after <-',editable);self.assertEqual(sum(x['placement']=='selected_reference' for x in profile['semantic']['selected_spans_source_order']),2)
   self.assertIn('helper_before <-',row['prompt_text']);self.assertIn('helper_after <-',row['prompt_text'])
 def test_missing_target_function_expands_full_contiguous_source(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,source,target,_=fixture(Path(d),full_function=False);row,profile=m.materialize_row(s,p,k,self.tokenizer)
   self.assertEqual(profile['geometry']['mode'],'expanded_full_source_contiguous');rr=profile['geometry']['replacement_range'];self.assertNotEqual(rr['content_sha256'],k['result']['context']['replacement_range']['content_sha256'])
   before=m.GEOMETRY.derive_full_before(source.encode(),target,s['context_closure']['target_definition_span']);self.assertEqual(rr['content_sha256'],before['before_sha256'])
   self.assertEqual(m.GEOMETRY.apply_zero_width(before['geometry'],target),source);self.assertTrue(profile['geometry']['application_exact'])
 def test_target_leakage_in_existing_evidence_holds(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,target,_=fixture(Path(d));k['result']['context']['selected_references']=[{'path':'R/leak.R','content':'\r\n'.join(target)}]
   with self.assertRaisesRegex(m.Hold,'target_roxygen_leaked_into_prompt'):m.materialize_row(s,p,k,self.tokenizer)
 def test_target_duplication_holds(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,source,target,_=fixture(Path(d));source=source+'\n'.join(target)+'\n';Path(s['source_path']).write_text(source);digest=hashlib.sha256(source.encode()).hexdigest();s['source_sha256']=digest;s['parse_bytes_sha256']=digest;s['scope']['parsed_source_sha256']=digest;k['validation']['source_sha256']=digest
   with self.assertRaisesRegex(m.Hold,'target_not_exactly_once'):m.materialize_row(s,p,k,self.tokenizer)
 def test_long_target_never_truncated(self):
  with tempfile.TemporaryDirectory() as d:
   target=["#' @title Long target","#' @param x input"]+["#' detail "+str(i) for i in range(1200)];s,p,k,_,_,_=fixture(Path(d),target);row,profile=m.materialize_row(s,p,k,self.tokenizer)
   self.assertTrue(profile['geometry']['target_gt_1024']);self.assertEqual(row['target_body_text'],'\n'.join(target));self.assertTrue(row['target_body_text'].endswith("#' detail 1199"))
 def test_formal_scope_stale_closure_and_nontrain_hold(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_,_=fixture(Path(d));s['scope']['function_dependencies']['helper_before']['formals']=[];s['scope']['function_dependencies']['helper_before']['variables']=['x']
   with self.assertRaisesRegex(m.Hold,'semantic_v6_recheck_failed'):m.materialize_row(s,p,k,self.tokenizer)
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_,_=fixture(Path(d));k['row_ref']['split']='unknown'
   with self.assertRaisesRegex(m.Hold,'train_split'):m.materialize_row(s,p,k,self.tokenizer)
 def test_length_buckets_no_exclusion(self):self.assertEqual([m.length_bucket(x) for x in (4096,4097,8192,8193,16384,16385,32768,32769)],['le_4096','4097_8192','4097_8192','8193_16384','8193_16384','16385_32768','16385_32768','gt_32768'])
 def test_cli_atomic_denominator_and_bindings(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);semantic,provenance,packet,_,_,_=fixture(root);sp=root/'semantic.jsonl';pp=root/'provenance.jsonl';kp=root/'packets.jsonl';ap=root/'accepted.json';mp=root/'semantic-manifest.json';out=root/'out'
   for path,row in ((sp,semantic),(pp,provenance),(kp,packet)):path.write_text(json.dumps(row)+'\n')
   ap.write_text(json.dumps({'ids':[semantic['row_id']]}));mp.write_text(json.dumps({'status':'complete_review_only','exact_id_closure':True,'code':{'analyzer_sha256':m.SEMANTIC_SHA,'scope_helper_sha256':m.SEMANTIC_SCOPE_SHA,'namespace_helper_sha256':m.SEMANTIC_NAMESPACE_SHA},'output':{'sha256':m.sha(sp)}}))
   argv=['materialize_semantic.py','--semantic-manifest',str(mp),'--expected-semantic-manifest-sha256',m.sha(mp),'--semantic-ledger',str(sp),'--expected-semantic-sha256',m.sha(sp),'--provenance-ledger',str(pp),'--expected-provenance-sha256',m.sha(pp),'--candidate-packets',str(kp),'--expected-packets-sha256',m.sha(kp),'--accepted-ids',str(ap),'--expected-accepted-ids-sha256',m.sha(ap),'--output',str(out)]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'NORMALIZED_ROOT',root):m.main()
   manifest=json.loads((out/'manifest.json').read_text());self.assertTrue(manifest['exact_denominator_closure']);self.assertEqual((manifest['materialized_rows'],manifest['hold_rows']),(1,0));self.assertEqual(manifest['inputs']['semantic_context_geometry_sha256'],m.GEOMETRY_SHA)
if __name__=='__main__':unittest.main()
