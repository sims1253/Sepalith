import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('semantic_materializer_v3',HERE/'materialize_semantic.py');m=importlib.util.module_from_spec(spec);sys.modules['semantic_materializer_v3']=m;spec.loader.exec_module(m)

def fixture(root:Path,target_lines=None,crlf=False,helper=False,long_source=True):
 if target_lines is None:target_lines=["#' @title Target docs","#' @param x input"]
 filler_before=[f'# context before {i:03d} '+('x'*30) for i in range(100 if long_source else 2)]
 filler_after=[f'# context after {i:03d} '+('y'*30) for i in range(100 if long_source else 2)]
 helper_lines=['helper <- function(x) x + 2',''] if helper else []
 function=['target <- function(x) {','  helper(x)' if helper else '  x + 1','}']
 lines=filler_before+helper_lines+target_lines+function+filler_after+[''];lf='\n'.join(lines);raw=lf.replace('\n','\r\n').encode() if crlf else lf.encode();source_path=root/'source.R';source_path.write_bytes(raw)
 target_start=len(filler_before)+len(helper_lines);function_start0=target_start+len(target_lines);function_span=[function_start0+1,function_start0+len(function)]
 top=['target']+(['helper'] if helper else []);spans={'target':function_span};counts={'target':1};deps={'target':{'codetools_ok':True,'formals':['x'],'variables':[],'call_heads':['helper'] if helper else ['+','{'],'base_bound':[] if helper else ['+','{']}}
 if helper:spans['helper']=[len(filler_before)+1,len(filler_before)+1];counts['helper']=1;deps['helper']={'codetools_ok':True,'formals':['x'],'variables':[],'call_heads':['+'],'base_bound':['+']}
 normalized=raw.replace(b'\r\n',b'\n');scope={'status':'scope_inventory_complete','target_definition_span':function_span,'function_formals':['x'],'variable_references':[],'call_heads':['helper'] if helper else ['+','{'],'base_bound':[] if helper else ['+','{'],'top_level_definitions':top,'top_level_definition_spans':spans,'top_level_definition_counts':counts,'function_dependencies':deps,'parsed_source_sha256':hashlib.sha256(normalized).hexdigest()}
 item={'target_occurrences':1,'target_definition_name':'target','documented_params':['x'],'imported_symbols':[],'prior_reviewed_recovery':False};reasons,resolution,closure=m.SEMANTIC.decide(item,scope,normalized.decode());assert reasons==[],reasons
 target=('\n'.join(target_lines)+'\n').encode();rid='row-v3-fixture';source_sha=hashlib.sha256(raw).hexdigest();semantic={'row_id':rid,'status':'semantic_supported_context_closure_root_review_required','reasons':[],'source_path':str(source_path),'source_sha256':source_sha,'parse_bytes_sha256':hashlib.sha256(normalized).hexdigest(),'target_occurrences':1,'target_occurrence_method':'uniform_crlf_to_lf' if crlf else 'exact_bytes','target_definition_name':'target','documented_params':['x'],'imported_symbols':[],'namespace':{'stat_stable':True,'status':'namespace_inventory_complete'},'prior_reviewed_recovery':False,'target_sha256':hashlib.sha256(target).hexdigest(),'target_bytes':len(target),'scope':scope,'reference_resolution':resolution,'context_closure':closure}
 local_prefix=filler_before[-8:]+helper_lines;local_suffix=function+filler_after[:2];selection='\n'.join([*local_prefix,'',*local_suffix]);sel_sha=hashlib.sha256(selection.encode()).hexdigest();context={'schema_version':'sepalith.prompt.prm03.v1','path':'R/source.R','prefix':local_prefix,'selected_references':[],'history':[],'diagnostics':[],'retrieval':[],'scope_mode':'off','scope_lines':[],'suffix_lines':local_suffix,'region_old':[],'cursor':{'region_line_index':-1,'code_point_column':None,'utf16_column':None},'replacement_range':{'uri':'file:///workspace/R/source.R','document_version':4,'content_sha256':sel_sha,'start':{'line':len(local_prefix),'character':0},'end':{'line':len(local_prefix),'character':0}},'document_eol':'lf'}
 packet={'family':'roxygen_drafting','row_ref':{'row_id':rid,'family':'roxygen_drafting','package_id':'fixturepkg','group_id':'g-fixture','source':'scenario_roxygen_drafting','file':'/mnt/h/sepalith/datasets/scenarios_v1/roxygen_drafting.jsonl','line':1,'raw_line_sha256':'1'*64,'split':'train_group'},'validation':{'source_path':str(source_path),'source_sha256':source_sha,'parent_group_split':'train_group','full_buffer_application':True,'normalized_parent_R_parse':True},'result':{'operation':'replace','target_body':target_lines,'selection_source':{'text':selection,'content_sha256':sel_sha},'context':context}}
 provenance={'row_id':rid,'status':'provenance_pass_semantic_analyzer_queued','family':'roxygen_drafting','package_id':'fixturepkg','license_decision':{'ok':True,'reason':'positive_reviewed_allowed_license'},'source_stat_stable':True,'description_stat_stable':True}
 return semantic,provenance,packet,raw,target_lines

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.tokenizer=m.TokenizerAdapter();cls.old=m.NORMALIZED_ROOT;m.NORMALIZED_ROOT=Path(tempfile.gettempdir())
 @classmethod
 def tearDownClass(cls):m.NORMALIZED_ROOT=cls.old
 def test_target_only_bounded_view_full_identity_global_range_and_apply(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,raw,target=fixture(Path(d));row,profile=m.materialize_row(s,p,k,self.tokenizer);g=profile['geometry'];target_global=raw[:raw.index(('\n'.join(target)+'\n').encode())].count(b'\n')
   self.assertEqual(g['global_replacement_range']['start'],{'line':target_global,'character':0});self.assertEqual(g['full_preedit_sha256'],g['global_replacement_range']['content_sha256']);self.assertNotEqual(g['selected_view_document_sha256'],g['full_preedit_sha256']);self.assertFalse(g['selected_view_is_full_document']);self.assertTrue(g['application_exact']);self.assertEqual(g['selection_policy_id'],'prm05-source-balanced-v1');self.assertEqual(row['target_body_text'],'\n'.join(target));self.assertEqual(m.STRICT.validate_token_row(row,full_text=True),[])
 def test_small_source_naturally_selects_full_document(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_=fixture(Path(d),long_source=False);_,profile=m.materialize_row(s,p,k,self.tokenizer);self.assertTrue(profile['geometry']['selected_view_is_full_document']);self.assertEqual(profile['geometry']['selected_view_document_sha256'],profile['geometry']['full_preedit_sha256'])
 def test_crlf_full_identity_and_application(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,raw,_=fixture(Path(d),crlf=True);row,profile=m.materialize_row(s,p,k,self.tokenizer);self.assertEqual(profile['geometry']['global_replacement_range']['content_sha256'],profile['geometry']['full_preedit_sha256']);self.assertTrue(profile['geometry']['application_exact']);self.assertIn('\n',row['target_body_text']);self.assertNotIn('\r',row['target_body_text'])
 def test_stale_candidate_hash_and_range_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_=fixture(Path(d));k['result']['context']['replacement_range']['content_sha256']='0'*64
   with self.assertRaisesRegex(m.Hold,'candidate_geometry_invalid:context_content_sha256_mismatch'):m.materialize_row(s,p,k,self.tokenizer)
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_=fixture(Path(d));k['result']['context']['replacement_range']['start']['line']+=1;k['result']['context']['replacement_range']['end']['line']+=1
   with self.assertRaisesRegex(m.Hold,'candidate_geometry_invalid:expected_zero_width_blank_anchor'):m.materialize_row(s,p,k,self.tokenizer)
 def test_helper_dependency_named_hold(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_=fixture(Path(d),helper=True)
   with self.assertRaisesRegex(m.Hold,'helper_evidence_production_wiring_unavailable:helper'):m.materialize_row(s,p,k,self.tokenizer)
 def test_existing_typed_reference_named_hold(self):
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_=fixture(Path(d));k['result']['context']['selected_references']=[{'path':'R/helper.R','content':'x <- 1'}]
   with self.assertRaisesRegex(m.Hold,'candidate_typed_evidence_production_wiring_unavailable'):m.materialize_row(s,p,k,self.tokenizer)
 def test_long_target_not_truncated(self):
  target=["#' @title Long","#' @param x input"]+["#' detail "+str(i) for i in range(1200)]
  with tempfile.TemporaryDirectory() as d:
   s,p,k,_,_=fixture(Path(d),target);row,profile=m.materialize_row(s,p,k,self.tokenizer);self.assertTrue(profile['geometry']['target_gt_1024']);self.assertTrue(row['target_body_text'].endswith("#' detail 1199"))
 def test_actual_train_row_uses_full_identity_global_line(self):
  rid='227626e3a7a234b808a096b1';sp=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-roxy-semantic-v6/bounded-root-01/semantic-ledger.jsonl');pp=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v2/bounded-05/bounded-ledger.jsonl');kp=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl')
  def get(path):
   with path.open() as stream:
    for line in stream:
     row=json.loads(line)
     if (row.get('row_id') or row.get('row_ref',{}).get('row_id'))==rid:return row
   raise KeyError(rid)
  old=m.NORMALIZED_ROOT;m.NORMALIZED_ROOT=Path('/mnt/h/sepalith/normalized')
  try:
   row,profile=m.materialize_row(get(sp),get(pp),get(kp),self.tokenizer)
  finally:m.NORMALIZED_ROOT=old
  self.assertEqual(profile['geometry']['global_replacement_range']['start']['line'],290);self.assertEqual(profile['geometry']['full_preedit_sha256'],'8bb73f454eb7931b1e838bf10e4b2208eb3d9b8914fe8a67b24b7bd22619d559');self.assertTrue(profile['geometry']['application_exact']);self.assertEqual(m.STRICT.validate_token_row(row,full_text=True),[])
 def test_cli_denominator_and_source_bindings(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s,p,k,_,_=fixture(root,long_source=False);sp=root/'s.jsonl';pp=root/'p.jsonl';kp=root/'k.jsonl';ap=root/'a.json';mp=root/'m.json';out=root/'out'
   for path,row in ((sp,s),(pp,p),(kp,k)):path.write_text(json.dumps(row)+'\n')
   ap.write_text(json.dumps({'ids':[s['row_id']]}));mp.write_text(json.dumps({'status':'complete_review_only','exact_id_closure':True,'code':{'analyzer_sha256':m.SEMANTIC_SHA,'scope_helper_sha256':m.SEMANTIC_SCOPE_SHA,'namespace_helper_sha256':m.SEMANTIC_NAMESPACE_SHA},'output':{'sha256':m.sha(sp)}}))
   argv=['x','--semantic-manifest',str(mp),'--expected-semantic-manifest-sha256',m.sha(mp),'--semantic-ledger',str(sp),'--expected-semantic-sha256',m.sha(sp),'--provenance-ledger',str(pp),'--expected-provenance-sha256',m.sha(pp),'--candidate-packets',str(kp),'--expected-packets-sha256',m.sha(kp),'--accepted-ids',str(ap),'--expected-accepted-ids-sha256',m.sha(ap),'--output',str(out)]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'NORMALIZED_ROOT',root):m.main()
   manifest=json.loads((out/'manifest.json').read_text());self.assertEqual((manifest['materialized_rows'],manifest['hold_rows']),(1,0));self.assertEqual(manifest['inputs']['campaign_selection_sha256'],m.SELECTION_SHA)
if __name__=='__main__':unittest.main()
