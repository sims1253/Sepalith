import copy,hashlib,importlib.util,json,math,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=load('review706',P/'prepare.py');R=load('evalreview706',P/'review_eval.py')
def fixture():
 result={'status':'root_admitted_execution_stopped','global_step':706,'global_optimizer_step_offset':66,'stage_step':640,'initial_cursor':6144,'observed_draws':4096,'last_draw_position':10239,'train_loss':1.0}
 events=[]
 for step in range(451,707):events += [{'event':'pre_optimizer','next_step':step,'finite_gradient_tensors':381,'nonzero_gradient_tensors':381},{'event':'log','step':step,'logs':{'loss':1.0,'grad_norm':2.0}}]
 return result,events
class TestReview(unittest.TestCase):
 def test_exact_accounting_and_failed_startups_do_not_count(self):
  result,events=fixture();events=[{'event':'phase','phase':'load_bound_start'},{'event':'phase','phase':'load_bound_start'}]+events
  self.assertEqual(M.validate_accounting(result,events),{'updates':256,'draws':4096,'cursor':10240})
  bad=copy.deepcopy(result);bad['last_draw_position']=10238
  with self.assertRaises(AssertionError):M.validate_accounting(bad,events)
  with self.assertRaises(AssertionError):M.validate_accounting(result,events[:-2])
 def test_manifest_contract_identity_and_inventory(self):
  identity={'x':1};wanted=M.canonical_sha(identity);manifest={'full':True,'checkpoint_kind':'full_weights','step':706,'files':{x:{} for x in M.FULL},'identity':identity}
  self.assertTrue(M.validate_manifest_contract(manifest,wanted))
  with self.assertRaises(AssertionError):M.validate_manifest_contract(manifest,'0'*64)
  manifest['files'].pop('optimizer.pt')
  with self.assertRaises(AssertionError):M.validate_manifest_contract(manifest,wanted)
 def test_actual_transition_admissions_and_cadence(self):
  recipe=M.read(M.RECIPE);self.assertTrue(M.validate_transition(recipe))
  bad=copy.deepcopy(recipe);bad['runtime']['checkpoint_every']=24
  with self.assertRaises(AssertionError):M.validate_transition(bad)
 def test_runtime_handles_exact(self):
  value=json.loads((P/'runtime-handles.template.json').read_text());self.assertEqual((value['controller_pid'],value['guard_pid'],value['attestation_pid'],value['model_child_pid']),(3038867,3040840,3042470,3044342))
 def test_internal_eval_math_and_order_control(self):
  rows=[{'row_id':'a','loss_tokens':2,'loss_sum':2.0,'mean_causal_nll':1.0},{'row_id':'b','loss_tokens':3,'loss_sum':6.0,'mean_causal_nll':2.0}]
  def small(metrics,ids):
   assert [x['row_id'] for x in metrics]==ids
   total=sum(x['loss_tokens'] for x in metrics);loss=sum(x['loss_sum'] for x in metrics)
   assert all(math.isclose(x['mean_causal_nll'],x['loss_sum']/x['loss_tokens'],rel_tol=0,abs_tol=1e-12) for x in metrics)
   return loss/total
  self.assertEqual(small(rows,['a','b']),1.6)
  with self.assertRaises(AssertionError):small(list(reversed(rows)),['a','b'])
 def test_saved_precision_closure_and_all_panels(self):
  self.assertEqual(hashlib.sha256((P/'saved_precision.py').read_bytes()).hexdigest(),'3a46d3f301d2f6ade8332b0426edd182a16a7b52c44f28aee60a7b27af144629')
  source=(P/'evaluate_matched.py').read_text();self.assertIn("for label in ('anchor2k', '8k', '16k')",source);compile(source,str(P/'evaluate_matched.py'),'exec')
 def test_eval_exact_order_and_recomputed_aggregate(self):
  with tempfile.TemporaryDirectory() as td:
   fixture_path=Path(td)/'rows.jsonl';fixture_rows=[{'row_id':'a','document_id':'d1'},{'row_id':'b','document_id':'d2'}]
   fixture_path.write_text(''.join(json.dumps(x)+'\n' for x in fixture_rows));ids_sha=hashlib.sha256(b'a\nb\n').hexdigest()
   old=R.FIXTURES;R.FIXTURES={'x':(fixture_path,R.sha(fixture_path),2,5,ids_sha)}
   rows=[{'row_id':'a','loss_tokens':2,'loss_sum':2.0,'mean_causal_nll':1.0},{'row_id':'b','loss_tokens':3,'loss_sum':6.0,'mean_causal_nll':2.0}]
   panel={'step':706,'binding_sha256':'a'*64,'length_stratum':'x','case_ids':['d1','d2'],'denominators':{'validation_rows':2,'validation_loss_tokens':5},'row_metrics':rows,'metrics':{'mean_causal_nll':1.6}}
   try:
    self.assertEqual(R.validate_panel(panel,'x','a'*64),1.6)
    for mutation in ('substitution','permutation','aggregate'):
     bad=copy.deepcopy(panel)
     if mutation=='substitution':bad['row_metrics'][0]['row_id']='z'
     elif mutation=='permutation':bad['row_metrics'].reverse()
     else:bad['metrics']['mean_causal_nll']=1.61
     with self.assertRaises(AssertionError):R.validate_panel(bad,'x','a'*64)
   finally:R.FIXTURES=old
 def test_root_commands_are_dormant_and_exact(self):
  c=json.loads((P/'root-commands.json').read_text());self.assertEqual(c['status'],'manual_cpu_prepare_then_root_admitted_cuda_evaluation');self.assertIn(str(P/'prepare.py'),c['prepare_manual_after_terminal']);self.assertIn(str(P/'review_eval.py'),c['review_eval_manual_after_evaluation'])
if __name__=='__main__':unittest.main()
