import copy,hashlib,importlib.util,json,math,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]
S=importlib.util.spec_from_file_location('review354',P/'prepare.py');M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

def fixture():
 result={'status':'root_admitted_execution_stopped','global_step':354,'global_optimizer_step_offset':66,'initial_cursor':4224,'observed_draws':384,'last_draw_position':4607,'train_loss':1.0}
 rows=[]
 for step in range(331,355):
  rows += [{'event':'pre_optimizer','next_step':step,'finite_gradient_tensors':381,'nonzero_gradient_tensors':381},{'event':'log','step':step,'logs':{'loss':1.0,'grad_norm':2.0}}]
 return result,rows

class TestReview(unittest.TestCase):
 def test_exact_24_update_accounting(self):
  result,rows=fixture();self.assertEqual(M.validate_accounting(result,rows),{'updates':24,'draws':384,'cursor':4608})
 def test_missing_update_rejected(self):
  result,rows=fixture();rows=[x for x in rows if not(x.get('event')=='pre_optimizer' and x.get('next_step')==340)]
  with self.assertRaises(AssertionError):M.validate_accounting(result,rows)
 def test_gradient_or_cursor_corruption_rejected(self):
  result,rows=fixture();rows[0]['nonzero_gradient_tensors']=380
  with self.assertRaises(AssertionError):M.validate_accounting(result,rows)
  result,rows=fixture();result['last_draw_position']=4606
  with self.assertRaises(AssertionError):M.validate_accounting(result,rows)
 def test_long_evaluator_does_not_repeat_anchor(self):
  source=(P/'evaluate_long_only.py').read_text();self.assertIn("for label in ('8k', '16k')",source);self.assertNotIn("for label in ('anchor2k'",source);compile(source,str(P/'evaluate_long_only.py'),'exec')
 def test_runtime_handles_fail_closed_placeholders(self):
  value=json.loads((P/'runtime-handles.template.json').read_text());self.assertEqual(value['status'],'terminal_success');self.assertEqual((value['controller_pid'],value['guard_pid'],value['attestation_pid'],value['model_child_pid']),(2630015,2630027,2630218,2634519))
 def test_actual_evaluator_preflight_has_complete_precision_closure(self):
  spec=importlib.util.spec_from_file_location('longonly',P/'evaluate_long_only.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/'source';source.mkdir();model=root/'model';model.mkdir();reference=root/'reference';reference.mkdir()
   manifest=source/'source-manifest.json';manifest.write_text(json.dumps({'files':[]})+'\n');E.SOURCE=source;E.SOURCE_SHA=E.digest(manifest)
   rows={}
   for name in ('8k','16k'):
    p=root/f'{name}.jsonl';p.write_text('{}\n');rows[name]={'validation_rows':{'path':str(p),'sha256':E.digest(p)},'validation_batch_size':1}
   panels=root/'panels.json';panels.write_text(json.dumps({'panels':rows})+'\n');E.PANELS=panels;E.PANELS_SHA=E.digest(panels)
   for name in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json'):(model/name).write_text(name)
   (reference/'tokenizer.json').write_text('tokenizer.json');E.REFERENCE=reference;E.TOKENIZER_SHA=E.digest(model/'tokenizer.json')
   binding={'schema':'sepalith.cpt.long-eval-root-binding.v1','status':'admitted','training_authorized':False,'runner_sha256':E.digest(P/'evaluate_long_only.py'),'dtype_restoration_sha256':'3a46d3f301d2f6ade8332b0426edd182a16a7b52c44f28aee60a7b27af144629','model_path':str(model),'model_files':{n:E.digest(model/n) for n in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json')}}
   observed,loaded=E.preflight(binding);self.assertEqual(observed,model);self.assertEqual(set(loaded),{'8k','16k'})

if __name__=='__main__':unittest.main()
