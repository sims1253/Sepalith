import importlib.util,itertools,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location('m',ROOT/'source/throughput_candidate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row(i,n,p='p',d=None):
 ids=[0]+[10+i]*(n-2)+[1]; labels=[-100]+ids[1:-1]+[1]
 return {'row_id':f'r{i}','package':p,'document_id':d or f'd{i}','input_ids':ids,'labels':labels,'attention_mask':[1]*n}
class T(unittest.TestCase):
 def test_packed_slices_reconstruct_every_original_tensor(self):
  rows=[row(1,4),row(2,5)];p=m.pack_rows(rows,16)
  self.assertEqual(len(p['members']),2)
  for x,member in zip(rows,p['members']):
   a,b=member['start'],member['end'];self.assertEqual(p['input_ids'][a:b],x['input_ids']);self.assertEqual(p['labels'][a:b],x['labels'])
  self.assertEqual(sum(v!=-100 for v in p['labels'][1:]),sum(sum(v!=-100 for v in x['labels'][1:]) for x in rows))
 def test_packed_boundary_retains_bos_eos_and_masks_boundary_prediction(self):
  p=m.pack_rows([row(1,4),row(2,5)],16); boundary=p['members'][1]['start']
  self.assertEqual((p['input_ids'][boundary-1],p['input_ids'][boundary]),(1,0));self.assertEqual(p['labels'][boundary],-100)
  self.assertEqual(p['cross_document_attention'],'allowed_causal')
 def test_length_pairing_minimizes_pair_padding(self):
  rows=[{'row_id':f'r{i}','length':n} for i,n in enumerate([3,4,9,10,14,16,20,21,29,31,35,36,40,42,50,51])]
  s=m.paired_schedule(rows,7); by={x['row_id']:x['length'] for x in rows}; got=sum(2*max(by[s['row_ids'][i]],by[s['row_ids'][i+1]])-by[s['row_ids'][i]]-by[s['row_ids'][i+1]] for i in range(0,16,2))
  expected=sum(b-a for a,b in zip(sorted(x['length'] for x in rows)[::2],sorted(x['length'] for x in rows)[1::2]));self.assertEqual(got,expected)
 def test_unique_before_terminal_replay_and_block_mix(self):
  rows=[{'row_id':f'r{i}','length':100+i} for i in range(19)]
  s=m.paired_schedule(rows,3407);seen=set();first_repeat=None
  for i,x in enumerate(s['row_ids']):
   if x in seen:first_repeat=i;break
   seen.add(x)
  self.assertEqual(first_repeat,19);self.assertEqual(len(seen),19);self.assertEqual(s['named_replays'],13);self.assertEqual(s['micro_batch'],2);self.assertEqual(s['gradient_accumulation'],8)
 def test_pack_overflow_rejected(self):
  with self.assertRaises(ValueError):m.pack_rows([row(1,7),row(2,7)],10)
 def test_benchmark_rejects_wrong_fixture_hash_before_model_import(self):
  import subprocess,sys
  bench=ROOT/'source/microbatch_benchmark.py'
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);rows=td/'rows';rows.write_text(json.dumps(row(1,4))+'\n')
   p=subprocess.run([sys.executable,str(bench),'--model',str(td/'absent-model'),'--rows',str(rows),'--rows-sha256','0'*64,'--report',str(td/'report'),'--steps','1','--optimizer','aurora_mix','--max-sequence-tokens','2048','--micro-batch','1','--accumulation','16'],capture_output=True,text=True,env={'CUDA_VISIBLE_DEVICES':'','PYTHONNOUSERSITE':'1'})
   self.assertNotEqual(p.returncode,0);self.assertIn('rows hash mismatch',p.stderr);self.assertFalse((td/'report').exists())
 def test_holdout_overlap_main_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);tr=td/'tr';ho=td/'ho';tr.write_text(json.dumps(row(1,4,p='held',d='dtrain'))+'\n');ho.write_text(json.dumps(row(2,4,p='held',d='dval'))+'\n')
   import subprocess,sys
   p=subprocess.run([sys.executable,str(ROOT/'source/throughput_candidate.py'),'--rows',str(tr),'--rows-sha256',m.sha(tr),'--heldout',str(ho),'--heldout-sha256',m.sha(ho),'--schedule-output',str(td/'s'),'--report-output',str(td/'r')],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0);self.assertIn('heldout overlap',p.stderr)
if __name__=='__main__':unittest.main()
