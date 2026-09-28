from pathlib import Path
import copy,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from varlen_contract import *
def rows(ns):
 return [{'row_id':f'r{i}','_draw_position':50+i,'input_ids':[0]+[7]*(n-2)+[1],'labels':[-100]+[7]*(n-2)+[1]} for i,n in enumerate(ns)]
class T(unittest.TestCase):
 def test_conserves_exact_update(self):
  r=rows([3,4,5,6]*4);before=copy.deepcopy(r);g=pack_update(r,first_position=50,token_cap=18);v=verify(r,g,first_position=50)
  self.assertEqual(r,before);self.assertEqual(v['rows'],16);self.assertLess(v['physical_groups'],16)
  self.assertTrue(all('packed_seq_lengths' in x and 'attention_mask' not in x for x in g))
 def test_denominator_excludes_each_boundary(self):
  r=rows([4]*16);g=pack_update(r,first_position=50,token_cap=16);self.assertEqual(verify(r,g,first_position=50)['loss_denominator'],48)
 def test_rejects_gap_long_unmasked_or_wrong_count(self):
  cases=[]
  x=rows([4]*16);x[2]['_draw_position']=90;cases.append((x,10))
  x=rows([4]*16);cases.append((x,3))
  x=rows([4]*16);x[1]['labels'][0]=0;cases.append((x,10))
  cases.append((rows([4]*15),10))
  for x,cap in cases:
   with self.assertRaises(ContractError):pack_update(x,first_position=50,token_cap=cap)
 def test_tampered_flat_slice_rejected(self):
  r=rows([4]*16);g=pack_update(r,first_position=50,token_cap=16);g[0]['input_ids'][0][2]=99
  with self.assertRaises(ContractError):verify(r,g,first_position=50)
 def test_each_flat_geometry_dimension_is_checked(self):
  for key in ('input_ids','labels','position_ids'):
   r=rows([4]*16);g=pack_update(r,first_position=50,token_cap=16);g[0][key][0].pop()
   with self.assertRaises(ContractError):verify(r,g,first_position=50)
 def test_pack_is_deterministic_and_preserves_original_order_metadata(self):
  r=rows([9,4,8,3,7,6,5,4,3,9,8,7,6,5,4,3])
  a=pack_update(r,first_position=50,token_cap=16);b=pack_update(copy.deepcopy(r),first_position=50,token_cap=16)
  self.assertEqual(a,b)
  observed=[m['draw_position'] for g in a for m in g['members']]
  self.assertEqual(observed,list(range(50,66)))
 def test_resume_view_rebuilds_same_next_update(self):
  base=rows([4]*32)
  for i,r in enumerate(base):r['_draw_position']=i
  uninterrupted=PackedUpdateView(base)
  resumed=PackedUpdateView(base,initial_cursor=16)
  self.assertEqual(uninterrupted[1],resumed[0])
  self.assertEqual([m['draw_position'] for g in resumed[0] for m in g['members']],list(range(16,32)))
  with self.assertRaises(ContractError):PackedUpdateView(base,initial_cursor=3)
if __name__=='__main__':unittest.main()
