from pathlib import Path
import copy,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from varlen_pack_candidate import *

def rows(lengths):
 out=[]
 for i,n in enumerate(lengths):out.append({'row_id':f'r{i}','_draw_position':100+i,'input_ids':[0]+[3]*(n-2)+[1],'labels':[-100]+[3]*(n-2)+[1]})
 return out

class TestPack(unittest.TestCase):
 def test_exact_tokens_labels_cursor_and_boundaries(self):
  source=rows([6,7,8,9]*4);before=copy.deepcopy(source);groups=pack_update(source,first_position=100,physical_token_cap=30)
  self.assertLess(len(groups),16)
  self.assertEqual(source,before)
  seen=[]
  for group in groups:
   packed=collate_varlen(group);verify_conservation(group,packed);seen.extend(x['_draw_position'] for x in group)
   self.assertNotIn('attention_mask',packed)
  self.assertEqual(sorted(seen),list(range(100,116)))
 def test_never_crosses_optimizer_update(self):
  with self.assertRaisesRegex(PackError,'effective batch'):pack_update(rows([4]*15),first_position=100)
 def test_rejects_cursor_gap_long_row_and_unmasked_boundary(self):
  for mutate,cap in [(lambda x:x[3].update(_draw_position=999),100),(lambda x:None,3),(lambda x:x[2]['labels'].__setitem__(0,0),100)]:
   value=rows([4]*16);mutate(value)
   with self.assertRaises(PackError):pack_update(value,first_position=100,physical_token_cap=cap)
 def test_loss_denominator_conserved(self):
  value=rows([4]*16);groups=pack_update(value,first_position=100,physical_token_cap=20)
  self.assertEqual(sum(collate_varlen(g)['loss_denominator'] for g in groups),sum(sum(v!=-100 for v in r['labels'][1:]) for r in value))

if __name__=='__main__':unittest.main()
