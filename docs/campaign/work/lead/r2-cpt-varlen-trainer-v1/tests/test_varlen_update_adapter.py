import sys, unittest
from pathlib import Path

PACKET=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKET/'source/experiments/training'))
from varlen_update_adapter import (PackedOptimizerWindowDataset,VarlenContractError,
    pack_optimizer_window)


def row(position,length):
    ids=[(position+i)%19 for i in range(length)]
    return {'input_ids':ids,'labels':[-100]+ids[1:],'attention_mask':[1]*length,
            '_draw_position':position}


class Rows:
    def __init__(self,values):self.values=values
    def __len__(self):return len(self.values)
    def __getitem__(self,i):return self.values[i]


class VarlenContractTests(unittest.TestCase):
    def test_exact_window_conservation_positions_and_denominator(self):
        rows=[row(i,n) for i,n in enumerate((3,4,6,2))]
        update=pack_optimizer_window(rows,first_position=0,token_cap=7,effective_batch=4)
        self.assertEqual([p['packed_seq_lengths'] for p in update['physical_packs']],[[3,4],[6],[2]])
        self.assertEqual(update['draw_positions'],[0,1,2,3])
        self.assertEqual(update['loss_denominator'],sum(len(x['input_ids'])-1 for x in rows))
        ids=[];labels=[]
        for pack in update['physical_packs']:
            ids.extend(pack['input_ids'][0]);labels.extend(pack['labels'][0])
            cursor=0
            for length in pack['packed_seq_lengths']:
                self.assertEqual(pack['position_ids'][0][cursor:cursor+length],list(range(length)))
                self.assertEqual(pack['labels'][0][cursor],-100);cursor+=length
        self.assertEqual(ids,[v for x in rows for v in x['input_ids']])
        self.assertEqual(labels,[v for x in rows for v in x['labels']])

    def test_dataset_preserves_optimizer_windows_after_resume_cursor(self):
        base=Rows([row(i,2+i%3) for i in range(4,12)])
        view=PackedOptimizerWindowDataset(base,first_position=4,token_cap=8,effective_batch=4)
        self.assertEqual(len(view),2)
        self.assertEqual(view[0]['draw_positions'],[4,5,6,7])
        self.assertEqual(view[1]['draw_positions'],[8,9,10,11])

    def test_rejects_boundary_exposure_order_cap_and_partial_update(self):
        rows=[row(i,3) for i in range(4)]
        bad=[dict(x) for x in rows];bad[1]=dict(bad[1],_draw_position=8)
        with self.assertRaisesRegex(VarlenContractError,'draw order'):pack_optimizer_window(bad,first_position=0,token_cap=8,effective_batch=4)
        bad=[dict(x) for x in rows];bad[2]=dict(bad[2],labels=[2,2,2])
        with self.assertRaisesRegex(VarlenContractError,'boundary'):pack_optimizer_window(bad,first_position=0,token_cap=8,effective_batch=4)
        with self.assertRaisesRegex(VarlenContractError,'exceeds'):pack_optimizer_window(rows,first_position=0,token_cap=2,effective_batch=4)
        with self.assertRaisesRegex(VarlenContractError,'whole optimizer'):PackedOptimizerWindowDataset(Rows(rows[:3]),first_position=0,token_cap=8,effective_batch=4)


if __name__=='__main__':unittest.main(verbosity=2)
