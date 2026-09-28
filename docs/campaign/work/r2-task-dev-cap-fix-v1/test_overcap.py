import json,tempfile,unittest
from pathlib import Path
from test_campaign_eval import TextTokenizer,ControlledModel,context
from campaign_checkpoint import digest
from campaign_eval import development_evaluator
from sepalith.campaign_protocol import RENDERER_ID,serialize_target
class OverCap(unittest.TestCase):
 def test_long_gold_retained_and_capped_generation_scored(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);tok=TextTokenizer(); gold=['x <- '+('1 + '*40)+'1']
   cases=[{'id':'long','split':'dev','family':'finish','package_id':'p','operation':'replace','region_new':gold,'context':context().to_dict()},{'id':'noop','split':'dev','family':'no_op','package_id':'q','operation':'no_op','region_new':['x <- 1'],'context':context().to_dict()}]
   panel=root/'panel.jsonl';panel.write_text(''.join(json.dumps(x)+'\n' for x in cases))
   recipe={'development_panel':{'path':str(panel),'sha256':digest(panel)},'renderer_id':RENDERER_ID,'development_case_ids':['long','noop'],'development_max_new_tokens':64,'parameters':{'max_sequence_tokens':4096}}
   class Capped(ControlledModel):
    def generate(self,input_ids,**kw):
     assert kw['max_new_tokens']==64
     return super().generate(input_ids,**kw)
   outputs=[tok.encode(serialize_target('replace',gold))[:64],tok.encode(serialize_target('no_op',['x <- 1']))+[1]]
   result=development_evaluator(recipe)(Capped(outputs),tok,root/'archive/full/checkpoint-250',250)
   self.assertEqual(result['denominators']['cases'],2);self.assertEqual(result['counts']['reference_over_generation_cap'],1);self.assertEqual(result['counts']['exact_region'],1)
   rows=json.loads(Path(result['cases_path']).read_text())['results'];self.assertTrue(rows[0]['cap_hit']);self.assertGreater(rows[0]['reference_tokens'],64);self.assertEqual(rows[0]['loss']['target_tokens'],rows[0]['reference_tokens'])
   recipe['parameters']['max_sequence_tokens']=64
   with self.assertRaisesRegex(ValueError,'context capacity'):
    development_evaluator(recipe)(Capped(outputs),tok,root/'archive/full/checkpoint-500',500)
if __name__=='__main__':unittest.main()
