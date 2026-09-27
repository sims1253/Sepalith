import copy,hashlib,json,unittest
from bind_task_parent import validate,PARENTS,TOK,TOKCFG
class Tests(unittest.TestCase):
 def fixture(self,source):
  i={'source':source,'policy':{'initialization':PARENTS[source]},'parent':{'revision':'8dc5f6055b90fe4b9422340810b270b9569f37f3'}}
  p={'stage':'cpt_raw_r_v1','identity':i,'draw_schedule':{'sha256':'a'*64},'split_id':'TRAIN'};c={'full':True,'step':250,'identity':i}
  inv=[{'path':'model.safetensors','bytes':1,'sha256':'b'*64}]
  m={'schema_version':'sepalith.merged-cpt.parent-manifest.v1','kind':'merged_cpt','cpt_identity':i,'cpt_checkpoint_manifest':c,'source_cursor':4000,'previous_sampler':{'consumed_draws':4000,'schedule_sha256':'a'*64,'split_id':'TRAIN'},'source_script_sha256':'c'*64,'base_model_revision':i['parent']['revision'],'tokenizer_original_bytes_restored':True,'tokenizer':{'tokenizer_json_sha256':TOK,'tokenizer_config_sha256':TOKCFG},'merge_verification':dict(device='cpu',dtype='bfloat16',accumulation_dtype='float32',rounding='one final BF16 cast',adapter_tensors_exact=588,independent_lora_matrix_merges_exact=294,cuda_started=False),'weight_inventory':inv,'merged_weights_sha256':'b'*64,'weight_inventory_sha256':hashlib.sha256(json.dumps(inv,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
  return m,p,c
 def test_both_parents(self):
  for s in PARENTS:validate(*self.fixture(s),'c'*64)
 def test_bad_cursor(self):
  m,p,c=self.fixture(next(iter(PARENTS)));m['source_cursor']=3999
  with self.assertRaises(ValueError):validate(m,p,c,'c'*64)
 def test_bad_helper(self):
  with self.assertRaises(ValueError):validate(*self.fixture(next(iter(PARENTS))),'d'*64)
 def test_bad_rounding(self):
  m,p,c=self.fixture(next(iter(PARENTS)));m['merge_verification']['accumulation_dtype']='bfloat16'
  with self.assertRaises(ValueError):validate(m,p,c,'c'*64)
 def test_wrong_source(self):
  m,p,c=self.fixture(next(iter(PARENTS)));p['identity']['source']='e'*64
  with self.assertRaises(ValueError):validate(m,p,c,'c'*64)
if __name__=='__main__':unittest.main()
