from __future__ import annotations
import hashlib,json,random,sys,tempfile,unittest
from pathlib import Path
PACKET=Path(__file__).resolve().parents[1];TRAINING=PACKET/"source/experiments/training";sys.path.insert(0,str(TRAINING))
import bind_representative_cpt as binder
import full_weight_cpt_trainer as trainer
from campaign_cpt_data import validate_materialized_rows
from saved_precision import restore_saved_fp32
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def row(rid,doc,pkg,payload,idx=0,start=0,overlap=0,final=True):
 ids=[0,*payload,1];labels=[-100,*payload,1 if final else -100]
 for i in range(overlap):labels[1+i]=-100
 return {"schema":1,"row_id":rid,"document_id":doc,"package":pkg,"group_id":"g-"+doc,"cpt_partition":"cpt_train","source_path":"TRAIN/"+doc+".R","source_sha256":hashlib.sha256(doc.encode()).hexdigest(),"chunk_index":idx,"input_ids":ids,"labels":labels,"attention_mask":[1]*len(ids),"source_token_start":start,"source_token_end":start+len(payload)-overlap,"overlap_context_tokens":overlap,"is_document_end":final,"supervised_tokens":sum(x!=-100 for x in labels)}
def fixtures():
 rows=[row("d0-c0","d0","p0",[4,5],0,0,0,False),row("d0-c1","d0","p0",[5,6,7],1,2,1,True)]
 rows += [row(f"d{i}-c0",f"d{i}",f"p{i%3}",list(range(10,10+i))) for i in range(1,10)]
 return rows
class TestRepresentative(unittest.TestCase):
 def dataset(self,rows):
  root=Path(self.tmp.name);rp=root/"rows.jsonl";rp.write_text("".join(json.dumps(x)+"\n" for x in rows));summary=validate_materialized_rows(rows,max_sequence_tokens=8192)
  ids=[x["row_id"] for x in rows];random.Random(3407).shuffle(ids);replay=ids[:(-len(ids))%16];draws=ids+replay
  schedule={"split_id":"representative","method":"one_pass_plus_named_replay_v1","seed":3407,"max_steps":len(draws)//16,"effective_batch":16,"token_rows_sha256":sha(rp),"row_ids":draws,"replay_row_ids":replay};sp=root/"schedule.json";sp.write_text(json.dumps(schedule))
  cohort={"rows":{"path":str(rp),"sha256":sha(rp)},"draw_schedule":{"path":str(sp),"sha256":sha(sp)},"max_sequence_tokens":8192,"unique_rows":len(rows),"documents":summary["documents"],"packages":len(summary["packages"]),"input_tokens":summary["input_tokens"],"payload_tokens":summary["payload_tokens"],"loss_tokens":summary["loss_tokens"],"updates":len(draws)//16,"named_replays":len(replay)}
  return trainer.FrozenTokenRowDataset({"cohort":cohort}),cohort
 def setUp(self):self.tmp=tempfile.TemporaryDirectory()
 def tearDown(self):self.tmp.cleanup()
 def test_variable_length_complete_documents_and_exact_coverage(self):
  rows=fixtures();ds,c=self.dataset(rows);self.assertEqual(len(ds),16);self.assertEqual(len(set(ds.draws[:11])),11);self.assertEqual(len(ds.draws)-11,5);self.assertGreater(max(len(ds[i]["input_ids"]) for i in range(11)),min(len(ds[i]["input_ids"]) for i in range(11)))
  for rid in {x["row_id"] for x in rows}:self.assertEqual(ds.draws[:11].count(rid),1)
 def test_incomplete_document_and_bad_mask_fail(self):
  with self.assertRaises(Exception):self.dataset(fixtures()[1:])
  rows=fixtures();rows[2]["labels"][1]=-100
  with self.assertRaises(Exception):self.dataset(rows)
 def test_mandatory_stop_and_resume_cursor(self):
  runtime={"mandatory_stop_step":8,"max_steps":16}
  self.assertEqual(trainer.milestone_action(8,0,runtime),{"save":True,"stop":True,"cursor":128})
  self.assertEqual(trainer.milestone_action(8,128,runtime),{"save":True,"stop":False,"cursor":128})
  self.assertEqual(trainer.milestone_action(16,128,runtime),{"save":True,"stop":False,"cursor":256})
 def test_resume_requires_exact_root_continuation(self):
  root=Path(self.tmp.name);recipe=root/"bound.json";recipe.write_text("{}\n");checkpoint=root/"checkpoint-8";checkpoint.mkdir();manifest=checkpoint/"campaign-manifest.json";manifest.write_text("{}\n");r={"runtime":{"mandatory_stop_step":8}}
  with self.assertRaisesRegex(ValueError,"requires a root continuation"):trainer.validate_continuation(recipe,r,checkpoint,None)
  admission=root/"continue.json";admission.write_text(json.dumps({"schema":"sepalith.sft11.cpt-representative-continuation-admission.v2","status":"admitted","decision":"continue","bound_recipe_sha256":sha(recipe),"checkpoint":str(checkpoint),"checkpoint_manifest_sha256":sha(manifest),"step":8}))
  self.assertEqual(trainer.validate_continuation(recipe,r,checkpoint,admission)["sha256"],sha(admission))
  value=json.loads(admission.read_text());value["checkpoint_manifest_sha256"]="0"*64;admission.write_text(json.dumps(value))
  with self.assertRaisesRegex(ValueError,"manifest differs"):trainer.validate_continuation(recipe,r,checkpoint,admission)
 def test_missing_root_admission_fails(self):
  root=Path(self.tmp.name);template=root/"template.json";template.write_text(json.dumps({"schema":"sepalith.sft11.full-weight-cpt-representative-template.v2","launch_authorized":False}));admission=root/"admission.json";admission.write_text(json.dumps({"schema":"sepalith.sft11.full-weight-cpt-representative-root-admission.v2","status":"pending","launch_authorized":False}))
  with self.assertRaisesRegex(ValueError,"root admission missing"):binder.bind(template,admission,root/"bound.json")
 def test_runtime_is_not_fixed_to_old_384_rows_or_24_steps(self):
  recipe={"seed":3407,"runtime":{"micro_batch":1,"gradient_accumulation":16,"max_steps":73,"learning_rate":3e-5,"scheduler":"cosine","warmup_ratio":.02,"checkpoint_every":8}}
  kwargs=trainer.training_arguments_kwargs(recipe,Path(self.tmp.name));self.assertEqual(kwargs["max_steps"],73);self.assertEqual(kwargs["save_steps"],8);self.assertEqual(kwargs["gradient_accumulation_steps"],16)
 def test_source_retains_fullweight_resume_and_tokenizer_gates(self):
  source=(TRAINING/"full_weight_cpt_trainer.py").read_text();self.assertIn('expected_checkpoint_kind="full_weights"',source);self.assertIn('restore_trainer_eog_alignment(model, tokenizer, reference)',source);self.assertIn('EXPECTED_PARAMETER_TENSORS = 381',source);self.assertIn('initial_cursor = prior["step"] * 16',source);self.assertNotIn('max_steps"] == 24',source)
  self.assertLess(source.index("FastLanguageModel.for_training"),source.index("restore_saved_fp32(model"));self.assertLess(source.index("restore_saved_fp32(model"),source.index("named = list(model.named_parameters())"))
  self.assertEqual(sha(TRAINING/"saved_precision.py"),"3a46d3f301d2f6ade8332b0426edd182a16a7b52c44f28aee60a7b27af144629")

class TestSavedPrecision(unittest.TestCase):
 def test_exact_restore_is_idempotent_and_bf16_is_untouched(self):
  import torch
  from safetensors.torch import save_file
  class Model(torch.nn.Module):
   def __init__(self):
    super().__init__();self.norm=torch.nn.Parameter(torch.tensor([1.0,2.0],dtype=torch.bfloat16));self.dense=torch.nn.Parameter(torch.tensor([3.0,4.0],dtype=torch.bfloat16))
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/"model.safetensors";saved=torch.tensor([1.0001,2.0003],dtype=torch.float32);save_file({"norm":saved,"dense":torch.tensor([9.0,10.0],dtype=torch.bfloat16)},str(path))
   model=Model();dense_before=model.dense.detach().clone();first=restore_saved_fp32(model,path)
   self.assertEqual(first["fp32_tensors_restored"],1);self.assertEqual(model.norm.dtype,torch.float32);self.assertTrue(torch.equal(model.norm.detach(),saved));self.assertEqual(model.dense.dtype,torch.bfloat16);self.assertTrue(torch.equal(model.dense.detach(),dense_before))
   second=restore_saved_fp32(model,path);self.assertEqual(second["tensors"][0]["changed_elements"],0);self.assertTrue(torch.equal(model.dense.detach(),dense_before))
 def test_invalid_saved_fp32_norms_fail_closed(self):
  import torch
  from safetensors.torch import save_file
  class Model(torch.nn.Module):
   def __init__(self):super().__init__();self.norm=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16))
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/"bad.safetensors";save_file({"norm":torch.tensor([float("nan"),1.0],dtype=torch.float32)},str(path))
   with self.assertRaises(AssertionError):restore_saved_fp32(Model(),path)
   save_file({"norm":torch.ones(3,dtype=torch.float32)},str(path))
   with self.assertRaises(AssertionError):restore_saved_fp32(Model(),path)
if __name__=="__main__":unittest.main(verbosity=2)
