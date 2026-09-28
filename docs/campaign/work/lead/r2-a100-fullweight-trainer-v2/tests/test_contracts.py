import hashlib, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

PACKET=Path(__file__).parents[1]; SOURCE=PACKET/"source"/"experiments"/"training"
sys.path.insert(0,str(SOURCE))
from a100_distributed import (checkpoint_rng_contract, create_initial_resume_view,
    ddp_training_arguments, rounding_seed, validate_ddp_window, validate_gathered_window)
from a100_execution_transition import validate

class Contracts(unittest.TestCase):
 def test_geometry_and_seed_contracts(self):
    a=ddp_training_arguments(output_dir="/tmp/x",max_steps=2,learning_rate=3e-6,warmup_steps=2,seed=3407)
    assert (a["per_device_train_batch_size"],a["gradient_accumulation_steps"],a["average_tokens_across_devices"])==(2,1,True)
    assert rounding_seed(3407,91)==rounding_seed(3407,91)!=rounding_seed(3407,92)
    ranks=[validate_ddp_window([384+r*2,385+r*2],initial_cursor=384,local_update=0,rank=r) for r in range(8)]
    assert validate_gathered_window(ranks,first=384)==list(range(384,400))
    with self.assertRaises(ValueError): validate_ddp_window([384,386],initial_cursor=384,local_update=0,rank=0)

 def test_resume_view_and_rng_alias(self):
    temp=tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")); tmp_path=Path(temp.name)
    src=tmp_path/"source"; src.mkdir()
    files={}
    for name,data in {"model.safetensors":b"m","optimizer.pt":b"o","scheduler.pt":b"s",
                      "rng_state.pth":b"rng","trainer_state.json":b"{}"}.items():
        (src/name).write_bytes(data); files[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    manifest={"checkpoint_kind":"full_weights","full":True,"step":90,"files":files}
    (src/"campaign-manifest.json").write_text(json.dumps(manifest))
    digest=hashlib.sha256((src/"campaign-manifest.json").read_bytes()).hexdigest()
    out=tmp_path/"view"; receipt=create_initial_resume_view(src,out,manifest_sha256=digest)
    assert receipt["world_size"]==8 and all((out/f"rng_state_{r}.pth").read_bytes()==b"rng" for r in range(8))
    assert checkpoint_rng_contract(out)["rank_rng_sha256"]["0"]==hashlib.sha256(b"rng").hexdigest()
    with self.assertRaises(ValueError): create_initial_resume_view(src,tmp_path/"bad",manifest_sha256="0"*64)
    temp.cleanup()

 def test_preparation_template_exact_checkpoint(self):
    value=json.loads((PACKET/"recipe.template.json").read_text())
    assert validate(value)=={"status":"prepared_not_admitted","source_step":90,"cursor":384,"world_size":8,"large_payload_verified":False}
    bad=json.loads(json.dumps(value)); bad["runtime"]["initial_cursor"]=400
    with self.assertRaises(ValueError): validate(bad)

 def test_pinned_transformers_and_accelerate_geometry(self):
    from transformers import TrainingArguments
    from torch.utils.data import BatchSampler, SequentialSampler
    from accelerate.data_loader import BatchSamplerShard
    value=json.loads((PACKET/"recipe.template.json").read_text())
    from a100_distributed import ddp_training_arguments
    kwargs=ddp_training_arguments(output_dir=os.environ["TMPDIR"],max_steps=2,
        learning_rate=3e-6,warmup_steps=2,seed=3407)
    kwargs["use_cpu"]=True
    args=TrainingArguments(**kwargs)
    self.assertTrue(args.average_tokens_across_devices)
    base=BatchSampler(SequentialSampler(range(32)),batch_size=2,drop_last=True)
    rank_batches=[]
    for rank in range(8):
      shard=BatchSamplerShard(base,num_processes=8,process_index=rank,split_batches=False)
      rank_batches.append(next(iter(shard)))
    self.assertEqual(rank_batches,[list(range(r*2,r*2+2)) for r in range(8)])

 def test_real_eight_process_ddp_resume(self):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
      env=dict(os.environ); env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TMPDIR":tmp})
      cmd=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=8",str(PACKET/"tests"/"ddp_runtime_worker.py")]
      run=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=120)
      self.assertEqual(run.returncode,0,run.stdout+run.stderr)
      self.assertIn('"resume_exact": true',run.stdout)

if __name__=="__main__": unittest.main()
