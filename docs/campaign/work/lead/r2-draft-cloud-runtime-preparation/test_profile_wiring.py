"""Exercise upstream loader wiring without constructing a campaign model."""
from pathlib import Path
import os,sys,runpy,json,tempfile
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/DeepSpec'),str(ROOT/'dependency-overlay')]
for key in ['SEPALITH_PUBLIC_DRAFT_WEIGHTS','SEPALITH_DRAFT_OUTPUT_ROOT','SEPALITH_DRAFT_LOG_ROOT','SEPALITH_DRAFT_CACHE','SEPALITH_MINICPM5_TARGET']:
 os.environ[key]='/UNBOUND_TEST/'+key
import torch
from deepspec.utils.config import to_config_node
from deepspec.trainer.ckpt_manager import discover_latest_checkpoint
from deepspec.utils.constant import auto_eval_command
from warmstart_trainer import SepalithWarmstartTrainer

def main():
 cfg=runpy.run_path(str(ROOT/'profile_config.py'))
 raw={k:cfg[k] for k in ['model','train','data','logging','project_name','exp_name']}
 raw['logging']['checkpoint_dir']='/EXPLICIT_OVERRIDE'
 args=to_config_node(cfg['finalize_cfg'](raw))
 assert args.logging.checkpoint_dir=='/EXPLICIT_OVERRIDE'
 assert args.data.num_workers==1 and args.logging.checkpointing_steps==5
 assert args.train.max_train_steps==8 and args.model.allow_resume is False
 assert auto_eval_command is None
 trainer=object.__new__(SepalithWarmstartTrainer)
 trainer.args=args;trainer.world_size=1;trainer.global_rank=0;trainer.samples_per_epoch=2
 sample={'input_ids':torch.tensor([0,4,1]),'loss_mask':torch.tensor([0,1,1]),'target_hidden_states':torch.zeros(3,10,dtype=torch.bfloat16),'target_last_hidden_states':torch.zeros(3,2,dtype=torch.bfloat16)}
 trainer.train_dataset=[sample,sample]
 loader=trainer._build_train_dataloader();it=iter(loader)
 try:
  batch=next(it);assert batch['input_ids'].tolist()==[[0,4,1]]
  assert batch['attention_mask'].tolist()==[[1,1,1]]
 finally:
  it._shutdown_workers()
 trainer.resume_checkpoint_dir='/EXISTING_CHECKPOINT'
 try:trainer._build_draft_model(target_config=None,model_args=args.model)
 except ValueError as e:assert 'explicit resume admission' in str(e)
 else:raise AssertionError('implicit resume was not rejected')
 with tempfile.TemporaryDirectory() as directory:
  Path(directory,'sepalith-warmstart.json').write_text('{}')
  assert discover_latest_checkpoint(directory) is None
 result={'status':'pass','actual_upstream_dataloader_batch':True,'native_eog1_mask_preserved':True,'explicit_output_override_preserved':True,'implicit_resume_rejected_before_model_creation':True,'receipt_ignored_by_checkpoint_discovery':True,'auto_eval_command':None,'model_created':False,'cuda_started':False,'torch':torch.__version__}
 (ROOT/'profile-wiring-test.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
