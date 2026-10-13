import sys,os,tempfile,json
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'r2-full-weight-cpt-stage-transition-v1/tests'
sys.path.insert(0,str(p))
import test_hf_trainer_transition as t
with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
 root=Path(td);t.torch.manual_seed(818)
 def args(path,steps,ignore):
  a=t.arguments(path,steps,ignore);a.gradient_accumulation_steps=16;return a
 source=t.TrackTrainer(model=t.Tiny(),args=args(root/'source',2,False),train_dataset=t.Rows(range(100,132)),data_collator=t.collate,callbacks=[t.SaveStop(2)])
 source.train(); checkpoint=root/'source/checkpoint-2'
 def run(name,cursor,resume,stop=None):
  seen=[]; cb=t.SaveStop(stop or 5,stop is not None)
  trainer=t.TrackTrainer(model=t.Tiny(),args=args(root/name,5,True),train_dataset=t.StageCursorView(t.Rows(range(48)),cursor),data_collator=t.collate,callbacks=[cb],seen=seen)
  trainer.train(resume_from_checkpoint=str(resume));return trainer,seen,cb
 direct,rows,cb=run('direct',0,checkpoint)
 first,firstrows,_=run('split',0,checkpoint,3)
 resumed,lastrows,_=run('resume',16,root/'split/checkpoint-3')
 assert rows==list(range(48)) and firstrows==list(range(16)) and lastrows==list(range(16,48))
 assert direct.state.global_step==resumed.state.global_step==5
 assert t.state_equal(direct.model.state_dict(),resumed.model.state_dict())
 assert t.state_equal(direct.optimizer.state_dict(),resumed.optimizer.state_dict())
 assert t.state_equal(direct.lr_scheduler.state_dict(),resumed.lr_scheduler.state_dict())
 assert all(x==1e-3 for x in cb.lrs[0])
 print(json.dumps({'status':'pass','gradient_accumulation':16,'source_steps':2,'destination_draws':48,'interrupted_after_draws':16,'resumed_draws':32,'terminal_global_step':5,'exact_model_optimizer_scheduler_parity':True}))
