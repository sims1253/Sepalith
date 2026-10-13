"""Real installed Dataset/SFTTrainer seams, using an in-memory CPU toy model."""
import copy, datetime, json, os, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager

os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'source/experiments/training'))
import torch
torch.set_num_threads(1)
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, PretrainedConfig, TrainerCallback
from trl import SFTConfig
from campaign_sft import sequential_sft_trainer_class, runtime_options, training_configuration_guard
from campaign_sft_data import target_only_collator
from campaign_checkpoint import preserve_random_state
from campaign_control import control_callback
from target_only_gate import target_only_startup_gate


def rows(count=16):
    return [{'input_ids':[0,2+i,20]+[21]*(i%3)+[31]*(1+i%3)+[41,42,1],
             'target_start':3+i%3,'target_body_tokens':[31]*(1+i%3),'target_terminal_tokens':[41,42]}
            for i in range(count)]


def tokenizer():
    tok=Tokenizer(WordLevel({str(i):i for i in range(64)},unk_token='63'))
    return PreTrainedTokenizerFast(tokenizer_object=tok,bos_token='0',eos_token='1',pad_token='1',unk_token='63')


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scalar=torch.nn.Parameter(torch.tensor(0.1));self.config=PretrainedConfig(vocab_size=64,bos_token_id=0,eos_token_id=1,pad_token_id=1,use_cache=True)
        self.config.use_cache=True
        self.config._name_or_path='local-synthetic';self.witness=None;self.seen=[];self.resume_mode=False
    def forward(self,input_ids,attention_mask=None,labels=None,num_items_in_batch=None,num_logits_to_keep=0,**kwargs):
        if self.resume_mode:
            self.seen.extend(input_ids[:,1].tolist())
            # Order-sensitive differentiable CPU objective; not a model-quality test.
            loss=((self.scalar-input_ids[:,1].float()/16000)**2).sum()/max(1,len(input_ids))
            return {'loss':loss,'logits':None}
        width=num_logits_to_keep or input_ids.shape[1]
        logits=torch.zeros((len(input_ids),width,64))+self.scalar*0
        if labels is None:return SimpleNamespace(logits=logits)
        count=num_items_in_batch if num_items_in_batch is not None else labels[:,1:].ne(-100).sum()
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,64),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/count
        if self.witness is not None:self.witness.append({'denominator':int(count)})
        return {'loss':loss,'logits':object()}


def trainer(path, data, *, microbatch=2, model=None, max_steps=501, save_steps=250):
    return sequential_sft_trainer_class()(
        model=model or Tiny(),processing_class=tokenizer(),train_dataset=Dataset.from_list(data),
        data_collator=target_only_collator,args=SFTConfig(
            output_dir=str(path),use_cpu=True,bf16=False,fp16=False,gradient_checkpointing=False,per_device_train_batch_size=microbatch,
            gradient_accumulation_steps=16//microbatch,max_steps=max_steps,learning_rate=0.001,
            warmup_ratio=.03,lr_scheduler_type='cosine',optim='adamw_torch',weight_decay=0,
            seed=3407,data_seed=3407,remove_unused_columns=False,dataloader_num_workers=0,
            dataset_kwargs={'skip_prepare_dataset':True},packing=False,completion_only_loss=False,
            max_length=4096,save_steps=save_steps,save_strategy='steps',save_only_model=False,
            ignore_data_skip=False,logging_strategy='no',report_to='none',disable_tqdm=True,
        ))


class RealTrainerTests(unittest.TestCase):
    def test_real_dataset_trainer_collator_and_denominator_all_geometries(self):
        self.assertFalse(torch.cuda.is_initialized())
        for micro in (4,2,1):
            with tempfile.TemporaryDirectory(dir=HERE.parent) as temp:
                t=trainer(Path(temp),rows(),microbatch=micro)
                t.create_optimizer();loader=t.get_train_dataloader()
                admitted=copy.deepcopy([t.train_dataset[i] for i in range(16)])
                @contextmanager
                def observe(labels,count):
                    calls=[];t.model.witness=calls
                    try:yield calls
                    finally:t.model.witness=None
                # Same production Trainer.compute_loss path; replace only the CUDA
                # fused-symbol observer with a toy-model call witness.
                original=t.compute_loss
                def loss(*args,**kwargs):
                    value,output=original(*args,**kwargs)
                    return value,SimpleNamespace(logits=output['logits'])
                t.compute_loss=loss
                result=target_only_startup_gate(t,t.model,loader,gate_clock_step=0,
                    admitted_rows=admitted,preserve_state=preserve_random_state,
                    observe=observe,allowed_start_steps=(0,250,500),expected_accumulation=16//micro)
                self.assertEqual(result['microbatches_checked'],16//micro)
                self.assertEqual(result['reference_rows_checked'],micro)
                self.assertEqual(result['target_denominator'],79)
                self.assertEqual(len(t.optimizer.state),0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_runtime_change_is_task_only(self):
        self.assertEqual(runtime_options({'stage':'task_sft_prm03_v1'}),{
            'loader_kwargs':{'use_gradient_checkpointing':True},'gradient_checkpointing':True,'logging_steps':1})
        for stage in ('sft','finish_correction_target_only_pilot50_v1'):
            self.assertEqual(runtime_options({'stage':stage}),{
                'loader_kwargs':{},'gradient_checkpointing':'unsloth','logging_steps':20})

    def test_eval_failure_restores_rng_mode_cache_and_checkpointing(self):
        model=Tiny();model.gradient_checkpointing=True
        rng=torch.get_rng_state().clone();before=float(model.scalar.detach())
        def restore(m,use_gradient_checkpointing):
            m.gradient_checkpointing=use_gradient_checkpointing;m.train()
        with self.assertRaisesRegex(RuntimeError,'synthetic eval'):
            with preserve_random_state(model):
                with training_configuration_guard(model,restore):
                    model.config.use_cache=False;model.gradient_checkpointing=False
                    torch.rand(9);raise RuntimeError('synthetic eval')
        self.assertTrue(model.training);self.assertTrue(model.config.use_cache)
        self.assertTrue(model.gradient_checkpointing);self.assertEqual(float(model.scalar.detach()),before)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_actual_trainer_resume250_and500_restores_draws_optimizer_scheduler(self):
        data=rows(16000);begin=[]
        class BeginWitness(TrainerCallback):
            def on_train_begin(self,args,state,control,**kwargs):
                optimizer=kwargs['optimizer'];scheduler=kwargs['lr_scheduler']
                begin.append({'step':state.global_step,'optimizer_steps':[
                    int(value['step']) for value in optimizer.state.values()],
                    'scheduler_last_epoch':scheduler.last_epoch,'model_training':kwargs['model'].training})
        with tempfile.TemporaryDirectory(dir=HERE.parent) as temp:
            base=Path(temp)
            def run(name,stop,resume=None):
                m=Tiny();m.resume_mode=True
                t=trainer(base/name,data,model=m,max_steps=1000)
                t.add_callback(control_callback(telemetry_path=base/(name+'.jsonl'),
                    identity={'synthetic_cpu_resume_only':True},
                    deadline=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=5)).isoformat(),
                    reserve_seconds=5,stop_steps=[stop]))
                t.add_callback(BeginWitness())
                t.train(resume_from_checkpoint=str(resume) if resume else None)
                self.assertEqual(t.state.global_step,stop)
                return t,m
            baseline,bmodel=run('baseline',501)
            first,m1=run('first',250)
            checkpoint250=base/'first/checkpoint-250'
            second,m2=run('second',500,checkpoint250)
            checkpoint500=base/'second/checkpoint-500'
            third,m3=run('third',501,checkpoint500)
            expected=list(range(2,8018))
            self.assertEqual(bmodel.seen,expected)
            self.assertEqual(m1.seen,list(range(2,4002)))
            self.assertEqual(m2.seen,list(range(4002,8002)))
            self.assertEqual(m3.seen,list(range(8002,8018)))
            self.assertTrue(torch.equal(bmodel.scalar,m3.scalar))
            self.assertEqual(baseline.lr_scheduler.state_dict(),third.lr_scheduler.state_dict())
            left=baseline.optimizer.state_dict();right=third.optimizer.state_dict()
            self.assertEqual(left['param_groups'],right['param_groups'])
            for key,value in left['state'][0].items():
                if torch.is_tensor(value):self.assertTrue(torch.equal(value,right['state'][0][key]),key)
                else:self.assertEqual(value,right['state'][0][key])
            for witness in begin:
                self.assertTrue(witness['model_training'])
                self.assertEqual(witness['scheduler_last_epoch'],witness['step'])
                self.assertEqual(witness['optimizer_steps'],[witness['step']] if witness['step'] else [])
            for step,path in [(250,checkpoint250),(500,checkpoint500)]:
                self.assertEqual(json.loads((path/'trainer_state.json').read_text())['global_step'],step)
                for name in ('optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin'):
                    self.assertGreater((path/name).stat().st_size,0)
            (HERE.parent/'cpu-resume-evidence.json').write_text(json.dumps({
                'status':'PASS','synthetic_only':True,'actual_class':type(third).__name__,
                'max_steps_horizon':1000,'effective_batch':16,'microbatch':2,'gradient_accumulation':8,
                'checkpoints':[250,500],'next_draw_offsets':[4000,8000],
                'baseline_and_resumed_terminal_step':501,'exact_parameter_optimizer_scheduler_equality':True,
                'actual_train_begin_state':begin,
                'limitations':['Toy CPU objective and adamw_torch; no real LoRA, CUDA fused loss or model quality.',
                    'Real checkpoint sealing and dataset/source identity admission remain root gates.',
                    'Startup target witness reads dataset offset0; this separate test verifies Trainer resume skips.']},indent=2)+'\n')
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main(verbosity=2)
