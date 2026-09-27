from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'corrective-target-only-preparation-v1/candidate'),'/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/26a07c58eebc1769a224a193a7b163e989125e4117236cc367b940e706d7af6f/source/experiments/training']
from target_only_gate import target_only_startup_gate, verify_batch, observe_fused_call
from campaign_sft_data import target_only_collator
from transformers import Trainer


def rows():
    return [{'input_ids':[0,23,24]+[25]*(i%3)+[31,41,42,1], 'target_start':3+i%3,
             'target_body_tokens':[31], 'target_terminal_tokens':[41,42]} for i in range(16)]


class LogitFixture:
    def __init__(self):self.training=True;self.config=SimpleNamespace(use_cache=True);self.scalar=torch.tensor(0.0)
    def parameters(self):return iter([self.scalar])
    def modules(self):return iter([self])
    def train(self,mode=True):self.training=mode;return self
    def __call__(self,input_ids,attention_mask,**kwargs):
        return SimpleNamespace(logits=torch.zeros((len(input_ids),kwargs.get('num_logits_to_keep',input_ids.shape[1]),64)))


class TrainerFixture:
    def __init__(self):
        self.train_dataset=rows();self.admitted_rows=deepcopy(self.train_dataset);self.model_accepts_loss_kwargs=True;self.compute_loss_func=None;self.label_smoother=None
        self.optimizer=SimpleNamespace(state={});self.accelerator=SimpleNamespace()
        self.args=SimpleNamespace(remove_unused_columns=False,dataloader_num_workers=0,gradient_accumulation_steps=4,
                                 device=torch.device('cpu'),average_tokens_across_devices=False,n_gpu=1)
        self.generator=torch.Generator().manual_seed(3407)
        self.loader=torch.utils.data.DataLoader(self.train_dataset,batch_size=4,collate_fn=target_only_collator,generator=self.generator)
        self.calls=None;self.factor=1;self.denominator_delta=0;self.witness=True;self.return_nan=False
    def _get_num_items_in_batch(self,*args):return Trainer._get_num_items_in_batch(self,*args)+self.denominator_delta
    def get_batch_samples(self,*args):return Trainer.get_batch_samples(self,*args)
    def _prepare_inputs(self,batch):return batch
    def compute_loss_context_manager(self):return nullcontext()
    def compute_loss(self,model,batch,return_outputs,num_items_in_batch):
        if self.witness:self.calls.append({'denominator':int(num_items_in_batch)})
        logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,64),batch['labels'][:,1:].reshape(-1),ignore_index=-100,reduction='sum')/num_items_in_batch
        if self.return_nan:loss=loss*float('nan')
        return loss*self.factor,SimpleNamespace(logits=object())
    @contextmanager
    def observe(self,labels,count):
        self.calls=[]
        try:yield self.calls
        finally:self.calls=None


@contextmanager
def preserve(model):
    state=torch.get_rng_state();mode=model.training
    try:model.train(False);yield
    finally:torch.set_rng_state(state);model.train(mode)


class GateTests(unittest.TestCase):
    def run_gate(self,trainer=None,model=None):
        trainer=trainer or TrainerFixture();model=model or LogitFixture()
        return target_only_startup_gate(trainer,model,trainer.loader,gate_clock_step=0,admitted_rows=trainer.admitted_rows,preserve_state=preserve,observe=trainer.observe)
    def test_actual_dataloader_and_installed_hf_denominator_pass(self):
        trainer=TrainerFixture();model=LogitFixture();rng=torch.get_rng_state();gen=trainer.generator.get_state()
        result=self.run_gate(trainer,model)
        self.assertEqual(result['target_denominator'],64);self.assertEqual(result['actual_fused_calls'],1)
        self.assertEqual(result['microbatch_target_denominators'],[16]*4)
        self.assertAlmostEqual(result['fused_loss'],result['reference_loss'],places=5)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(torch.equal(gen,trainer.generator.get_state()))
        self.assertTrue(model.training);self.assertTrue(model.config.use_cache);self.assertFalse(result['resume_equivalence_proven'])
    def test_mask_mutations_fail(self):
        expected=rows()[:4]
        for coordinate in [(0,0),(0,3),(0,6),(0,7)]:
            batch=target_only_collator(expected);batch['labels'][coordinate]=23
            with self.assertRaises(ValueError):verify_batch(batch,expected)
    def test_changed_data_order_or_missing_metadata_fails(self):
        trainer=TrainerFixture();trainer.train_dataset[0]=deepcopy(trainer.train_dataset[0]);del trainer.train_dataset[0]['target_start']
        with self.assertRaisesRegex(ValueError,'post_trainer_row_differs'):self.run_gate(trainer)
        batch=target_only_collator(rows()[:4]);batch['input_ids'][0,1]=26
        with self.assertRaisesRegex(ValueError,'changed_tokens'):verify_batch(batch,rows()[:4])
        trainer=TrainerFixture();trainer.train_dataset[0]['input_ids'][1]=26
        with self.assertRaisesRegex(ValueError,'post_trainer_row_differs'):self.run_gate(trainer)
    def test_bad_denominator_fails_before_forward(self):
        trainer=TrainerFixture();trainer.denominator_delta=1
        with self.assertRaisesRegex(ValueError,'trainer_denominator'):self.run_gate(trainer)
    def test_bad_fused_loss_or_unwitnessed_path_fails_closed(self):
        for attr,value,pattern in [('factor',2,'loss_mismatch'),('return_nan',True,'nonfinite'),('witness',False,'not_witnessed')]:
            trainer=TrainerFixture();setattr(trainer,attr,value)
            with self.assertRaisesRegex(ValueError,pattern):self.run_gate(trainer)
    def test_fresh_optimizer_state_and_objective_are_gated(self):
        for mutate,pattern in [(lambda t:t.optimizer.state.update(old='state'),'restored_optimizer'),
                              (lambda t:setattr(t,'model_accepts_loss_kwargs',False),'drop_denominator'),
                              (lambda t:setattr(t.args,'remove_unused_columns',True),'columns')]:
            trainer=TrainerFixture();mutate(trainer)
            with self.assertRaisesRegex(ValueError,pattern):self.run_gate(trainer)
    def test_failure_restores_runtime_state(self):
        trainer=TrainerFixture();trainer.factor=2;model=LogitFixture();model.training=False
        gen=trainer.generator.get_state();rng=torch.get_rng_state()
        with self.assertRaises(ValueError):self.run_gate(trainer,model)
        self.assertFalse(model.training);self.assertTrue(model.config.use_cache)
        self.assertTrue(torch.equal(gen,trainer.generator.get_state()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_installed_symbol_observer_checks_arguments_and_restores_on_failure(self):
        import types
        from unittest.mock import patch
        top=types.ModuleType('unsloth');models=types.ModuleType('unsloth.models');llama=types.ModuleType('unsloth.models.llama')
        top.models=models;models.llama=llama
        original=lambda **kwargs:torch.tensor(0.5)
        llama.unsloth_fused_ce_loss=original;labels=torch.tensor([[-100,31,1]])
        with patch.dict(sys.modules,{'unsloth':top,'unsloth.models':models,'unsloth.models.llama':llama}):
            with observe_fused_call(labels,2) as calls:
                self.assertEqual(float(llama.unsloth_fused_ce_loss(labels=labels,n_items=2)),0.5)
                self.assertEqual(len(calls),1)
            self.assertIs(llama.unsloth_fused_ce_loss,original)
            with self.assertRaisesRegex(ValueError,'denominator_changed'):
                with observe_fused_call(labels,2):llama.unsloth_fused_ce_loss(labels=labels,n_items=3)
            self.assertIs(llama.unsloth_fused_ce_loss,original)


if __name__=='__main__':unittest.main(verbosity=2)
