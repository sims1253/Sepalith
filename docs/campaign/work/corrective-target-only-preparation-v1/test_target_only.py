import ast
import copy
import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/26a07c58eebc1769a224a193a7b163e989125e4117236cc367b940e706d7af6f/source/experiments/training')
sys.path[:0] = [str(HERE/'candidate'), str(SOURCE)]
from campaign_sft_data import full_text_collator, target_only_collator, target_only_pilot_policy
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def row(prompt=(23,24), body=(31,32), terminal=(41,42)):
    return {'input_ids':[0,*prompt,*body,*terminal,1], 'target_start':1+len(prompt),
            'target_body_tokens':list(body), 'target_terminal_tokens':list(terminal)}


class TargetOnlyTests(unittest.TestCase):
    def test_variable_lengths_prompt_attention_eos_and_padding(self):
        rows=[row(),row(prompt=(23,24,25,26),body=(33,)),row(body=())]
        batch=target_only_collator(rows)
        for i,r in enumerate(rows):
            start,end=r['target_start'],len(r['input_ids'])
            self.assertEqual(batch['input_ids'][i,:end].tolist(),r['input_ids'])
            self.assertEqual(batch['attention_mask'][i,:end].tolist(),[1]*end)
            self.assertEqual(batch['labels'][i,:start].tolist(),[-100]*start)
            self.assertEqual(batch['labels'][i,start:end].tolist(),r['input_ids'][start:])
            self.assertTrue((batch['labels'][i,end:]==-100).all())
            self.assertTrue((batch['attention_mask'][i,end:]==0).all())
            self.assertEqual(batch['labels'][i,end-1].item(),1)
        self.assertEqual((batch['labels']!=-100).sum().item(),sum(len(r['input_ids'])-r['target_start'] for r in rows))

    def test_causal_shift_loss_and_logit_gradients(self):
        rows=[row(),row(body=())]
        labels=target_only_collator(rows)['labels']
        logits=torch.zeros((*labels.shape,64),dtype=torch.float64,requires_grad=True)
        loss=torch.nn.functional.cross_entropy(logits[:,:-1,:].reshape(-1,64),labels[:,1:].reshape(-1),ignore_index=-100)
        self.assertAlmostEqual(loss.item(),math.log(64),places=12)
        loss.backward()
        supervised=torch.nn.functional.pad(labels[:,1:]!=-100,(0,1),value=False)
        self.assertTrue((logits.grad[~supervised]==0).all())
        self.assertTrue((logits.grad[supervised].abs().sum(-1)>0).all())
        for i,r in enumerate(rows):
            self.assertTrue(supervised[i,r['target_start']-1])
            self.assertTrue(supervised[i,len(r['input_ids'])-2])
        changed=logits.detach().clone();changed[~supervised]=torch.arange(64,dtype=torch.float64)*100
        other=torch.nn.functional.cross_entropy(changed[:,:-1,:].reshape(-1,64),labels[:,1:].reshape(-1),ignore_index=-100)
        self.assertEqual(other.item(),loss.item())

    def test_full_text_behavior_unchanged_and_inputs_not_mutated(self):
        rows=[row()];before=copy.deepcopy(rows)
        full=full_text_collator(rows);target=target_only_collator(rows)
        self.assertEqual(rows,before)
        self.assertEqual(full['labels'][0].tolist(),rows[0]['input_ids'])
        self.assertTrue(torch.equal(full['input_ids'],target['input_ids']))

    def test_installed_hf_accumulation_denominator_counts_only_targets(self):
        from types import SimpleNamespace
        from transformers import Trainer
        batches=[target_only_collator([row()]),target_only_collator([row(body=()),row(body=(31,))])]
        owner=SimpleNamespace(model_accepts_loss_kwargs=True,compute_loss_func=None,
            args=SimpleNamespace(average_tokens_across_devices=False,n_gpu=1),accelerator=SimpleNamespace())
        count=Trainer._get_num_items_in_batch(owner,batches,torch.device('cpu'))
        self.assertEqual(count.item(),sum((b['labels'][:,1:]!=-100).sum().item() for b in batches))
        self.assertEqual(count.item(),12)

    def test_noop_edit_and_deletion_targets_remain_fully_supervised(self):
        for r in [row(body=(31,32,33)),row(body=(34,)),row(body=())]:
            b=target_only_collator([r]);self.assertEqual(b['labels'][0,r['target_start']:].tolist(),r['target_body_tokens']+r['target_terminal_tokens']+[1])

    def test_reject_boundary_truncation_pre_padding_and_incomplete_targets(self):
        changes=[lambda r:r['input_ids'].pop(),lambda r:r['input_ids'].append(1),lambda r:r['input_ids'].__setitem__(0,2),
                 lambda r:r.update(target_start=0),lambda r:r.update(target_start=True),lambda r:r.update(target_start=len(r['input_ids'])),
                 lambda r:r['target_body_tokens'].append(35),lambda r:r.update(target_terminal_tokens=[]),
                 lambda r:r['input_ids'].__setitem__(1,True),lambda r:r['input_ids'].__setitem__(1,130560)]
        for change in changes:
            r=row();change(r)
            with self.assertRaises(ValueError):target_only_collator([r])
        self.assertEqual(target_only_collator([row(prompt=(23,)*4090)])['input_ids'].shape[1],4096)
        with self.assertRaises(ValueError):target_only_collator([row(prompt=(23,)*4091)])
        with self.assertRaises(ValueError):target_only_collator([])

    def test_policy_requires_fresh_theta0_then_matched_full25_resume(self):
        import json
        recipe=json.loads((HERE/'recipe-candidate.json').read_text())
        self.assertTrue(target_only_pilot_policy(recipe))
        resumed=copy.deepcopy(recipe);resumed['decision_steps']=[50];resumed['resume_from']='/ROOT_ACCEPTED_PILOT_FULL25'
        self.assertTrue(target_only_pilot_policy(resumed))
        for key,value in [('learning_rate',2e-4),('max_steps',50),('max_sequence_tokens',2048)]:
            bad=copy.deepcopy(recipe);bad['parameters'][key]=value
            with self.assertRaises(ValueError):target_only_pilot_policy(bad)
        for change in [lambda r:r.update(decision_steps=[100]),lambda r:r.update(resume_from='/latest-corrective'),
                       lambda r:r['identity']['parent'].update(revision='latest-corrective'),lambda r:r['identity']['policy'].update(full_text_labels=True)]:
            bad=copy.deepcopy(recipe);change(bad)
            with self.assertRaises(ValueError):target_only_pilot_policy(bad)

    def test_production_wiring_retains_boundaries_and_enforces_stop50(self):
        source=(HERE/'candidate/campaign_sft.py').read_text();tree=ast.parse(source)
        self.assertIn('"target_start", "target_body_tokens", "target_terminal_tokens"',source)
        self.assertIn('remove_unused_columns=not target_only',source)
        self.assertIn('data_collator=target_only_collator if target_only else full_text_collator',source)
        self.assertIn('({50} if target_only else set())',source)
        self.assertIn('restored["step"] != 25',source)
        self.assertIn('learning_rate=params["learning_rate"]',source)
        self.assertIsInstance(tree,ast.Module)


if __name__=='__main__':unittest.main(verbosity=2)
