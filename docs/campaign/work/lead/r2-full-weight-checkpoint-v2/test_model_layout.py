import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from model_layout import validate_dense_weights

class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/'config.json').write_text('{}')
    def tearDown(self):
        self.tmp.cleanup()
    def save(self, name, tensor='model.weight'):
        save_file({tensor: np.ones((2,3),dtype=np.float32)}, str(self.root/name))
    def index(self, mapping):
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))
    def test_single_dense(self):
        self.save('model.safetensors')
        self.assertEqual(validate_dense_weights(self.root)['tensor_count'],1)
    def test_shards(self):
        self.save('part1.safetensors','a');self.save('part2.safetensors','b')
        self.index({'a':'part1.safetensors','b':'part2.safetensors'})
        self.assertEqual(validate_dense_weights(self.root)['tensor_count'],2)
    def test_missing_shard(self):
        self.index({'a':'missing.safetensors'})
        with self.assertRaises(ValueError):validate_dense_weights(self.root)
    def test_path_traversal(self):
        self.index({'a':'../external.safetensors'})
        with self.assertRaises(ValueError):validate_dense_weights(self.root)
    def test_wrong_tensor_mapping(self):
        self.save('part1.safetensors','a');self.index({'b':'part1.safetensors'})
        with self.assertRaises(ValueError):validate_dense_weights(self.root)
    def test_adapter_rejected(self):
        self.save('model.safetensors','layer.lora_A.default.weight')
        with self.assertRaises(ValueError):validate_dense_weights(self.root)
    def test_ambiguous_layout(self):
        self.save('model.safetensors');self.index({'model.weight':'model.safetensors'})
        with self.assertRaises(ValueError):validate_dense_weights(self.root)


class DenseSealTests(unittest.TestCase):
    setUp = LayoutTests.setUp
    tearDown = LayoutTests.tearDown
    save = LayoutTests.save
    def test_full_resume_requires_states_and_detects_tamper(self):
        from campaign_checkpoint import seal_checkpoint, verify_checkpoint, FULL_STATE_FILES
        self.save('model.safetensors')
        identity={key: {'id':'fixture'} for key in ('parent','tokenizer','renderer','data','source','policy','schedule')}
        with self.assertRaisesRegex(ValueError,'not resumable'):
            seal_checkpoint(self.root,identity,1,full=True,sampler={'cursor':1},checkpoint_kind='full_weights')
        for name in FULL_STATE_FILES:
            (self.root/name).write_text('{"global_step":1}' if name=='trainer_state.json' else 'fixture')
        manifest=seal_checkpoint(self.root,identity,1,full=True,sampler={'cursor':1},checkpoint_kind='full_weights')
        self.assertEqual(manifest['checkpoint_kind'],'full_weights')
        verify_checkpoint(self.root,identity,require_full=True)
        (self.root/'optimizer.pt').write_text('changed')
        with self.assertRaisesRegex(ValueError,'bytes differ'):
            verify_checkpoint(self.root,identity,require_full=True)

if __name__=='__main__':unittest.main()

class KindBindingTests(unittest.TestCase):
    setUp = LayoutTests.setUp
    tearDown = LayoutTests.tearDown
    save = LayoutTests.save
    def seal(self,kind):
        from campaign_checkpoint import seal_checkpoint,FULL_STATE_FILES
        if kind=='full_weights':self.save('model.safetensors')
        else:
            (self.root/'adapter_model.safetensors').write_text('adapterfixture')
            (self.root/'adapter_config.json').write_text('{}')
        for name in FULL_STATE_FILES:
            (self.root/name).write_text('{"global_step":1}' if name=='trainer_state.json' else 'fixture')
        identity={key:{'id':'fixture'} for key in ('parent','tokenizer','renderer','data','source','policy','schedule')}
        return seal_checkpoint(self.root,identity,1,full=True,sampler={'cursor':1},checkpoint_kind=kind)
    def test_adapter_cannot_resume_dense(self):
        from campaign_checkpoint import verify_checkpoint
        self.seal('adapter')
        with self.assertRaisesRegex(ValueError,'kind differs'):
            verify_checkpoint(self.root,require_full=True,expected_checkpoint_kind='full_weights')
    def test_manifest_kind_tampering_rejected(self):
        from campaign_checkpoint import verify_checkpoint
        manifest=self.seal('full_weights');manifest['checkpoint_kind']='adapter'
        (self.root/'campaign-manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'kind differs'):
            verify_checkpoint(self.root,require_full=True)
    def test_manifest_step_tampering_rejected(self):
        from campaign_checkpoint import verify_checkpoint
        manifest=self.seal('full_weights');manifest['step']=2
        (self.root/'campaign-manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'state and manifest disagree'):
            verify_checkpoint(self.root,require_full=True,expected_checkpoint_kind='full_weights')
    def test_dense_archive_preserves_kind(self):
        from campaign_checkpoint import archive_checkpoint,verify_checkpoint
        self.seal('full_weights')
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'archive';archive_checkpoint(self.root,dest)
            self.assertEqual(verify_checkpoint(dest,require_full=True,expected_checkpoint_kind='full_weights')['checkpoint_kind'],'full_weights')
