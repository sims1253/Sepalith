"""CPU checks for the exact C/full250 to D/full500 recovery packet."""
import copy, hashlib, json, sys, tempfile, unittest
from pathlib import Path
HERE=Path(__file__).resolve(); WORK=HERE.parents[1]
TRAIN=WORK/'source/experiments/training'; sys.path.insert(0,str(TRAIN))
import campaign_checkpoint
import campaign_expanded_sft as expanded
import campaign_sft

class Resume250DPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe=json.loads((WORK/'recipe.json').read_text())
        cls.checkpoint=Path(cls.recipe['resume_from'])

    def test_actual_full250_optimizer_rng_sampler_continuity(self):
        r=self.recipe; root=self.checkpoint
        predecessor=campaign_sft.resume_checkpoint_identity(r,r['identity'])
        restored=campaign_checkpoint.verify_checkpoint(root,predecessor,require_full=True)
        state=json.loads((root/'campaign-state.json').read_text())
        self.assertEqual(restored['step'],250)
        self.assertEqual(state['sampler']['consumed_draws'],4000)
        self.assertEqual(state['sampler']['schedule_sha256'],r['draw_schedule']['sha256'])
        for name in ('adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'):
            self.assertTrue((root/name).is_file(),name)
        self.assertTrue(expanded.expanded_target_only_policy(r))

    def test_source_migration_is_only_identity_difference(self):
        r=self.recipe; predecessor=campaign_sft.resume_checkpoint_identity(r,r['identity'])
        current=copy.deepcopy(r['identity']); old=copy.deepcopy(predecessor)
        self.assertNotEqual(old.pop('source'),current.pop('source'))
        self.assertEqual(old,current)

    def test_rejects_manifest_substitution(self):
        r=copy.deepcopy(self.recipe)
        r['resume_identity_compatibility']['checkpoint_manifest_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'manifest hash'):
            campaign_sft.resume_checkpoint_identity(r,r['identity'])

    def test_packet_has_fresh_paths_and_next_boundary(self):
        r=self.recipe
        self.assertEqual(r['mandatory_stop_steps'],[500])
        self.assertEqual(r['decision_steps'],[500])
        self.assertEqual(r['parameters']['max_steps'],1000)
        self.assertEqual(r['runtime_horizon']['maximum_attempt_seconds'],6084)
        self.assertEqual(r['runtime_horizon']['outer_guard_seconds'],6144)
        self.assertNotIn('expanded-postsft500-c',r['output_dir'])
        self.assertFalse(Path(r['output_dir']).exists())
        self.assertFalse(Path(r['archive_dir']).exists())

    def test_only_evaluator_bytes_differ_from_c_source(self):
        c=WORK.parent/'r2-expanded-sft-c/source'; d=WORK/'source'
        cfiles={p.relative_to(c):hashlib.sha256(p.read_bytes()).hexdigest() for p in c.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
        dfiles={p.relative_to(d):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
        changed={str(k) for k in cfiles.keys()|dfiles.keys() if cfiles.get(k)!=dfiles.get(k)}
        self.assertEqual(changed,{'experiments/training/campaign_eval.py'})

if __name__=='__main__': unittest.main(verbosity=2)
