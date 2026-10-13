"""Negative access-order regressions with synthetic metadata only."""
import argparse,datetime,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import construct_final as constructor
import evaluate_final as native
import entry_gate as g

class ReleasedClock(datetime.datetime):
    @classmethod
    def now(cls,tz=None):return cls(2026,9,14,10,1,tzinfo=datetime.timezone.utc)

class EntryGateTests(unittest.TestCase):
    def freeze(self):
        return {'status':'frozen','weights_frozen':True,'harness_frozen':True,'final_access_unlocked':True,
            'weights_sha256':g.Q8,'harness_sha256':'a'*64,'weights_frozen_at':'2026-09-14T10:00:00Z','harness_frozen_at':'2026-09-14T10:00:00Z'}
    def args(self):
        return argparse.Namespace(freeze=Path('/synthetic/freeze.json'),harness_sha256='a'*64,
            source_manifest=Path('/synthetic/source-graph.json'),source_sha256='b'*64,
            requests=Path('/synthetic/requests.json'),case_specs=Path('/synthetic/specs.json'),
            source_authorization=Path('/synthetic/authorization.json'),train_identities=Path('/synthetic/train.json'),dev_identities=Path('/synthetic/dev.json'),
            allowed_root=[Path('/synthetic/source')],rows_manifest=Path('/synthetic/rows-manifest.json'),rows=Path('/synthetic/rows.json'),
            native_admission=Path('/synthetic/native.json'),manifest_sha256='c'*64,output=Path('/synthetic/output'),deadline_seconds=3600)
    def assert_only_freeze_touched(self,entry,freeze,clock=None):
        touched=[]
        def read(path,*args,**kwargs):
            touched.append(str(path))
            if str(path)!='/synthetic/freeze.json':raise AssertionError('non-freeze input opened before gate')
            return json.dumps(freeze)
        patch_clock=mock.patch.object(g.datetime,'datetime',clock) if clock else mock.patch.object(g,'Q8',g.Q8)
        with patch_clock,mock.patch.object(Path,'read_text',read),mock.patch.object(Path,'read_bytes',side_effect=AssertionError('row bytes touched')),mock.patch.object(Path,'stat',side_effect=AssertionError('protected path metadata touched')):
            with self.assertRaises(g.gate.FinalEvaluatorError):entry.execute(self.args())
        self.assertEqual(touched,['/synthetic/freeze.json'])
    def test_constructor_actual_clock_blocks_before_metadata(self):self.assert_only_freeze_touched(constructor,self.freeze())
    def test_native_actual_clock_blocks_before_manifest(self):self.assert_only_freeze_touched(native,self.freeze())
    def test_constructor_missing_freeze_after_release(self):
        for flag in ('weights_frozen','harness_frozen','final_access_unlocked'):
            f=self.freeze();f[flag]=False
            with self.subTest(flag=flag):self.assert_only_freeze_touched(constructor,f,ReleasedClock)
    def test_native_missing_freeze_after_release(self):
        for flag in ('weights_frozen','harness_frozen','final_access_unlocked'):
            f=self.freeze();f[flag]=False
            with self.subTest(flag=flag):self.assert_only_freeze_touched(native,f,ReleasedClock)
    def test_wrong_weight_blocks_both_entries(self):
        f=self.freeze();f['weights_sha256']='d'*64
        for entry in (constructor,native):self.assert_only_freeze_touched(entry,f,ReleasedClock)
    def test_wrong_harness_blocks_both_entries(self):
        f=self.freeze();f['harness_sha256']='d'*64
        for entry in (constructor,native):self.assert_only_freeze_touched(entry,f,ReleasedClock)
    def test_future_frozen_timestamp_blocks(self):
        f=self.freeze();f['harness_frozen_at']='2026-09-14T11:00:00Z'
        self.assert_only_freeze_touched(constructor,f,ReleasedClock)
    def test_constructor_graph_must_be_frozen_before_graph_read(self):
        with mock.patch.object(Path,'read_text',side_effect=AssertionError('graph read')):
            with self.assertRaisesRegex(ValueError,'not explicitly frozen'):g.source_graph('/synthetic/graph','b'*64,{},'constructor_source_closure_sha256')
    def test_native_identity_gate_precedes_rows_manifest(self):
        a=self.args();seen=[]
        def read(path,*args,**kwargs):
            seen.append(str(path))
            if str(path)!='/synthetic/native.json':raise AssertionError('rows manifest touched')
            return '{}'
        with mock.patch.object(g,'release',return_value=self.freeze()),mock.patch.object(g,'source_graph',return_value={}),mock.patch.object(Path,'read_text',read),mock.patch.object(native.evaluator,'verify_native_freeze',side_effect=ValueError('native frozen identity differs')):
            with self.assertRaisesRegex(ValueError,'native frozen identity'):native.execute(a)
        self.assertEqual(seen,['/synthetic/native.json'])
    def test_source_epoch_rejects_mutation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            f=Path(temp)/'source.py';f.write_text('a=1\n');graph={'files':{'source':{'path':str(f)}}}
            epoch=g.source_epoch(graph);f.write_text('a=2222\n')
            with self.assertRaisesRegex(ValueError,'source changed'):g.check_epoch(graph,epoch)
    def test_immutable_manifest_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            f=Path(temp)/'manifest.json';g.immutable_json(f,{'synthetic':True})
            with self.assertRaises(FileExistsError):g.immutable_json(f,{'synthetic':False})
    def test_no_public_time_parameter(self):
        import inspect
        self.assertNotIn('now',inspect.signature(g.release).parameters)
        for entry in (constructor,native):
            text=Path(entry.__file__).read_text()
            self.assertNotIn("add_argument('--now",text)
            self.assertNotIn('add_argument("--now',text)

if __name__=='__main__':
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});unittest.main(verbosity=2)
