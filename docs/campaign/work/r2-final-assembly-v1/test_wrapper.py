"""Synthetic refusal and identity regressions; no final/source/model inputs."""
import argparse,datetime,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
import assemble_final as entry
class Before(datetime.datetime):
    @classmethod
    def now(cls,tz=None):return cls(2026,9,14,9,59,tzinfo=datetime.timezone.utc)
class After(datetime.datetime):
    @classmethod
    def now(cls,tz=None):return cls(2026,9,14,10,1,tzinfo=datetime.timezone.utc)
class Tests(unittest.TestCase):
    def freeze(self):
        return dict(status='frozen',weights_frozen=True,harness_frozen=True,final_access_unlocked=True,
            weights_sha256=entry.g.Q8,harness_sha256='a'*64,weights_frozen_at='2026-09-14T10:00:00Z',harness_frozen_at='2026-09-14T10:00:00Z')
    def args(self):
        return argparse.Namespace(freeze=Path('/synthetic/freeze'),harness_sha256='a'*64,
            source_manifest=Path('/never/graph'),source_sha256='b'*64,selection=Path('/never/selection'),
            registry=Path('/never/registry'),output=Path('/never/output'))
    def denied(self,freeze,clock):
        seen=[]
        def read(path,*args,**kw):
            seen.append(str(path))
            if str(path)!='/synthetic/freeze':raise AssertionError('non-freeze metadata read')
            return json.dumps(freeze)
        with mock.patch.object(entry.g.datetime,'datetime',clock),mock.patch.object(Path,'read_text',read),mock.patch.object(Path,'stat',side_effect=AssertionError('source/path stat')),mock.patch.object(entry.assembly,'checked_json',side_effect=AssertionError('protected metadata')):
            with self.assertRaises(entry.g.gate.FinalEvaluatorError):entry.execute(self.args())
        self.assertEqual(seen,['/synthetic/freeze'])
    def test_real_clock_gate_before_inputs(self):self.denied(self.freeze(),Before)
    def test_each_missing_freeze_flag_after_time(self):
        for flag in ('weights_frozen','harness_frozen','final_access_unlocked'):
            f=self.freeze();f[flag]=False
            with self.subTest(flag=flag):self.denied(f,After)
    def test_wrong_weight_and_harness(self):
        for field in ('weights_sha256','harness_sha256'):
            f=self.freeze();f[field]='c'*64
            with self.subTest(field=field):self.denied(f,After)
    def test_graph_requires_specific_assembly_binding(self):
        with mock.patch.object(entry.g,'release',return_value=self.freeze()),mock.patch.object(Path,'read_text',side_effect=AssertionError('graph opened')):
            with self.assertRaisesRegex(ValueError,'assembly_source_closure_sha256'):entry.execute(self.args())
    def test_frozen_component_exact_copy(self):
        original=entry.assembly.CAM/'work/lead/r2-final-assembly-preparation-v1/assemble_inputs.py'
        self.assertEqual(original.read_bytes(),(HERE/'assemble_inputs.py').read_bytes())
    def test_no_public_clock_or_synthetic_input_option(self):
        names={action.dest for action in entry.make_parser()._actions}
        self.assertEqual(names,{'help','freeze','source_manifest','selection','registry','output','harness_sha256','source_sha256'})
    def test_real_synthetic_guard_and_construction(self):
        result=entry.synthetic_work()
        self.assertEqual(result['guarded_probe_count'],5);self.assertEqual(result['supported'],1)
        self.assertFalse(result['final_source_access']);self.assertEqual(len(result['families_preloaded']),6)
    def test_source_epoch_and_immutable_output(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            p=Path(temp)/'source.py';p.write_text('x=1\n');graph={'files':{'x':{'path':str(p)}}}
            epoch=entry.g.source_epoch(graph);p.write_text('x=2\n')
            with self.assertRaisesRegex(ValueError,'source changed'):entry.g.check_epoch(graph,epoch)
            dest=Path(temp)/'result.json';entry.g.immutable_json(dest,{'synthetic':True})
            with self.assertRaises(FileExistsError):entry.g.immutable_json(dest,{'synthetic':False})
if __name__=='__main__':
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});unittest.main(verbosity=2)
