import ast
import datetime
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import qualification as q

HERE = Path(__file__).resolve().parent
TREE = ast.parse((HERE/'run_editor.py').read_text())

class Preparation(unittest.TestCase):
    def test_exact_local_package_capsule_guard_and_actual_argument_function(self):
        pins, argv = q.verify_local_inputs()
        self.assertEqual(len(pins['capsule_runtime_files']), 9)
        self.assertEqual(len(pins['package_files']), 10)
        self.assertTrue(q.verify_ngram_argv(argv))

    def test_ngram_mutations_and_draft_model_rejected(self):
        _, argv = q.verify_local_inputs()
        for flag in ['--spec-type','--spec-draft-n-max','--spec-ngram-mod-n-match','--spec-ngram-mod-n-min','--spec-ngram-mod-n-max','-m']:
            with self.subTest(flag=flag):
                changed = list(argv); changed[changed.index(flag)+1] = 'wrong'
                with self.assertRaises(ValueError): q.verify_ngram_argv(changed)
        for changed in [argv + ['--spec-type','ngram-mod'], argv + ['--model-draft','unexpected']]:
            with self.assertRaises(ValueError): q.verify_ngram_argv(changed)

    def test_process_argv_and_identity_recheck(self):
        _, argv = q.verify_local_inputs()
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            p=Path(directory); (p/'77').mkdir(); (p/'77/cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
            expected={'pid':77,'startTick':'99','uid':1000}; (p/'native-identity.json').write_text(json.dumps(expected))
            with patch.object(q,'Path',lambda value:p if value=='/proc' else Path(value)):
                self.assertEqual(q.verify_owned_native(p,lambda _:expected)['argv'],argv)
                with self.assertRaisesRegex(ValueError,'identity_changed'): q.verify_owned_native(p,lambda _:None)
                calls=iter([expected,None])
                with self.assertRaisesRegex(ValueError,'identity_changed'): q.verify_owned_native(p,lambda _:next(calls))
                changed=list(argv);changed[changed.index('--spec-ngram-mod-n-match')+1]='23';(p/'77/cmdline').write_bytes(('\0'.join(changed)+'\0').encode())
                with self.assertRaisesRegex(ValueError,'argument_mismatch'):q.verify_owned_native(p,lambda _:expected)

    def test_controller_launch_args_are_bounded_and_preserve_capsule(self):
        call=next(n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen')
        argv=eval(compile(ast.Expression(call.args[0]),'<argv-only>','eval'),{'sys':sys,'PACK':HERE.parent/'r2-ngram-lan-v1','STATE':Path('/synthetic/state'),'HERE':HERE})
        self.assertEqual(argv[argv.index('--session-seconds')+1],'300')
        node=next(n for n in ast.walk(TREE) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets))
        current=Path('/synthetic/fresh-uuid');remote='/home/m0hawk/.local/share/sepalith-r2-step500-checks'
        args=eval(compile(ast.Expression(node.value),'<editor-argv-only>','eval'),{'REMOTE':remote,'binding_sha':'synthetic-sha','current':current,'run':remote+'/runs/'+current.name})
        self.assertIn('230s',args)
        self.assertEqual(args[args.index('--timeout-ms')+1],'180000')
        self.assertEqual(args[args.index('--debug-port')+1],'19413')
        self.assertEqual(args[args.index('--debounce-ms')+1],'350')
        self.assertIn(remote+'/capsule-v3/run_remote_editor.mjs',args)

    def cleanup_case(self, terminal, alive=False):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            p=Path(directory);lock=p/'synthetic-lock';lock.touch()
            if terminal is not None:(p/'terminal.json').write_text(json.dumps(terminal))
            class Child:
                def poll(self):return 0
                def wait(self,timeout):return 0
            saved={};identity={'pid':77,'startTick':'99','uid':1000}
            env={'child':Child(),'current':p,'failure':None,'cleanup':None,'supervisor':identity,'ident':lambda _:identity if alive else None,'save':lambda n,o:saved.update({n:o}),'signal':signal,'subprocess':subprocess,'json':json,'LOCK':lock,'fcntl':fcntl,'datetime':datetime,'time':time,'started':time.monotonic()}
            body=next(n.finalbody for n in TREE.body if isinstance(n,ast.Try))
            exec(compile(ast.Module(body=body,type_ignores=[]),'<actual-controller-finally>','exec'),env)
            return saved['controller-terminal.json']

    def test_actual_cleanup_accepts_released_and_rejects_missing_failed_alive(self):
        good={'failure':None,'resourceReleaseProven':True,'ledgerRelease':{'status':'released'},'cleanup':[]}
        self.assertIsNone(self.cleanup_case(good)['failure'])
        self.assertTrue(self.cleanup_case(None)['failure'])
        self.assertTrue(self.cleanup_case({**good,'ledgerRelease':{'status':'retained_ledger_mismatch'}})['failure'])
        self.assertTrue(self.cleanup_case({**good,'failure':'native failed'})['failure'])
        self.assertTrue(self.cleanup_case(good,alive=True)['failure'])

if __name__=='__main__':unittest.main(verbosity=2)
