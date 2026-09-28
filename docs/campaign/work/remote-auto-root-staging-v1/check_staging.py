import ast, copy, hashlib, io, json, shlex, tarfile, unittest
from pathlib import Path
from transfer_capsules import validate_archive, check_plan, command, REMOTE_RECEIVER
HERE=Path(__file__).resolve().parent
C=HERE.parents[1]
PLANS=json.loads((HERE/'staging-plan.json').read_text())['arms']

def archive(entries):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w') as t:
        for name,data,kind in entries:
            m=tarfile.TarInfo(name);m.type=kind;m.size=len(data)
            if kind==tarfile.SYMTYPE:m.linkname='/forbidden'
            t.addfile(m,io.BytesIO(data) if kind==tarfile.REGTYPE else None)
    return out.getvalue()
class Checks(unittest.TestCase):
    def test_complete_pinned_archives_and_bindings(self):
        for p in PLANS:
            data,expected=check_plan(p);self.assertEqual(len(expected),15)
            self.assertLess(len(data),200000)
            self.assertEqual(len(set(expected)),15)
    def test_wrong_arm_and_identity_fail(self):
        for key,value in [('arm_ms',0),('instance_id','00000000-0000-0000-0000-000000000000'),('binding_sha256','0'*64),('remote_capsule','/tmp/unowned')]:
            p=copy.deepcopy(PLANS[0]);p[key]=value
            with self.assertRaises(AssertionError):check_plan(p)
    def test_archive_negative_controls(self):
        good=b'approved';expected={'safe':hashlib.sha256(good).hexdigest()}
        self.assertEqual(validate_archive(archive([('safe',good,tarfile.REGTYPE)]),expected),{'safe':good})
        for rows in [[('../safe',good,tarfile.REGTYPE)],[('/safe',good,tarfile.REGTYPE)],[('safe',good,tarfile.SYMTYPE)],[('safe',good,tarfile.REGTYPE),('safe',good,tarfile.REGTYPE)],[('asset-key.pem',good,tarfile.REGTYPE)],[('safe',b'wrong',tarfile.REGTYPE)],[]]:
            with self.assertRaises(ValueError):validate_archive(archive(rows),expected)
        with self.assertRaises(ValueError):validate_archive(b'x'*(2*1024*1024+1),expected)
    def test_controller_path_and_guard_only_expected_delta(self):
        original=(C/'work/lead/remote-primary-a/notebook-capsule/notebook_editor_supervisor.py').read_text()
        for p in PLANS:
            guard=(HERE/p['name']/'notebook_editor_supervisor.py').read_text();ast.parse(guard)
            normalized=guard.replace(repr(p['name']+'-capsule'),"'remote-editor-a-capsule'").replace(repr('runs/'+p['name']),"'runs/remote-editor-a'").replace(repr('runs/'+p['name']+'-supervision'),"'runs/remote-editor-a-supervision'").replace(p['binding_sha256'],'26cff670485b2384758af0c478e78aa040e5f1f36d10e3c753257011dd55ce94').replace(p['instance_id'],'223bdcf2-c958-4477-93e2-72a80c769152').replace(", '--debounce-ms', '"+str(p['arm_ms'])+"'",'').replace(",'--debounce-ms','"+str(p['arm_ms'])+"'",'')
            self.assertEqual(normalized,original)
            self.assertIn("REMOTE + '/"+p['name']+"-capsule/notebook_editor_supervisor.py'",Path(p['local_root_controller']).read_text())
    def test_runtime_sources_identical_and_metadata_specific(self):
        first=HERE/PLANS[0]['name'];second=HERE/PLANS[1]['name']
        for name in ['observe_renderer.mjs','run_remote_editor.mjs','analyze_renderer.mjs','analyze_auto.mjs','check_merged.mjs','candidate.vsix','primary-editor-harness/extension.js','primary-editor-harness/acceptance-v2.js','primary-editor-harness/package.json']:
            self.assertEqual((first/name).read_bytes(),(second/name).read_bytes())
        for p in PLANS:
            metadata=json.loads((HERE/p['name']/'root-supplied-identities.json').read_text())
            self.assertEqual(metadata['instance_id'],p['instance_id']);self.assertEqual(metadata['debounce_ms'],p['arm_ms'])
    def test_transfer_is_file_only_and_safe_shell_roundtrip(self):
        tree=ast.parse(REMOTE_RECEIVER)
        self.assertFalse(any(isinstance(n,ast.Import) and any(a.name in ['subprocess','socket'] for a in n.names) for n in ast.walk(tree)))
        for p in PLANS:
            _,expected=check_plan(p);cmd=command(p,expected)
            self.assertNotIn('-R',cmd);self.assertNotIn('-L',cmd)
            remote=shlex.split(cmd[-1]);self.assertEqual(remote[:2],['python3','-c']);self.assertEqual(remote[2],REMOTE_RECEIVER)
            self.assertEqual(remote[3],p['remote_capsule']);self.assertEqual(json.loads(remote[6]),expected)
            self.assertNotIn('asset-key.pem',expected)
    def test_no_private_key_model_or_extra_archive_files(self):
        for p in PLANS:
            _,expected=check_plan(p)
            self.assertEqual(set(expected),set(p['files'])|{'capsule-manifest.json'})
            self.assertFalse(any(name.endswith(('.pem','.gguf','.safetensors')) for name in expected))
if __name__=='__main__':unittest.main(verbosity=2)
