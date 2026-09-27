"""CPU-only FINAL controller gate/cleanup tests. No model or server launch."""
import contextlib,copy,datetime,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import root_controller as c
class ControllerTests(unittest.TestCase):
    def test_native_profile_args_remain_256(self):
            argv=c.native_argv(18403)
            for flag,wanted in (('-b','256'),('-ub','256'),('-c','4096'),('--parallel','1'),('-t','6')):
                self.assertEqual(argv[argv.index(flag)+1],wanted)
    def test_actual_bundle_aliases_pass_real_schema(self):
            # Hash installed ELF/library bytes only. The model row stays synthetic.
            bundle=[c.record(c.BUNDLE/f['name'],f['sha256']) for f in c.identity.PROFILE['bundle']['files']]
            self.assertEqual([Path(f['path']).name for f in bundle],[f['name'] for f in c.identity.PROFILE['bundle']['files']])
            self.assertTrue(any(f['path']!=f['resolved_path'] for f in bundle))
            a=json.loads((c.CANDIDATE/'native-admission.template.json').read_text())
            a.update(status='root_admitted',pid=42,start_tick='1',fresh_process=True,exclusive_owner=True,no_other_clients=True,root_verified_model_before_launch=True,cuda_backend_observed=True,all_layers_offloaded=True,load_evidence_sha256='a'*64,bundle_files=bundle,argv=c.native_argv(18403),mapped_code_files=[bundle[-1]])
            a['model']['path']=str(c.MODEL)
            c.identity.validate_admission(a)
            broken=copy.deepcopy(a)
            for row in broken['bundle_files']:row['path']=row['resolved_path']
            with self.assertRaisesRegex(ValueError,'bundle differs'):c.identity.validate_admission(broken)
    def test_record_rejects_hash_drift_and_preserves_alias(self):
            with tempfile.TemporaryDirectory(dir=HERE) as temp:
                target=Path(temp)/'library.real';target.write_bytes(b'synthetic ELF placeholder')
                alias=Path(temp)/'library.so.0';alias.symlink_to(target)
                row=c.record(alias)
                self.assertEqual(row['path'],str(alias));self.assertEqual(row['resolved_path'],str(target))
                self.assertEqual(row['stat'],c.identity.stat_identity(target))
                with self.assertRaises(ValueError):c.record(alias,'0'*64)
    def test_partial_gpu_load_rejected(self):
            with self.assertRaises(ValueError):c.load_evidence('offloaded 42/43 layers to GPU\nCUDA0 model buffer size = 2200.0 MiB')
            self.assertEqual(c.load_evidence('offloaded 43/43 layers to GPU\nCUDA0 model buffer size = 2279.73 MiB')['offloaded'],43)
    def test_cleanup_pid_reuse_never_signals(self):
            row={'pid':999999,'start_tick':'old','role':'synthetic'}
            with mock.patch.object(c.Path,'exists',return_value=True),mock.patch.object(c.Path,'read_text',return_value='unused'),mock.patch.object(c.identity,'proc_stat',return_value=('S','new')),mock.patch.object(c.os,'killpg') as kill:
                result=c.stop_owned(row);self.assertIn('PID reused',result['action']);kill.assert_not_called()
    def test_cleanup_rejects_foreign_process_group(self):
            row={'pid':999999,'start_tick':'1','role':'synthetic'}
            with mock.patch.object(c.Path,'exists',return_value=True),mock.patch.object(c.Path,'read_text',return_value='unused'),mock.patch.object(c.identity,'proc_stat',return_value=('S','1')),mock.patch.object(c.os,'getpgid',return_value=12),mock.patch.object(c.os,'killpg') as kill:
                result=c.stop_owned(row);self.assertIn('not owned',result['action']);kill.assert_not_called()
    def test_cleanup_disappearance_is_success(self):
            with mock.patch.object(c.Path,'exists',return_value=True),mock.patch.object(c.Path,'read_text',side_effect=FileNotFoundError):
                self.assertTrue(c.stop_owned({'pid':999999,'start_tick':'1'})['absent'])
    def test_cleanup_terms_and_kills_matching_group(self):
            row={'pid':999999,'start_tick':'1','role':'synthetic'}
            with mock.patch.object(c.Path,'exists',return_value=True),mock.patch.object(c.Path,'read_text',return_value='unused'),mock.patch.object(c.identity,'proc_stat',return_value=('S','1')),mock.patch.object(c.os,'getpgid',return_value=999999),mock.patch.object(c.time,'monotonic',side_effect=[0,11]),mock.patch.object(c.os,'killpg') as kill:
                c.stop_owned(row)
                self.assertEqual(kill.call_args_list,[mock.call(999999,c.signal.SIGTERM),mock.call(999999,c.signal.SIGKILL)])
    def approval(self):
        return {'schema':'dat08.native-final-root-launch.v1','status':'root_admitted',
            'source_policy_reviewed':True,'exclusive_gpu_owner':True,'no_other_clients':True,'native_loading_trust_accepted':True,'final_only':True,'authorize_derived_runtime_freeze':True,
            'run_id':'synthetic','lease_id':'synthetic','cuda_lock_path':str(c.CUDA_LOCK),'profile_sha256':c.identity.digest(c.identity.PROFILE),'port':18403,'client_seconds':3600,'constructor_seconds':600,'root_review_seconds':300,'host_seconds':4860,'client_rss_bytes':c.CLIENT_RSS_BYTES,'server_rss_bytes':c.SERVER_RSS_BYTES,
            'created_at':'2026-09-14T10:00:00+00:00','expires_at':'2026-09-14T12:00:00+00:00',
            'freeze_path':'/synthetic/freeze.json','harness_sha256':'a'*64,'source_closure_sha256':'b'*64,'constructor_source_closure_sha256':'c'*64,'prelaunch_freeze_sha256':'d'*64,
            'input_files':{},'constructor_input_files':{},'source_manifest':'/synthetic/native-graph.json','constructor_source_manifest':'/synthetic/constructor-graph.json'}
    def freeze(self):
        return {'status':'frozen','weights_frozen':True,'harness_frozen':True,'final_access_unlocked':True,'weights_sha256':c.identity.Q8,'harness_sha256':'a'*64,'weights_frozen_at':'2026-09-14T10:00:00+00:00','harness_frozen_at':'2026-09-14T10:00:00+00:00','model_binding_kind':'fresh-pinned-native-process.v1','native_profile_sha256':c.identity.digest(c.identity.PROFILE),'source_closure_sha256':'b'*64,'constructor_source_closure_sha256':'c'*64}
    def released(self):return datetime.datetime(2026,9,14,10,1,tzinfo=datetime.timezone.utc)
    def test_actual_pre_release_blocks_all_other_paths(self):
        a=self.approval();freeze=self.freeze();a['prelaunch_freeze_sha256']=c.identity.digest(freeze)
        def read(path,*args,**kwargs):
            self.assertEqual(str(path),a['freeze_path']);return json.dumps(freeze)
        with mock.patch.object(c.Path,'read_text',read),mock.patch.object(c,'record') as hashed,mock.patch.object(c.subprocess,'Popen') as launch:
            with self.assertRaises(Exception):c.release(a)
            hashed.assert_not_called();launch.assert_not_called()
    def test_flags_fail_before_final_stat_hash_or_launch(self):
        for flag in ('weights_frozen','harness_frozen','final_access_unlocked'):
            freeze=self.freeze();freeze[flag]=False;a=self.approval();a['prelaunch_freeze_sha256']=c.identity.digest(freeze)
            with self.subTest(flag=flag),mock.patch.object(c,'now',return_value=self.released()),mock.patch.object(c.Path,'read_text',return_value=json.dumps(freeze)),mock.patch.object(c.Path,'stat',side_effect=AssertionError('unexpected path stat')),mock.patch.object(c,'record') as hashed,mock.patch.object(c.subprocess,'Popen') as launch:
                with self.assertRaises(Exception):c.release(a)
                hashed.assert_not_called();launch.assert_not_called()
    def test_wrong_weight_harness_and_graph_freeze_reject(self):
        for key in ('weights_sha256','harness_sha256','source_closure_sha256','constructor_source_closure_sha256','native_profile_sha256'):
            freeze=self.freeze();freeze[key]='f'*64;a=self.approval();a['prelaunch_freeze_sha256']=c.identity.digest(freeze)
            with self.subTest(key=key),mock.patch.object(c,'now',return_value=self.released()),mock.patch.object(c.Path,'read_text',return_value=json.dumps(freeze)):
                with self.assertRaises(Exception):c.release(a)
    def test_dev_or_missing_final_permission_rejects_before_release(self):
        for key,value in [('schema','dat08.native-dev-root-launch.v1'),('final_only',False),('authorize_derived_runtime_freeze',False)]:
            a=self.approval();a[key]=value
            with self.subTest(key=key),mock.patch.object(c,'release') as release:
                with self.assertRaises(ValueError):c.validate_approval(a)
                release.assert_not_called()
    def test_time_gate_precedes_all_input_hashes(self):
        a=self.approval();a['input_files']={'/synthetic/final-manifest.json':'a'*64}
        with mock.patch.object(c,'now',return_value=self.released()),mock.patch.object(c,'release',side_effect=ValueError('closed time gate')),mock.patch.object(c,'record') as record:
            with self.assertRaisesRegex(ValueError,'closed time gate'):c.validate_approval(a)
            record.assert_not_called()
    def test_child_gate_failure_never_reads_graph_or_launches(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            with mock.patch.object(c,'release',side_effect=ValueError('closed gate')),mock.patch.object(c,'free_port'),mock.patch.object(c.subprocess,'Popen') as spawn:
                self.assertEqual(c.child(self.approval(),Path(temp)),1);spawn.assert_not_called()
            terminal=json.loads((Path(temp)/'controller-terminal.json').read_text());self.assertFalse(terminal['final_access_attempted']);self.assertEqual(terminal['cleanup'],[])
    def test_derived_freeze_changes_only_authorized_runtime_fields(self):
        a=self.approval();f=self.freeze();admission={'purpose':'final-evaluation','run_id':'synthetic','lease_id':'synthetic','pid':42}
        with mock.patch.object(c,'release',return_value=f):out=c.derive_run_freeze(a,admission)
        for key,value in f.items():self.assertEqual(out[key],value)
        self.assertEqual(out['native_process_admission_sha256'],c.identity.digest(admission));self.assertEqual(out['prelaunch_freeze_sha256'],c.identity.digest(f))
        self.assertNotIn('native_process_admission_sha256',f)
    def test_derived_freeze_rejects_wrong_run_or_dev_purpose(self):
        for key,value in [('purpose','dev-rehearsal'),('run_id','other'),('lease_id','other')]:
            admission={'purpose':'final-evaluation','run_id':'synthetic','lease_id':'synthetic'};admission[key]=value
            with self.subTest(key=key),mock.patch.object(c,'release',return_value=self.freeze()):
                with self.assertRaises(ValueError):c.derive_run_freeze(self.approval(),admission)
    def review(self):
        a=self.approval();f=self.freeze();n={'expires_at':'2026-09-14T11:30:00+00:00'};m={'manifest_sha256':'e'*64,'rows_artifact_sha256':'f'*64,'row_count':2}
        r={'schema':'dat08.native-final-case-review.v1','status':'root_admitted','coverage_reviewed':True,'run_id':'synthetic','lease_id':'synthetic','freeze_sha256':c.identity.digest(f),'native_admission_sha256':c.identity.digest(n),'manifest_sha256':'e'*64,'rows_sha256':'f'*64,'expected_cases':2,'created_at':'2026-09-14T10:00:00+00:00','expires_at':'2026-09-14T11:00:00+00:00'}
        return a,f,n,m,r
    def test_same_run_root_manifest_review_accepts(self):
        a,f,n,m,r=self.review()
        with mock.patch.object(c,'release'),mock.patch.object(c,'now',return_value=self.released()):self.assertEqual(c.validate_review(r,a,f,n,m)['expected_cases'],2)
    def test_root_manifest_review_mismatch_rejects(self):
        for key in ('run_id','lease_id','freeze_sha256','native_admission_sha256','manifest_sha256','rows_sha256','expected_cases'):
            a,f,n,m,r=self.review();r[key]='wrong'
            with self.subTest(key=key),mock.patch.object(c,'release'),mock.patch.object(c,'now',return_value=self.released()):
                with self.assertRaises(ValueError):c.validate_review(r,a,f,n,m)
    def test_review_expiry_cannot_outlive_native(self):
        a,f,n,m,r=self.review();r['expires_at']='2026-09-14T12:00:00+00:00'
        with mock.patch.object(c,'release'),mock.patch.object(c,'now',return_value=self.released()):
            with self.assertRaises(ValueError):c.validate_review(r,a,f,n,m)
    def test_guard_constants_and_no_public_clock(self):
        self.assertEqual((c.CHILD_SECONDS,c.HOST_SECONDS),(4800,4860));self.assertEqual(c.rss_ceiling('constructor'),1536*1024**2)
        source=(HERE/'root_controller.py').read_text();self.assertNotIn("add_argument('--now",source);self.assertIn("'--manifest-sha256',manifest['manifest_sha256']",source)
        a=self.approval()
        for key,value in [('client_seconds',3601),('host_seconds',4861),('root_review_seconds',301),('constructor_seconds',601)]:
            b=copy.deepcopy(a);b[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_approval(b)
    def test_separate_process_graph_hashes_without_wrong_main_origin(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            source=Path(temp)/'child.py';source.write_text('# synthetic different child __main__\n')
            graph={'schema':'dat08.source-closure.v1','status':'root_admitted','unresolved':[],'roots':['child'],'files':{'child':{'path':str(source),'sha256':c.sha(source),'modules':['__main__']}}}
            self.assertEqual(c.verify_process_bundle_files(graph,c.identity.digest(graph)),graph)
            graph['status']='preparation_only'
            with self.assertRaises(ValueError):c.verify_process_bundle_files(graph,c.identity.digest(graph))
    def test_global_cuda_lock_rejects_competitor_and_is_not_unlinked(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path=Path(temp)/'cuda0.lock';path.touch()
            with mock.patch.object(c,'CUDA_LOCK',path):
                fd=c.acquire_cuda_lock()
                try:
                    c.validate_cuda_lock(fd)
                    with self.assertRaises(BlockingIOError):c.acquire_cuda_lock()
                    self.assertTrue(path.exists())
                finally:os.close(fd)
                again=c.acquire_cuda_lock();os.close(again);self.assertTrue(path.exists())
    def test_cuda_lock_requires_initialized_file_and_exact_inherited_inode(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path=Path(temp)/'cuda0.lock';other=Path(temp)/'other';other.touch()
            with mock.patch.object(c,'CUDA_LOCK',path):
                with self.assertRaises(ValueError):c.acquire_cuda_lock()
                path.touch();fd=os.open(other,os.O_RDWR)
                try:
                    with self.assertRaises(ValueError):c.validate_cuda_lock(fd)
                finally:os.close(fd)
    def test_happy_path_reaches_distinct_constructor_and_evaluator_with_same_freeze(self):
        # Execute the real child controller to terminal. Only external native
        # processes/HTTP/ownership and sealed-data validation are synthetic.
        # No argv builder is mocked; both Python command lines are exercised.
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root=Path(temp);run=root/'run';run.mkdir();output=root/'constructed'
            a=self.approval();a.update(constructor_output=str(output),rows_manifest=str(output/'rows-manifest.json'),rows=str(output/'canonical-rows.json'),allowed_roots=[str(root/'synthetic-source')])
            for name in ('requests','case_specs','source_authorization','train_identities','dev_identities'):
                path=root/(name+'.json');path.write_text('{}');a[name]=str(path);a['constructor_input_files'][str(path)]='a'*64
            for key in ('source_manifest','constructor_source_manifest'):
                path=root/(key+'.json');path.write_text('{}');a[key]=str(path)
            (root/'native-mapped-code-allowlist.json').write_text(json.dumps({'files':[{'path':'/synthetic/code.so','sha256':'b'*64}]}))
            (root/'elf-loader-metadata.json').write_text(json.dumps({'missing_system_cache_names':[],'loader_metadata_pins':{}}))
            calls=[];processes=[];manifest={'schema':c.row_gate.MANIFEST_SCHEMA,'manifest_sha256':'e'*64,'rows_artifact_sha256':'f'*64,'row_count':2,'case_ids':['synthetic-1','synthetic-2']}
            def spawn(argv,**kwargs):
                calls.append(list(argv));proc=mock.Mock(pid=9999900+len(calls),returncode=0);processes.append(proc)
                proc.poll.return_value=None if len(calls)==1 else 0
                if len(calls)==1:
                    kwargs['stdout'].write(b'offloaded 43/43 layers to GPU\nCUDA0 model buffer size = 2279.73 MiB\n');kwargs['stdout'].flush()
                elif len(calls)==2:
                    output.mkdir();(output/'constructor-complete.json').write_text(json.dumps({'status':'constructed_and_validated','rows_manifest_sha256':manifest['manifest_sha256']}));(output/'rows-manifest.json').write_text(json.dumps(manifest));(output/'canonical-rows.json').write_text('{"synthetic_only":true}')
                elif len(calls)==3:
                    results=run/'final-results';results.mkdir();(results/'results.json').write_text(json.dumps({'status':'complete','denominators':{'attempted_cases':2},'completed_case_ids':manifest['case_ids']}))
                else:raise AssertionError('unexpected launch')
                return proc
            real_read=c.Path.read_text;real_save=c.save
            def read(path,*args,**kwargs):
                if str(path)=='/proc/9999901/maps':return '1000-2000 r-xp 0000 08:01 1 /synthetic/code.so\n'
                return real_read(path,*args,**kwargs)
            def save(path,value):
                real_save(path,value)
                if path.name=='awaiting-root-case-review.json':
                    review=dict(value,status='root_admitted',coverage_reviewed=True,created_at=self.released().isoformat(),expires_at=(self.released()+datetime.timedelta(seconds=600)).isoformat())
                    real_save(run/'final-evaluation-approval.json',review)
            def request(endpoint,path,payload,timeout):
                if path=='/props':return 200,{'model_path':str(c.MODEL),'total_slots':1,'default_generation_settings':{'n_ctx':4096}},0
                if path=='/tokenize':return 200,{'tokens':[17,18]},0
                raise AssertionError('unexpected HTTP path')
            def artifact(path,expected=None):return {'path':str(path),'sha256':expected,'stat':{}}
            owner=mock.Mock();owner.verify.return_value={'synthetic_native_owner':True}
            with contextlib.ExitStack() as stack:
                patches=[mock.patch.object(c,'HERE',root),mock.patch.object(c,'now',return_value=self.released()),mock.patch.object(c,'release',return_value=self.freeze()),mock.patch.object(c,'verify_process_bundle_files'),mock.patch.object(c,'controller_source_guard'),mock.patch.object(c,'check'),mock.patch.object(c,'record',side_effect=artifact),mock.patch.object(c,'free_port'),mock.patch.object(c.subprocess,'Popen',side_effect=spawn),mock.patch.object(c,'process_record',side_effect=lambda proc,role:{'pid':proc.pid,'start_tick':'1','role':role}),mock.patch.object(c.Path,'read_text',read),mock.patch.object(c,'save',side_effect=save),mock.patch.object(c.transport,'_request_json',side_effect=request),mock.patch.object(c.transport,'_stream_completion',return_value={'stream_complete':True}),mock.patch.object(c.identity,'NativeOwner',return_value=owner),mock.patch.object(c.row_gate,'_validate_manifest',return_value=manifest),mock.patch.object(c,'stop_owned',side_effect=lambda row:dict(row,absent=True))]
                for patch in patches:stack.enter_context(patch)
                self.assertEqual(c.child(a,run),0,json.loads((run/'controller-terminal.json').read_text()))
            self.assertEqual(len(calls),3)
            constructor,evaluator=calls[1:]
            self.assertEqual(constructor[:5],['/home/m0hawk/Documents/Sepalith/.venv/bin/python','-I','-S','-B',str(c.ENTRY/'construct_final.py')])
            self.assertEqual(evaluator[:5],['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-I','-S','-B',str(c.ENTRY/'evaluate_final.py')])
            def arg(argv,key):return argv[argv.index(key)+1]
            expected_freeze=str(run/'final-freeze.json')
            self.assertEqual(arg(constructor,'--freeze'),expected_freeze);self.assertEqual(arg(evaluator,'--freeze'),expected_freeze)
            self.assertEqual(arg(constructor,'--source-manifest'),a['constructor_source_manifest']);self.assertEqual(arg(evaluator,'--source-manifest'),a['source_manifest'])
            self.assertEqual(arg(constructor,'--harness-sha256'),arg(evaluator,'--harness-sha256'))
            self.assertEqual(arg(evaluator,'--manifest-sha256'),manifest['manifest_sha256']);self.assertEqual(arg(evaluator,'--rows'),a['rows'])
            freeze=json.loads((run/'final-freeze.json').read_text());admission=json.loads((run/'native-admission.json').read_text())
            self.assertEqual(freeze['native_process_admission_sha256'],c.identity.digest(admission));self.assertEqual(admission['purpose'],'final-evaluation')
            terminal=json.loads((run/'controller-terminal.json').read_text());self.assertIsNone(terminal['failure']);self.assertEqual(len(terminal['cleanup']),3)
if __name__=='__main__':
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});unittest.main(verbosity=2)
