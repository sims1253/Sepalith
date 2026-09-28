"""Native PRM03 final evaluator candidate. No server/model load or public clock override."""
import copy, datetime, json, os, time
from pathlib import Path
import final_row_gate as gate
import final_binding as binding
import native_transport as transport
from native_identity import NativeOwner, PROFILE, Q8, digest, stat_identity


def verify_native_freeze(freeze, admission, source_manifest, harness_sha):
    # This is the native route's replacement for the HF registered-tensor binding.
    # The existing HF helper and its tensor route remain unchanged.
    gate._validate_freeze(freeze, expected_weights_sha256=Q8,
                         expected_harness_sha256=harness_sha,
                         now=datetime.datetime.now(datetime.timezone.utc))
    if freeze.get('native_process_admission_sha256')!=digest(admission):raise ValueError('native admission not frozen')
    if freeze.get('native_profile_sha256')!=digest(PROFILE):raise ValueError('delivered native profile not frozen')
    if freeze.get('model_binding_kind')!='fresh-pinned-native-process.v1':raise ValueError('explicit native binding kind required')
    binding.verify_source_closure(source_manifest,freeze['source_closure_sha256'])


def summary(records, rows, status, audit, error=None):
    q=lambda r:r.get('quality',{})
    result={'schema':'dat08.native-final-quality.v1','status':status,'gate':audit,
        'expected_case_ids':[r['id'] for r in rows],'completed_case_ids':[r['id'] for r in records],
        'denominators':{'expected_cases':len(rows),'attempted_cases':len(records),
                       'edit_cases':sum(r['operation']!='no_op' for r in rows),
                       'strict_noop_cases':sum(r['operation']=='no_op' for r in rows),
                       'complete_responses':sum(bool(r.get('response_complete')) for r in records)},
        'counts':{'edit_exact':sum(bool(q(r).get('edit_exact')) for r in records),
                  'strict_noop_correct':sum(bool(q(r).get('strict_noop_correct')) for r in records),
                  'false_suggestions':sum(bool(q(r).get('noop_false_positive')) for r in records),
                  'protocol_valid':sum(bool(q(r).get('protocol_valid')) for r in records),
                  'cap_hit':sum(bool(r.get('cap',{}).get('hit')) for r in records),
                  'mechanical_or_transport_failed':sum(r.get('status')!='accepted' for r in records)},
        'results':records,'loss_metrics':{'status':'unavailable_native_no_logits'},
        'quality_acceptance_claim':False,'direct_in_memory_tensor_proof':False}
    if error is not None:result['error']=error
    return result


class NativeFinalEvaluator:
    def __init__(self, rows, protocol, freeze, admission, sources, audit, owner):
        self.rows=copy.deepcopy(rows);self.protocol=protocol;self.freeze=copy.deepcopy(freeze)
        self.admission=copy.deepcopy(admission);self.sources=copy.deepcopy(sources);self.audit=copy.deepcopy(audit)
        self.owner=owner
    def evaluate(self, output_dir, tokenizer_dir, *, global_deadline_seconds=3600):
        def recheck():
            gate._validate_freeze(self.freeze,expected_weights_sha256=Q8,
                expected_harness_sha256=self.audit['harness_sha256'],now=datetime.datetime.now(datetime.timezone.utc))
        return evaluate_rows(self.rows,self.protocol,self.owner,self.sources,self.audit,output_dir,
            tokenizer_dir,global_deadline_seconds,recheck,
            lambda:verify_native_freeze(self.freeze,self.admission,self.sources,self.audit['harness_sha256']))
    def close(self):self.owner.close()


def prepare_final_native_evaluator(rows_artifact_path, *, final_rows_manifest, expected_manifest_sha256,
        freeze_receipt, expected_harness_sha256, native_admission, source_manifest):
    # Actual clock and both root freeze flags run before any final path metadata/read.
    verify_native_freeze(freeze_receipt,native_admission,source_manifest,expected_harness_sha256)
    if native_admission.get('purpose')!='final-evaluation':raise ValueError('final-purpose native admission required')
    manifest=gate._validate_manifest(final_rows_manifest,expected_manifest_sha256=expected_manifest_sha256,
                                     freeze_sha256=gate.sha256_json(freeze_receipt))
    owner=NativeOwner(native_admission);owner.acquire()
    try:
        native_audit=owner.verify(full_hash=True)
        path=gate._safe_metadata_path(Path(rows_artifact_path))
        protocol=gate._load_protocol()
        rows,row_audit,_=gate._validate_rows_file(path,manifest,protocol)
        audit={'status':'native_final_rows_gate_passed','weights_sha256':Q8,'harness_sha256':expected_harness_sha256,
               'freeze_receipt_sha256':gate.sha256_json(freeze_receipt),'manifest_sha256':expected_manifest_sha256,
               'rows':row_audit,'native':native_audit,'profile_sha256':digest(PROFILE),
               'source_closure_sha256':freeze_receipt['source_closure_sha256'],'binding_kind':'fresh-pinned-native-process.v1'}
        return NativeFinalEvaluator(rows,protocol,freeze_receipt,native_admission,source_manifest,audit,owner)
    except BaseException:
        owner.close();raise


def evaluate_rows(rows,protocol,owner,sources,audit,output_dir,tokenizer_dir,global_seconds,recheck,full_recheck):
    if not 1<=global_seconds<=7200:raise ValueError('bounded global deadline required')
    output=Path(output_dir)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():raise ValueError('fresh absolute output directory required')
    output.mkdir(mode=0o700)
    records=[];started=time.monotonic();destination=output/'results.json'
    transport._write_atomic(destination,summary(records,rows,'partial',audit))
    source_epoch={k:stat_identity(v['path']) for k,v in sources['files'].items()}
    def before_request(deadline=None):
        recheck()
        if time.monotonic()-started>global_seconds-10:raise TimeoutError('global deadline reserve reached')
        if any(stat_identity(v['path'])!=source_epoch[k] for k,v in sources['files'].items()):raise ValueError('source changed during native evaluation')
        owner.verify(deadline=deadline)
    try:
        full_recheck();transport._write_atomic(output/'native-before.json',owner.verify(full_hash=True))
        tokenizer,tokenizer_audit=transport._load_tokenizer(Path(tokenizer_dir))
        cases=transport._prepare_cases([{'row':r,'context':protocol.PromptContext.from_mapping(r['context'])} for r in rows],protocol)
        encoding=transport._encode_cases(cases,tokenizer,protocol)
        transport._write_atomic(output/'tokenizer.json',{'identity':tokenizer_audit,'encoding':encoding})
        for row,case in zip(rows,cases):
            record=transport._run_case(owner.a['endpoint'],case,protocol,tokenizer,PROFILE['case_deadline_seconds'],before_request=before_request)
            record['source_provenance']=copy.deepcopy(row.get('source_provenance',{}))
            records.append(record)
            transport._write_atomic(destination,summary(records,rows,'partial',audit))
            if record.get('status')=='failed' or not record.get('response_complete'):raise RuntimeError('native case failed: '+row['id'])
            # Complete protocol-invalid responses remain scored failures, preserving denominator.
        before_request();full_recheck()
        transport._write_atomic(output/'native-after.json',owner.verify(full_hash=True))
        if [r['id'] for r in records]!=[r['id'] for r in rows]:raise ValueError('native final denominator drift')
        result=summary(records,rows,'complete',audit)
        result['elapsed_seconds']=time.monotonic()-started
        transport._write_atomic(destination,result)
        return result
    except BaseException as error:
        transport._write_atomic(destination,summary(records,rows,'partial',audit,{'type':type(error).__name__,'message':str(error)}))
        raise
    finally:owner.close()


__all__=['prepare_final_native_evaluator','NativeFinalEvaluator']
