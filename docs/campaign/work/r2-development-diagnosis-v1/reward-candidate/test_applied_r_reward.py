"""Real old/new reward + real R parser, deterministic decoder controls only."""
import copy,hashlib,os,sys,unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
SOURCE=Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source')
assert hashlib.sha256((SOURCE/'experiments/training/campaign_rl_train.py').read_bytes()).hexdigest()=='78d27aa98cebc80292d1871a39821eee5a1a705b5a8f412ce270d1707d724443'
sys.path.insert(0,str(SOURCE/'experiments/training'))
from applied_r_reward import AppliedDocumentExactReward,BindingError,bind_train_records,sha_text,offset,prepare_reward_bindings,ADMISSION_POLICY
from pinned_r_parser import create_r_parser
from sepalith.campaign_protocol import PromptContext
PARSE=create_r_parser()
assert 'torch' not in sys.modules and 'transformers' not in sys.modules

def fixture(*,old='  x + 1',target='  x + 2',operation='replace',suffix='}',eol='\n',prefix='f <- function(x) {'):
    document=eol.join([prefix,old]+([suffix] if suffix else []))
    digest=sha_text(document);uri='file:///synthetic-r2-control.R'
    mapping={'schema_version':'sepalith.prompt.prm03.v1','path':'synthetic-r2-control.R','prefix':[prefix],'region_old':[old] if old else [],'suffix_lines':[suffix] if suffix else [],'cursor':{'region_line_index':0,'code_point_column':0,'utf16_column':0} if old else {'region_line_index':-1,'code_point_column':None,'utf16_column':None},'history':[],'diagnostics':[],'scope_lines':[],'retrieval':[],'selected_references':[],'scope_mode':'off','document_eol':'crlf' if eol=='\r\n' else 'lf','replacement_range':{'uri':uri,'document_version':0,'content_sha256':digest,'start':{'line':1,'character':0},'end':{'line':1,'character':len(old.encode('utf-16-le'))//2}}}
    return SimpleNamespace(row={'id':'synthetic-control','family':'no_op' if operation=='no_op' else 'finish_block','package_id':'synthetic-test-only','split':'train','target_operation':operation,'target_body_text':target},context=PromptContext.from_mapping(mapping),capture=SimpleNamespace(uri=uri,version=0,content_sha256=digest),source_identity={'source_provenance':{'selection_source':{'document_text':document,'content_sha256':digest,'document_version':0}}})

def reward(record,text,ids=None,parser=PARSE):
    bound=bind_train_records([record],parser)
    r=AppliedDocumentExactReward(bindings=bound,parse_r=parser,decoder=lambda _ids,**kw:text)
    row=record.row
    score,details=r.score_one(record.context,row['target_operation'],row['target_body_text'],[100,1] if ids is None else ids,row_id=row['id'],family=row['family'],package_id=row['package_id'])
    return score,details

class RewardTests(unittest.TestCase):
    def test_exact_valid(self):
        score,r=reward(fixture(),'  x + 2\n>>>>>>> UPDATED');self.assertEqual(score,1.2);self.assertTrue(r['applied_r_parse']);self.assertEqual(r['semantic_status'],'exact_reference_parse_valid')
    def test_exact_noop(self):
        score,r=reward(fixture(operation='no_op',target='[NO_EDIT]'),'[NO_EDIT]\n>>>>>>> UPDATED');self.assertEqual(score,1.2);self.assertTrue(r['predicted_noop'])
    def test_copy_old_normalizes_to_noop(self):
        score,r=reward(fixture(operation='no_op',target='[NO_EDIT]'),'  x + 1\n>>>>>>> UPDATED');self.assertEqual(score,1.2);self.assertTrue(r['predicted_noop'])
    def test_false_noop_on_edit(self):
        score,r=reward(fixture(),'[NO_EDIT]\n>>>>>>> UPDATED');self.assertEqual(score,0);self.assertEqual(r['semantic_status'],'nonexact_parse_valid_semantics_unverified')
    def test_false_edit_on_noop(self):
        score,r=reward(fixture(operation='no_op',target='[NO_EDIT]'),'  x + 2\n>>>>>>> UPDATED');self.assertEqual(score,0);self.assertTrue(r['applied_r_parse'])
    def test_plausible_alternative_not_semantic_proof(self):
        score,r=reward(fixture(),'  2 + x\n>>>>>>> UPDATED');self.assertEqual(score,0);self.assertEqual(r['semantic_status'],'nonexact_parse_valid_semantics_unverified')
    def test_whitespace_alternative_unverified(self):
        score,r=reward(fixture(),'  x+2\n>>>>>>> UPDATED');self.assertEqual(score,0);self.assertTrue(r['applied_r_parse'])
    def test_line_overlap_removed(self):
        record=fixture(target='  y <- x + 2\n  y');score,r=reward(record,'  y <- x + 2\n  99\n>>>>>>> UPDATED');self.assertGreater(r['legacy_reward_diagnostic'],0);self.assertEqual(score,0);self.assertFalse(r['line_f1_used_for_reward'])
    def test_valid_protocol_invalid_full_document(self):
        score,r=reward(fixture(),'  if (x) {\n>>>>>>> UPDATED');self.assertEqual(score,0);self.assertTrue(r['protocol_valid']);self.assertFalse(r['applied_r_parse'])
    def test_incomplete_before_valid_finish(self):
        score,r=reward(fixture(old='',target='  x + 2\n}',suffix=''),'  x + 2\n}\n>>>>>>> UPDATED');self.assertEqual(score,1.2)
    def test_delete(self):
        score,r=reward(fixture(operation='delete',target=''),'\n>>>>>>> UPDATED');self.assertEqual(score,1.2)
    def test_crlf(self):
        self.assertEqual(reward(fixture(eol='\r\n'),'  x + 2\n>>>>>>> UPDATED')[0],1.2)
    def test_unicode_utf16(self):
        self.assertEqual(reward(fixture(old='  "🦊"',target='  "🦉"'),'  "🦉"\n>>>>>>> UPDATED')[0],1.2)
    def test_protocol_failures_do_not_parse_prediction(self):
        for text,ids in [('  x + 2\n>>>>>>> UPDATED',[100]),('  x + 2\n>>>>>>> UPDATED',[100,130073]),('  x + 2\n>>>>>>> UPDATED',[0,1]),('  x + 2\n>>>>>>> UPDATED',[100,1,100,1]),('bad terminal',[100,1]),('  x + 2\n>>>>>>> UPDATED',[100]*192+[1])]:
            with self.subTest(ids_len=len(ids),text=text):
                calls=[]
                def p(doc):calls.append(doc);return PARSE(doc)
                score,r=reward(fixture(),text,ids,parser=p);self.assertEqual(score,0);self.assertEqual(len(calls),1);self.assertIsNone(r['applied_r_parse'])
    def test_invalid_gold_stops_admission(self):
        with self.assertRaises(BindingError):bind_train_records([fixture(target='  if (x) {')],PARSE)
    def test_missing_document_stops_admission(self):
        record=fixture();record.source_identity={}
        with self.assertRaises(BindingError):bind_train_records([record],PARSE)
    def test_document_hash_mismatch(self):
        record=fixture();record.source_identity['source_provenance']['selection_source']['document_text']+=' '
        with self.assertRaises(BindingError):bind_train_records([record],PARSE)
    def test_capture_version_mismatch(self):
        record=fixture();record.capture.version=2
        with self.assertRaises(BindingError):bind_train_records([record],PARSE)
    def test_nontrain_and_duplicate_rejected(self):
        record=fixture();record.row['split']='dev'
        with self.assertRaises(BindingError):bind_train_records([record],PARSE)
        record=fixture()
        with self.assertRaises(BindingError):bind_train_records([record,record],PARSE)
    def test_wrong_region_selected_rejected(self):
        record=fixture();mapping=record.context.to_dict();mapping['region_old']=['  y + 9'];record.context=PromptContext.from_mapping(mapping)
        with self.assertRaises(BindingError):bind_train_records([record],PARSE)
    def test_context_target_id_drift_rejected(self):
        record=fixture();r=AppliedDocumentExactReward(bindings=bind_train_records([record],PARSE),parse_r=PARSE,decoder=lambda _ids,**kw:'  x + 2\n>>>>>>> UPDATED')
        for field,value in [('row_id','unknown'),('target_body_text','  x + 3'),('package_id','other')]:
            args={'context_value':record.context,'target_operation':'replace','target_body_text':'  x + 2','generated_ids':[100,1],'row_id':record.row['id'],'family':record.row['family'],'package_id':record.row['package_id']};args[field]=value
            with self.subTest(field=field),self.assertRaises(BindingError):r.score_one(**args)
    def test_actual_batch_interface_and_sink(self):
        record=fixture();events=[];r=AppliedDocumentExactReward(bindings=bind_train_records([record],PARSE),parse_r=PARSE,decoder=lambda _ids,**kw:'  x + 2\n>>>>>>> UPDATED',event_sink=events.append)
        args={'prompts':[{'text':'SOURCE ONLY','ids':[0,100]}],'completions':['unused'],'completion_ids':[[100,1]],'context':[record.context.to_dict()],'target_operation':['replace'],'target_body_text':['  x + 2'],'id':[record.row['id']],'family':[record.row['family']],'package_id':[record.row['package_id']],'split':['train']}
        self.assertEqual(r(**args),[1.2]);self.assertEqual(events,r.last_records);self.assertEqual(events[0]['reward_policy'],'r2_applied_buffer_parse_and_exact_v1')
        args['split']=['dev']
        with self.assertRaises(BindingError):r(**args)
    def test_no_R_execution(self):
        # Side-effect-looking R is parsed only; no file is created.
        path=Path(__file__).with_name('MUST_NOT_EXIST');self.assertFalse(path.exists());self.assertTrue(PARSE('writeLines("bad", "'+str(path)+'")'));self.assertFalse(path.exists())
    def test_legacy_implementation_pin_rejected(self):
        record=fixture();bindings=bind_train_records([record],PARSE)
        with mock.patch('applied_r_reward.inspect.getsource',return_value='changed implementation'):
            with self.assertRaises(BindingError):AppliedDocumentExactReward(bindings=bindings,parse_r=PARSE,decoder=lambda ids,**kw:'')

    def test_partial_buffer_noop_needs_separate_policy(self):
        with self.assertRaises(BindingError):bind_train_records([fixture(operation='no_op',target='[NO_EDIT]',suffix='')],PARSE)

    def test_surrogate_midpoint_rejected(self):
        record=fixture(old='  \"🦊\"',target='  \"🦉\"');document=record.source_identity['source_provenance']['selection_source']['document_text']
        with self.assertRaises(BindingError):offset(document,SimpleNamespace(line=1,character=4))

    def test_document_ceiling_rejected(self):
        with mock.patch('applied_r_reward.MAX_DOCUMENT_BYTES',10):
            with self.assertRaises(BindingError):bind_train_records([fixture()],PARSE)

    def test_absent_or_changed_objective_not_admitted(self):
        for policy in [None,{},dict(ADMISSION_POLICY,other_reward=0.1)]:
            with self.subTest(policy=policy),self.assertRaises(BindingError):prepare_reward_bindings([fixture()],policy,PARSE)
        self.assertEqual(len(prepare_reward_bindings([fixture()],dict(ADMISSION_POLICY),PARSE)),1)

    def test_budget_is_inclusive_and_valid_EOS_at_192_is_allowed(self):
        score,r=reward(fixture(),'  x + 2\n>>>>>>> UPDATED',[100]*191+[1]);self.assertEqual(score,1.2);self.assertTrue(r['canonical_eos']);self.assertTrue(r['cap_hit'])
        score,r=reward(fixture(),'  x + 2\n>>>>>>> UPDATED',[100]*192);self.assertEqual(score,0);self.assertEqual(r['failure'],'missing_canonical_eos')

    def test_no_model_framework_import(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
