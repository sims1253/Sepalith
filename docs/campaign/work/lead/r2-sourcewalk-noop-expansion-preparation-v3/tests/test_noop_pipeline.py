import copy,hashlib,json,os,sys,tempfile,unittest
from unittest import mock
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'source'))
import prepare_noop_inputs as prep
import materialize_noops as mat
RID='43b24d15b32aae89fdf72245';SHARD=10
PACK=Path(f'/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-{SHARD:04d}/structured-materialization-v1/candidate-packets.jsonl')
LEDGER=Path(f'/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01/shards/shard-{SHARD:04d}/ledger.jsonl')
def one(path):
 with path.open()as handle:
  for line in handle:
   x=json.loads(line)
   if (x.get('row_id')or x.get('row_ref',{}).get('row_id'))==RID:return x
 raise AssertionError(RID)
class Test(unittest.TestCase):
 def test_duplicate_geometry_different_workspace_is_retained_and_counted(self):
  base={'preedit_sha256':'a'*64,'cursor':{'line':7,'character':0},'path':'R/same.R','absolute_document_path':'/pkg-a/R/same.R'}
  a=dict(base,row_id='row-a',workspace_root='/pkg-a')
  b=dict(base,row_id='row-b',workspace_root='/pkg-b',absolute_document_path='/pkg-b/R/same.R')
  groups=prep.validate_prediction_set([a,b])
  self.assertEqual(len(groups),1);self.assertEqual(groups[0]['count'],2);self.assertEqual(groups[0]['row_ids'],['row-a','row-b']);self.assertEqual(groups[0]['workspace_roots'],['/pkg-a','/pkg-b']);self.assertEqual(groups[0]['disposition'],'retain_all_until_provider_prompt_target_dedup')
 def test_duplicate_row_identity_still_fails(self):
  a={'row_id':'same','preedit_sha256':'a'*64,'cursor':{'line':1,'character':0},'path':'R/a.R','workspace_root':'/a','absolute_document_path':'/a/R/a.R'}
  b=dict(a,workspace_root='/b',absolute_document_path='/b/R/a.R')
  with self.assertRaisesRegex(ValueError,'duplicate row identity'):prep.validate_prediction_set([a,b])
 def test_real_provenance_reconstructs_full_source_and_global_cursor(self):
  packet=one(PACK);provenance=one(LEDGER);prediction,sidecar=prep.recover(packet,provenance);source=Path(packet['validation']['source_path']);self.assertEqual(prediction['preedit_text'].encode(),source.read_bytes());self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),prediction['preedit_sha256']);self.assertGreater(prediction['cursor']['line'],packet['result']['context']['replacement_range']['start']['line']);self.assertNotIn('target',prediction);self.assertFalse(prediction['selection_target_or_gold_used']);self.assertEqual(sidecar['target_operation'],'no_op');self.assertEqual(sidecar['target_body_lines'],[])
 def test_bad_or_ambiguous_geometry_fails(self):
  packet=one(PACK);provenance=one(LEDGER);bad=copy.deepcopy(provenance);bad['noop_geometry']['window_occurrences']=2
  with self.assertRaisesRegex(prep.RowValidationError,'provenance geometry'):prep.recover(packet,bad)
  bad=copy.deepcopy(packet);bad['result']['context']['replacement_range']['end']['line']+=1
  with self.assertRaisesRegex(prep.RowValidationError,'zero-width'):prep.recover(bad,provenance)
 def test_geometry_failure_becomes_explicit_row_hold(self):
  packet=one(PACK);provenance=one(LEDGER);bad=copy.deepcopy(provenance);bad['noop_geometry']['window_occurrences']=2
  item={'row_id':RID,'shard':SHARD,'status':prep.SUPPORTED}
  prediction,sidecar,hold=prep.candidate_output(item,packet,bad)
  self.assertIsNone(prediction);self.assertIsNone(sidecar);self.assertEqual(hold['status'],'hold');self.assertIn('provenance geometry',hold['reason']);self.assertFalse(hold['silent_drop'])
 def test_missing_source_is_infrastructure_failure_not_hold(self):
  packet=copy.deepcopy(one(PACK));provenance=one(LEDGER);packet['validation']['source_path']='/definitely/missing/noop-source.R';item={'row_id':RID,'shard':SHARD,'status':prep.SUPPORTED}
  with self.assertRaises(FileNotFoundError):prep.candidate_output(item,packet,provenance)
 def test_resource_failure_is_not_converted_to_hold(self):
  packet=one(PACK);provenance=one(LEDGER);item={'row_id':RID,'shard':SHARD,'status':prep.SUPPORTED}
  with mock.patch.object(prep,'recover',side_effect=MemoryError('bounded control')):
   with self.assertRaises(MemoryError):prep.candidate_output(item,packet,provenance)
 def test_main_missing_source_aborts_and_removes_temporary_output(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);replay=root/'replay/shard-0010';pack_root=root/'packets/shard-0010/structured-materialization-v1';replay.mkdir(parents=True);pack_root.mkdir(parents=True)
   provenance=one(LEDGER);packet=copy.deepcopy(one(PACK));packet['validation']['source_path']=str(root/'missing-source.R')
   ledger=replay/'ledger.jsonl';packets=pack_root/'candidate-packets.jsonl';ledger.write_text(json.dumps(provenance)+'\n');packets.write_text(json.dumps(packet)+'\n')
   receipt={'status':'complete','shard':SHARD,'rows':1,'outputs':[{'path':str(ledger),'rows':1,'sha256':prep.sha(ledger)}],'binding':{'candidate_packets_sha256':prep.sha(packets)}}
   receipt_path=replay/'receipt.json';receipt_path.write_text(json.dumps(receipt)+'\n')
   candidate=root/'candidate.jsonl';candidate.write_text(json.dumps({'row_id':RID,'shard':SHARD,'status':prep.SUPPORTED})+'\n')
   coverage=root/'coverage.json';coverage.write_text(json.dumps({'partial':True,'completed_shards':1,'pending_shards':[x for x in range(41) if x!=SHARD],'not_in_existing_pool_by_id':1,'receipts':[{'shard':SHARD,'sha256':prep.sha(receipt_path)}]})+'\n')
   output=root/'prepared';argv=['prepare_noop_inputs.py','--coverage',str(coverage),'--coverage-sha256',prep.sha(coverage),'--candidate-ids',str(candidate),'--candidate-ids-sha256',prep.sha(candidate),'--replay-root',str(root/'replay'),'--packet-root',str(root/'packets'),'--output',str(output)]
   with mock.patch.object(sys,'argv',argv):
    with self.assertRaises(FileNotFoundError):prep.main()
   self.assertFalse(output.exists());self.assertEqual(list(root.glob('.prepared.*')),[])
 def test_candidate_status_must_match_pinned_ledger(self):
  packet=one(PACK);provenance=one(LEDGER);item={'row_id':RID,'shard':SHARD,'status':'hold_independent_provenance_failure'}
  with self.assertRaisesRegex(ValueError,'status differs'):prep.candidate_output(item,packet,provenance)
 def test_prompt_projection_uses_source_geometry_and_not_metadata(self):
  path=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/train-token-rows.jsonl')
  with path.open()as handle:row=next(json.loads(x)for x in handle if json.loads(x)['family']=='no_op')
  a=mat.prompt_source_projection(row['prompt_text']);self.assertEqual(len(a),64);changed=row['prompt_text'].replace('<filename>selected_references','ignored metadata\n<filename>selected_references',1);self.assertNotEqual(a,mat.prompt_source_projection(changed));self.assertEqual(mat.digest(row['target_text']),mat.digest('[NO_EDIT]\n>>>>>>> UPDATED'))
 def test_existing_union_is_exact_20191(self):
  v=json.loads((P/'existing-union-manifest.json').read_text());self.assertEqual(v['rows'],20191);self.assertEqual(sum(x['rows']for x in v['cohorts']),20191);self.assertEqual(len(v['cohorts']),5)
 def test_partial_scope_never_claims_global_closure(self):
  partial=json.loads(Path('docs/campaign/work/lead/r2-noop-pool-coverage-root-v1/result.json').read_text());self.assertFalse(prep.is_full_closure(partial));full={'partial':False,'completed_shards':41,'pending_shards':[],'receipts':[{}]*41};self.assertTrue(prep.is_full_closure(full));full['pending_shards']=[40];self.assertFalse(prep.is_full_closure(full))
 def test_actual_noop_row_uses_standard_target_and_strict_protocol(self):
  packet=one(PACK);ctx=mat.protocol.PromptContext.from_mapping(packet['result']['context']);tok=mat.Tok(Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json'));row=mat.protocol.build_training_row(ctx,operation='no_op',region_new=[],tokenizer=tok,row_id=RID,family='no_op',package_id='adagio',split='train');mat.validate_row(row,tok);self.assertEqual(row['target_text'],'[NO_EDIT]\n>>>>>>> UPDATED');self.assertEqual(row['target_body_text'],'[NO_EDIT]')
if __name__=='__main__':unittest.main(verbosity=2)
