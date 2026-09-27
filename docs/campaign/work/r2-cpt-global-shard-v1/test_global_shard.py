import os,sys,unittest
from pathlib import Path
sys.dont_write_bytecode=True;os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_global_shard as b
TRAIN=next('synthetic-'+str(i) for i in range(100) if b.c.partition('synthetic-'+str(i))=='cpt_train')
VAL=next('synthetic-'+str(i) for i in range(100) if b.c.partition('synthetic-'+str(i))=='cpt_validation')
class Tests(unittest.TestCase):
 def test_registry_train_positive(self):
  e={'name':'synthetic','group_id':TRAIN,'split':'train_group'};self.assertEqual(b.admit_group(e,{'synthetic':(TRAIN,'train_group')},{TRAIN:'cpt_train'},set())[0],'synthetic')
 def test_reserved_validation_rejected(self):
  e={'name':'synthetic','group_id':VAL,'split':'train_group'}
  with self.assertRaisesRegex(ValueError,'validation'):b.admit_group(e,{'synthetic':(VAL,'train_group')},{VAL:'cpt_validation'},set())
 def test_wrong_registry_split_rejected(self):
  e={'name':'synthetic','group_id':TRAIN,'split':'train_group'}
  with self.assertRaisesRegex(ValueError,'registry'):b.admit_group(e,{'synthetic':(TRAIN,'not_train')},{TRAIN:'cpt_train'},set())
 def test_emitted_group_rejected(self):
  e={'name':'synthetic','group_id':TRAIN,'split':'train_group'}
  with self.assertRaisesRegex(ValueError,'already emitted'):b.admit_group(e,{'synthetic':(TRAIN,'train_group')},{TRAIN:'cpt_train'},{TRAIN})
 def test_prior_broad_document_rejected(self):
  self.assertEqual(b.reject_reason({'sha256':'x','sha1':'y'},set(),{'x'},set(),set()),'already_emitted_broad_document_duplicate')
 def test_reserved_document_cross_package_duplicate_rejected(self):
  self.assertEqual(b.reject_reason({'sha256':'x','sha1':'y'},set(),set(),{'x'},set()),'reserved_CPT_validation_document_duplicate')
 def test_nontrain_sha1_or_git_blob_rejected(self):
  for key in ['sha1','git_blob_sha1']:
   self.assertEqual(b.reject_reason({'sha256':'x',key:'protected'},{'protected'},set(),set(),set()),'known_nontrain_parent_hash_match')
 def test_within_shard_duplicate_rejected(self):
  self.assertEqual(b.reject_reason({'sha256':'x'},set(),set(),set(),{'x'}),'duplicate_within_global_shard')
 def test_complete_chunk_tail_and_exact_loss_denominator(self):
  for n in [1,2046,2047,4092,4093,6001]:
   source=[2+(i%100) for i in range(n)];rows=list(b.c.chunks(source,2048));actual=[]
   for i,r in enumerate(rows):
    self.assertEqual(r['input_ids'][0],0);self.assertEqual(r['input_ids'][-1],1);self.assertLessEqual(len(r['input_ids']),2048)
    actual.extend(r['input_ids'][1+r['overlap_context_tokens']:-1]);self.assertEqual(r['labels'][-1],1 if i==len(rows)-1 else -100)
   self.assertEqual(actual,source);self.assertEqual(sum(r['supervised_tokens'] for r in rows),n+1)
 def test_no_framework_import(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
