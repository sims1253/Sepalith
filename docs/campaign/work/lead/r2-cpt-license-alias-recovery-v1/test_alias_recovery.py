from __future__ import annotations
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
mat=load('alias_recovery_test_materializer',HERE/'source/materialize_alias_recovery.py')
aliases=load('alias_recovery_test_aliases',HERE/'source/license_aliases.py')
raw=load('alias_recovery_test_raw',HERE/'source/raw_cpt_broader.py')
validator=load('alias_recovery_test_validator',HERE/'source/campaign_cpt_data.py')

class AliasRecoveryTests(unittest.TestCase):
 def test_exact_source_pins(self):
  for path,digest in mat.PINS.items():self.assertEqual(mat.sha(path),digest)

 def test_only_existing_family_aliases_recover(self):
  for original in mat.RECOVERABLE:
   normalized=mat.recoverable_license(original,aliases,raw)
   self.assertIsNotNone(normalized);self.assertFalse(raw.allowed_license(original));self.assertTrue(raw.allowed_license(normalized));self.assertNotEqual(original,normalized)
  for value in ('EUPL','CC BY-NC 4.0','file LICENSE','BSL-1.0','BSD_3_clause_custom','MyBSD_2_clause','MIT + file LICENSE'):
   self.assertIsNone(mat.recoverable_license(value,aliases,raw))

 def test_source_and_description_provenance_preserves_original_license(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=root/'x.R';src.write_text('x <- 1\n');desc=root/'DESCRIPTION';license='BSD_2_clause + file LICENSE';desc.write_text(f'Package: pkg\nLicense: {license}\nLicense_is_FOSS: yes\nLicense_restricts_use: no\n')
   st=src.stat();meta={'package':'pkg','group_id':'g-train','path':str(src),'license':license,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev,'description_path':str(desc),'description_sha256':hashlib.sha256(desc.read_bytes()).hexdigest()}
   text,failure,data,actual=mat.read_candidate({'package':'pkg','group_id':'g-train','path':str(src),'license':license}, {str(src):meta})
   self.assertEqual(text,'x <- 1\n');self.assertIsNone(failure);self.assertEqual(data,src.read_bytes());self.assertEqual(actual['license'],license)

 def test_package_restrictions_fail_closed(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=root/'x.R';src.write_text('x <- 1\n');desc=root/'DESCRIPTION';license='BSD_2_clause + file LICENSE';desc.write_text(f'Package: pkg\nLicense: {license}\nLicense_restricts_use: yes\n')
   st=src.stat();meta={'package':'pkg','group_id':'g','path':str(src),'license':license,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev,'description_path':str(desc),'description_sha256':hashlib.sha256(desc.read_bytes()).hexdigest()}
   with self.assertRaisesRegex(ValueError,'restrictions prohibit'):mat.read_candidate({'package':'pkg','group_id':'g','path':str(src),'license':license},{str(src):meta})

 def test_frozen_chunks_validate_as_schema1_complete_document(self):
  tokens=list(range(10,5000));doc='a'*64;rows=[]
  for index,chunk in enumerate(raw.chunks(tokens,2048)):
   rows.append({'schema':1,'row_id':f'{doc}:{index}','document_id':doc,'package':'pkg','group_id':'g-train','cpt_partition':'cpt_train','source_path':'x.R','source_sha256':doc,'chunk_index':index,**chunk})
  result=validator.validate_materialized_rows(rows,max_sequence_tokens=2048,require_complete_documents=True)
  self.assertEqual(result['payload_tokens'],len(tokens));self.assertEqual(result['documents'],1);self.assertEqual(result['rows'],3)

 def test_resume_reconciles_existing_documents_into_dedup(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);doc='b'*64;documents=folder/'documents.jsonl';documents.write_text(json.dumps({'sha256':doc})+'\n')
   empty=folder/'empty';empty.write_text('')
   artifacts={p.name:{'bytes':p.stat().st_size,'sha256':mat.sha(p)} for p in (documents,empty)}
   (folder/'receipt.json').write_text(json.dumps({'artifacts':artifacts}))
   seen=set();recovered=set();conflicts=[];mat.reconcile_existing(folder,seen,recovered,conflicts);self.assertEqual(seen,{doc});self.assertEqual(conflicts,[])
   # A fresh main/base snapshot containing the recovery document records a
   # terminal conflict without losing the ability to scan later groups.
   seen={doc};recovered=set();conflicts=[];mat.reconcile_existing(folder,seen,recovered,conflicts)
   self.assertEqual(conflicts,[{'document_id':doc,'recovery_group':folder.name,'reason':'now_present_in_captured_main_or_base'}])

 def test_split_and_protected_partition_rejected(self):
  registry={'pkg':('g-train','train_group'),'held':('g-held','train_group')};partitions={'g-train':'cpt_train','g-held':'cpt_validation'}
  mat.verify_split('g-train','pkg',registry,partitions)
  with self.assertRaisesRegex(ValueError,'protected'):mat.verify_split('g-held','held',registry,partitions)
  with self.assertRaisesRegex(ValueError,'not in exact TRAIN'):mat.verify_split('g-x','pkg',registry,partitions)

if __name__=='__main__':unittest.main()
