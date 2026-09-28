import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P));import audit_expanded_union as A
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,rows):p.write_text(''.join(json.dumps(x)+'\n' for x in rows));return {'path':str(p),'sha256':sha(p),'rows':len(rows)}
class TestV2(unittest.TestCase):
 def fixture(self,d):
  tokens=dump(d/'tokens.jsonl',[{'id':'a','prompt_text':'p1','target_text':'t1','split':'train'},{'id':'b','prompt_text':'p2','target_text':'t2','split':'train'}]);provs=dump(d/'provenance.jsonl',[{'row_id':'a'},{'row_id':'b'}]);ledger=dump(d/'ledger.jsonl',[{'row_id':'a','status':'candidate'},{'row_id':'b','status':'candidate'},{'row_id':'h','status':'hold'},{'row_id':'z','status':'duplicate'}])
  manifest={'outputs':{Path(x['path']).name:{'path':Path(x['path']).name,'sha256':x['sha256'],'rows':x['rows']} for x in (tokens,provs,ledger)}};mp=d/'manifest.json';mp.write_text(json.dumps(manifest)+'\n')
  c={'name':'x','input_rows':4,'candidate_rows':2,'terminal_manifest':{'path':str(mp),'sha256':sha(mp),'rows':1},'token_rows':[tokens],'provenance_rows':[provs],'decision_ledger':ledger,'decision_status_classes':{'candidate':['candidate'],'hold':['hold'],'exact_duplicate':['duplicate']},'partition_counts':{'candidate':2,'hold':1,'exact_duplicate':1}}
  return c
 def patched(self):old=A.INPUTS;A.INPUTS={'x':4};self.addCleanup(setattr,A,'INPUTS',old)
 def test_candidate_hold_duplicate_partition_and_unresolved_geometry(self):
  self.patched()
  with tempfile.TemporaryDirectory() as td:
   c=self.fixture(Path(td));value=A.validate(c);self.assertEqual((value[0],{k:len(v) for k,v in value[1].items()}),(2,{'candidate':2,'hold':1,'exact_duplicate':1}))
   rows,report=A.audit({'schema':A.SCHEMA,'cohorts':[c]});self.assertEqual((report['input_rows'],report['candidate_rows']),(4,2));self.assertEqual(len(rows),4);self.assertFalse(report['admission_ready']);self.assertGreater(report['missing_geometry_rows'],0)
 def test_wrong_input_accounting_rejected(self):
  self.patched()
  with tempfile.TemporaryDirectory() as td:
   c=self.fixture(Path(td));c['input_rows']=5
   with self.assertRaisesRegex(A.AuditError,'input_denominator'):A.validate(c)
 def test_substituted_file_pin_not_in_manifest_rejected(self):
  self.patched()
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);c=self.fixture(d);other=d/'other.jsonl';other.write_text(Path(c['token_rows'][0]['path']).read_text());c['token_rows'][0]={'path':str(other),'sha256':sha(other),'rows':2}
   with self.assertRaisesRegex(A.AuditError,'not_in_terminal_manifest'):A.validate(c)
 def test_wrong_partition_and_candidate_ids_rejected(self):
  self.patched()
  with tempfile.TemporaryDirectory() as td:
   c=self.fixture(Path(td));c['partition_counts']['hold']=0
   with self.assertRaisesRegex(A.AuditError,'partition_counts'):A.validate(c)
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);c=self.fixture(d);ledger=Path(c['decision_ledger']['path']);rows=[json.loads(x) for x in ledger.read_text().splitlines()];rows[0]['row_id']='other';c['decision_ledger']=dump(ledger,rows);m=Path(c['terminal_manifest']['path']);manifest=json.loads(m.read_text());manifest['outputs']['ledger.jsonl'].update(sha256=c['decision_ledger']['sha256']);m.write_text(json.dumps(manifest)+'\n');c['terminal_manifest']['sha256']=sha(m)
   with self.assertRaisesRegex(A.AuditError,'candidate_id_partition'):A.validate(c)
if __name__=='__main__':unittest.main()
